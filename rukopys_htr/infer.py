from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from PIL import Image
from tqdm import tqdm

from .constants import (
    DEFAULT_PAGE_MAX_NEW_TOKENS,
    DEFAULT_REGION_MAX_NEW_TOKENS,
    EMPTY_TEXT_TYPES,
    READING_ORDER_ROW_BAND,
    REGION_TYPES,
    TRANSCRIBED_TYPES,
)
from .ensemble import merge_page_and_detector_regions
from .geometry import clamp_bbox
from .io import load_split
from .jsonl import read_jsonl, write_jsonl
from .postprocess import regions_from_model_json
from .prompts import page_to_regions_json_prompt, transcribe_region_prompt
from .schemas import PageRecord, Region
from .vlm_loading import load_vision_model_and_processor, resolve_pixel_budget, run_vlm_generation

logger = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, image_path: Path) -> list[Region]: ...


class Transcriber(Protocol):
    def transcribe(
        self,
        image_path: Path,
        region: Region,
        page: PageRecord | None = None,
    ) -> str: ...


class EmptyDetector:
    def detect(self, image_path: Path) -> list[Region]:
        return []


class YoloDetector:
    def __init__(self, model_path: Path, confidence: float = 0.25, iou: float = 0.5):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Install detector extras with `pip install -e '.[detector]'`."
            ) from exc

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.iou = iou

    def detect(self, image_path: Path) -> list[Region]:
        results = self.model.predict(
            str(image_path),
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )
        regions: list[Region] = []
        for result in results:
            names = result.names
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = [int(round(v)) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0].item())
                default_type = REGION_TYPES[cls_id] if cls_id < len(REGION_TYPES) else "handwritten"
                region_type = names.get(cls_id, default_type)
                conf = float(box.conf[0].item())
                regions.append(
                    Region(
                        bbox=xyxy,
                        type=region_type,
                        text="",
                        confidence=conf,
                    )
                )
        return regions


class EmptyTranscriber:
    def transcribe(
        self,
        image_path: Path,
        region: Region,
        page: PageRecord | None = None,
    ) -> str:
        return ""


class VisionTextGenerationTranscriber:
    def __init__(
        self,
        model_path: Path,
        prompt: str | None = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = DEFAULT_REGION_MAX_NEW_TOKENS,
        max_pixels: int | None = None,
    ):
        self.torch, self.processor, self.model = load_vision_model_and_processor(
            model_path,
            load_in_4bit=load_in_4bit,
            max_pixels=max_pixels,
        )
        self.max_pixels, self.min_pixels = resolve_pixel_budget(self.processor, max_pixels)
        self.default_prompt = prompt
        self.max_new_tokens = max_new_tokens

    def transcribe(
        self,
        image_path: Path,
        region: Region,
        page: PageRecord | None = None,
    ) -> str:
        prompt = self.default_prompt or transcribe_region_prompt(
            source=page.source if page else None,
            region_type=region.type,
        )
        with Image.open(image_path) as image:
            crop = image.crop(tuple(region.bbox)).convert("RGB")
        return run_vlm_generation(
            self.torch,
            self.processor,
            self.model,
            crop,
            prompt,
            max_new_tokens=self.max_new_tokens,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )


class VisionPageJsonDetector:
    def __init__(
        self,
        model_path: Path,
        prompt: str | None = None,
        max_new_tokens: int = DEFAULT_PAGE_MAX_NEW_TOKENS,
        load_in_4bit: bool = True,
        max_pixels: int | None = None,
    ):
        self.torch, self.processor, self.model = load_vision_model_and_processor(
            model_path,
            load_in_4bit=load_in_4bit,
            max_pixels=max_pixels,
        )
        self.max_pixels, self.min_pixels = resolve_pixel_budget(self.processor, max_pixels)
        self.default_prompt = prompt
        self.max_new_tokens = max_new_tokens

    def detect_for_page(
        self,
        image_path: Path,
        image_width: int,
        image_height: int,
        page: PageRecord | None = None,
    ) -> list[Region]:
        prompt = self.default_prompt or page_to_regions_json_prompt(
            source=page.source if page else None,
        )
        with Image.open(image_path) as image:
            page_image = image.convert("RGB")
        regions = self._generate_page_regions(
            page_image,
            prompt,
            image_width=image_width,
            image_height=image_height,
            max_new_tokens=self.max_new_tokens,
        )
        if not regions and self.max_new_tokens < DEFAULT_PAGE_MAX_NEW_TOKENS:
            retry_tokens = min(self.max_new_tokens * 2, DEFAULT_PAGE_MAX_NEW_TOKENS)
            logger.warning(
                "Retrying page-VLM for %s with max_new_tokens=%d",
                image_path,
                retry_tokens,
            )
            regions = self._generate_page_regions(
                page_image,
                prompt,
                image_width=image_width,
                image_height=image_height,
                max_new_tokens=retry_tokens,
            )
        if not regions:
            logger.warning("Page-VLM returned no regions for %s", image_path)
        return regions

    def _generate_page_regions(
        self,
        page_image: Image.Image,
        prompt: str,
        *,
        image_width: int,
        image_height: int,
        max_new_tokens: int,
    ) -> list[Region]:
        decoded = run_vlm_generation(
            self.torch,
            self.processor,
            self.model,
            page_image,
            prompt,
            max_new_tokens=max_new_tokens,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )
        return regions_from_model_json(
            decoded,
            image_width=image_width,
            image_height=image_height,
        )


def sort_regions_reading_order(regions: list[Region]) -> list[Region]:
    return sorted(
        regions,
        key=lambda region: (region.bbox[1] // READING_ORDER_ROW_BAND, region.bbox[0]),
    )


def _finalize_regions(
    detected_regions: list[Region],
    page: PageRecord,
    image_path: Path,
    transcriber: Transcriber,
) -> list[Region]:
    regions: list[Region] = []
    for region in detected_regions:
        bbox = clamp_bbox(region.bbox, page.image_width, page.image_height)
        if bbox is None:
            continue
        region.bbox = bbox
        if region.type in EMPTY_TEXT_TYPES:
            region.text = ""
        elif not region.text and region.type in TRANSCRIBED_TYPES:
            region.text = transcriber.transcribe(image_path, region, page=page)
        regions.append(region)
    return sort_regions_reading_order(regions)


def run_inference(
    test_dir: Path,
    output_jsonl: Path,
    detector: Detector | None = None,
    transcriber: Transcriber | None = None,
    page_detector: VisionPageJsonDetector | None = None,
    *,
    ensemble: bool = False,
    ensemble_iou_threshold: float = 0.5,
) -> int:
    detector = detector or EmptyDetector()
    transcriber = transcriber or EmptyTranscriber()

    raw_root = test_dir.parent if test_dir.name == "test" else test_dir
    pages = load_split(raw_root, "test")
    if not pages and (test_dir / "metadata.jsonl").exists():
        pages = [
            PageRecord.from_dict(row, split="test", base_dir=raw_root)
            for row in read_jsonl(test_dir / "metadata.jsonl")
        ]

    rows = []
    json_failures = 0
    for page in tqdm(pages, desc="Inference"):
        image_path = Path(page.image_path or test_dir / page.file_name)
        if ensemble and page_detector is not None:
            page_regions = page_detector.detect_for_page(
                image_path,
                page.image_width,
                page.image_height,
                page=page,
            )
            if not page_regions:
                json_failures += 1
            detector_regions = detector.detect(image_path)
            detected_regions = merge_page_and_detector_regions(
                page_regions,
                detector_regions,
                iou_threshold=ensemble_iou_threshold,
            )
        elif page_detector is not None:
            detected_regions = page_detector.detect_for_page(
                image_path,
                page.image_width,
                page.image_height,
                page=page,
            )
            if not detected_regions:
                json_failures += 1
        else:
            detected_regions = detector.detect(image_path)

        regions = _finalize_regions(detected_regions, page, image_path, transcriber)
        rows.append(
            {
                "image": page.image_name,
                "regions": [region.to_full_dict() for region in regions],
            }
        )

    if json_failures:
        logger.warning("Page-VLM produced empty regions on %d pages", json_failures)

    write_jsonl(output_jsonl, rows)
    return len(rows)
