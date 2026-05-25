from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image
from tqdm import tqdm

from .constants import REGION_TYPES, TRANSCRIBED_TYPES
from .geometry import clamp_bbox
from .io import load_split
from .jsonl import read_jsonl, write_jsonl
from .postprocess import regions_from_model_json
from .schemas import PageRecord, Region


class Detector(Protocol):
    def detect(self, image_path: Path) -> list[Region]: ...


class Transcriber(Protocol):
    def transcribe(self, image_path: Path, region: Region) -> str: ...


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
    def transcribe(self, image_path: Path, region: Region) -> str:
        return ""


def _load_vision_model_and_processor(model_path: Path, load_in_4bit: bool = True):
    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install VLM extras with `pip install -e '.[vlm]'`.") from exc

    try:
        from transformers import AutoModelForImageTextToText as AutoVisionModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVisionModel

    model_id = str(model_path)
    processor_id = model_id
    cuda_available = torch.cuda.is_available()
    cuda_bf16 = cuda_available and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if cuda_bf16 else torch.float16
    quantization_config = None
    if load_in_4bit and cuda_available:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    if (model_path / "adapter_config.json").exists():
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise RuntimeError("Install VLM extras with `pip install -e '.[vlm]'`.") from exc

        peft_config = PeftConfig.from_pretrained(model_id)
        processor_id = peft_config.base_model_name_or_path
        base_model = AutoVisionModel.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype if cuda_available else torch.float32,
            device_map="auto" if cuda_available else None,
        )
        model = PeftModel.from_pretrained(base_model, model_id)
    else:
        model = AutoVisionModel.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype if cuda_available else torch.float32,
            device_map="auto" if cuda_available else None,
        )

    processor = AutoProcessor.from_pretrained(processor_id)
    model.eval()
    return torch, processor, model


def _decode_new_tokens(processor, generated, input_ids) -> str:
    generated_ids = generated[:, input_ids.shape[-1] :]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


class VisionTextGenerationTranscriber:
    def __init__(
        self,
        model_path: Path,
        prompt: str | None = None,
        load_in_4bit: bool = True,
    ):
        self.torch, self.processor, self.model = _load_vision_model_and_processor(
            model_path,
            load_in_4bit=load_in_4bit,
        )
        self.prompt = prompt or "Transcribe this Ukrainian document region exactly."

    def transcribe(self, image_path: Path, region: Region) -> str:
        with Image.open(image_path) as image:
            crop = image.crop(tuple(region.bbox)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=[crop], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=192)
        return _decode_new_tokens(self.processor, generated, inputs["input_ids"])


class VisionPageJsonDetector:
    def __init__(
        self,
        model_path: Path,
        prompt: str | None = None,
        max_new_tokens: int = 2048,
        load_in_4bit: bool = True,
    ):
        self.torch, self.processor, self.model = _load_vision_model_and_processor(
            model_path,
            load_in_4bit=load_in_4bit,
        )
        self.prompt = prompt or (
            "Return a JSON array of document regions for this page. "
            "Each item must contain bbox [x1,y1,x2,y2], type, and text. "
            "Use exact transcription. Use empty text for image and graph regions."
        )
        self.max_new_tokens = max_new_tokens

    def detect_for_page(
        self, image_path: Path, image_width: int, image_height: int
    ) -> list[Region]:
        with Image.open(image_path) as image:
            page_image = image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=[page_image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = _decode_new_tokens(self.processor, generated, inputs["input_ids"])
        return regions_from_model_json(decoded, image_width=image_width, image_height=image_height)


def sort_regions_reading_order(regions: list[Region]) -> list[Region]:
    return sorted(regions, key=lambda region: (region.bbox[1] // 40, region.bbox[0]))


def run_inference(
    test_dir: Path,
    output_jsonl: Path,
    detector: Detector | None = None,
    transcriber: Transcriber | None = None,
    page_detector: VisionPageJsonDetector | None = None,
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
    for page in tqdm(pages, desc="Inference"):
        image_path = Path(page.image_path or test_dir / page.file_name)
        regions = []
        detected_regions = (
            page_detector.detect_for_page(image_path, page.image_width, page.image_height)
            if page_detector
            else detector.detect(image_path)
        )
        for region in detected_regions:
            bbox = clamp_bbox(region.bbox, page.image_width, page.image_height)
            if bbox is None:
                continue
            region.bbox = bbox
            if not region.text and region.type in TRANSCRIBED_TYPES:
                region.text = transcriber.transcribe(image_path, region)
            elif region.type not in TRANSCRIBED_TYPES:
                region.text = ""
            regions.append(region)
        regions = sort_regions_reading_order(regions)
        rows.append(
            {
                "image": page.image_name,
                "regions": [region.to_full_dict() for region in regions],
            }
        )

    write_jsonl(output_jsonl, rows)
    return len(rows)
