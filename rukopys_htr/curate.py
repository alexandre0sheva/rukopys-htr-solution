from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from .constants import QUALITY_WEIGHTS, REGION_TYPES, TRANSCRIBED_TYPES
from .geometry import clamp_bbox, yolo_bbox
from .io import load_split
from .jsonl import write_jsonl
from .schemas import PageRecord, Region


def _quality_weight(annotation_source: str | None) -> float:
    return QUALITY_WEIGHTS.get(annotation_source or "", 0.5)


def _copy_image(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return True


def _clean_region(region: Region, page: PageRecord) -> Region | None:
    bbox = clamp_bbox(region.bbox, page.image_width, page.image_height)
    if bbox is None:
        return None
    region_type = region.type if region.type in REGION_TYPES else "handwritten"
    text = region.text or ""
    return Region(
        bbox=bbox,
        type=region_type,
        text=text,
        language=region.language or "uk",
        legibility=region.legibility or "legible",
        confidence=region.confidence,
    )


def _load_pages(raw_dir: Path, include_silver: bool, max_silver: int | None) -> list[PageRecord]:
    pages = load_split(raw_dir, "train")
    if include_silver:
        silver = load_split(raw_dir, "silver")
        if max_silver is not None:
            silver = silver[:max_silver]
        pages.extend(silver)
    pages.extend(load_split(raw_dir, "test"))
    return pages


def _assign_yolo_split(pages: list[PageRecord], val_fraction: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    train_pages = [page for page in pages if page.split in {"train", "silver"} and page.regions]
    by_source: dict[str, list[PageRecord]] = {}
    for page in train_pages:
        by_source.setdefault(page.source or "unknown", []).append(page)

    assignments: dict[str, str] = {}
    for source_pages in by_source.values():
        source_pages = list(source_pages)
        rng.shuffle(source_pages)
        val_count = max(1, round(len(source_pages) * val_fraction)) if len(source_pages) > 1 else 0
        val_names = {page.file_name for page in source_pages[:val_count]}
        for page in source_pages:
            assignments[page.file_name] = "val" if page.file_name in val_names else "train"
    return assignments


def _write_yolo_artifacts(
    pages: list[PageRecord],
    output_dir: Path,
    val_fraction: float,
    seed: int,
) -> None:
    yolo_dir = output_dir / "yolo"
    assignments = _assign_yolo_split(pages, val_fraction=val_fraction, seed=seed)
    type_to_id = {name: idx for idx, name in enumerate(REGION_TYPES)}

    for page in tqdm(pages, desc="YOLO labels", leave=False):
        yolo_split = assignments.get(page.file_name)
        if yolo_split is None or not page.image_path:
            continue
        src = Path(page.image_path)
        image_dst = yolo_dir / "images" / yolo_split / page.image_name
        if not _copy_image(src, image_dst):
            continue

        label_path = yolo_dir / "labels" / yolo_split / f"{Path(page.image_name).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for region in page.regions:
            if region.type not in type_to_id:
                continue
            cx, cy, bw, bh = yolo_bbox(region.bbox, page.image_width, page.image_height)
            lines.append(f"{type_to_id[region.type]} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    data_yaml = yolo_dir / "data.yaml"
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(REGION_TYPES))
    data_yaml.write_text(
        f"path: {yolo_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names_block}\n",
        encoding="utf-8",
    )


def _crop_region(src_path: Path, bbox: list[int], dst_path: Path) -> bool:
    if not src_path.exists():
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as image:
        crop = image.crop(tuple(bbox))
        crop.save(dst_path)
    return True


def _write_dataset_card(output_dir: Path, stats: dict[str, Any]) -> None:
    card_dir = output_dir / "dataset_card"
    card_dir.mkdir(parents=True, exist_ok=True)
    type_counts = "\n".join(f"- `{k}`: {v}" for k, v in sorted(stats["region_types"].items()))
    source_counts = "\n".join(f"- `{k}`: {v}" for k, v in sorted(stats["sources"].items()))
    card = f"""---
license: cc-by-nc-sa-4.0
task_categories:
- object-detection
- image-to-text
language:
- uk
tags:
- handwriting-recognition
- htr
- ocr
- ukrainian
- document-analysis
---

# RUKOPYS Curated MVP

Curated derivative of `UkrainianCatholicUniversity/rukopys` for the Kaggle
Handwritten to Data challenge.

## Contents

- `metadata.jsonl`: normalized page records.
- `regions.jsonl`: one row per region.
- `vlm_sft.jsonl`: crop-level transcription examples.
- `page_sft.jsonl`: full-page image to structured JSON examples.
- `yolo/`: YOLO-format layout detection dataset.
- `crops/`: region crops for transcription fine-tuning, if exported.

## Stats

- Pages: {stats["pages"]}
- Regions: {stats["regions"]}
- Crop SFT examples: {stats["vlm_examples"]}
- Page SFT examples: {stats["page_sft_examples"]}

## Sources

{source_counts}

## Region Types

{type_counts}

## Notes

Quality weights are assigned by annotation source:

- `annotator`: 1.0
- `volunteer`: 0.75
- `auto`: 0.35
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")
    (card_dir / "README.md").write_text(card, encoding="utf-8")


def curate_dataset(
    raw_dir: Path,
    output_dir: Path,
    include_silver: bool = False,
    max_silver: int | None = None,
    crop_images: bool = False,
    page_sft: bool = True,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = _load_pages(raw_dir, include_silver=include_silver, max_silver=max_silver)

    normalized_pages: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    vlm_rows: list[dict[str, Any]] = []
    page_sft_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    for page in tqdm(pages, desc="Curating pages"):
        cleaned_regions = []
        for region in page.regions:
            cleaned = _clean_region(region, page)
            if cleaned:
                cleaned_regions.append(cleaned)
        page.regions = cleaned_regions
        source_counts[page.source or "unknown"] += 1
        type_counts.update(region.type for region in cleaned_regions)

        if page.image_path:
            image_dst = output_dir / "images" / page.split / page.image_name
            _copy_image(Path(page.image_path), image_dst)
            page.image_path = str(image_dst)

        quality_weight = _quality_weight(page.annotation_source)
        normalized_pages.append(page.to_dict() | {"quality_weight": quality_weight})

        for idx, region in enumerate(page.regions):
            region_id = f"{page.split}:{Path(page.image_name).stem}:{idx:04d}"
            crop_rel = None
            if crop_images and page.image_path and region.type in TRANSCRIBED_TYPES:
                crop_rel = f"crops/{page.split}/{Path(page.image_name).stem}_{idx:04d}.jpg"
                _crop_region(Path(page.image_path), region.bbox, output_dir / crop_rel)

            row = {
                "region_id": region_id,
                "image": page.image_name,
                "page_file_name": page.file_name,
                "split": page.split,
                "source": page.source,
                "annotation_source": page.annotation_source,
                "quality_weight": quality_weight,
                "bbox": region.bbox,
                "type": region.type,
                "language": region.language,
                "legibility": region.legibility,
                "text": region.text,
                "crop": crop_rel,
            }
            region_rows.append(row)

            if crop_rel and region.text and region.legibility == "legible":
                vlm_rows.append(
                    {
                        "id": region_id,
                        "image": crop_rel,
                        "task": "transcribe_region",
                        "region_type": region.type,
                        "source": page.source,
                        "quality_weight": quality_weight,
                        "prompt": (
                            "Transcribe this Ukrainian document region exactly. "
                            "Preserve punctuation, correction markers, and LaTeX where applicable."
                        ),
                        "answer": region.text,
                    }
                )

        if page_sft and page.split in {"train", "silver"} and page.image_path and page.regions:
            page_sft_rows.append(
                {
                    "id": f"{page.split}:{Path(page.image_name).stem}:page",
                    "image": f"images/{page.split}/{page.image_name}",
                    "task": "page_to_regions_json",
                    "source": page.source,
                    "annotation_source": page.annotation_source,
                    "quality_weight": quality_weight,
                    "prompt": (
                        "Return a JSON array of document regions for this page. "
                        "Each item must contain bbox [x1,y1,x2,y2], type, and text. "
                        "Use exact transcription. Use empty text for image and graph regions."
                    ),
                    "answer": [
                        region.to_submission_dict()
                        for region in sorted(
                            page.regions,
                            key=lambda item: (item.bbox[1] // 40, item.bbox[0]),
                        )
                    ],
                }
            )

    write_jsonl(output_dir / "metadata.jsonl", normalized_pages)
    write_jsonl(output_dir / "regions.jsonl", region_rows)
    write_jsonl(output_dir / "vlm_sft.jsonl", vlm_rows)
    write_jsonl(output_dir / "page_sft.jsonl", page_sft_rows)
    _write_yolo_artifacts(
        [page for page in pages if page.split != "test"],
        output_dir=output_dir,
        val_fraction=val_fraction,
        seed=seed,
    )

    stats = {
        "pages": len(normalized_pages),
        "regions": len(region_rows),
        "vlm_examples": len(vlm_rows),
        "page_sft_examples": len(page_sft_rows),
        "sources": dict(source_counts),
        "region_types": dict(type_counts),
    }
    _write_dataset_card(output_dir, stats)
    return stats
