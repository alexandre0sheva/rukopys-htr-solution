from __future__ import annotations

import csv
import json
from pathlib import Path

from .jsonl import read_jsonl
from .schemas import PageRecord, Region


def load_split(raw_dir: Path, split: str) -> list[PageRecord]:
    metadata_path = raw_dir / split / "metadata.jsonl"
    if not metadata_path.exists():
        return []
    return [
        PageRecord.from_dict(row, split=split, base_dir=raw_dir)
        for row in read_jsonl(metadata_path)
    ]


def load_all_available(raw_dir: Path, include_silver: bool = False) -> list[PageRecord]:
    pages = load_split(raw_dir, "train")
    if include_silver:
        pages.extend(load_split(raw_dir, "silver"))
    pages.extend(load_split(raw_dir, "test"))
    return pages


def load_predictions_jsonl(path: Path) -> dict[str, list[Region]]:
    rows = read_jsonl(path)
    out: dict[str, list[Region]] = {}
    for row in rows:
        image = Path(str(row["image"])).name
        out[image] = [Region.from_dict(region) for region in row.get("regions", [])]
    return out


def write_submission(
    predictions: dict[str, list[Region]],
    sample_submission: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_submission.open("r", encoding="utf-8", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        fieldnames = reader.fieldnames or ["image", "regions"]
        rows = list(reader)

    with output_path.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            image = Path(row["image"]).name
            regions = predictions.get(image, [])
            row["regions"] = json.dumps(
                [region.to_submission_dict() for region in regions],
                ensure_ascii=False,
            )
            writer.writerow(row)
