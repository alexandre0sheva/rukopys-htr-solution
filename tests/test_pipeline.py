from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from rukopys_htr.curate import curate_dataset
from rukopys_htr.evaluate import evaluate_predictions
from rukopys_htr.infer import EmptyDetector, EmptyTranscriber, run_inference
from rukopys_htr.io import load_predictions_jsonl, write_submission
from rukopys_htr.jsonl import write_jsonl
from rukopys_htr.schemas import PageRecord, Region


def _make_raw_dataset(root: Path) -> None:
    for split in ["train", "test"]:
        (root / split / "images").mkdir(parents=True)

    Image.new("RGB", (200, 100), "white").save(root / "train" / "images" / "train-1.jpg")
    Image.new("RGB", (200, 100), "white").save(root / "test" / "images" / "test-1.jpg")

    write_jsonl(
        root / "train" / "metadata.jsonl",
        [
            {
                "file_name": "images/train-1.jpg",
                "image_width": 200,
                "image_height": 100,
                "source": "dictation",
                "annotation_source": "annotator",
                "year": 2024,
                "regions": [
                    {
                        "bbox": [10, 20, 110, 50],
                        "type": "handwritten",
                        "language": "uk",
                        "legibility": "legible",
                        "text": "Тест",
                    }
                ],
            }
        ],
    )
    write_jsonl(
        root / "test" / "metadata.jsonl",
        [
            {
                "file_name": "images/test-1.jpg",
                "image_width": 200,
                "image_height": 100,
                "source": "dictation",
            }
        ],
    )
    with (root / "sample_submission.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "regions"])
        writer.writeheader()
        writer.writerow({"image": "test-1.jpg", "regions": "[]"})


def test_curate_dataset_exports_core_artifacts(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    curated = tmp_path / "curated"
    _make_raw_dataset(raw)

    stats = curate_dataset(raw, curated, crop_images=True)

    assert stats["pages"] == 2
    assert stats["regions"] == 1
    assert (curated / "metadata.jsonl").exists()
    assert (curated / "regions.jsonl").exists()
    assert (curated / "vlm_sft.jsonl").exists()
    assert (curated / "page_sft.jsonl").exists()
    assert (curated / "yolo" / "data.yaml").exists()
    assert list((curated / "crops" / "train").glob("*.jpg"))

    page_rows = [
        json.loads(line)
        for line in (curated / "page_sft.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert page_rows[0]["task"] == "page_to_regions_json"
    assert page_rows[0]["answer"][0]["text"] == "Тест"


def test_empty_inference_and_submission(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_raw_dataset(raw)
    predictions = tmp_path / "predictions.jsonl"
    submission = tmp_path / "submission.csv"

    count = run_inference(
        raw / "test",
        predictions,
        detector=EmptyDetector(),
        transcriber=EmptyTranscriber(),
    )
    assert count == 1

    write_submission(
        load_predictions_jsonl(predictions),
        raw / "sample_submission.csv",
        submission,
    )
    rows = list(csv.DictReader(submission.open("r", encoding="utf-8")))
    assert rows[0]["image"] == "test-1.jpg"
    assert json.loads(rows[0]["regions"]) == []


def test_page_vlm_inference_batches_pages(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_raw_dataset(raw)
    Image.new("RGB", (200, 100), "white").save(raw / "test" / "images" / "test-2.jpg")
    write_jsonl(
        raw / "test" / "metadata.jsonl",
        [
            {
                "file_name": "images/test-1.jpg",
                "image_width": 200,
                "image_height": 100,
                "source": "dictation",
            },
            {
                "file_name": "images/test-2.jpg",
                "image_width": 200,
                "image_height": 100,
                "source": "dictation",
            },
        ],
    )
    predictions = tmp_path / "page_predictions.jsonl"

    class FakePageDetector:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def detect_batch_for_pages(
            self,
            pages: list[tuple[Path, PageRecord]],
        ) -> list[list[Region]]:
            self.batch_sizes.append(len(pages))
            return [
                [Region(bbox=[10, 20, 110, 50], type="handwritten", text=page.image_name)]
                for _, page in pages
            ]

    page_detector = FakePageDetector()
    count = run_inference(
        raw / "test",
        predictions,
        page_detector=page_detector,  # type: ignore[arg-type]
        batch_size=2,
    )

    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert count == 2
    assert page_detector.batch_sizes == [2]
    assert [row["image"] for row in rows] == ["test-1.jpg", "test-2.jpg"]
    assert rows[1]["regions"][0]["text"] == "test-2.jpg"


def test_proxy_evaluation(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_raw_dataset(raw)
    predictions = tmp_path / "train_predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "image": "train-1.jpg",
                "regions": [
                    Region(bbox=[10, 20, 110, 50], type="handwritten", text="Тест").to_full_dict()
                ],
            }
        ],
    )

    metrics = evaluate_predictions(raw, predictions)
    assert metrics["f1"] == 1.0
    assert metrics["cer"] == 0.0


def test_quality_weight_filters_training_rows(tmp_path: Path) -> None:
    from rukopys_htr.jsonl import write_jsonl
    from rukopys_htr.train_vlm import JsonlVisionSFTDataset

    jsonl_path = tmp_path / "vlm_sft.jsonl"
    write_jsonl(
        jsonl_path,
        [
            {"image": "a.jpg", "answer": "A", "quality_weight": 1.0},
            {"image": "b.jpg", "answer": "B", "quality_weight": 0.35},
        ],
    )
    dataset = JsonlVisionSFTDataset(jsonl_path, max_length=128, min_quality_weight=0.75)
    assert len(dataset) == 1
    assert dataset.rows[0]["answer"] == "A"
