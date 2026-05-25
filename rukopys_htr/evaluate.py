from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from .geometry import bbox_iou
from .io import load_predictions_jsonl, load_split
from .jsonl import read_jsonl
from .schemas import PageRecord, Region

logger = logging.getLogger(__name__)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            substitute_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, substitute_cost))
        previous = current
    return previous[-1]


def cer(pred: str, truth: str) -> float:
    denom = max(1, len(truth))
    return levenshtein(pred, truth) / denom


def _match_regions(
    predicted: list[Region],
    truth: list[Region],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predicted):
        for truth_idx, gt in enumerate(truth):
            if pred.type != gt.type:
                continue
            score = bbox_iou(pred.bbox, gt.bbox)
            if score >= iou_threshold:
                candidates.append((score, pred_idx, truth_idx))
    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, pred_idx, truth_idx in candidates:
        if pred_idx in used_pred or truth_idx in used_truth:
            continue
        used_pred.add(pred_idx)
        used_truth.add(truth_idx)
        matches.append((pred_idx, truth_idx))
    return matches


def _compute_page_metrics(
    truth_pages: dict[str, PageRecord],
    predictions: dict[str, list[Region]],
    iou_threshold: float,
) -> dict[str, float]:
    tp = fp = fn = 0
    cer_sum = 0.0
    text_matches = 0

    for image, page in truth_pages.items():
        pred_regions = predictions.get(image, [])
        matches = _match_regions(pred_regions, page.regions, iou_threshold=iou_threshold)
        tp += len(matches)
        fp += max(0, len(pred_regions) - len(matches))
        fn += max(0, len(page.regions) - len(matches))
        for pred_idx, truth_idx in matches:
            cer_sum += cer(pred_regions[pred_idx].text, page.regions[truth_idx].text)
            text_matches += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    mean_cer = cer_sum / max(1, text_matches)
    return {
        "matched_regions": float(tp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cer": mean_cer,
        "proxy_score": f1 * (1.0 - min(mean_cer, 1.0)),
    }


def _grouped_metrics(
    truth_pages: dict[str, PageRecord],
    predictions: dict[str, list[Region]],
    iou_threshold: float,
    key_fn,
) -> dict[str, dict[str, float]]:
    grouped_truth: dict[str, dict[str, PageRecord]] = defaultdict(dict)
    grouped_predictions: dict[str, dict[str, list[Region]]] = defaultdict(dict)
    for image, page in truth_pages.items():
        key = str(key_fn(page) or "unknown")
        grouped_truth[key][image] = page
        grouped_predictions[key][image] = predictions.get(image, [])

    return {
        key: _compute_page_metrics(grouped_truth[key], grouped_predictions[key], iou_threshold)
        for key in sorted(grouped_truth)
    }


def load_yolo_val_pages(curated_dir: Path) -> dict[str, PageRecord]:
    val_dir = curated_dir / "yolo" / "images" / "val"
    if not val_dir.exists():
        return {}

    val_names = {path.name for path in val_dir.glob("*") if path.is_file()}
    metadata_path = curated_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return {}

    pages: dict[str, PageRecord] = {}
    for row in read_jsonl(metadata_path):
        page = PageRecord.from_dict(row, split=str(row.get("split", "train")), base_dir=curated_dir)
        if page.image_name in val_names:
            pages[page.image_name] = page
    return pages


def evaluate_predictions(
    raw_dir: Path,
    predictions_jsonl: Path,
    iou_threshold: float = 0.5,
    *,
    split: str = "train",
    by_source: bool = False,
    by_annotation_source: bool = False,
) -> dict[str, float | dict[str, dict[str, float]]]:
    if split == "yolo_val":
        raise ValueError("Use evaluate_curated_val() for split='yolo_val'")

    truth_pages = {page.image_name: page for page in load_split(raw_dir, split)}
    predictions = load_predictions_jsonl(predictions_jsonl)
    metrics = _compute_page_metrics(truth_pages, predictions, iou_threshold)
    if by_source:
        metrics["by_source"] = _grouped_metrics(
            truth_pages,
            predictions,
            iou_threshold,
            key_fn=lambda page: page.source,
        )
    if by_annotation_source:
        metrics["by_annotation_source"] = _grouped_metrics(
            truth_pages,
            predictions,
            iou_threshold,
            key_fn=lambda page: page.annotation_source,
        )
    return metrics


def evaluate_curated_val(
    curated_dir: Path,
    predictions_jsonl: Path,
    iou_threshold: float = 0.5,
    *,
    by_source: bool = False,
    by_annotation_source: bool = False,
) -> dict[str, float | dict[str, dict[str, float]]]:
    truth_pages = load_yolo_val_pages(curated_dir)
    if not truth_pages:
        logger.warning("No YOLO val pages found under %s", curated_dir)
        return {
            "matched_regions": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "cer": 0.0,
            "proxy_score": 0.0,
        }

    predictions = load_predictions_jsonl(predictions_jsonl)
    metrics = _compute_page_metrics(truth_pages, predictions, iou_threshold)
    if by_source:
        metrics["by_source"] = _grouped_metrics(
            truth_pages,
            predictions,
            iou_threshold,
            key_fn=lambda page: page.source,
        )
    if by_annotation_source:
        metrics["by_annotation_source"] = _grouped_metrics(
            truth_pages,
            predictions,
            iou_threshold,
            key_fn=lambda page: page.annotation_source,
        )
    return metrics
