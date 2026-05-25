from __future__ import annotations

from pathlib import Path

from .geometry import bbox_iou
from .io import load_predictions_jsonl, load_split
from .schemas import Region


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


def evaluate_predictions(
    raw_dir: Path,
    predictions_jsonl: Path,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    truth_pages = {page.image_name: page for page in load_split(raw_dir, "train")}
    predictions = load_predictions_jsonl(predictions_jsonl)

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
