from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from .constants import EMPTY_TEXT_TYPES, READING_ORDER_ROW_BAND
from .geometry import bbox_iou
from .io import load_predictions_jsonl, load_split
from .jsonl import read_jsonl
from .schemas import PageRecord, Region

logger = logging.getLogger(__name__)

LATEX_SYMBOLS = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ε",
    r"\theta": "θ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\pi": "π",
    r"\sigma": "σ",
    r"\phi": "φ",
    r"\omega": "ω",
    r"\cdot": "·",
    r"\times": "×",
    r"\div": "÷",
    r"\pm": "±",
    r"\leq": "≤",
    r"\geq": "≥",
    r"\neq": "≠",
    r"\rightarrow": "→",
    r"\to": "→",
    r"\leftarrow": "←",
    r"\uparrow": "↑",
    r"\downarrow": "↓",
    r"\vee": "∨",
    r"\wedge": "∧",
    r"\setminus": "\\",
    r"\mid": "|",
    r"\lvert": "|",
    r"\rvert": "|",
    r"\lfloor": "⌊",
    r"\rfloor": "⌋",
    r"\lceil": "⌈",
    r"\rceil": "⌉",
    r"\male": "♂",
    r"\female": "♀",
}

SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
LATIN_CYRILLIC_LOOKALIKES = str.maketrans({"c": "с", "o": "о", "p": "р", "x": "х"})


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


def _strip_latex_wrapper(text: str, command: str) -> str:
    pattern = re.compile(rf"\\{command}\s*\{{([^{{}}]*)\}}")
    while True:
        updated = pattern.sub(r"\1", text)
        if updated == text:
            return text
        text = updated


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"~~.*?~~\{(.*?)\}", r"\1", text)
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    text = re.sub(r"\\xrightarrow(?:\[[^\]]*\])?\{[^{}]*\}", "→", text)
    text = re.sub(r"\\xleftarrow(?:\[[^\]]*\])?\{[^{}]*\}", "←", text)

    for command in (
        "mathrm",
        "mathbf",
        "mathit",
        "mathbb",
        "mathcal",
        "mathfrak",
        "operatorname",
        "boldsymbol",
        "underline",
        "cancel",
    ):
        text = _strip_latex_wrapper(text, command)

    text = re.sub(r"\\(?:over|under)set\{[^{}]*\}\{([^{}]*)\}", r"\1", text)
    text = re.sub(
        r"\\(?:big|Big|bigg|Bigg|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr)\b",
        "",
        text,
    )
    text = re.sub(r"\\(?:hline|cline\{[^{}]*\}|phantom\{[^{}]*\})", "", text)

    for source, replacement in LATEX_SYMBOLS.items():
        text = text.replace(source, replacement)
    text = re.sub(
        r"\\(sin|cos|tan|log|ln|lim|arcsin|arccos|arctan|sinh|cosh|tanh)\b",
        r"\1 ",
        text,
    )
    text = re.sub(r"([A-Za-zА-Яа-яІіЇїЄєҐґ0-9])_\{([^{}]+)\}", r"\1_\2", text)
    text = re.sub(r"([A-Za-zА-Яа-яІіЇїЄєҐґ0-9])\^\{([^{}]+)\}", r"\1^\2", text)
    text = re.sub(
        r"([A-Za-zА-Яа-яІіЇїЄєҐґ])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda m: f"{m.group(1)}^{m.group(2).translate(SUPERSCRIPTS)}",
        text,
    )
    text = re.sub(
        r"([A-Za-zА-Яа-яІіЇїЄєҐґ])([₀₁₂₃₄₅₆₇₈₉]+)",
        lambda m: f"{m.group(1)}_{m.group(2).translate(SUBSCRIPTS)}",
        text,
    )
    text = text.replace("''", "'")
    text = text.translate(
        str.maketrans({"—": "-", "–": "-", "«": '"', "»": '"', "“": '"', "”": '"'})
    )
    text = text.translate(LATIN_CYRILLIC_LOOKALIKES)
    return re.sub(r"\s+", " ", text).strip()


def _reading_order_key(region: Region) -> tuple[int, int]:
    center_y = (region.bbox[1] + region.bbox[3]) // 2
    return (center_y // READING_ORDER_ROW_BAND, region.bbox[0])


def _is_scorable(region: Region) -> bool:
    return (
        region.language == "uk"
        and region.legibility == "legible"
        and region.type not in EMPTY_TEXT_TYPES
    )


def _page_text(regions: list[Region], *, truth: bool) -> str:
    parts: list[str] = []
    for region in sorted(regions, key=_reading_order_key):
        if truth and not _is_scorable(region):
            continue
        if not truth and region.type in EMPTY_TEXT_TYPES:
            continue
        text = normalize_text(region.text)
        if text:
            parts.append(text)
    return " ".join(parts)


def _match_regions(
    predicted: list[Region],
    truth: list[Region],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predicted):
        for truth_idx, gt in enumerate(truth):
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
    class_correct = 0
    cer_sum = 0.0
    text_matches = 0
    page_cer_sum = 0.0

    for image, page in truth_pages.items():
        pred_regions = predictions.get(image, [])
        matches = _match_regions(pred_regions, page.regions, iou_threshold=iou_threshold)
        tp += len(matches)
        fp += max(0, len(pred_regions) - len(matches))
        fn += max(0, len(page.regions) - len(matches))
        for pred_idx, truth_idx in matches:
            if pred_regions[pred_idx].type == page.regions[truth_idx].type:
                class_correct += 1
            if not _is_scorable(page.regions[truth_idx]):
                continue
            cer_sum += cer(
                normalize_text(pred_regions[pred_idx].text),
                normalize_text(page.regions[truth_idx].text),
            )
            text_matches += 1
        page_cer_sum += cer(
            _page_text(pred_regions, truth=False),
            _page_text(page.regions, truth=True),
        )

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    class_acc = class_correct / max(1, tp)
    mean_cer = cer_sum / max(1, text_matches)
    mean_page_cer = page_cer_sum / max(1, len(truth_pages))
    official_score = (
        0.15 * f1
        + 0.05 * class_acc
        + 0.30 * (1.0 - min(mean_cer, 1.0))
        + 0.50 * (1.0 - min(mean_page_cer, 1.0))
    )
    return {
        "matched_regions": float(tp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_acc": class_acc,
        "cer": mean_cer,
        "page_cer": mean_page_cer,
        "score": official_score,
        "proxy_score": f1 * (1.0 - min(mean_cer, 1.0)),
    }


def _grouped_metrics(
    truth_pages: dict[str, PageRecord],
    predictions: dict[str, list[Region]],
    iou_threshold: float,
    key_fn: Callable[[PageRecord], object],
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
