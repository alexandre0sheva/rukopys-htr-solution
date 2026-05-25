from __future__ import annotations

from .geometry import bbox_iou
from .schemas import Region


def _best_iou_match(
    region: Region,
    candidates: list[Region],
    used: set[int],
    iou_threshold: float,
) -> int | None:
    best_idx: int | None = None
    best_score = iou_threshold
    for idx, candidate in enumerate(candidates):
        if idx in used:
            continue
        if region.type != candidate.type:
            continue
        score = bbox_iou(region.bbox, candidate.bbox)
        if score >= best_score:
            best_score = score
            best_idx = idx
    return best_idx


def merge_page_and_detector_regions(
    page_regions: list[Region],
    detector_regions: list[Region],
    *,
    iou_threshold: float = 0.5,
) -> list[Region]:
    merged: list[Region] = []
    used_detector: set[int] = set()

    for page_region in page_regions:
        match_idx = _best_iou_match(page_region, detector_regions, used_detector, iou_threshold)
        if match_idx is not None:
            used_detector.add(match_idx)
            detector_region = detector_regions[match_idx]
            merged.append(
                Region(
                    bbox=page_region.bbox,
                    type=page_region.type,
                    text=page_region.text or detector_region.text,
                    language=page_region.language,
                    legibility=page_region.legibility,
                    confidence=page_region.confidence or detector_region.confidence,
                )
            )
        else:
            merged.append(page_region)

    for idx, detector_region in enumerate(detector_regions):
        if idx in used_detector:
            continue
        merged.append(
            Region(
                bbox=detector_region.bbox,
                type=detector_region.type,
                text=detector_region.text,
                language=detector_region.language,
                legibility=detector_region.legibility,
                confidence=detector_region.confidence,
            )
        )

    return merged
