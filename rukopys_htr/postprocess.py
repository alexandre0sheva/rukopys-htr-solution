from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from .constants import REGION_TYPES
from .geometry import clamp_bbox
from .schemas import Region

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JsonParseResult:
    items: list[Any]
    ok: bool
    error: str | None = None


def extract_json_array(text: str) -> list[Any]:
    return extract_json_array_with_status(text).items


def extract_json_array_with_status(text: str) -> JsonParseResult:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        snippet = cleaned[:200]
        logger.warning("Failed to locate JSON array in model output: %r", snippet)
        return JsonParseResult(items=[], ok=False, error="json_array_not_found")
    payload = cleaned[start : end + 1]
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON array from model output: %s", exc)
        return JsonParseResult(items=[], ok=False, error=str(exc))
    if not isinstance(value, list):
        logger.warning("Model JSON payload is not a list: %r", type(value).__name__)
        return JsonParseResult(items=[], ok=False, error="json_not_list")
    return JsonParseResult(items=value, ok=True)


def _normalize_region_type(region_type: str) -> str:
    if region_type in REGION_TYPES:
        return region_type
    logger.warning("Unknown region type %r; defaulting to handwritten", region_type)
    return "handwritten"


def regions_from_model_json(text: str, image_width: int, image_height: int) -> list[Region]:
    parsed = extract_json_array_with_status(text)
    regions: list[Region] = []
    for item in parsed.items:
        if not isinstance(item, dict):
            continue
        bbox = clamp_bbox(item.get("bbox", []), image_width, image_height)
        if bbox is None:
            continue
        region_type = _normalize_region_type(str(item.get("type") or "handwritten"))
        regions.append(
            Region(
                bbox=bbox,
                type=region_type,
                text="" if item.get("text") is None else str(item.get("text")),
                language=str(item.get("language") or "uk"),
                legibility=str(item.get("legibility") or "legible"),
                confidence=item.get("confidence"),
            )
        )
    return regions
