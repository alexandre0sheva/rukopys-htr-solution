from __future__ import annotations

import json
import re
from typing import Any

from .geometry import clamp_bbox
from .schemas import Region


def extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def regions_from_model_json(text: str, image_width: int, image_height: int) -> list[Region]:
    regions: list[Region] = []
    for item in extract_json_array(text):
        if not isinstance(item, dict):
            continue
        bbox = clamp_bbox(item.get("bbox", []), image_width, image_height)
        if bbox is None:
            continue
        region_type = str(item.get("type") or "handwritten")
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
