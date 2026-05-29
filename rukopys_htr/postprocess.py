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


def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    return re.sub(r"```$", "", cleaned).strip()


def _find_balanced_array_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
    return None


def _salvage_json_array_items(text: str, array_start: int) -> list[Any]:
    decoder = json.JSONDecoder()
    items: list[Any] = []
    index = array_start + 1
    while index < len(text):
        while index < len(text) and text[index] in " \t\n\r,":
            index += 1
        if index >= len(text) or text[index] == "]":
            break
        if text[index] != "{":
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            items.append(value)
        index = end
    return items


def extract_json_array_with_status(text: str) -> JsonParseResult:
    cleaned = _strip_markdown_fence(text)
    start = cleaned.find("[")
    if start == -1:
        snippet = cleaned[:200]
        logger.warning("Failed to locate JSON array in model output: %r", snippet)
        return JsonParseResult(items=[], ok=False, error="json_array_not_found")

    end = _find_balanced_array_end(cleaned, start)
    if end is not None:
        payload = cleaned[start : end + 1]
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON array from model output: %s", exc)
        else:
            if isinstance(value, list):
                return JsonParseResult(items=value, ok=True)
            logger.warning("Model JSON payload is not a list: %r", type(value).__name__)
            return JsonParseResult(items=[], ok=False, error="json_not_list")

    salvaged = _salvage_json_array_items(cleaned, start)
    if salvaged:
        logger.warning(
            "Recovered %d region(s) from truncated or malformed page-VLM JSON",
            len(salvaged),
        )
        return JsonParseResult(items=salvaged, ok=True, error="salvaged_partial_array")

    if end is None:
        logger.warning(
            "Page-VLM JSON array appears truncated (no closing bracket); 0 complete regions"
        )
        return JsonParseResult(items=[], ok=False, error="json_array_truncated")
    return JsonParseResult(items=[], ok=False, error="json_decode_error")


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


def text_lines_from_model_output(text: str) -> list[str]:
    parsed = extract_json_array_with_status(text)
    if parsed.items:
        lines: list[str] = []
        for item in parsed.items:
            if isinstance(item, str):
                line = item.strip()
            elif isinstance(item, dict):
                line = str(item.get("text") or "").strip()
            else:
                line = str(item).strip()
            if line:
                lines.append(line)
        return lines

    cleaned = _strip_markdown_fence(text)
    return [line.strip() for line in cleaned.splitlines() if line.strip()]
