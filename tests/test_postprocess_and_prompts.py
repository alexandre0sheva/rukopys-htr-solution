from __future__ import annotations

import json

from rukopys_htr.ensemble import merge_page_and_detector_regions
from rukopys_htr.postprocess import (
    extract_json_array,
    regions_from_model_json,
    text_lines_from_model_output,
)
from rukopys_htr.prompts import (
    TRANSCRIBE_REGION_DEFAULT,
    page_to_regions_json_prompt,
    page_to_text_lines_json_prompt,
    transcribe_region_prompt,
)
from rukopys_htr.schemas import Region


def test_transcribe_prompt_matches_curated_default() -> None:
    prompt = transcribe_region_prompt()
    assert prompt.startswith(TRANSCRIBE_REGION_DEFAULT)


def test_source_specific_transcribe_prompt() -> None:
    prompt = transcribe_region_prompt(source="math", region_type="formula")
    assert "LaTeX" in prompt
    assert "mathematical notation" in prompt


def test_source_specific_page_prompt() -> None:
    prompt = page_to_regions_json_prompt(source="table")
    assert "tables" in prompt


def test_source_specific_page_text_prompt() -> None:
    prompt = page_to_text_lines_json_prompt(source="school")
    assert "JSON array" in prompt
    assert "homework" in prompt


def test_extract_json_array_from_markdown_fence() -> None:
    text = '```json\n[{"bbox": [1, 2, 3, 4], "type": "handwritten", "text": "A"}]\n```'
    items = extract_json_array(text)
    assert len(items) == 1
    assert items[0]["text"] == "A"


def test_extract_json_array_returns_empty_on_invalid_payload() -> None:
    assert extract_json_array("not json at all") == []


def test_extract_json_array_salvages_truncated_page_output() -> None:
    complete = (
        '{"bbox": [1, 2, 3, 4], "type": "handwritten", "text": "first"}, '
        '{"bbox": [5, 6, 7, 8'
    )
    text = f"[{complete}]"
    items = extract_json_array(text)
    assert len(items) == 1
    assert items[0]["text"] == "first"


def test_extract_json_array_parses_balanced_outer_brackets() -> None:
    text = '[{"bbox": [1, 2, 3, 4], "type": "handwritten", "text": "ok"}]'
    items = extract_json_array(text)
    assert len(items) == 1
    assert items[0]["bbox"] == [1, 2, 3, 4]


def test_regions_from_model_json_clamps_and_normalizes_type() -> None:
    payload = json.dumps(
        [
            {"bbox": [-5, 0, 250, 120], "type": "unknown_type", "text": "Hi"},
            {"bbox": [0, 0, 0, 0], "type": "handwritten", "text": "Skip"},
        ]
    )
    regions = regions_from_model_json(payload, image_width=200, image_height=100)
    assert len(regions) == 1
    assert regions[0].bbox == [0, 0, 200, 100]
    assert regions[0].type == "handwritten"
    assert regions[0].text == "Hi"


def test_text_lines_from_model_output_accepts_json_and_plain_text() -> None:
    assert text_lines_from_model_output('["А", {"text": "Б"}]') == ["А", "Б"]
    assert text_lines_from_model_output("А\n\nБ") == ["А", "Б"]


def test_merge_page_and_detector_regions_prefers_page_text() -> None:
    page_regions = [
        Region(bbox=[10, 10, 50, 50], type="handwritten", text="page text"),
    ]
    detector_regions = [
        Region(bbox=[12, 12, 48, 48], type="handwritten", text="detector text"),
        Region(bbox=[100, 10, 150, 50], type="printed", text=""),
    ]
    merged = merge_page_and_detector_regions(
        page_regions,
        detector_regions,
        iou_threshold=0.5,
    )
    assert len(merged) == 2
    assert merged[0].text == "page text"
    assert merged[1].bbox == [100, 10, 150, 50]
