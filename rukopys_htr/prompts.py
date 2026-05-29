from __future__ import annotations

TRANSCRIBE_REGION_DEFAULT = (
    "Transcribe this Ukrainian document region exactly. "
    "Most text is Ukrainian; prefer Ukrainian Cyrillic letters for ambiguous handwriting, "
    "including і, ї, є, and ґ. "
    "Preserve punctuation, correction markers, and LaTeX where applicable."
)

PAGE_TO_REGIONS_JSON_DEFAULT = (
    "Return a JSON array of document regions for this page. "
    "Each item must contain bbox [x1,y1,x2,y2], type, and text. "
    "Most text is Ukrainian; prefer Ukrainian Cyrillic letters for ambiguous handwriting. "
    "Use exact transcription. Use empty text for image and graph regions."
)

PAGE_TO_TEXT_LINES_JSON_DEFAULT = (
    "Return only a JSON array of transcribed text lines for this page in reading order. "
    "Do not include bounding boxes, explanations, markdown, or extra keys. "
    "Most text is Ukrainian; prefer Ukrainian Cyrillic letters for ambiguous handwriting, "
    "including і, ї, є, and ґ. "
    "Preserve Ukrainian text, punctuation, corrections, formulas, and table cell text exactly."
)

SOURCE_TRANSCRIBE_HINTS: dict[str, str] = {
    "dictation": (
        "This is dictated handwritten text; preserve spoken punctuation and corrections."
    ),
    "document_fragment": (
        "This is a document fragment; preserve layout-sensitive punctuation."
    ),
    "math": (
        "This page contains mathematical notation; preserve LaTeX and formula symbols."
    ),
    "table": (
        "This region is part of a table; preserve cell structure and separators."
    ),
    "graph": (
        "Transcribe labels or legends; leave non-text graphical content empty if needed."
    ),
    "drawing": "Transcribe captions or labels; preserve sketch annotations exactly.",
}

SOURCE_PAGE_HINTS: dict[str, str] = {
    "dictation": (
        "The page is dictated handwriting; preserve correction markers and punctuation."
    ),
    "document_fragment": (
        "The page is a document fragment; detect all text blocks including marginalia."
    ),
    "math": (
        "The page contains formulas; preserve LaTeX and use type formula where needed."
    ),
    "table": (
        "The page contains tables; use type table for tabular regions and preserve cells."
    ),
    "graph": (
        "The page contains graphs; use type graph with empty text for chart-only areas."
    ),
    "drawing": (
        "The page contains drawings; use type image or annotation for non-text sketches."
    ),
}

SOURCE_PAGE_TEXT_HINTS: dict[str, str] = {
    "dictation": (
        "This is a national dictation page; keep the dictated prose in natural line order."
    ),
    "archive": (
        "This may contain historical Ukrainian orthography, stamps, and printed headers; "
        "transcribe visible textual content."
    ),
    "university": (
        "This may contain formulas, chemistry, tables, and prose; "
        "keep each visual line as one item."
    ),
    "school": (
        "This may contain homework, formulas, teacher annotations, and tables; "
        "keep each visual line as one item."
    ),
    "math": (
        "This page contains formulas; preserve LaTeX-like notation and symbols."
    ),
    "table": (
        "This page contains tables; preserve rows in reading order and use spaces or "
        "separators inside a row."
    ),
}

REGION_TYPE_TRANSCRIBE_HINTS: dict[str, str] = {
    "formula": "Preserve LaTeX notation and mathematical symbols exactly.",
    "table": "Preserve row/column structure using consistent separators.",
    "printed": "Preserve printed punctuation and capitalization exactly.",
    "annotation": "Preserve marginal notes and correction markers.",
}


def transcribe_region_prompt(
    source: str | None = None,
    region_type: str | None = None,
) -> str:
    parts = [TRANSCRIBE_REGION_DEFAULT]
    if source:
        hint = SOURCE_TRANSCRIBE_HINTS.get(source)
        if hint:
            parts.append(hint)
    if region_type:
        hint = REGION_TYPE_TRANSCRIBE_HINTS.get(region_type)
        if hint:
            parts.append(hint)
    return " ".join(parts)


def page_to_regions_json_prompt(source: str | None = None) -> str:
    parts = [PAGE_TO_REGIONS_JSON_DEFAULT]
    if source:
        hint = SOURCE_PAGE_HINTS.get(source)
        if hint:
            parts.append(hint)
    return " ".join(parts)


def page_to_text_lines_json_prompt(source: str | None = None) -> str:
    parts = [PAGE_TO_TEXT_LINES_JSON_DEFAULT]
    if source:
        hint = SOURCE_PAGE_TEXT_HINTS.get(source)
        if hint:
            parts.append(hint)
    return " ".join(parts)
