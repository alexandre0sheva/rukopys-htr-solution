from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ID = "AlexandreSheva/rukopys-curated-mvp"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _format_value(value: Any) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _release_copy(repo_id: str | None) -> tuple[str, str, str]:
    repo_text = (repo_id or "").lower()
    if "a100-v2" in repo_text or repo_text.endswith("-v2"):
        return (
            "RUKOPYS Qwen3-VL 8B Page LoRA (A100 v2)",
            "Second public page-level adapter for Ukrainian handwritten document parsing. "
            "This is the preferred release over the original page adapter when comparing the "
            "two public 8B LoRA checkpoints.",
            "Improved A100-class training run focused on full-page image-to-JSON extraction.",
        )
    return (
        "RUKOPYS Qwen3-VL 8B Page LoRA",
        "Initial public page-level LoRA adapter for Ukrainian handwritten document parsing. "
        "It adapts Qwen3-VL 8B to read a full scanned page and return structured text regions.",
        "Baseline public 8B page adapter for the RUKOPYS HTR pipeline.",
    )


def write_model_card(
    output_dir: Path,
    *,
    repo_id: str | None = None,
    qlora_config: Any | None = None,
    train_examples: int | None = None,
    eval_examples: int | None = None,
    overwrite: bool = True,
) -> Path:
    """Write a Hub README for a RUKOPYS PEFT/LoRA adapter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir / "README.md"
    if readme_path.exists() and not overwrite:
        return readme_path

    adapter_config = _read_json(output_dir / "adapter_config.json")
    base_model = (
        getattr(qlora_config, "base_model", None)
        or adapter_config.get("base_model_name_or_path")
        or DEFAULT_BASE_MODEL
    )
    lora_r = getattr(qlora_config, "lora_r", None) or adapter_config.get("r")
    lora_alpha = getattr(qlora_config, "lora_alpha", None) or adapter_config.get("lora_alpha")
    max_steps = getattr(qlora_config, "max_steps", None)
    learning_rate = getattr(qlora_config, "learning_rate", None)
    batch_size = getattr(qlora_config, "batch_size", None)
    grad_accum_steps = getattr(qlora_config, "grad_accum_steps", None)
    max_length = getattr(qlora_config, "max_length", None)
    max_pixels = getattr(qlora_config, "max_pixels", None)
    min_quality_weight = getattr(qlora_config, "min_quality_weight", None)
    weighted_sampling = getattr(qlora_config, "use_weighted_sampling", None)

    title, summary, release_note = _release_copy(repo_id)
    repo_heading = f"`{repo_id}`" if repo_id else "this adapter"
    effective_batch = None
    if batch_size is not None and grad_accum_steps is not None:
        effective_batch = batch_size * grad_accum_steps

    card = f"""---
base_model: {base_model}
library_name: peft
pipeline_tag: image-text-to-text
license: apache-2.0
datasets:
- {DEFAULT_DATASET_ID}
language:
- uk
tags:
- peft
- lora
- qwen3-vl
- document-analysis
- handwriting-recognition
- htr
- ukrainian
- image-to-text
---

# {title}

{summary}

{repo_heading} contains a PEFT/LoRA adapter, not a standalone model. Load it on top of
[`{base_model}`](https://huggingface.co/{base_model}) to run page-level Ukrainian handwriting
recognition and document-structure extraction.

## What It Does

- Takes a full-page manuscript or handwriting image as input.
- Produces structured JSON regions with bounding boxes, region types, language metadata, and text.
- Targets Ukrainian handwritten text recognition (HTR), OCR post-processing, and document AI workflows.
- Fits into the RUKOPYS pipeline as the page-level vision-language model.

## Release Positioning

{release_note}

The adapter is intended for experimentation, portfolio review, and reproducible HTR pipeline
development. For production use, validate on your own scans because handwriting style, scan quality,
page layout, and annotation source can shift model behavior.

## Training Data

Trained on the curated RUKOPYS MVP dataset:
[`{DEFAULT_DATASET_ID}`](https://huggingface.co/datasets/{DEFAULT_DATASET_ID}).

The dataset is a cleaned derivative of `UkrainianCatholicUniversity/rukopys` prepared for:

- page-to-regions JSON supervised fine-tuning,
- crop-level text transcription fine-tuning,
- layout detection experiments,
- repeatable Kaggle-style evaluation and submission generation.

## Training Setup

- Base model: `{base_model}`
- Method: 4-bit QLoRA / PEFT LoRA adapter fine-tuning
- LoRA rank: `{_format_value(lora_r)}`
- LoRA alpha: `{_format_value(lora_alpha)}`
- Max steps: `{_format_value(max_steps)}`
- Learning rate: `{_format_value(learning_rate)}`
- Per-device batch size: `{_format_value(batch_size)}`
- Gradient accumulation steps: `{_format_value(grad_accum_steps)}`
- Effective batch size: `{_format_value(effective_batch)}`
- Max sequence length: `{_format_value(max_length)}`
- Max image pixels: `{_format_value(max_pixels)}`
- Minimum quality weight: `{_format_value(min_quality_weight)}`
- Weighted sampling: `{_format_value(weighted_sampling)}`
- Training examples used: `{_format_value(train_examples)}`
- Evaluation examples held out: `{_format_value(eval_examples)}`

## Quick Use

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base_model_id = "{base_model}"
adapter_id = "{repo_id or '<your-adapter-repo>'}"

processor = AutoProcessor.from_pretrained(adapter_id)
base_model = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()
```

Use the project inference CLI for end-to-end page prediction and Kaggle submission generation.

## Output Format

The expected assistant response is JSON compatible with the RUKOPYS page schema:

```json
[
  {{
    "bbox": [10, 20, 300, 80],
    "type": "handwritten",
    "language": "uk",
    "text": "..."
  }}
]
```

## Limitations

- The adapter was trained for Ukrainian handwriting and may not generalize to other languages.
- It is sensitive to page resolution and preprocessing; match the training pixel budget when possible.
- Bounding boxes and text should be evaluated together, not as independent OCR text only.
- The training dataset inherits a non-commercial CC BY-NC-SA 4.0 license from the source data.

## Project Context

This model is part of a practical HTR system: raw RUKOPYS data curation, dataset packaging,
LoRA fine-tuning, inference, evaluation, and Kaggle-ready submission export. The goal is not only a
checkpoint, but a reproducible document-AI workflow for Ukrainian handwritten archives.
"""
    readme_path.write_text(card, encoding="utf-8")
    return readme_path
