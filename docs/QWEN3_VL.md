# Qwen3-VL Model Choice

Last checked: 2026-05-25.

## Current Generation

The current Qwen vision-language generation is Qwen3-VL. Official Qwen sources describe it
as the strongest Qwen vision-language series to date, with upgraded OCR, long-document
structure parsing, visual grounding, spatial reasoning, and video understanding.

Official lineup used for this project:

| Preset | Model | Intended use |
| --- | --- | --- |
| `colab_t4_fast` | `Qwen/Qwen3-VL-2B-Instruct` | Colab T4 smoke tests |
| `colab_t4_quality` | `Qwen/Qwen3-VL-4B-Instruct` | Colab T4 quality runs with 4-bit QLoRA |
| `colab_l4_balanced` | `Qwen/Qwen3-VL-4B-Instruct` | Colab L4 practical runs |
| `a100_quality` | `Qwen/Qwen3-VL-8B-Instruct` | default quality QLoRA target |
| `multi_gpu_32b` | `Qwen/Qwen3-VL-32B-Instruct` | multi-GPU QLoRA experiments |
| `teacher_moe_235b` | `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` | teacher inference / pseudo-labeling |

## Decision

Use `Qwen/Qwen3-VL-8B-Instruct` as the default model for the Kaggle solution. It is new-generation,
Apache-2.0, strong for OCR/document parsing, and still realistic for QLoRA on high-memory single-GPU
hardware. Use 2B/4B for Colab, and reserve 32B/235B for larger hardware or API/server-based teacher
generation.

The project trains VLMs with 4-bit NF4 QLoRA on CUDA. This is intentional: for this task, a larger
model in 4-bit is usually a better trade-off than a smaller model loaded in fp16, as long as the
image token budget fits in VRAM. Inference also loads VLMs in 4-bit on CUDA by default.

The 235B-A22B model should not be the default fine-tuning target for this repository because it is
an MoE-scale deployment model. It is better used to generate or verify silver labels, ensemble
predictions, and repair JSON outputs.

## Sources

- Qwen3-VL GitHub release log: https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL 8B model card: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Qwen3-VL 235B FP8 model card: https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
