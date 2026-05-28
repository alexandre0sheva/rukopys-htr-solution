# Google Colab Runbook

This runbook is optimized for Colab GPU runtimes. Use T4 for smoke tests and L4/A100 for useful
training runs.

Before you run publishing cells, choose your own public repository and Hub namespace. In local
development these values can live in `.env` and the CLI reads that file automatically; in Colab,
set them in the notebook variables.

```bash
export REPO_URL="https://github.com/OWNER/REPO.git"
export HF_NAMESPACE="your-hf-username-or-org"
```

## 1. Runtime

Select `Runtime -> Change runtime type -> GPU`.

Recommended mapping:

- T4: `Qwen/Qwen3-VL-2B-Instruct` for smoke tests, `Qwen/Qwen3-VL-4B-Instruct` for 4-bit quality runs.
- L4: `Qwen/Qwen3-VL-4B-Instruct`, practical MVP training.
- A100: `Qwen/Qwen3-VL-8B-Instruct` for the main quality run; `Qwen/Qwen3-VL-32B-Instruct` for high-quality experiments on 80GB.

## 2. Clone and Install

```bash
git clone "$REPO_URL" rukopys-htr
cd rukopys-htr
pip install -U pip
pip install -e ".[data,detector,vlm]"
```

If Colab upgrades core CUDA/PyTorch packages, restart the runtime once after installation.

For faster Hub transfers on current Hugging Face Hub/Xet backends:

```bash
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
```

## 3. Persistent Storage

```python
from google.colab import drive

drive.mount("/content/drive")
```

Recommended folders:

```bash
export WORK_ROOT="/content/drive/MyDrive/rukopys-htr"
export RAW_DIR="/content/rukopys-htr/data/raw/rukopys"
export CURATED_DIR="${WORK_ROOT}/data/curated/rukopys_mvp"
export RUNS_DIR="${WORK_ROOT}/runs"
export OUTPUTS_DIR="${WORK_ROOT}/outputs"
mkdir -p /content/rukopys-htr/data "${WORK_ROOT}/data" "${RUNS_DIR}" "${OUTPUTS_DIR}"
```

## 4. Hugging Face Login

Log in only if you plan to download private repos or push datasets/models:

```bash
hf auth login
hf auth whoami
```

## 5. Download and Curate

Default path: download the public source dataset and curate it locally.

```bash
rukopys download \
  --output "${RAW_DIR}" \
  --max-workers 32

rukopys curate \
  --raw-dir "${RAW_DIR}" \
  --output-dir "${CURATED_DIR}" \
  --include-silver \
  --max-silver 1000 \
  --crop-images
```

For a fast smoke test, reduce `--max-silver` to `100` or omit `--include-silver`.

If you published a curated dataset in your own namespace, you can reuse it:

```bash
rukopys download-curated \
  --output "${CURATED_DIR}" \
  --repo-id "${HF_NAMESPACE}/rukopys-curated-mvp"

rukopys download \
  --output "${RAW_DIR}" \
  --max-workers 32 \
  --allow-pattern "test/**" \
  --allow-pattern "sample_submission.csv"
```

The second command is still needed for `sample_submission.csv` and the hidden test images.

## 6. Train QLoRA on T4

```bash
rukopys train-vlm-qlora \
  --train-jsonl "${CURATED_DIR}/page_sft.jsonl" \
  --preset colab_t4_fast \
  --output-dir "${RUNS_DIR}/qwen3_vl_2b_page_t4" \
  --sample-limit 300 \
  --max-steps 100 \
  --batch-size 1
```

To push the adapter to your Hub namespace, add:

```bash
--push-to-hub --hub-model-id "${HF_NAMESPACE}/rukopys-qwen3-vl-2b-page-t4"
```

For a stronger T4 run, switch to the 4-bit 4B preset:

```bash
rukopys train-vlm-qlora \
  --train-jsonl "${CURATED_DIR}/page_sft.jsonl" \
  --preset colab_t4_quality \
  --output-dir "${RUNS_DIR}/qwen3_vl_4b_page_t4" \
  --sample-limit 500 \
  --max-steps 200
```

## 7. Train QLoRA on L4/A100

```bash
rukopys train-vlm-qlora \
  --train-jsonl "${CURATED_DIR}/page_sft.jsonl" \
  --preset a100_quality \
  --output-dir "${RUNS_DIR}/qwen3_vl_8b_page" \
  --max-steps 1200 \
  --eval-steps 100 \
  --min-quality-weight 0.75
```

For an A100 80GB 32B run:

```bash
rukopys train-vlm-qlora \
  --train-jsonl "${CURATED_DIR}/page_sft.jsonl" \
  --preset a100_32b_quality \
  --output-dir "${RUNS_DIR}/qwen3_vl_32b_page" \
  --max-steps 1200 \
  --eval-steps 100 \
  --min-quality-weight 0.75
```

Add `--push-to-hub --hub-model-id "${HF_NAMESPACE}/MODEL_REPO_NAME"` to either command when you
want to preserve the adapter outside Colab.

## 8. Optional Detector

```bash
rukopys train-detector \
  --data-yaml "${CURATED_DIR}/yolo/data.yaml" \
  --model yolo11n.pt \
  --output-dir "${RUNS_DIR}/detector" \
  --epochs 30 \
  --image-size 1280 \
  --batch 4
```

## 9. Inference and Submission CSV

```bash
rukopys infer \
  --mode page-vlm \
  --test-dir "${RAW_DIR}/test" \
  --vlm-model "${RUNS_DIR}/qwen3_vl_8b_page" \
  --max-pixels 401408 \
  --batch-size 4 \
  --output-jsonl "${OUTPUTS_DIR}/page_predictions.jsonl"

rukopys make-submission \
  --predictions "${OUTPUTS_DIR}/page_predictions.jsonl" \
  --sample-submission "${RAW_DIR}/sample_submission.csv" \
  --output "${OUTPUTS_DIR}/submission.csv"
```

Inference loads VLMs in 4-bit on CUDA by default. Add `--no-load-in-4bit` only for fp16/bf16 inference.

## Practical Notes

- If you hit OOM, restart the runtime first so model downloads are not competing with stale GPU allocations.
- Then lower `--max-length` (try `768`) and `--max-pixels` (try `131072`), or move from the 4B T4 preset to the 2B smoke-test preset.
- For page-level QLoRA, always pass `--max-pixels`. Full-page scans can otherwise produce tens of thousands of vision tokens and break image-token alignment.
- During inference, always pass `--max-pixels` on T4/L4. Without it, full-page images can request tens of GB of VRAM.
- On A100 80GB, use `--batch-size 4` for page-VLM inference first, then try `6` or `8` if VRAM allows.
- `--sample-limit` only reduces dataset size, not per-step VRAM.
- Keep `--batch-size 1` for training and scale effective batch with `--grad-accum-steps`.
- Start with `page_sft.jsonl`; use `vlm_sft.jsonl` only for detector+crop recognizer experiments.
