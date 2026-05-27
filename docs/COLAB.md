# Google Colab Runbook

This runbook is optimized for Colab GPU runtimes. Use T4 for smoke tests and L4/A100 for useful training runs.

## 1. Runtime

Select `Runtime -> Change runtime type -> GPU`.

Recommended mapping:

- T4: `Qwen/Qwen3-VL-2B-Instruct` for smoke tests, `Qwen/Qwen3-VL-4B-Instruct` for 4-bit quality runs.
- L4: `Qwen/Qwen3-VL-4B-Instruct`, practical MVP training.
- A100: `Qwen/Qwen3-VL-8B-Instruct` runs with longer context and more steps.

## 2. Clone and Install

```bash
git clone https://github.com/alexandre0sheva/rukopys-htr-solution.git
cd rukopys-htr-solution
pip install -U pip
pip install -e ".[data,detector,vlm,kaggle]"
```

If Colab upgrades core CUDA/PyTorch packages, restart the runtime once after installation.

## 3. Persistent Storage

```python
from google.colab import drive
drive.mount("/content/drive")
```

Recommended folders:

```bash
mkdir -p /content/drive/MyDrive/rukopys-htr/data
mkdir -p /content/drive/MyDrive/rukopys-htr/runs
mkdir -p /content/drive/MyDrive/rukopys-htr/outputs
```

## 4. Tokens

For Hugging Face:

```bash
export HF_TOKEN=hf_your_write_token
hf auth login --token "$HF_TOKEN" --add-to-git-credential
```

For Kaggle:

```bash
mkdir -p ~/.kaggle
# upload kaggle.json through the Colab file picker, then:
cp /content/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Accept the Kaggle competition rules in the web UI before submitting.

## 5. Download and Curate

```bash
rukopys download \
  --output /content/drive/MyDrive/rukopys-htr/data/raw/rukopys

rukopys curate \
  --raw-dir /content/drive/MyDrive/rukopys-htr/data/raw/rukopys \
  --output-dir /content/drive/MyDrive/rukopys-htr/data/curated/rukopys_mvp \
  --include-silver \
  --max-silver 1000 \
  --crop-images
```

For a fast smoke test, reduce `--max-silver` to `100` or omit `--include-silver`.

## 6. Train QLoRA on T4

```bash
rukopys train-vlm-qlora \
  --train-jsonl /content/drive/MyDrive/rukopys-htr/data/curated/rukopys_mvp/page_sft.jsonl \
  --preset colab_t4_fast \
  --output-dir /content/drive/MyDrive/rukopys-htr/runs/qwen3_vl_2b_page_t4 \
  --sample-limit 300 \
  --max-steps 100 \
  --batch-size 1 \
  --push-to-hub \
  --hub-model-id AlexandreSheva/rukopys-qwen3-vl-2b-page-t4
```

For a stronger T4 run, switch to the 4-bit 4B preset:

```bash
rukopys train-vlm-qlora \
  --train-jsonl /content/drive/MyDrive/rukopys-htr/data/curated/rukopys_mvp/page_sft.jsonl \
  --preset colab_t4_quality \
  --output-dir /content/drive/MyDrive/rukopys-htr/runs/qwen3_vl_4b_page_t4 \
  --sample-limit 500 \
  --max-steps 200 \
  --push-to-hub \
  --hub-model-id AlexandreSheva/rukopys-qwen3-vl-4b-page-t4
```

## 7. Train QLoRA on L4/A100

```bash
rukopys train-vlm-qlora \
  --train-jsonl /content/drive/MyDrive/rukopys-htr/data/curated/rukopys_mvp/page_sft.jsonl \
  --base-model Qwen/Qwen3-VL-8B-Instruct \
  --output-dir /content/drive/MyDrive/rukopys-htr/runs/qwen3_vl_8b_page \
  --max-steps 600 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --max-length 6144 \
  --max-pixels 802816 \
  --lora-r 16 \
  --lora-alpha 32 \
  --push-to-hub \
  --hub-model-id AlexandreSheva/rukopys-qwen3-vl-8b-page
```

## 8. Optional Detector

```bash
rukopys train-detector \
  --data-yaml /content/drive/MyDrive/rukopys-htr/data/curated/rukopys_mvp/yolo/data.yaml \
  --model yolo11n.pt \
  --output-dir /content/drive/MyDrive/rukopys-htr/runs/detector \
  --epochs 30 \
  --image-size 1280 \
  --batch 4
```

## 9. Inference and Submit

```bash
rukopys infer \
  --mode page-vlm \
  --test-dir /content/drive/MyDrive/rukopys-htr/data/raw/rukopys/test \
  --vlm-model /content/drive/MyDrive/rukopys-htr/runs/qwen3_vl_8b_page \
  --max-pixels 401408 \
  --output-jsonl /content/drive/MyDrive/rukopys-htr/outputs/page_predictions.jsonl

rukopys make-submission \
  --predictions /content/drive/MyDrive/rukopys-htr/outputs/page_predictions.jsonl \
  --sample-submission /content/drive/MyDrive/rukopys-htr/data/raw/rukopys/sample_submission.csv \
  --output /content/drive/MyDrive/rukopys-htr/outputs/submission.csv

rukopys submit-kaggle \
  --submission /content/drive/MyDrive/rukopys-htr/outputs/submission.csv \
  --message "page-vlm qlora colab"
```

Inference loads VLMs in 4-bit on CUDA by default. Add `--no-load-in-4bit` only for fp16/bf16 inference.

## Practical Notes

- If you hit OOM, restart the runtime first so the 17GB model download is not competing with stale GPU allocations.
- Then lower `--max-length` (try `768`) and `--max-pixels` (try `131072`), or move from the 4B T4 preset to the 2B smoke-test preset.
- For page-level QLoRA, always pass `--max-pixels`. Without it, full-page scans can produce tens of thousands of vision tokens; truncating `--max-length` then breaks Qwen3-VL with `Image features and image tokens do not match`.
- During inference, always pass `--max-pixels` on T4/L4. Without it, full-page images can request tens of GB of VRAM.
- `--sample-limit` only reduces dataset size, not per-step VRAM.
- Keep `--batch-size 1` and scale effective batch with `--grad-accum-steps`.
- Always push to Hub during real runs. Colab sessions can disconnect.
- Start with `page_sft.jsonl`; use `vlm_sft.jsonl` only for detector+crop recognizer experiments.
