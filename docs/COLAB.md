# Google Colab Runbook

This runbook is optimized for Colab GPU runtimes. Use T4 for smoke tests and L4/A100 for useful training runs.

## 1. Runtime

Select `Runtime -> Change runtime type -> GPU`.

Recommended mapping:

- T4: `Qwen/Qwen2-VL-2B-Instruct`, page SFT smoke tests, low sequence length.
- L4: `Qwen/Qwen2.5-VL-3B-Instruct`, practical MVP training.
- A100: 3B or 7B runs with longer context and more steps.

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
  --output-dir /content/drive/MyDrive/rukopys-htr/runs/vlm_page_t4 \
  --sample-limit 300 \
  --max-steps 100 \
  --batch-size 1 \
  --push-to-hub \
  --hub-model-id AlexandreSheva/rukopys-page-vlm-t4
```

## 7. Train QLoRA on L4/A100

```bash
rukopys train-vlm-qlora \
  --train-jsonl /content/drive/MyDrive/rukopys-htr/data/curated/rukopys_mvp/page_sft.jsonl \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --output-dir /content/drive/MyDrive/rukopys-htr/runs/vlm_page_3b \
  --max-steps 600 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --max-length 1536 \
  --lora-r 16 \
  --lora-alpha 32 \
  --push-to-hub \
  --hub-model-id AlexandreSheva/rukopys-page-vlm-3b
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
  --vlm-model /content/drive/MyDrive/rukopys-htr/runs/vlm_page_3b \
  --output-jsonl /content/drive/MyDrive/rukopys-htr/outputs/page_predictions.jsonl

rukopys make-submission \
  --predictions /content/drive/MyDrive/rukopys-htr/outputs/page_predictions.jsonl \
  --sample-submission /content/drive/MyDrive/rukopys-htr/data/raw/rukopys/sample_submission.csv \
  --output /content/drive/MyDrive/rukopys-htr/outputs/submission.csv

rukopys submit-kaggle \
  --submission /content/drive/MyDrive/rukopys-htr/outputs/submission.csv \
  --message "page-vlm qlora colab"
```

## Practical Notes

- If you hit OOM, lower `--max-length`, set `--sample-limit`, or use the 2B model.
- Keep `--batch-size 1` and scale effective batch with `--grad-accum-steps`.
- Always push to Hub during real runs. Colab sessions can disconnect.
- Start with `page_sft.jsonl`; use `vlm_sft.jsonl` only for detector+crop recognizer experiments.
