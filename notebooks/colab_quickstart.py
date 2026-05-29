# %% [markdown]
# # RUKOPYS HTR Colab Quickstart
#
# Run this notebook on a GPU runtime. T4 is enough for smoke tests; L4/A100 is better for useful
# QLoRA runs.
#
# Replace `REPO_URL` and `HF_NAMESPACE` with values you control before publishing anything. In
# local development, keep these in `.env`; in Colab, set them directly in this notebook.

# %%
!nvidia-smi

# %% [markdown]
# ## Clone and install

# %%
REPO_URL = "https://github.com/OWNER/REPO.git"
REPO_DIR = "/content/rukopys-htr"

!git clone {REPO_URL} {REPO_DIR}
%cd {REPO_DIR}
!pip install -U pip
!pip install -e ".[data,detector,vlm]"

# %%
%env HF_XET_HIGH_PERFORMANCE=1
%env HF_HUB_DISABLE_PROGRESS_BARS=1

# %% [markdown]
# If Colab upgraded CUDA or PyTorch packages, restart the runtime once, then continue from here.

# %% [markdown]
# ## Mount Drive

# %%
from google.colab import drive

drive.mount("/content/drive")

WORK_ROOT = "/content/drive/MyDrive/rukopys-htr"
RAW_DIR = "/content/rukopys-htr/data/raw/rukopys"
CURATED_DIR = f"{WORK_ROOT}/data/curated/rukopys_mvp"
RUNS_DIR = f"{WORK_ROOT}/runs"
OUTPUTS_DIR = f"{WORK_ROOT}/outputs"
HF_NAMESPACE = "your-hf-username-or-org"
HF_DATASET_ID = f"{HF_NAMESPACE}/rukopys-curated-mvp"
HF_PAGE_TEXT_MODEL_ID = f"{HF_NAMESPACE}/rukopys-qwen3-vl-8b-page-text"
HF_DETECTOR_MODEL_ID = f"{HF_NAMESPACE}/rukopys-yolo11m-detector"

!mkdir -p /content/rukopys-htr/data {WORK_ROOT}/data {RUNS_DIR} {OUTPUTS_DIR}

# %% [markdown]
# ## Hugging Face login
#
# Log in only if you plan to download private repos or push datasets/models.

# %%
!hf auth login
!hf auth whoami

# %% [markdown]
# ## Prepare data
#
# Default path: download the public source dataset and curate it locally.

# %%
!rukopys download --output {RAW_DIR} --max-workers 32

# %%
!rukopys curate \
  --raw-dir {RAW_DIR} \
  --output-dir {CURATED_DIR} \
  --include-silver \
  --max-silver 1000 \
  --crop-images

# Upload the curated dataset so future Colab runs can skip curation.
!rukopys upload-dataset \
  --dataset-dir {CURATED_DIR} \
  --repo-id {HF_DATASET_ID} \
  --pack \
  --replace-existing

# %% [markdown]
# ### Optional: reuse your own curated dataset
#
# Run these cells instead of the local curation cells if you previously published a curated dataset
# to your own Hugging Face namespace.

# %%
!rukopys download-curated \
  --output {CURATED_DIR} \
  --repo-id {HF_DATASET_ID}

# %%
# Still needed for sample_submission.csv and test inference.
!rukopys download \
  --output {RAW_DIR} \
  --max-workers 32 \
  --allow-pattern "test/**" \
  --allow-pattern "sample_submission.csv"

# %% [markdown]
# ## T4 smoke-test page-text QLoRA
#
# Restart the runtime before this cell if you loaded other models earlier.
# The `colab_t4_fast` preset uses conservative memory settings
# (`max_length=1024`, `max_pixels=262144`).

# %%
!rukopys train-vlm-qlora \
  --train-jsonl {CURATED_DIR}/page_text_sft.jsonl \
  --preset colab_t4_fast \
  --output-dir {RUNS_DIR}/qwen3_vl_2b_page_text_t4 \
  --sample-limit 300 \
  --max-steps 100 \
  --batch-size 1

# %% [markdown]
# ## Train detector
#
# The detector predicts layout boxes and region types. The page-text VLM predicts text lines; the
# inference pipeline assigns the generated text back to detector boxes in reading order.

# %%
!rukopys train-detector \
  --data-yaml {CURATED_DIR}/yolo/data.yaml \
  --model yolo11m.pt \
  --output-dir {RUNS_DIR}/detector_yolo11m \
  --epochs 80 \
  --image-size 1536 \
  --batch 4

!rukopys upload-model \
  --model-dir {RUNS_DIR}/detector_yolo11m \
  --repo-id {HF_DETECTOR_MODEL_ID}

# %% [markdown]
# ## T4 quality page-text QLoRA
#
# This uses 4-bit QLoRA with a larger 4B model. Use this after the smoke test works.

# %%
!rukopys train-vlm-qlora \
  --train-jsonl {CURATED_DIR}/page_text_sft.jsonl \
  --preset colab_t4_quality \
  --output-dir {RUNS_DIR}/qwen3_vl_4b_page_text_t4 \
  --sample-limit 500 \
  --max-steps 200

# %% [markdown]
# ## A100 practical page-text QLoRA
#
# Use this cell for the main Kaggle experiment. It pushes the adapter to Hugging Face.

# %%
!rukopys train-vlm-qlora \
  --train-jsonl {CURATED_DIR}/page_text_sft.jsonl \
  --preset a100_quality \
  --output-dir {RUNS_DIR}/qwen3_vl_8b_page_text \
  --max-steps 1800 \
  --eval-steps 100 \
  --min-quality-weight 0.75 \
  --push-to-hub \
  --hub-model-id {HF_PAGE_TEXT_MODEL_ID}

# %% [markdown]
# ## A100/H100 80GB 32B experiment
#
# Use this after the 8B run establishes a good validation/submission baseline. It is slower and
# needs more VRAM, but may improve Ukrainian handwriting and formula robustness.

# %%
HF_PAGE_TEXT_32B_MODEL_ID = f"{HF_NAMESPACE}/rukopys-qwen3-vl-32b-page-text"

!rukopys train-vlm-qlora \
  --train-jsonl {CURATED_DIR}/page_text_sft.jsonl \
  --preset a100_32b_quality \
  --output-dir {RUNS_DIR}/qwen3_vl_32b_page_text \
  --max-steps 1800 \
  --eval-steps 100 \
  --min-quality-weight 0.75 \
  --push-to-hub \
  --hub-model-id {HF_PAGE_TEXT_32B_MODEL_ID}

# %% [markdown]
# ## Inference and submission

# %%
!rukopys infer \
  --mode detector-page-text \
  --test-dir {RAW_DIR}/test \
  --detector-model {RUNS_DIR}/detector_yolo11m/weights/best.pt \
  --vlm-model {RUNS_DIR}/qwen3_vl_8b_page_text \
  --max-pixels 1605632 \
  --page-max-new-tokens 1024 \
  --output-jsonl {OUTPUTS_DIR}/detector_page_text_predictions.jsonl

# %%
!rukopys make-submission \
  --predictions {OUTPUTS_DIR}/detector_page_text_predictions.jsonl \
  --sample-submission {RAW_DIR}/sample_submission.csv \
  --output {OUTPUTS_DIR}/submission.csv

# %% [markdown]
