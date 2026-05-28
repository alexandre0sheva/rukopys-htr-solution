# %% [markdown]
# # RUKOPYS HTR Colab Quickstart
#
# Run this notebook on a GPU runtime. T4 is enough for smoke tests; L4/A100 is better for useful QLoRA runs.

# %%
!nvidia-smi

# %% [markdown]
# ## Clone and install

# %%
REPO_URL = "https://github.com/alexandre0sheva/rukopys-htr-solution.git"
!git clone {REPO_URL}
%cd rukopys-htr-solution
!pip install -U pip
!pip install -e ".[data,detector,vlm,kaggle]"

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

!mkdir -p /content/rukopys-htr/data {WORK_ROOT}/data {RUNS_DIR} {OUTPUTS_DIR}

# %% [markdown]
# ## Hugging Face login
#
# Create a write token at https://huggingface.co/settings/tokens.

# %%
import getpass
import os

os.environ["HF_TOKEN"] = getpass.getpass("HF_TOKEN: ")
!hf auth login --token "$HF_TOKEN" --add-to-git-credential
!hf auth whoami

# %% [markdown]
# ## Prepare data — pick one path
#
# Run **either** Option A **or** Option B below (not both).
#
# - **Option A (fast):** download the pre-curated dataset from Hugging Face. Also download raw for `sample_submission.csv` and the test split used at inference.
# - **Option B (fresh curation):** download the original RUKOPYS dataset and curate locally. Use this when you changed curation settings or want to publish a new curated build.

# %% [markdown]
# ### Option A: Download pre-curated dataset
#
# Packed tar shards are downloaded from Hugging Face and unpacked automatically.

# %%
HF_DATASET_ID = "AlexandreSheva/rukopys-curated-mvp"

!rukopys download-curated \
  --output {CURATED_DIR} \
  --repo-id {HF_DATASET_ID}

# %%
# Still needed for sample_submission.csv and test inference, even when using Option A.
!rukopys download \
  --output {RAW_DIR} \
  --max-workers 32 \
  --allow-pattern "test/**" \
  --allow-pattern "sample_submission.csv"

# %% [markdown]
# ### Option B: Download raw and curate
#
# Skip Option A if you run these cells instead.

# %%
!rukopys download --output {RAW_DIR} --max-workers 32

# %%
!rukopys curate \
  --raw-dir {RAW_DIR} \
  --output-dir {CURATED_DIR} \
  --include-silver \
  --max-silver 1000 \
  --crop-images

# %% [markdown]
# ## T4 smoke-test QLoRA
#
# Restart the runtime before this cell if you loaded other models earlier.
# The `colab_t4_fast` preset uses conservative memory settings
# (`max_length=1024`, `max_pixels=262144`).

# %%
HF_MODEL_ID = "AlexandreSheva/rukopys-qwen3-vl-2b-page-t4"
!rukopys train-vlm-qlora \
  --train-jsonl {CURATED_DIR}/page_sft.jsonl \
  --preset colab_t4_fast \
  --output-dir {RUNS_DIR}/qwen3_vl_2b_page_t4 \
  --sample-limit 300 \
  --max-steps 100 \
  --batch-size 1 \
  --push-to-hub \
  --hub-model-id {HF_MODEL_ID}

# %% [markdown]
# ## T4 quality QLoRA
#
# This uses 4-bit QLoRA with a larger 4B model. Use this after the smoke test works.

# %%
HF_MODEL_ID = "AlexandreSheva/rukopys-qwen3-vl-4b-page-t4"
!rukopys train-vlm-qlora \
  --train-jsonl {CURATED_DIR}/page_sft.jsonl \
  --preset colab_t4_quality \
  --output-dir {RUNS_DIR}/qwen3_vl_4b_page_t4 \
  --sample-limit 500 \
  --max-steps 200 \
  --push-to-hub \
  --hub-model-id {HF_MODEL_ID}

# %% [markdown]
# ## L4/A100 practical QLoRA
#
# Use this cell instead of the T4 smoke-test when you have enough VRAM.

# %%
HF_MODEL_ID = "AlexandreSheva/rukopys-qwen3-vl-8b-page"
!rukopys train-vlm-qlora \
  --train-jsonl {CURATED_DIR}/page_sft.jsonl \
  --preset a100_quality \
  --output-dir {RUNS_DIR}/qwen3_vl_8b_page \
  --max-steps 1200 \
  --eval-steps 100 \
  --min-quality-weight 0.75 \
  --push-to-hub \
  --hub-model-id {HF_MODEL_ID}

# %% [markdown]
# ## A100 80GB high-quality QLoRA

# %%
HF_MODEL_ID = "AlexandreSheva/rukopys-qwen3-vl-32b-page"
!rukopys train-vlm-qlora \
  --train-jsonl {CURATED_DIR}/page_sft.jsonl \
  --preset a100_32b_quality \
  --output-dir {RUNS_DIR}/qwen3_vl_32b_page \
  --max-steps 1200 \
  --eval-steps 100 \
  --min-quality-weight 0.75 \
  --push-to-hub \
  --hub-model-id {HF_MODEL_ID}

# %% [markdown]
# ## Upload curated dataset
#
# Run this after **Option B** curation, or whenever you want to publish an updated curated build.
# Stage to Colab local SSD with `rsync --delete`, pack into tar shards, then upload.
# `upload-dataset --pack` validates layout and repacks if staging is inconsistent.

# %%
from pathlib import Path

HF_DATASET_ID = "AlexandreSheva/rukopys-curated-mvp"
STAGING_DIR = Path("/content/hf_upload_staging/rukopys_mvp")

STAGING_DIR.parent.mkdir(parents=True, exist_ok=True)
# Mirror curated data exactly; without --delete stale shards/manifest can block repacking.
!rsync -a --delete --info=progress2 "{CURATED_DIR}/" "{STAGING_DIR}/"
!rm -rf "{STAGING_DIR}/.cache"

!du -sh "{STAGING_DIR}"
!find "{STAGING_DIR}" -type f | wc -l
!rukopys pack-curated --dataset-dir {STAGING_DIR}

!rukopys upload-dataset \
  --dataset-dir {STAGING_DIR} \
  --repo-id {HF_DATASET_ID} \
  --private \
  --pack \
  --replace-existing

print(f"https://huggingface.co/datasets/{HF_DATASET_ID}")

# %% [markdown]
# ## Inference and submission

# %%
!rukopys infer \
  --mode page-vlm \
  --test-dir {RAW_DIR}/test \
  --vlm-model {RUNS_DIR}/qwen3_vl_2b_page_t4 \
  --max-pixels 262144 \
  --page-max-new-tokens 1536 \
  --batch-size 4 \
  --output-jsonl {OUTPUTS_DIR}/page_predictions.jsonl

# %%
!rukopys make-submission \
  --predictions {OUTPUTS_DIR}/page_predictions.jsonl \
  --sample-submission {RAW_DIR}/sample_submission.csv \
  --output {OUTPUTS_DIR}/submission.csv

# %% [markdown]
# ## Kaggle submit
#
# Upload `kaggle.json` to Colab first, and accept the competition rules in the Kaggle web UI.

# %%
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

# %%
!rukopys submit-kaggle \
  --submission {OUTPUTS_DIR}/submission.csv \
  --message "page-vlm qlora colab"
