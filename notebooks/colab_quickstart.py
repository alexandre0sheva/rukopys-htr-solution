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

# %% [markdown]
# If Colab upgraded CUDA or PyTorch packages, restart the runtime once, then continue from here.

# %% [markdown]
# ## Mount Drive

# %%
from google.colab import drive

drive.mount("/content/drive")

WORK_ROOT = "/content/drive/MyDrive/rukopys-htr"
RAW_DIR = f"{WORK_ROOT}/data/raw/rukopys"
CURATED_DIR = f"{WORK_ROOT}/data/curated/rukopys_mvp"
RUNS_DIR = f"{WORK_ROOT}/runs"
OUTPUTS_DIR = f"{WORK_ROOT}/outputs"

!mkdir -p {WORK_ROOT}/data {RUNS_DIR} {OUTPUTS_DIR}

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
# ## Download curated dataset
#
# Skip local curation by downloading the packed curated dataset from Hugging Face.
# It is unpacked automatically into the normal training layout.
#
# You still need the raw dataset for `sample_submission.csv` and the test split used at inference time.

# %%
HF_DATASET_ID = "AlexandreSheva/rukopys-curated-mvp"

!rukopys download-curated \
  --output {CURATED_DIR} \
  --repo-id {HF_DATASET_ID}

# %%
!rukopys download --output {RAW_DIR}

# %% [markdown]
# ## T4 smoke-test QLoRA

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
  --base-model Qwen/Qwen3-VL-8B-Instruct \
  --output-dir {RUNS_DIR}/qwen3_vl_8b_page \
  --max-steps 600 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --max-length 1536 \
  --lora-r 16 \
  --lora-alpha 32 \
  --push-to-hub \
  --hub-model-id {HF_MODEL_ID}

# %% [markdown]
# ## Upload curated dataset
#
# Stage to Colab local SSD first, then upload with tar-shard packing.
# `upload-dataset` packs image folders, deletes the previous Hub version, and uploads the new one.
#
# Run the curate step locally or in Colab only when you need to publish a fresh curated build:
#
# ```bash
# rukopys curate --raw-dir {RAW_DIR} --output-dir {CURATED_DIR} \
#   --include-silver --max-silver 1000 --crop-images
# ```

# %%
from pathlib import Path

HF_DATASET_ID = "AlexandreSheva/rukopys-curated-mvp"
STAGING_DIR = Path("/content/hf_upload_staging/rukopys_mvp")

STAGING_DIR.parent.mkdir(parents=True, exist_ok=True)
!rsync -a --info=progress2 "{CURATED_DIR}/" "{STAGING_DIR}/"

!du -sh "{STAGING_DIR}"
!find "{STAGING_DIR}" -type f | wc -l

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
  --vlm-model {RUNS_DIR}/qwen3_vl_8b_page \
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
