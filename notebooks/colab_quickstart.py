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
# ## Download and curate

# %%
!rukopys download --output {RAW_DIR}

# %%
!rukopys curate \
  --raw-dir {RAW_DIR} \
  --output-dir {CURATED_DIR} \
  --include-silver \
  --max-silver 1000 \
  --crop-images

# %% [markdown]
# ## T4 smoke-test QLoRA

# %%
HF_MODEL_ID = "AlexandreSheva/rukopys-page-vlm-t4"
!rukopys train-vlm-qlora \
  --train-jsonl {CURATED_DIR}/page_sft.jsonl \
  --preset colab_t4_fast \
  --output-dir {RUNS_DIR}/vlm_page_t4 \
  --sample-limit 300 \
  --max-steps 100 \
  --batch-size 1 \
  --push-to-hub \
  --hub-model-id {HF_MODEL_ID}

# %% [markdown]
# ## L4/A100 practical QLoRA
#
# Use this cell instead of the T4 smoke-test when you have enough VRAM.

# %%
HF_MODEL_ID = "AlexandreSheva/rukopys-page-vlm-3b"
!rukopys train-vlm-qlora \
  --train-jsonl {CURATED_DIR}/page_sft.jsonl \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --output-dir {RUNS_DIR}/vlm_page_3b \
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

# %%
HF_DATASET_ID = "AlexandreSheva/rukopys-curated-mvp"
!rukopys upload-dataset \
  --dataset-dir {CURATED_DIR} \
  --repo-id {HF_DATASET_ID} \
  --private

# %% [markdown]
# ## Inference and submission

# %%
!rukopys infer \
  --mode page-vlm \
  --test-dir {RAW_DIR}/test \
  --vlm-model {RUNS_DIR}/vlm_page_3b \
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
