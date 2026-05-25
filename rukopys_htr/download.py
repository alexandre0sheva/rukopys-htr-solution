from __future__ import annotations

from pathlib import Path

from .constants import SOURCE_DATASET


def download_dataset(output_dir: Path, repo_id: str = SOURCE_DATASET) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub or run `pip install -e .` first.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
    )
