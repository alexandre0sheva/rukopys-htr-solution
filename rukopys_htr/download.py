from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import DEFAULT_CURATED_DATASET, SOURCE_DATASET
from .pack import is_packed, unpack_curated


def download_dataset(
    output_dir: Path,
    repo_id: str = SOURCE_DATASET,
    *,
    max_workers: int = 16,
    allow_patterns: list[str] | None = None,
) -> Path:
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
            max_workers=max_workers,
            allow_patterns=allow_patterns,
        )
    )


def download_curated_dataset(
    output_dir: Path,
    repo_id: str = DEFAULT_CURATED_DATASET,
    unpack: bool = True,
    max_workers: int = 16,
) -> tuple[Path, dict[str, Any]]:
    path = download_dataset(output_dir=output_dir, repo_id=repo_id, max_workers=max_workers)
    if not unpack:
        return path, {
            "unpacked_dirs": 0,
            "restored_files": 0,
            "already_unpacked": not is_packed(path),
            "unpack_skipped": True,
        }
    if not is_packed(path):
        print(
            f"Dataset at {path} is already unpacked (no {path.name}/_pack_manifest.json).",
            flush=True,
        )
        return path, {
            "unpacked_dirs": 0,
            "restored_files": 0,
            "already_unpacked": True,
        }
    print(f"Download finished. Starting tar-shard unpack in {path} ...", flush=True)
    stats = unpack_curated(path)
    return path, stats
