from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import DEFAULT_CURATED_DATASET, DEFAULT_DETECTOR_REPO, SOURCE_DATASET
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


def download_model(
    output_dir: Path,
    repo_id: str,
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
            repo_type="model",
            local_dir=str(output_dir),
            max_workers=max_workers,
            allow_patterns=allow_patterns,
        )
    )


def find_detector_checkpoint(model_dir: Path) -> Path:
    candidates = [
        model_dir / "weights" / "best.pt",
        model_dir / "best.pt",
        model_dir / "weights" / "last.pt",
        model_dir / "last.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checkpoints = sorted(model_dir.glob("**/*.pt"))
    if checkpoints:
        return checkpoints[0]
    raise FileNotFoundError(
        f"No detector checkpoint found under {model_dir}. "
        "Expected weights/best.pt or another .pt file."
    )


def download_detector_model(
    output_dir: Path,
    repo_id: str = DEFAULT_DETECTOR_REPO,
    *,
    max_workers: int = 16,
) -> tuple[Path, Path]:
    if repo_id == DEFAULT_DETECTOR_REPO:
        raise ValueError(
            "No public detector model is configured. Pass --repo-id with a Hugging Face model "
            "repo you control, for example your-hf-username-or-org/rukopys-yolo11m-detector."
        )
    path = download_model(
        output_dir=output_dir,
        repo_id=repo_id,
        max_workers=max_workers,
        allow_patterns=[
            "README.md",
            "args.yaml",
            "*.yaml",
            "*.pt",
            "weights/*.pt",
        ],
    )
    return path, find_detector_checkpoint(path)


def download_vlm_model(
    output_dir: Path,
    repo_id: str,
    *,
    max_workers: int = 16,
) -> Path:
    return download_model(
        output_dir=output_dir,
        repo_id=repo_id,
        max_workers=max_workers,
        allow_patterns=[
            "README.md",
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.json",
            "generation_config.json",
            "config.json",
            "merges.txt",
            "vocab.json",
            "vocab.txt",
        ],
    )


def download_curated_dataset(
    output_dir: Path,
    repo_id: str = DEFAULT_CURATED_DATASET,
    unpack: bool = True,
    max_workers: int = 16,
    unpack_workers: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    if repo_id == DEFAULT_CURATED_DATASET:
        raise ValueError(
            "No public curated dataset is configured. Pass --repo-id with a Hugging Face dataset "
            "repo you control, for example your-hf-username-or-org/rukopys-curated-mvp."
        )
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
    stats = unpack_curated(path, num_workers=unpack_workers or max_workers)
    return path, stats
