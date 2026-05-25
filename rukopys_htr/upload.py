from __future__ import annotations

import logging
from pathlib import Path

from .pack import is_packed, pack_curated

logger = logging.getLogger(__name__)

DELETE_BATCH_SIZE = 100


def _require_hf_api():
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub or run `pip install -e .` first.") from exc
    return HfApi, RepositoryNotFoundError


def clear_hub_repo(repo_id: str, repo_type: str) -> int:
    HfApi, RepositoryNotFoundError = _require_hf_api()
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        return 0

    if not files:
        return 0

    for start in range(0, len(files), DELETE_BATCH_SIZE):
        batch = files[start : start + DELETE_BATCH_SIZE]
        api.delete_files(
            repo_id=repo_id,
            repo_type=repo_type,
            delete_patterns=batch,
            commit_message="Remove previous dataset version",
        )
    logger.info("Deleted %d files from %s", len(files), repo_id)
    return len(files)


def upload_folder_to_hub(
    local_dir: Path,
    repo_id: str,
    repo_type: str,
    private: bool = False,
    commit_message: str | None = None,
    *,
    pack: bool = False,
    replace_existing: bool = False,
    max_files_per_shard: int | None = None,
) -> str:
    HfApi, _ = _require_hf_api()

    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    upload_dir = local_dir.resolve()
    if pack and not is_packed(upload_dir):
        pack_kwargs = {}
        if max_files_per_shard is not None:
            pack_kwargs["max_files_per_shard"] = max_files_per_shard
        pack_stats = pack_curated(upload_dir, **pack_kwargs)
        logger.info("Packed curated dataset before upload: %s", pack_stats)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

    if replace_existing:
        clear_hub_repo(repo_id=repo_id, repo_type=repo_type)

    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(upload_dir),
        commit_message=commit_message or f"Upload {repo_type} artifacts",
    )
    return f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
