from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from .cards import write_model_card
from .pack import ensure_packed_for_hub, validate_hub_layout

logger = logging.getLogger(__name__)

T = TypeVar("T")

LARGE_FOLDER_FILE_THRESHOLD = 500
LARGE_FOLDER_BYTES_THRESHOLD = 1_000_000_000
GITATTRIBUTES = ".gitattributes"


def _require_hf_api():
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub or run `pip install -e .` first.") from exc
    return HfApi, HfHubHTTPError, RepositoryNotFoundError


def _retry_on_rate_limit(func: Callable[[], T], *, action: str, max_retries: int = 6) -> T:
    _, HfHubHTTPError, _ = _require_hf_api()
    for attempt in range(max_retries):
        try:
            return func()
        except HfHubHTTPError as exc:
            response = exc.response
            if response is None or response.status_code != 429 or attempt == max_retries - 1:
                raise
            retry_after = response.headers.get("Retry-After")
            wait_seconds = int(retry_after) if retry_after else min(300, 30 * (attempt + 1))
            logger.warning(
                "Hugging Face rate limit while %s; retrying in %ss (%d/%d)",
                action,
                wait_seconds,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"Failed to {action} after rate-limit retries")


def _folder_upload_stats(local_dir: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for path in local_dir.rglob("*"):
        if path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
    return file_count, total_bytes


def _split_top_level_paths(files: list[str]) -> tuple[list[str], list[str]]:
    folders: set[str] = set()
    root_files: set[str] = set()
    for path in files:
        if path == GITATTRIBUTES:
            continue
        if "/" in path:
            folders.add(path.split("/", 1)[0])
        else:
            root_files.add(path)
    return sorted(folders), sorted(root_files)


def clear_hub_repo(
    repo_id: str,
    repo_type: str,
    *,
    private: bool = False,
    recreate: bool = False,
) -> str:
    """Clear Hub repo contents before a full re-upload.

    By default deletes top-level folders/files in a handful of commits so the repo
    entity (download stats, likes, URL) is preserved. Pass ``recreate=True`` to
    delete and recreate the repo, which resets Hub stats but is more aggressive.
    """
    HfApi, _, RepositoryNotFoundError = _require_hf_api()
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        files = []

    if not files:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
        return "empty"

    if recreate:
        logger.info(
            "Recreating %s (%d existing files); Hub stats will reset",
            repo_id,
            len(files),
        )

        def _recreate() -> None:
            api.delete_repo(repo_id=repo_id, repo_type=repo_type, missing_ok=True)
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

        _retry_on_rate_limit(_recreate, action=f"recreating {repo_id}")
        return "recreated"

    folders, root_files = _split_top_level_paths(files)
    logger.info(
        "Clearing %s (%d files) via %d folder delete(s) and %d root file(s)",
        repo_id,
        len(files),
        len(folders),
        len(root_files),
    )

    for folder in folders:
        def _delete_folder(path_in_repo: str = folder) -> None:
            api.delete_folder(
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Remove previous {repo_type} folder {path_in_repo}",
            )

        _retry_on_rate_limit(_delete_folder, action=f"deleting {repo_id}/{folder}")

    if root_files:
        def _delete_root_files() -> None:
            api.delete_files(
                repo_id=repo_id,
                repo_type=repo_type,
                delete_patterns=root_files,
                commit_message="Remove previous dataset root files",
            )

        _retry_on_rate_limit(_delete_root_files, action=f"deleting root files in {repo_id}")

    return f"cleared:{len(folders) + (1 if root_files else 0)}"


def upload_folder_to_hub(
    local_dir: Path,
    repo_id: str,
    repo_type: str,
    private: bool = False,
    commit_message: str | None = None,
    *,
    pack: bool = False,
    replace_existing: bool = False,
    recreate_repo: bool = False,
    max_files_per_shard: int | None = None,
    num_workers: int = 8,
) -> str:
    HfApi, _, _ = _require_hf_api()

    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    upload_dir = local_dir.resolve()
    if repo_type == "model":
        write_model_card(upload_dir, repo_id=repo_id)

    if pack:
        pack_kwargs = {}
        if max_files_per_shard is not None:
            pack_kwargs["max_files_per_shard"] = max_files_per_shard
        pack_stats = ensure_packed_for_hub(upload_dir, **pack_kwargs)
        logger.info("Packed curated dataset before upload: %s", pack_stats)
        validate_hub_layout(upload_dir)

    api = HfApi()
    if replace_existing:
        clear_hub_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            recreate=recreate_repo,
        )
    else:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

    file_count, total_bytes = _folder_upload_stats(upload_dir)
    use_large_uploader = (
        repo_type == "dataset"
        or file_count >= LARGE_FOLDER_FILE_THRESHOLD
        or total_bytes >= LARGE_FOLDER_BYTES_THRESHOLD
    )

    if use_large_uploader:
        logger.info(
            "Uploading with upload_large_folder (%d files, %.2f GB)",
            file_count,
            total_bytes / (1024**3),
        )

        def _upload_large() -> None:
            api.upload_large_folder(
                repo_id=repo_id,
                folder_path=str(upload_dir),
                repo_type=repo_type,
                private=private,
                num_workers=num_workers,
            )

        _retry_on_rate_limit(_upload_large, action=f"uploading {repo_id}")
    else:

        def _upload() -> None:
            api.upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=str(upload_dir),
                commit_message=commit_message or f"Upload {repo_type} artifacts",
            )

        _retry_on_rate_limit(_upload, action=f"uploading {repo_id}")

    return f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
