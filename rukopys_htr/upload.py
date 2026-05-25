from __future__ import annotations

from pathlib import Path


def upload_folder_to_hub(
    local_dir: Path,
    repo_id: str,
    repo_type: str,
    private: bool = False,
    commit_message: str | None = None,
) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub or run `pip install -e .` first.") from exc

    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        commit_message=commit_message or f"Upload {repo_type} artifacts",
    )
    return f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
