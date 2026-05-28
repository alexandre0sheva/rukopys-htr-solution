from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from huggingface_hub.utils import RepositoryNotFoundError

from rukopys_htr.cards import write_model_card
from rukopys_htr.upload import _split_top_level_paths, clear_hub_repo, upload_folder_to_hub


def _mock_hf_api(api: MagicMock):
    hf_api_cls = MagicMock(return_value=api)
    return patch(
        "rukopys_htr.upload._require_hf_api",
        return_value=(hf_api_cls, Exception, RepositoryNotFoundError),
    )


def test_split_top_level_paths() -> None:
    folders, root_files = _split_top_level_paths(
        [
            "crops/train/a.jpg",
            "images/test/b.jpg",
            "metadata.jsonl",
            "README.md",
            ".gitattributes",
        ]
    )
    assert folders == ["crops", "images"]
    assert root_files == ["README.md", "metadata.jsonl"]


def test_clear_hub_repo_deletes_top_level_folders() -> None:
    api = MagicMock()
    api.list_repo_files.return_value = [
        "crops/train/a.jpg",
        "images/train/b.jpg",
        "metadata.jsonl",
        "README.md",
    ]

    with _mock_hf_api(api):
        action = clear_hub_repo("example-org/dataset", "dataset", private=True)

    assert action == "cleared:3"
    assert api.delete_folder.call_count == 2
    api.delete_folder.assert_any_call(
        path_in_repo="crops",
        repo_id="example-org/dataset",
        repo_type="dataset",
        commit_message="Remove previous dataset folder crops",
    )
    api.delete_files.assert_called_once()
    api.delete_repo.assert_not_called()


def test_clear_hub_repo_can_recreate_repo() -> None:
    api = MagicMock()
    api.list_repo_files.return_value = ["metadata.jsonl"]

    with _mock_hf_api(api):
        action = clear_hub_repo("example-org/dataset", "dataset", recreate=True)

    assert action == "recreated"
    api.delete_repo.assert_called_once()
    api.delete_folder.assert_not_called()


def test_clear_hub_repo_skips_delete_when_empty() -> None:
    api = MagicMock()
    api.list_repo_files.return_value = []

    with _mock_hf_api(api):
        action = clear_hub_repo("example-org/dataset", "dataset", private=False)

    assert action == "empty"
    api.delete_repo.assert_not_called()
    api.delete_folder.assert_not_called()
    api.create_repo.assert_called_once()


def test_upload_dataset_uses_large_folder_uploader(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "curated"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.jsonl").write_text("{}\n", encoding="utf-8")

    api = MagicMock()
    api.list_repo_files.return_value = ["crops/train/a.jpg"]

    with _mock_hf_api(api):
        url = upload_folder_to_hub(
            dataset_dir,
            repo_id="example-org/dataset",
            repo_type="dataset",
            replace_existing=True,
        )

    assert url.endswith("datasets/example-org/dataset")
    api.delete_folder.assert_called_once()
    api.delete_repo.assert_not_called()
    api.upload_large_folder.assert_called_once()
    api.upload_folder.assert_not_called()


def test_model_card_describes_a100_v2_adapter(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct", "r": 32, "lora_alpha": 64}',
        encoding="utf-8",
    )

    readme = write_model_card(
        model_dir,
        repo_id="example-org/rukopys-qwen3-vl-8b-page-a100-v2",
    )

    card = readme.read_text(encoding="utf-8")
    assert "RUKOPYS Qwen3-VL 8B Page LoRA (A100 v2)" in card
    assert "preferred release over the original page adapter" in card
    assert "page-level Ukrainian handwriting" in card
