from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rukopys_htr.download import (
    download_curated_dataset,
    download_detector_model,
    download_vlm_model,
)


def test_download_curated_requires_configured_repo_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Pass --repo-id"):
        download_curated_dataset(tmp_path)


def test_download_detector_requires_configured_repo_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Pass --repo-id"):
        download_detector_model(tmp_path)


def test_download_detector_returns_best_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        local_dir = Path(kwargs["local_dir"])
        weights_dir = local_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"fake")
        return str(local_dir)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    path, checkpoint = download_detector_model(
        tmp_path / "detector",
        repo_id="example-org/rukopys-yolo11m-detector",
    )

    assert path == tmp_path / "detector"
    assert checkpoint == tmp_path / "detector" / "weights" / "best.pt"
    assert calls["repo_type"] == "model"
    assert calls["allow_patterns"] == ["README.md", "args.yaml", "*.yaml", "*.pt", "weights/*.pt"]


def test_download_vlm_downloads_adapter_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    path = download_vlm_model(
        tmp_path / "vlm",
        repo_id="example-org/rukopys-qwen3-vl-8b-page-text",
    )

    assert path == tmp_path / "vlm"
    assert calls["repo_type"] == "model"
    assert "adapter_config.json" in calls["allow_patterns"]
    assert "adapter_model.safetensors" in calls["allow_patterns"]
    assert "processor_config.json" in calls["allow_patterns"]
