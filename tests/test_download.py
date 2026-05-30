from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rukopys_htr.download import download_curated_dataset, download_detector_model


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
