from __future__ import annotations

from pathlib import Path

import pytest

from rukopys_htr.train_detector import _validate_yolo_images


def _write_data_yaml(root: Path) -> Path:
    yolo_dir = root / "yolo"
    yolo_dir.mkdir(parents=True)
    data_yaml = yolo_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {yolo_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: handwritten\n",
        encoding="utf-8",
    )
    return data_yaml


def test_validate_yolo_images_accepts_loose_images(tmp_path: Path) -> None:
    data_yaml = _write_data_yaml(tmp_path)
    train_dir = tmp_path / "yolo" / "images" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "page.jpg").write_bytes(b"fake")

    _validate_yolo_images(data_yaml)


def test_validate_yolo_images_explains_packed_shards(tmp_path: Path) -> None:
    data_yaml = _write_data_yaml(tmp_path)
    train_dir = tmp_path / "yolo" / "images" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "shard-00000.tar").write_bytes(b"fake")

    with pytest.raises(RuntimeError, match="unpack-curated"):
        _validate_yolo_images(data_yaml)
