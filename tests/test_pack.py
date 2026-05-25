from __future__ import annotations

from pathlib import Path

from PIL import Image

from rukopys_htr.pack import (
    HUB_SAFE_FILES_PER_DIRECTORY,
    has_loose_packable_files,
    is_packed,
    pack_curated,
    unpack_curated,
    validate_hub_layout,
)


def _make_pack_fixture(root: Path, rel_dir: str, file_count: int) -> None:
    directory = root / rel_dir
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(file_count):
        Image.new("RGB", (16, 16), color=(index % 256, 0, 0)).save(
            directory / f"sample_{index:04d}.jpg"
        )


def test_pack_and_unpack_roundtrip(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "curated"
    _make_pack_fixture(dataset_dir, "crops/train", file_count=5)
    _make_pack_fixture(dataset_dir, "images/test", file_count=3)

    pack_stats = pack_curated(dataset_dir, max_files_per_shard=2)
    assert pack_stats["packed_dirs"] == 2
    assert pack_stats["total_files"] == 8
    assert pack_stats["shard_count"] == 5
    assert is_packed(dataset_dir)
    assert len(list((dataset_dir / "crops/train").glob("*.jpg"))) == 0
    assert len(list((dataset_dir / "crops/train").glob("shard-*.tar"))) == 3

    unpack_stats = unpack_curated(dataset_dir)
    assert unpack_stats["unpacked_dirs"] == 2
    assert unpack_stats["restored_files"] == 8
    assert not is_packed(dataset_dir)
    assert len(list((dataset_dir / "crops/train").glob("*.jpg"))) == 5
    assert len(list((dataset_dir / "images/test").glob("*.jpg"))) == 3
    assert not list((dataset_dir / "crops/train").glob("shard-*.tar"))


def test_pack_is_idempotent(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "curated"
    _make_pack_fixture(dataset_dir, "yolo/labels/train", file_count=4)

    first = pack_curated(dataset_dir, max_files_per_shard=2)
    second = pack_curated(dataset_dir, max_files_per_shard=2)

    assert first["packed_dirs"] == 1
    assert second["already_packed"] is True
    assert second["total_files"] == 4


def test_repacks_when_manifest_exists_with_loose_files(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "curated"
    _make_pack_fixture(dataset_dir, "crops/silver", file_count=6)

    first = pack_curated(dataset_dir, max_files_per_shard=2)
    assert first["shard_count"] == 3

    _make_pack_fixture(dataset_dir, "crops/silver", file_count=4)
    assert has_loose_packable_files(dataset_dir)

    second = pack_curated(dataset_dir, max_files_per_shard=2)
    assert "already_packed" not in second
    assert second["total_files"] == 4
    assert len(list((dataset_dir / "crops/silver").glob("shard-*.tar"))) == 2
    validate_hub_layout(dataset_dir)


def test_validate_hub_layout_rejects_too_many_loose_files(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "curated"
    directory = dataset_dir / "crops/silver"
    directory.mkdir(parents=True)
    for index in range(HUB_SAFE_FILES_PER_DIRECTORY + 1):
        (directory / f"sample_{index}.jpg").write_bytes(b"x")

    try:
        validate_hub_layout(dataset_dir, max_files_per_directory=100)
    except ValueError as exc:
        assert "crops/silver" in str(exc)
    else:
        raise AssertionError("expected ValueError")
