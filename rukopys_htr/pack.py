from __future__ import annotations

import json
import logging
import tarfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)

PACK_MANIFEST = "_pack_manifest.json"
PACK_VERSION = 1
SHARD_PREFIX = "shard-"
SHARD_SUFFIX = ".tar"
DEFAULT_MAX_FILES_PER_SHARD = 2000

PACKABLE_DIRS = [
    "crops/train",
    "crops/silver",
    "images/train",
    "images/silver",
    "images/test",
    "yolo/images/train",
    "yolo/images/val",
    "yolo/labels/train",
    "yolo/labels/val",
]


def _shard_name(index: int) -> str:
    return f"{SHARD_PREFIX}{index:05d}{SHARD_SUFFIX}"


def is_packed(dataset_dir: Path) -> bool:
    return (dataset_dir / PACK_MANIFEST).is_file()


def _is_shard(path: Path) -> bool:
    return path.name.startswith(SHARD_PREFIX) and path.name.endswith(SHARD_SUFFIX)


def _collect_loose_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.name == PACK_MANIFEST or _is_shard(path):
            continue
        files.append(path)
    return files


def _pack_directory(
    dataset_dir: Path,
    rel_dir: str,
    max_files_per_shard: int,
) -> dict[str, Any] | None:
    directory = dataset_dir / rel_dir
    if not directory.is_dir():
        return None

    files = _collect_loose_files(directory)
    if not files:
        return None

    shards: list[str] = []
    for shard_index, start in enumerate(range(0, len(files), max_files_per_shard)):
        batch = files[start : start + max_files_per_shard]
        shard_name = _shard_name(shard_index)
        shard_path = directory / shard_name
        with tarfile.open(shard_path, "w") as archive:
            for file_path in batch:
                archive.add(file_path, arcname=file_path.name)
        for file_path in batch:
            file_path.unlink()
        shards.append(shard_name)

    return {"shards": shards, "file_count": len(files)}


def pack_curated(
    dataset_dir: Path,
    max_files_per_shard: int = DEFAULT_MAX_FILES_PER_SHARD,
    dirs: list[str] | None = None,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)

    if is_packed(dataset_dir):
        logger.info("Dataset already packed: %s", dataset_dir)
        manifest = json.loads((dataset_dir / PACK_MANIFEST).read_text(encoding="utf-8"))
        return {
            "already_packed": True,
            "packed_dirs": len(manifest.get("packed_dirs", {})),
            "total_files": sum(
                entry["file_count"] for entry in manifest.get("packed_dirs", {}).values()
            ),
        }

    target_dirs = dirs or PACKABLE_DIRS
    packed_dirs: dict[str, Any] = {}

    for rel_dir in tqdm(target_dirs, desc="Packing curated dataset"):
        entry = _pack_directory(dataset_dir, rel_dir, max_files_per_shard=max_files_per_shard)
        if entry is not None:
            packed_dirs[rel_dir] = entry

    if not packed_dirs:
        return {"packed_dirs": 0, "total_files": 0, "shard_count": 0}

    manifest = {
        "version": PACK_VERSION,
        "max_files_per_shard": max_files_per_shard,
        "packed_dirs": packed_dirs,
    }
    (dataset_dir / PACK_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    shard_count = sum(len(entry["shards"]) for entry in packed_dirs.values())
    total_files = sum(entry["file_count"] for entry in packed_dirs.values())
    return {
        "packed_dirs": len(packed_dirs),
        "total_files": total_files,
        "shard_count": shard_count,
    }


def unpack_curated(dataset_dir: Path, keep_shards: bool = False) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / PACK_MANIFEST
    if not manifest_path.is_file():
        return {"unpacked_dirs": 0, "restored_files": 0, "already_unpacked": True}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packed_dirs: dict[str, Any] = manifest.get("packed_dirs", {})
    restored_files = 0

    for rel_dir, entry in tqdm(packed_dirs.items(), desc="Unpacking curated dataset"):
        directory = dataset_dir / rel_dir
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing packed directory: {directory}")

        for shard_name in entry["shards"]:
            shard_path = directory / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing shard: {shard_path}")
            with tarfile.open(shard_path, "r") as archive:
                if hasattr(tarfile, "data_filter"):
                    archive.extractall(path=directory, filter="data")
                else:
                    archive.extractall(path=directory)
            if not keep_shards:
                shard_path.unlink()

        restored_files += entry["file_count"]

    if not keep_shards:
        manifest_path.unlink()

    return {
        "unpacked_dirs": len(packed_dirs),
        "restored_files": restored_files,
    }
