from __future__ import annotations

import json
import logging
import tarfile
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)

PACK_MANIFEST = "_pack_manifest.json"
PACK_VERSION = 1
SHARD_PREFIX = "shard-"
SHARD_SUFFIX = ".tar"
DEFAULT_MAX_FILES_PER_SHARD = 2000
HUB_MAX_FILES_PER_DIRECTORY = 10_000
HUB_SAFE_FILES_PER_DIRECTORY = 9_000

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
        if _is_shard(path):
            continue
        files.append(path)
    return files


def has_loose_packable_files(dataset_dir: Path, dirs: list[str] | None = None) -> bool:
    target_dirs = dirs or PACKABLE_DIRS
    for rel_dir in target_dirs:
        directory = dataset_dir / rel_dir
        if directory.is_dir() and _collect_loose_files(directory):
            return True
    return False


def needs_repack(dataset_dir: Path, dirs: list[str] | None = None) -> bool:
    if not is_packed(dataset_dir):
        return True
    return has_loose_packable_files(dataset_dir, dirs)


def count_files_per_directory(dataset_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        rel_parent = path.parent.relative_to(dataset_dir).as_posix()
        key = rel_parent or "."
        counts[key] = counts.get(key, 0) + 1
    return counts


def validate_hub_layout(
    dataset_dir: Path,
    *,
    max_files_per_directory: int = HUB_SAFE_FILES_PER_DIRECTORY,
) -> dict[str, int]:
    counts = count_files_per_directory(dataset_dir)
    offenders = {
        directory: file_count
        for directory, file_count in sorted(counts.items())
        if file_count > max_files_per_directory
    }
    if offenders:
        worst_dir, worst_count = max(offenders.items(), key=lambda item: item[1])
        raise ValueError(
            "Dataset is not safe to upload to Hugging Face: "
            f"{worst_dir!r} contains {worst_count} files "
            f"(limit {max_files_per_directory}). "
            "This usually means tar-shard packing was skipped or staging mixed "
            "packed and unpacked files. Re-run upload with a clean staging copy."
        )
    return counts


def _remove_shards(directory: Path) -> None:
    for path in directory.iterdir():
        if path.is_file() and _is_shard(path):
            path.unlink()


def _reset_pack_state(dataset_dir: Path, dirs: list[str] | None = None) -> None:
    manifest_path = dataset_dir / PACK_MANIFEST
    if manifest_path.is_file():
        manifest_path.unlink()

    for rel_dir in dirs or PACKABLE_DIRS:
        directory = dataset_dir / rel_dir
        if directory.is_dir():
            _remove_shards(directory)


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

    _remove_shards(directory)

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
    *,
    force: bool = False,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)

    target_dirs = dirs or PACKABLE_DIRS
    if (
        is_packed(dataset_dir)
        and not force
        and not has_loose_packable_files(dataset_dir, target_dirs)
    ):
        logger.info("Dataset already packed: %s", dataset_dir)
        manifest = json.loads((dataset_dir / PACK_MANIFEST).read_text(encoding="utf-8"))
        return {
            "already_packed": True,
            "packed_dirs": len(manifest.get("packed_dirs", {})),
            "total_files": sum(
                entry["file_count"] for entry in manifest.get("packed_dirs", {}).values()
            ),
        }

    if is_packed(dataset_dir) or has_loose_packable_files(dataset_dir, target_dirs):
        logger.info("Resetting stale pack state in %s", dataset_dir)
        _reset_pack_state(dataset_dir, target_dirs)

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
    validate_hub_layout(dataset_dir)
    return {
        "packed_dirs": len(packed_dirs),
        "total_files": total_files,
        "shard_count": shard_count,
    }


def ensure_packed_for_hub(
    dataset_dir: Path,
    max_files_per_shard: int = DEFAULT_MAX_FILES_PER_SHARD,
    dirs: list[str] | None = None,
) -> dict[str, Any]:
    if needs_repack(dataset_dir, dirs):
        return pack_curated(
            dataset_dir,
            max_files_per_shard=max_files_per_shard,
            dirs=dirs,
            force=needs_repack(dataset_dir, dirs),
        )

    counts = validate_hub_layout(dataset_dir)
    shard_count = sum(
        1 for path in dataset_dir.rglob("*") if path.is_file() and _is_shard(path)
    )
    return {
        "already_packed": True,
        "packed_dirs": len(json.loads((dataset_dir / PACK_MANIFEST).read_text())["packed_dirs"]),
        "total_files": sum(counts.values()),
        "shard_count": shard_count,
    }


def _collect_unpack_tasks(
    dataset_dir: Path,
    packed_dirs: dict[str, Any],
) -> list[tuple[str, str, Path]]:
    tasks: list[tuple[str, str, Path]] = []
    for rel_dir, entry in packed_dirs.items():
        directory = dataset_dir / rel_dir
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing packed directory: {directory}")

        for shard_name in entry["shards"]:
            shard_path = directory / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing shard: {shard_path}")
            tasks.append((rel_dir, shard_name, shard_path))
    return tasks


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.2f} GB"
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1_000_000:.1f} MB"
    return f"{num_bytes / 1_000:.1f} KB"


def unpack_curated(dataset_dir: Path, keep_shards: bool = False) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / PACK_MANIFEST
    if not manifest_path.is_file():
        return {"unpacked_dirs": 0, "restored_files": 0, "already_unpacked": True}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packed_dirs: dict[str, Any] = manifest.get("packed_dirs", {})
    tasks = _collect_unpack_tasks(dataset_dir, packed_dirs)
    total_files = sum(int(entry["file_count"]) for entry in packed_dirs.values())
    total_shard_bytes = sum(shard_path.stat().st_size for _, _, shard_path in tasks)
    started_at = time.monotonic()

    summary = (
        f"Unpacking curated dataset at {dataset_dir}: "
        f"{len(packed_dirs)} dirs, {len(tasks)} shards, "
        f"~{total_files} files, {_format_bytes(total_shard_bytes)} compressed"
    )
    logger.info(summary)
    print(summary, flush=True)

    restored_files = 0
    progress = tqdm(tasks, desc="Unpacking shards", unit="shard")
    for rel_dir, shard_name, shard_path in progress:
        progress.set_postfix_str(f"{rel_dir}/{shard_name}", refresh=False)
        directory = dataset_dir / rel_dir
        with tarfile.open(shard_path, "r") as archive:
            if hasattr(tarfile, "data_filter"):
                archive.extractall(path=directory, filter="data")
            else:
                archive.extractall(path=directory)
        if not keep_shards:
            shard_path.unlink()

    restored_files = total_files

    if not keep_shards:
        manifest_path.unlink()

    elapsed_seconds = time.monotonic() - started_at
    done = (
        f"Unpack complete: {restored_files} files restored in "
        f"{elapsed_seconds / 60:.1f} min "
        f"({_format_bytes(total_shard_bytes)} -> loose files on disk)"
    )
    logger.info(done)
    print(done, flush=True)

    return {
        "unpacked_dirs": len(packed_dirs),
        "restored_files": restored_files,
        "shard_count": len(tasks),
        "compressed_bytes": total_shard_bytes,
        "elapsed_seconds": round(elapsed_seconds, 1),
    }
