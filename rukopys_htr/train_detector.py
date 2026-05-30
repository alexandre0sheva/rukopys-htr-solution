from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".dng",
    ".heic",
    ".heif",
    ".jp2",
    ".jpeg",
    ".jpeg2000",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _resolve_yolo_split_dir(data_yaml: Path, split: str) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML or run `pip install -e .` first.") from exc

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(str(data.get("path") or data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_value = data.get(split)
    if not split_value:
        raise ValueError(f"YOLO data file {data_yaml} is missing {split!r}")
    split_path = Path(str(split_value))
    return split_path if split_path.is_absolute() else root / split_path


def _validate_yolo_images(data_yaml: Path) -> None:
    train_dir = _resolve_yolo_split_dir(data_yaml, "train")
    if not train_dir.is_dir():
        raise FileNotFoundError(f"YOLO train image directory does not exist: {train_dir}")

    images = [
        path
        for path in train_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if images:
        return

    shards = sorted(train_dir.glob("shard-*.tar"))
    if shards:
        raise RuntimeError(
            "YOLO train images are packed as tar shards, but Ultralytics needs loose images. "
            f"Run: rukopys unpack-curated --dataset-dir {data_yaml.parent.parent}"
        )
    raise FileNotFoundError(f"No YOLO train images found in {train_dir}")


def train_detector(
    data_yaml: Path,
    model: str,
    output_dir: Path,
    epochs: int = 30,
    image_size: int = 1280,
    batch: int = 4,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install detector extras with `pip install -e '.[detector]'`.") from exc

    _validate_yolo_images(data_yaml)
    output_dir.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(model)
    result = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )
    save_dir = Path(getattr(result, "save_dir", output_dir))
    return save_dir
