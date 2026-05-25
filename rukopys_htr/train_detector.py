from __future__ import annotations

from pathlib import Path


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
