from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def submit_to_kaggle(
    submission_csv: Path,
    competition: str = "handwritten-to-data",
    message: str = "RUKOPYS HTR submission",
) -> None:
    if not submission_csv.exists():
        raise FileNotFoundError(submission_csv)
    command = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        competition,
        "-f",
        str(submission_csv),
        "-m",
        message,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        raise RuntimeError(
            f"Kaggle submit failed for competition={competition!r}, "
            f"file={submission_csv}: {detail}"
        ) from exc
    if completed.stdout.strip():
        logger.info("Kaggle submit stdout: %s", completed.stdout.strip())
