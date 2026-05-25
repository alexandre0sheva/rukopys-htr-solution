from __future__ import annotations

import subprocess
from pathlib import Path


def submit_to_kaggle(
    submission_csv: Path,
    competition: str = "handwritten-to-data",
    message: str = "RUKOPYS HTR submission",
) -> None:
    if not submission_csv.exists():
        raise FileNotFoundError(submission_csv)
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            competition,
            "-f",
            str(submission_csv),
            "-m",
            message,
        ],
        check=True,
    )
