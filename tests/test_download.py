from __future__ import annotations

from pathlib import Path

import pytest

from rukopys_htr.download import download_curated_dataset


def test_download_curated_requires_configured_repo_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Pass --repo-id"):
        download_curated_dataset(tmp_path)
