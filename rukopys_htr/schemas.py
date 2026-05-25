from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Region:
    bbox: list[int]
    type: str
    text: str = ""
    language: str = "uk"
    legibility: str = "legible"
    confidence: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Region:
        return cls(
            bbox=[int(v) for v in data.get("bbox", [])],
            type=str(data.get("type", "")),
            text="" if data.get("text") is None else str(data.get("text")),
            language=str(data.get("language", "uk")),
            legibility=str(data.get("legibility", "legible")),
            confidence=data.get("confidence"),
        )

    def to_submission_dict(self) -> dict[str, Any]:
        return {
            "bbox": [int(v) for v in self.bbox],
            "type": self.type,
            "text": self.text,
        }

    def to_full_dict(self) -> dict[str, Any]:
        out = {
            "bbox": [int(v) for v in self.bbox],
            "type": self.type,
            "language": self.language,
            "legibility": self.legibility,
            "text": self.text,
        }
        if self.confidence is not None:
            out["confidence"] = float(self.confidence)
        return out


@dataclass(slots=True)
class PageRecord:
    file_name: str
    image_width: int
    image_height: int
    source: str
    split: str
    annotation_source: str | None = None
    year: int | None = None
    regions: list[Region] = field(default_factory=list)
    fund: str | None = None
    content_type: str | None = None
    grade: int | None = None
    subject: str | None = None
    group: str | None = None
    image_path: str | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        split: str,
        base_dir: Path | None = None,
    ) -> PageRecord:
        file_name = str(data.get("file_name") or data.get("image", {}).get("path") or "")
        if not file_name and data.get("image", {}).get("src"):
            file_name = Path(str(data["image"]["src"])).name
        if file_name.startswith("images/"):
            rel_file = file_name
        else:
            rel_file = f"images/{Path(file_name).name}" if file_name else ""

        image_path = str(base_dir / split / rel_file) if base_dir and rel_file else None
        raw_regions = data.get("regions") or []
        regions = [Region.from_dict(item) for item in raw_regions if isinstance(item, dict)]

        return cls(
            file_name=rel_file,
            image_width=int(data.get("image_width") or data.get("image", {}).get("width") or 0),
            image_height=int(data.get("image_height") or data.get("image", {}).get("height") or 0),
            source=str(data.get("source", "")),
            split=split,
            annotation_source=data.get("annotation_source"),
            year=data.get("year"),
            regions=regions,
            fund=data.get("fund"),
            content_type=data.get("content_type"),
            grade=data.get("grade"),
            subject=data.get("subject"),
            group=data.get("group"),
            image_path=image_path,
        )

    @property
    def image_name(self) -> str:
        return Path(self.file_name).name

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "source": self.source,
            "split": self.split,
            "annotation_source": self.annotation_source,
            "year": self.year,
            "regions": [region.to_full_dict() for region in self.regions],
            "fund": self.fund,
            "content_type": self.content_type,
            "grade": self.grade,
            "subject": self.subject,
            "group": self.group,
            "image_path": self.image_path,
        }
