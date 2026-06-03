from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from rukopys_htr.vlm_loading import (
    _load_processor_with_chat_template,
    configure_processor_pixels,
    load_vision_model_and_processor,
    prepare_vlm_image,
    resolve_pixel_budget,
)


def test_configure_processor_pixels_updates_size_dict() -> None:
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            max_pixels=16_777_216,
            min_pixels=3_136,
            size={"longest_edge": 16_777_216, "shortest_edge": 3_136},
        )
    )

    configure_processor_pixels(processor, 262_144)

    image_processor = processor.image_processor
    assert image_processor.max_pixels == 262_144
    assert image_processor.min_pixels == 65_536
    assert image_processor.size == {"longest_edge": 262_144, "shortest_edge": 65_536}


def test_configure_processor_pixels_handles_none_min_pixels() -> None:
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            max_pixels=16_777_216,
            min_pixels=None,
            size={"longest_edge": 16_777_216, "shortest_edge": None},
        )
    )

    configure_processor_pixels(processor, 802_816)

    image_processor = processor.image_processor
    assert image_processor.max_pixels == 802_816
    assert image_processor.min_pixels == 200_704
    assert image_processor.size == {"longest_edge": 802_816, "shortest_edge": 200_704}


def test_resolve_pixel_budget_applies_override() -> None:
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            max_pixels=16_777_216,
            min_pixels=3_136,
            size={"longest_edge": 16_777_216, "shortest_edge": 3_136},
        )
    )

    max_pixels, min_pixels = resolve_pixel_budget(processor, 131_072)

    assert max_pixels == 131_072
    assert min_pixels == 32_768


def test_load_vision_model_explains_missing_absolute_path(tmp_path) -> None:
    missing = tmp_path / "missing_adapter"

    with pytest.raises(FileNotFoundError, match="rukopys download-vlm"):
        load_vision_model_and_processor(missing.resolve())


def test_load_processor_falls_back_when_local_processor_has_no_chat_template(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "processor_config.json").write_text("{}", encoding="utf-8")
    calls = []

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(source):
            calls.append(str(source))
            if str(source) == str(adapter_dir):
                return SimpleNamespace(chat_template=None)
            return SimpleNamespace(
                chat_template=(
                    "{% for message in messages %}{{ message.content }}{% endfor %}"
                ),
            )

    processor, source = _load_processor_with_chat_template(
        FakeAutoProcessor,
        adapter_dir,
        "Qwen/Qwen3-VL-8B-Instruct",
    )

    assert processor.chat_template
    assert source == "Qwen/Qwen3-VL-8B-Instruct"
    assert calls == [str(adapter_dir), "Qwen/Qwen3-VL-8B-Instruct"]


@pytest.mark.parametrize(
    ("width", "height", "max_pixels"),
    [
        (4000, 6000, 262_144),
        (8000, 12000, 131_072),
    ],
)
def test_prepare_vlm_image_downscales_large_pages(
    width: int,
    height: int,
    max_pixels: int,
) -> None:
    pytest.importorskip("qwen_vl_utils")
    image = Image.new("RGB", (width, height), color="white")
    resized = prepare_vlm_image(image, max_pixels=max_pixels)
    assert resized.width * resized.height <= max_pixels
