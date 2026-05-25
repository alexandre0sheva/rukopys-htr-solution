from __future__ import annotations

from pathlib import Path

from rukopys_htr.cli import _load_model_preset
from rukopys_htr.config import load_yaml_config, resolve_value
from rukopys_htr.train_vlm import VisionDataCollator, _mask_labels_to_assistant_only


class FakeTensor:
    def __init__(self, rows: list[list[int]]):
        self.rows = [row[:] for row in rows]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.rows[0])

    def clone(self) -> FakeTensor:
        return FakeTensor(self.rows)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_idx, col = key
            if isinstance(col, slice):
                indices = range(*col.indices(len(self.rows[row_idx])))
                return [self.rows[row_idx][index] for index in indices]
            return self.rows[row_idx][col]
        if hasattr(key, "rows"):
            matched: list[list[int]] = []
            for row_idx, row in enumerate(self.rows):
                matched.append([
                    value if key.rows[row_idx][col_idx] else row[col_idx]
                    for col_idx, value in enumerate(row)
                ])
            return FakeTensor(matched)
        return self.rows[key]

    def __setitem__(self, key, value) -> None:
        if hasattr(key, "rows"):
            for row_idx, row in enumerate(self.rows):
                for col_idx, keep in enumerate(key.rows[row_idx]):
                    if keep:
                        row[col_idx] = value
            return
        row_idx, col = key
        if isinstance(col, slice):
            indices = range(*col.indices(len(self.rows[row_idx])))
            if isinstance(value, (int, float)):
                for index in indices:
                    self.rows[row_idx][index] = int(value)
            else:
                for index, item in zip(indices, value, strict=False):
                    self.rows[row_idx][index] = item
        else:
            self.rows[row_idx][col] = value

    def __eq__(self, other: int) -> FakeTensor:
        return FakeTensor([[1 if value == other else 0 for value in row] for row in self.rows])


def test_load_default_config() -> None:
    config = load_yaml_config()
    assert config["models"]["vlm_base"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert config["inference"]["mode"] == "detector-vlm"


def test_resolve_value_prefers_cli_over_preset_and_config() -> None:
    config = {"models": {"vlm_base": "from-config"}}
    preset = {"vlm_base": "from-preset"}
    assert (
        resolve_value("from-cli", preset, config, "vlm_base", ("models", "vlm_base"), "default")
        == "from-cli"
    )
    assert (
        resolve_value(None, preset, config, "vlm_base", ("models", "vlm_base"), "default")
        == "from-preset"
    )
    preset = {}
    assert (
        resolve_value(None, preset, config, "vlm_base", ("models", "vlm_base"), "default")
        == "from-config"
    )


def test_unknown_preset_raises() -> None:
    try:
        _load_model_preset("does_not_exist")
    except ValueError as exc:
        assert "does_not_exist" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_mask_labels_to_assistant_only() -> None:
    processor = type(
        "Processor",
        (),
        {
            "__call__": lambda self, text, images, return_tensors="pt": {
                "input_ids": FakeTensor([[9, 9, 9]])
            }
        },
    )()
    labels = FakeTensor([[1, 2, 3, 4, 0]])
    _mask_labels_to_assistant_only(
        processor,
        labels,
        prompt_texts=["prompt"],
        images=[object()],
        pad_token_id=0,
    )
    assert labels[0, 0] == -100
    assert labels[0, 1] == -100
    assert labels[0, 2] == -100
    assert labels[0, 3] == 4
    assert labels[0, 4] == -100


def test_vision_data_collator_masks_prompt_tokens(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "crop.jpg"
    Image.new("RGB", (20, 10), "white").save(image_path)

    class FakeProcessor:
        class Tokenizer:
            pad_token_id = 0

        tokenizer = Tokenizer()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            if add_generation_prompt:
                return "user prompt tokens"
            return "user prompt tokens assistant answer tokens"

        def __call__(
            self,
            text,
            images,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ):
            if len(text) == 1 and text[0] == "user prompt tokens":
                return {"input_ids": FakeTensor([[9, 9]])}
            return {"input_ids": FakeTensor([[1, 2, 3, 4, 5, 0]])}

    processor = FakeProcessor()
    collator = VisionDataCollator(processor, root=tmp_path, max_length=32)
    batch = collator(
        [
            {
                "image": "crop.jpg",
                "answer": "answer",
                "prompt": "Transcribe exactly.",
            }
        ]
    )
    labels = batch["labels"]
    assert labels[0, 0] == -100
    assert labels[0, 1] == -100
    assert labels[0, -1] == -100
    assert labels[0, 2] == 3
    assert labels[0, 3] == 4
