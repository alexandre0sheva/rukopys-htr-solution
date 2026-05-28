from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from rukopys_htr.cli import (
    _default_hf_dataset_id,
    _default_hf_model_id,
    _load_model_preset,
    main,
)
from rukopys_htr.config import load_env_file, load_yaml_config, resolve_value
from rukopys_htr.train_vlm import (
    VisionDataCollator,
    _mask_labels_to_assistant_only,
    _truncate_batch_from_right,
    _weighted_sampler_trainer_class,
)


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
            if row_idx == slice(None) and isinstance(col, slice):
                return FakeTensor([row[col] for row in self.rows])
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
            if isinstance(value, int | float):
                for index in indices:
                    self.rows[row_idx][index] = int(value)
            else:
                for index, item in zip(indices, value, strict=False):
                    self.rows[row_idx][index] = item
        else:
            self.rows[row_idx][col] = value

    def __eq__(self, other: int) -> FakeTensor:
        return FakeTensor([[1 if value == other else 0 for value in row] for row in self.rows])


def test_weighted_sampler_trainer_uses_custom_sampler() -> None:
    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def _get_train_sampler(self):
            return "default"

    custom_sampler = object()
    trainer_cls = _weighted_sampler_trainer_class(FakeTrainer)
    trainer = trainer_cls(train_sampler=custom_sampler)
    assert trainer._get_train_sampler() is custom_sampler

    default_trainer = trainer_cls(train_sampler=None)
    assert default_trainer._get_train_sampler() == "default"


def test_load_default_config() -> None:
    config = load_yaml_config()
    assert config["models"]["vlm_base"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert config["inference"]["mode"] == "detector-vlm"


def test_load_env_file_sets_hub_defaults(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "HF_NAMESPACE=example-org",
                "HF_DATASET_ID=example-org/rukopys-curated",
                'HF_MODEL_ID="example-org/rukopys-model"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("HF_NAMESPACE", raising=False)
    monkeypatch.delenv("HF_DATASET_ID", raising=False)
    monkeypatch.delenv("HF_MODEL_ID", raising=False)

    loaded = load_env_file(env_file)

    assert loaded["HF_NAMESPACE"] == "example-org"
    assert _default_hf_dataset_id({}) == "example-org/rukopys-curated"
    assert _default_hf_model_id() == "example-org/rukopys-model"


def test_upload_dataset_uses_env_repo_id(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_DATASET_ID=example-org/rukopys-curated\n", encoding="utf-8")
    dataset_dir = tmp_path / "curated"
    dataset_dir.mkdir()
    monkeypatch.delenv("HF_DATASET_ID", raising=False)

    with patch(
        "rukopys_htr.cli.upload_folder_to_hub",
        return_value="https://example.test",
    ) as upload:
        assert (
            main(["--env-file", str(env_file), "upload-dataset", "--dataset-dir", str(dataset_dir)])
            == 0
        )

    assert upload.call_args.kwargs["repo_id"] == "example-org/rukopys-curated"


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


def test_truncate_batch_from_right() -> None:
    batch = {
        "input_ids": FakeTensor([[1, 2, 3, 4, 5]]),
        "attention_mask": FakeTensor([[1, 1, 1, 1, 1]]),
        "labels": FakeTensor([[9, 9, 9, 9, 9]]),
    }
    _truncate_batch_from_right(batch, max_length=3)
    assert batch["input_ids"].rows[0] == [1, 2, 3]
    assert batch["labels"].rows[0] == [9, 9, 9]


def test_truncate_batch_from_right_rejects_vision_batches() -> None:
    batch = {
        "input_ids": FakeTensor([[1, 2, 3, 4, 5]]),
        "pixel_values": FakeTensor([[1, 2, 3]]),
    }
    try:
        _truncate_batch_from_right(batch, max_length=3)
    except ValueError as exc:
        assert "image token alignment" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


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
