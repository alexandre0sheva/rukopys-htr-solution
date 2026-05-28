from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .cards import write_model_card
from .jsonl import read_jsonl
from .prompts import transcribe_region_prompt
from .vlm_loading import (
    load_vision_model_and_processor,
    prepare_vlm_image,
    resolve_pixel_budget,
)


@dataclass(slots=True)
class QLoRAConfig:
    train_jsonl: Path
    base_model: str
    output_dir: Path
    max_steps: int = 200
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum_steps: int = 8
    max_length: int = 1024
    max_pixels: int | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    sample_limit: int | None = None
    min_quality_weight: float | None = None
    use_weighted_sampling: bool = True
    eval_fraction: float = 0.1
    eval_steps: int = 50
    gradient_checkpointing: bool = True
    push_to_hub: bool = False
    hub_model_id: str | None = None


class JsonlVisionSFTDataset:
    def __init__(
        self,
        jsonl_path: Path,
        max_length: int,
        sample_limit: int | None = None,
        min_quality_weight: float | None = None,
        rows: list[dict[str, Any]] | None = None,
    ):
        self.root = jsonl_path.parent
        self.max_length = max_length
        if rows is not None:
            self.rows = rows
        else:
            self.rows = []
            for row in read_jsonl(jsonl_path):
                if not row.get("image") or row.get("answer") is None:
                    continue
                weight = float(row.get("quality_weight", 1.0))
                if min_quality_weight is not None and weight < min_quality_weight:
                    continue
                self.rows.append(row)
                if sample_limit is not None and len(self.rows) >= sample_limit:
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def sample_weights(self) -> list[float]:
        return [max(float(row.get("quality_weight", 1.0)), 0.01) for row in self.rows]


def _truncate_batch_from_right(batch: dict[str, Any], max_length: int) -> None:
    input_ids = batch.get("input_ids")
    if input_ids is None or input_ids.shape[1] <= max_length:
        return
    if batch.get("pixel_values") is not None or batch.get("image_grid_thw") is not None:
        raise ValueError(
            f"Tokenized batch length {input_ids.shape[1]} exceeds --max-length {max_length}. "
            "Vision inputs cannot be truncated without breaking image token alignment. "
            "Lower resolution with --max-pixels or raise --max-length."
        )
    for key in ("input_ids", "attention_mask", "labels"):
        tensor = batch.get(key)
        if tensor is not None and hasattr(tensor, "shape") and tensor.shape[1] > max_length:
            batch[key] = tensor[:, :max_length]


def _mask_labels_to_assistant_only(
    processor: Any,
    labels: Any,
    prompt_texts: list[str],
    images: list[Image.Image],
    pad_token_id: int | None,
) -> None:
    for index, prompt_text in enumerate(prompt_texts):
        prompt_inputs = processor(
            text=[prompt_text],
            images=[images[index]],
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[index, :prompt_len] = -100
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100


class VisionDataCollator:
    def __init__(
        self,
        processor: Any,
        root: Path,
        max_length: int,
        *,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
    ):
        self.processor = processor
        self.root = root
        self.max_length = max_length
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        prompt_texts: list[str] = []
        images: list[Image.Image] = []
        for row in features:
            with Image.open(self.root / row["image"]) as image_file:
                image = image_file.convert("RGB")
            if self.max_pixels is not None:
                image = prepare_vlm_image(
                    image,
                    max_pixels=self.max_pixels,
                    min_pixels=self.min_pixels,
                )
            answer = row["answer"]
            if not isinstance(answer, str):
                answer = json.dumps(answer, ensure_ascii=False)
            prompt = row.get("prompt") or transcribe_region_prompt(
                source=row.get("source"),
                region_type=row.get("region_type"),
            )
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            full_messages = user_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            prompt_texts.append(
                self.processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            texts.append(self.processor.apply_chat_template(full_messages, tokenize=False))
            images.append(image)

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        _mask_labels_to_assistant_only(
            self.processor,
            labels,
            prompt_texts,
            images,
            self.processor.tokenizer.pad_token_id,
        )
        batch["labels"] = labels
        _truncate_batch_from_right(batch, self.max_length)
        return batch


def _split_train_eval_rows(
    rows: list[dict[str, Any]],
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if eval_fraction <= 0 or len(rows) < 2:
        return rows, []
    rng = random.Random(seed)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row.get("source") or "unknown"), []).append(row)

    eval_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for source_rows in by_source.values():
        source_rows = list(source_rows)
        rng.shuffle(source_rows)
        eval_count = max(1, round(len(source_rows) * eval_fraction)) if len(source_rows) > 1 else 0
        eval_rows.extend(source_rows[:eval_count])
        train_rows.extend(source_rows[eval_count:])
    return train_rows, eval_rows


def _weighted_sampler_trainer_class(Trainer: type) -> type:
    class WeightedSamplerTrainer(Trainer):
        def __init__(self, *args, train_sampler=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._train_sampler = train_sampler

        def _get_train_sampler(self, train_dataset=None):
            if self._train_sampler is not None:
                return self._train_sampler
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                return super()._get_train_sampler()

    return WeightedSamplerTrainer


def train_vlm_qlora(config: QLoRAConfig) -> Path:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install VLM extras with `pip install -e '.[vlm]'`.") from exc

    cuda_available = torch.cuda.is_available()
    cuda_bf16 = cuda_available and torch.cuda.is_bf16_supported()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _, processor, model = load_vision_model_and_processor(
        config.base_model,
        load_in_4bit=True,
        for_training=True,
        max_pixels=config.max_pixels,
    )
    if cuda_available:
        model = prepare_model_for_kbit_training(model)
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    all_rows = JsonlVisionSFTDataset(
        config.train_jsonl,
        max_length=config.max_length,
        sample_limit=config.sample_limit,
        min_quality_weight=config.min_quality_weight,
    ).rows
    if not all_rows:
        raise ValueError(f"No trainable rows found in {config.train_jsonl}")

    train_rows, eval_rows = _split_train_eval_rows(all_rows, config.eval_fraction, seed=42)
    train_dataset = JsonlVisionSFTDataset(
        config.train_jsonl,
        max_length=config.max_length,
        rows=train_rows,
    )
    eval_dataset = (
        JsonlVisionSFTDataset(
            config.train_jsonl,
            max_length=config.max_length,
            rows=eval_rows,
        )
        if eval_rows
        else None
    )

    max_pixels, min_pixels = resolve_pixel_budget(processor, config.max_pixels)
    data_collator = VisionDataCollator(
        processor,
        root=config.train_jsonl.parent,
        max_length=config.max_length,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )

    train_sampler = None
    if config.use_weighted_sampling and len(train_dataset) > 1:
        try:
            from torch.utils.data import WeightedRandomSampler

            weights = train_dataset.sample_weights()
            train_sampler = WeightedRandomSampler(
                weights=torch.tensor(weights, dtype=torch.double),
                num_samples=len(train_dataset),
                replacement=True,
            )
        except ImportError:
            train_sampler = None

    WeightedSamplerTrainer = _weighted_sampler_trainer_class(Trainer)
    args = TrainingArguments(
        output_dir=str(config.output_dir),
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        logging_steps=10,
        save_steps=max(50, min(config.max_steps, 200)),
        save_total_limit=2,
        bf16=cuda_bf16,
        fp16=cuda_available and not cuda_bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=[],
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.eval_steps if eval_dataset is not None else None,
        per_device_eval_batch_size=config.batch_size,
    )

    trainer = WeightedSamplerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        train_sampler=train_sampler,
    )
    trainer.train()
    trainer.save_model(str(config.output_dir))
    processor.save_pretrained(str(config.output_dir))
    write_model_card(
        config.output_dir,
        repo_id=config.hub_model_id,
        qlora_config=config,
        train_examples=len(train_dataset),
        eval_examples=len(eval_dataset) if eval_dataset is not None else 0,
    )
    if config.push_to_hub:
        trainer.push_to_hub()
    return config.output_dir
