from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


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
    lora_r: int = 16
    lora_alpha: int = 32
    sample_limit: int | None = None
    gradient_checkpointing: bool = True
    push_to_hub: bool = False
    hub_model_id: str | None = None


class JsonlVisionSFTDataset:
    def __init__(self, jsonl_path: Path, max_length: int, sample_limit: int | None = None):
        self.root = jsonl_path.parent
        self.max_length = max_length
        self.rows: list[dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if row.get("image") and row.get("answer") is not None:
                        self.rows.append(row)
                    if sample_limit is not None and len(self.rows) >= sample_limit:
                        break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class VisionDataCollator:
    def __init__(self, processor: Any, root: Path, max_length: int):
        self.processor = processor
        self.root = root
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        for row in features:
            image = Image.open(self.root / row["image"]).convert("RGB")
            answer = row["answer"]
            if not isinstance(answer, str):
                answer = json.dumps(answer, ensure_ascii=False)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": row.get("prompt") or "Transcribe exactly."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False))
            images.append(image)

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        batch["labels"] = batch["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            batch["labels"][batch["labels"] == pad_token_id] = -100
        return batch


def train_vlm_qlora(config: QLoRAConfig) -> Path:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install VLM extras with `pip install -e '.[vlm]'`.") from exc

    try:
        from transformers import AutoModelForImageTextToText as AutoVisionModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVisionModel

    config.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(config.base_model)

    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoVisionModel.from_pretrained(
        config.base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if quantization_config is not None:
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

    dataset = JsonlVisionSFTDataset(
        config.train_jsonl,
        max_length=config.max_length,
        sample_limit=config.sample_limit,
    )
    if len(dataset) == 0:
        raise ValueError(f"No trainable rows found in {config.train_jsonl}")

    args = TrainingArguments(
        output_dir=str(config.output_dir),
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        logging_steps=10,
        save_steps=max(50, min(config.max_steps, 200)),
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=[],
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=VisionDataCollator(
            processor,
            root=config.train_jsonl.parent,
            max_length=config.max_length,
        ),
    )
    trainer.train()
    trainer.save_model(str(config.output_dir))
    processor.save_pretrained(str(config.output_dir))
    if config.push_to_hub:
        trainer.push_to_hub()
    return config.output_dir
