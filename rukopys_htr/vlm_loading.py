from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def configure_training_processor(processor: Any, max_pixels: int | None = None) -> None:
    if max_pixels is None:
        return
    image_processor = processor.image_processor
    image_processor.max_pixels = max_pixels
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        size["longest_edge"] = max_pixels


def load_vision_model_and_processor(
    model_path: str | Path,
    *,
    load_in_4bit: bool = True,
    for_training: bool = False,
    max_pixels: int | None = None,
) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install VLM extras with `pip install -e '.[vlm]'`.") from exc

    try:
        from transformers import AutoModelForImageTextToText as AutoVisionModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVisionModel

    model_id = str(model_path)
    model_path_obj = Path(model_path)
    processor_id = model_id
    cuda_available = torch.cuda.is_available()
    cuda_bf16 = cuda_available and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if cuda_bf16 else torch.float16
    quantization_config = None
    if load_in_4bit and cuda_available:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    if (model_path_obj / "adapter_config.json").exists():
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise RuntimeError("Install VLM extras with `pip install -e '.[vlm]'`.") from exc

        peft_config = PeftConfig.from_pretrained(model_id)
        processor_id = peft_config.base_model_name_or_path
        base_model = AutoVisionModel.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype if cuda_available else torch.float32,
            device_map="auto" if cuda_available else None,
        )
        model = PeftModel.from_pretrained(base_model, model_id)
    else:
        model = AutoVisionModel.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype if cuda_available else torch.float32,
            device_map="auto" if cuda_available else None,
        )

    processor = AutoProcessor.from_pretrained(processor_id)
    if for_training:
        configure_training_processor(processor, max_pixels)
    if not for_training:
        model.eval()
    return torch, processor, model


def decode_new_tokens(processor: Any, generated: Any, input_ids: Any) -> str:
    generated_ids = generated[:, input_ids.shape[-1] :]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def run_vlm_generation(
    torch_module: Any,
    processor: Any,
    model: Any,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int = 192,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch_module.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return decode_new_tokens(processor, generated, inputs["input_ids"])
