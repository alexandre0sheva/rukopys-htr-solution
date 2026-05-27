from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from .constants import DEFAULT_INFERENCE_MAX_PIXELS

logger = logging.getLogger(__name__)

QWEN3_VL_IMAGE_FACTOR = 32


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def configure_processor_pixels(processor: Any, max_pixels: int | None = None) -> None:
    if max_pixels is None:
        return
    image_processor = processor.image_processor
    min_pixels = min(max_pixels // 4, max_pixels)
    image_processor.max_pixels = max_pixels
    image_processor.min_pixels = min_pixels
    # Qwen VL resize reads size["longest_edge"/"shortest_edge"], not max_pixels alone.
    image_processor.size = {
        "longest_edge": max_pixels,
        "shortest_edge": min_pixels,
    }


def resolve_pixel_budget(
    processor: Any,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    if max_pixels is not None:
        configure_processor_pixels(processor, max_pixels)
    image_processor = processor.image_processor
    size = getattr(image_processor, "size", None) or {}
    resolved_max = (
        _positive_int_or_none(size.get("longest_edge"))
        or _positive_int_or_none(getattr(image_processor, "max_pixels", None))
        or DEFAULT_INFERENCE_MAX_PIXELS
    )
    resolved_min = (
        _positive_int_or_none(size.get("shortest_edge"))
        or _positive_int_or_none(getattr(image_processor, "min_pixels", None))
        or min(resolved_max // 4, resolved_max)
    )
    return resolved_max, resolved_min


def prepare_vlm_image(
    image: Image.Image,
    *,
    max_pixels: int,
    min_pixels: int | None = None,
    factor: int = QWEN3_VL_IMAGE_FACTOR,
) -> Image.Image:
    try:
        from qwen_vl_utils.vision_process import smart_resize
    except ImportError as exc:
        raise RuntimeError("Install VLM extras with `pip install -e '.[vlm]'`.") from exc

    min_pixels = min_pixels or min(max_pixels // 4, max_pixels)
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if (resized_width, resized_height) == (width, height):
        return image
    return image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)


def _processor_source(model_path_obj: Path, fallback_id: str) -> str:
    if any(
        (model_path_obj / name).exists()
        for name in ("preprocessor_config.json", "processor_config.json")
    ):
        return str(model_path_obj)
    return fallback_id


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
            dtype=compute_dtype if cuda_available else torch.float32,
            device_map="auto" if cuda_available else None,
        )
        model = PeftModel.from_pretrained(base_model, model_id)
    else:
        model = AutoVisionModel.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            dtype=compute_dtype if cuda_available else torch.float32,
            device_map="auto" if cuda_available else None,
        )

    processor_source = _processor_source(model_path_obj, processor_id)
    processor = AutoProcessor.from_pretrained(processor_source)
    effective_max, effective_min = resolve_pixel_budget(processor, max_pixels)
    if not for_training:
        logger.info(
            "VLM pixel budget: max=%s min=%s (processor=%s)",
            effective_max,
            effective_min,
            processor_source,
        )
        model.eval()
    return torch, processor, model


def decode_new_tokens(processor: Any, generated: Any, input_ids: Any) -> str:
    generated_ids = generated[:, input_ids.shape[-1] :]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def decode_new_tokens_batch(processor: Any, generated: Any, input_ids: Any) -> list[str]:
    generated_ids = generated[:, input_ids.shape[-1] :]
    return [
        item.strip()
        for item in processor.batch_decode(generated_ids, skip_special_tokens=True)
    ]


def _build_generation_text(processor: Any, image: Image.Image, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _to_model_device(inputs: Any, device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in inputs.items()}


def _generation_kwargs(processor: Any, max_new_tokens: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
    }
    pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def run_vlm_generation_batch(
    torch_module: Any,
    processor: Any,
    model: Any,
    images: list[Image.Image],
    prompts: list[str],
    *,
    max_new_tokens: int = 192,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
) -> list[str]:
    if len(images) != len(prompts):
        raise ValueError("images and prompts must have the same length")
    if not images:
        return []

    prepared_images = [
        prepare_vlm_image(image, max_pixels=max_pixels, min_pixels=min_pixels)
        if max_pixels is not None
        else image
        for image in images
    ]
    texts = [
        _build_generation_text(processor, image, prompt)
        for image, prompt in zip(prepared_images, prompts, strict=True)
    ]
    tokenizer = getattr(processor, "tokenizer", None)
    previous_padding_side = getattr(tokenizer, "padding_side", None)
    if tokenizer is not None and previous_padding_side is not None:
        tokenizer.padding_side = "left"
    try:
        inputs = processor(text=texts, images=prepared_images, padding=True, return_tensors="pt")
    finally:
        if tokenizer is not None and previous_padding_side is not None:
            tokenizer.padding_side = previous_padding_side
    inputs = _to_model_device(inputs, model.device)
    with torch_module.inference_mode():
        generated = model.generate(
            **inputs,
            **_generation_kwargs(processor, max_new_tokens=max_new_tokens),
        )
    return decode_new_tokens_batch(processor, generated, inputs["input_ids"])


def run_vlm_generation(
    torch_module: Any,
    processor: Any,
    model: Any,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int = 192,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
) -> str:
    return run_vlm_generation_batch(
        torch_module,
        processor,
        model,
        [image],
        [prompt],
        max_new_tokens=max_new_tokens,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )[0]
