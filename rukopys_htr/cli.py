from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .constants import DEFAULT_DETECTOR_MODEL, DEFAULT_VLM_BASE_MODEL
from .curate import curate_dataset
from .download import download_dataset
from .evaluate import evaluate_predictions
from .infer import (
    EmptyDetector,
    EmptyTranscriber,
    VisionPageJsonDetector,
    VisionTextGenerationTranscriber,
    YoloDetector,
    run_inference,
)
from .io import load_predictions_jsonl, write_submission
from .kaggle import submit_to_kaggle
from .train_detector import train_detector
from .train_vlm import QLoRAConfig, train_vlm_qlora
from .upload import upload_folder_to_hub


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _load_model_preset(name: str | None) -> dict[str, Any]:
    if not name:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML or run `pip install -e .` first.") from exc

    config_path = Path(__file__).resolve().parent.parent / "configs" / "model_presets.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        presets = (yaml.safe_load(handle) or {}).get("presets", {})
    if name not in presets:
        available = ", ".join(sorted(presets)) or "none"
        raise ValueError(f"Unknown preset {name!r}. Available presets: {available}")
    return presets[name] or {}


def _arg_or_preset(
    args: argparse.Namespace,
    preset: dict[str, Any],
    name: str,
    default: Any,
) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return preset.get(name, default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rukopys", description="RUKOPYS HTR MVP pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download RUKOPYS from Hugging Face")
    p.add_argument("--output", type=_path, required=True)
    p.add_argument("--repo-id", default="UkrainianCatholicUniversity/rukopys")

    p = sub.add_parser("curate", help="Curate raw RUKOPYS data into training artifacts")
    p.add_argument("--raw-dir", type=_path, required=True)
    p.add_argument("--output-dir", type=_path, required=True)
    p.add_argument("--include-silver", action="store_true")
    p.add_argument("--max-silver", type=int)
    p.add_argument("--crop-images", action="store_true")
    p.add_argument("--no-page-sft", action="store_true")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("train-detector", help="Train YOLO layout detector")
    p.add_argument("--data-yaml", type=_path, required=True)
    p.add_argument("--model", default=DEFAULT_DETECTOR_MODEL)
    p.add_argument("--output-dir", type=_path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--image-size", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4)

    p = sub.add_parser("train-vlm-qlora", help="Fine-tune VLM region transcriber with QLoRA")
    p.add_argument("--train-jsonl", type=_path, required=True)
    p.add_argument("--preset", help="Model/training preset from configs/model_presets.yaml")
    p.add_argument("--base-model")
    p.add_argument("--output-dir", type=_path, required=True)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--grad-accum-steps", type=int)
    p.add_argument("--max-length", type=int)
    p.add_argument("--lora-r", type=int)
    p.add_argument("--lora-alpha", type=int)
    p.add_argument("--sample-limit", type=int)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id")

    p = sub.add_parser("infer", help="Run inference on test split and write predictions JSONL")
    p.add_argument("--test-dir", type=_path, required=True)
    p.add_argument("--output-jsonl", type=_path, required=True)
    p.add_argument("--detector-model", type=_path)
    p.add_argument("--vlm-model", type=_path)
    p.add_argument(
        "--mode",
        choices=["empty", "detector-vlm", "page-vlm"],
        default="detector-vlm",
    )
    p.add_argument("--detector-confidence", type=float, default=0.25)
    p.add_argument("--detector-iou", type=float, default=0.5)
    p.add_argument("--page-max-new-tokens", type=int, default=2048)
    p.add_argument(
        "--no-load-in-4bit",
        action="store_true",
        help="Disable 4-bit bitsandbytes loading for VLM inference.",
    )

    p = sub.add_parser("make-submission", help="Create Kaggle submission.csv")
    p.add_argument("--predictions", type=_path, required=True)
    p.add_argument("--sample-submission", type=_path, required=True)
    p.add_argument("--output", type=_path, required=True)

    p = sub.add_parser(
        "evaluate",
        help="Evaluate predictions on train metadata with a proxy metric",
    )
    p.add_argument("--raw-dir", type=_path, required=True)
    p.add_argument("--predictions", type=_path, required=True)
    p.add_argument("--iou-threshold", type=float, default=0.5)

    p = sub.add_parser("upload-dataset", help="Upload curated dataset folder to Hugging Face")
    p.add_argument("--dataset-dir", type=_path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--private", action="store_true")

    p = sub.add_parser("upload-model", help="Upload model artifact folder to Hugging Face")
    p.add_argument("--model-dir", type=_path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--private", action="store_true")

    p = sub.add_parser("submit-kaggle", help="Submit submission.csv to Kaggle")
    p.add_argument("--submission", type=_path, required=True)
    p.add_argument("--competition", default="handwritten-to-data")
    p.add_argument("--message", default="RUKOPYS HTR submission")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        path = download_dataset(output_dir=args.output, repo_id=args.repo_id)
        print(path)
        return 0

    if args.command == "curate":
        stats = curate_dataset(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            include_silver=args.include_silver,
            max_silver=args.max_silver,
            crop_images=args.crop_images,
            page_sft=not args.no_page_sft,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    if args.command == "train-detector":
        path = train_detector(
            data_yaml=args.data_yaml,
            model=args.model,
            output_dir=args.output_dir,
            epochs=args.epochs,
            image_size=args.image_size,
            batch=args.batch,
        )
        print(path)
        return 0

    if args.command == "train-vlm-qlora":
        preset = _load_model_preset(args.preset)
        path = train_vlm_qlora(
            QLoRAConfig(
                train_jsonl=args.train_jsonl,
                base_model=args.base_model or preset.get("vlm_base") or DEFAULT_VLM_BASE_MODEL,
                output_dir=args.output_dir,
                max_steps=_arg_or_preset(args, preset, "max_steps", 200),
                learning_rate=_arg_or_preset(args, preset, "learning_rate", 2e-4),
                batch_size=_arg_or_preset(args, preset, "batch_size", 1),
                grad_accum_steps=_arg_or_preset(args, preset, "grad_accum_steps", 8),
                max_length=_arg_or_preset(args, preset, "max_length", 1024),
                lora_r=_arg_or_preset(args, preset, "lora_r", 16),
                lora_alpha=_arg_or_preset(args, preset, "lora_alpha", 32),
                sample_limit=args.sample_limit,
                gradient_checkpointing=not args.no_gradient_checkpointing,
                push_to_hub=args.push_to_hub,
                hub_model_id=args.hub_model_id,
            )
        )
        print(path)
        return 0

    if args.command == "infer":
        detector = (
            YoloDetector(
                model_path=args.detector_model,
                confidence=args.detector_confidence,
                iou=args.detector_iou,
            )
            if args.detector_model
            else EmptyDetector()
        )
        transcriber = EmptyTranscriber()
        page_detector = None
        if args.mode == "detector-vlm" and args.vlm_model:
            transcriber = VisionTextGenerationTranscriber(
                args.vlm_model,
                load_in_4bit=not args.no_load_in_4bit,
            )
        if args.mode == "page-vlm":
            if not args.vlm_model:
                raise ValueError("--mode page-vlm requires --vlm-model")
            page_detector = VisionPageJsonDetector(
                args.vlm_model,
                max_new_tokens=args.page_max_new_tokens,
                load_in_4bit=not args.no_load_in_4bit,
            )
        count = run_inference(
            test_dir=args.test_dir,
            output_jsonl=args.output_jsonl,
            detector=detector,
            transcriber=transcriber,
            page_detector=page_detector,
        )
        print(f"Wrote predictions for {count} images to {args.output_jsonl}")
        return 0

    if args.command == "make-submission":
        predictions = load_predictions_jsonl(args.predictions)
        write_submission(predictions, args.sample_submission, args.output)
        print(args.output)
        return 0

    if args.command == "evaluate":
        metrics = evaluate_predictions(
            args.raw_dir,
            args.predictions,
            iou_threshold=args.iou_threshold,
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.command == "upload-dataset":
        url = upload_folder_to_hub(
            local_dir=args.dataset_dir,
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            commit_message="Upload curated RUKOPYS MVP dataset",
        )
        print(url)
        return 0

    if args.command == "upload-model":
        url = upload_folder_to_hub(
            local_dir=args.model_dir,
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            commit_message="Upload RUKOPYS MVP model artifacts",
        )
        print(url)
        return 0

    if args.command == "submit-kaggle":
        submit_to_kaggle(
            submission_csv=args.submission,
            competition=args.competition,
            message=args.message,
        )
        print(f"Submitted {args.submission} to {args.competition}")
        return 0

    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
