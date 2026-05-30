from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import env_value, load_env_file, load_yaml_config, resolve_value
from .constants import (
    DEFAULT_CURATED_DATASET,
    DEFAULT_DETECTOR_CONFIDENCE,
    DEFAULT_DETECTOR_IOU,
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_DETECTOR_REPO,
    DEFAULT_HF_NAMESPACE,
    DEFAULT_INFERENCE_MAX_PIXELS,
    DEFAULT_PAGE_MAX_NEW_TOKENS,
    DEFAULT_REGION_MAX_NEW_TOKENS,
    DEFAULT_VLM_BASE_MODEL,
    SOURCE_DATASET,
)
from .curate import curate_dataset
from .download import download_curated_dataset, download_dataset, download_detector_model
from .evaluate import evaluate_curated_val, evaluate_predictions
from .infer import (
    EmptyDetector,
    EmptyTranscriber,
    VisionPageJsonDetector,
    VisionPageTextRecognizer,
    VisionTextGenerationTranscriber,
    YoloDetector,
    run_inference,
)
from .io import load_predictions_jsonl, write_submission
from .kaggle import submit_to_kaggle
from .pack import pack_curated, unpack_curated
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


def _arg_or_config(
    args: argparse.Namespace,
    preset: dict[str, Any],
    config: dict[str, Any],
    name: str,
    config_keys: tuple[str, ...],
    default: Any,
) -> Any:
    return resolve_value(
        getattr(args, name),
        preset,
        config,
        name,
        config_keys,
        default,
    )


def _hf_namespace() -> str | None:
    namespace = env_value("HF_NAMESPACE")
    if namespace and namespace != DEFAULT_HF_NAMESPACE:
        return namespace
    return None


def _default_hf_dataset_id(config: dict[str, Any]) -> str:
    env_dataset = env_value("HF_DATASET_ID") or env_value("RUKOPYS_CURATED_DATASET")
    if env_dataset:
        return env_dataset
    namespace = _hf_namespace()
    if namespace:
        return f"{namespace}/rukopys-curated-mvp"
    configured = config.get("curated_dataset")
    if configured:
        return configured
    return DEFAULT_CURATED_DATASET


def _default_hf_model_id() -> str | None:
    env_model = env_value("HF_MODEL_ID") or env_value("RUKOPYS_HF_MODEL_ID")
    if env_model:
        return env_model
    namespace = _hf_namespace()
    if namespace:
        return f"{namespace}/rukopys-qwen3-vl-8b-page-qlora"
    return None


def _default_hf_detector_model_id(config: dict[str, Any]) -> str:
    env_model = env_value("HF_DETECTOR_MODEL_ID") or env_value("RUKOPYS_HF_DETECTOR_MODEL_ID")
    if env_model:
        return env_model
    namespace = _hf_namespace()
    if namespace:
        return f"{namespace}/rukopys-yolo11m-detector"
    configured = config.get("detector_model") or config.get("detector_repo")
    if configured:
        return configured
    return DEFAULT_DETECTOR_REPO


def _env_private(default: bool = False) -> bool:
    value = env_value("HF_PRIVATE") or env_value("RUKOPYS_HF_PRIVATE")
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _require_configured_dataset_id(repo_id: str) -> str:
    if repo_id == DEFAULT_CURATED_DATASET:
        raise ValueError("Set HF_DATASET_ID in .env or pass --repo-id.")
    return repo_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rukopys", description="RUKOPYS HTR MVP pipeline")
    parser.add_argument(
        "--env-file",
        type=_path,
        default=Path(".env"),
        help="Load dotenv-style account/repo defaults before running the command (default: .env)",
    )
    parser.add_argument(
        "--config",
        type=_path,
        help="Path to default YAML config (merged before CLI args)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download RUKOPYS from Hugging Face")
    p.add_argument("--output", type=_path, required=True)
    p.add_argument("--repo-id")
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help="Only download matching Hub paths. Can be passed more than once.",
    )

    p = sub.add_parser(
        "download-curated",
        help="Download curated dataset from Hugging Face and unpack tar shards",
    )
    p.add_argument("--output", type=_path, required=True)
    p.add_argument("--repo-id")
    p.add_argument("--no-unpack", action="store_true")
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument(
        "--unpack-workers",
        type=int,
        help="Parallel tar-shard unpack workers. Defaults to --max-workers.",
    )

    p = sub.add_parser(
        "download-detector",
        help="Download trained YOLO detector model from Hugging Face",
    )
    p.add_argument("--output", type=_path, required=True)
    p.add_argument("--repo-id")
    p.add_argument("--max-workers", type=int, default=16)

    p = sub.add_parser("curate", help="Curate raw RUKOPYS data into training artifacts")
    p.add_argument("--raw-dir", type=_path, required=True)
    p.add_argument("--output-dir", type=_path, required=True)
    p.add_argument("--include-silver", action="store_true")
    p.add_argument("--max-silver", type=int)
    p.add_argument("--crop-images", action="store_true")
    p.add_argument("--no-page-sft", action="store_true")
    p.add_argument("--val-fraction", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--num-workers",
        type=int,
        help="Parallel image copy/crop workers for curation. Does not change labels or filtering.",
    )

    p = sub.add_parser("pack-curated", help="Pack curated image directories into tar shards")
    p.add_argument("--dataset-dir", type=_path, required=True)
    p.add_argument("--max-files-per-shard", type=int, default=2000)

    p = sub.add_parser("unpack-curated", help="Restore loose files from tar shards")
    p.add_argument("--dataset-dir", type=_path, required=True)
    p.add_argument("--keep-shards", action="store_true")
    p.add_argument("--num-workers", type=int, default=1)

    p = sub.add_parser("train-detector", help="Train YOLO layout detector")
    p.add_argument("--data-yaml", type=_path, required=True)
    p.add_argument("--model")
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
    p.add_argument("--max-pixels", type=int)
    p.add_argument("--lora-r", type=int)
    p.add_argument("--lora-alpha", type=int)
    p.add_argument("--sample-limit", type=int)
    p.add_argument("--min-quality-weight", type=float)
    p.add_argument("--no-weighted-sampling", action="store_true")
    p.add_argument("--eval-fraction", type=float)
    p.add_argument("--eval-steps", type=int)
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
        choices=[
            "empty",
            "detector-vlm",
            "page-vlm",
            "page-text-vlm",
            "detector-page-text",
            "ensemble",
        ],
    )
    p.add_argument("--detector-confidence", type=float)
    p.add_argument("--detector-iou", type=float)
    p.add_argument("--region-max-new-tokens", type=int)
    p.add_argument("--page-max-new-tokens", type=int)
    p.add_argument(
        "--batch-size",
        type=int,
        help="Number of pages to generate at once for page-vlm inference.",
    )
    p.add_argument(
        "--max-pixels",
        type=int,
        help="Cap image resolution for VLM inference (match training preset on low VRAM GPUs).",
    )
    p.add_argument("--ensemble-iou-threshold", type=float, default=0.5)
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
    p.add_argument("--raw-dir", type=_path)
    p.add_argument("--curated-dir", type=_path)
    p.add_argument("--predictions", type=_path, required=True)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--split", choices=["train", "yolo_val"], default="train")
    p.add_argument("--by-source", action="store_true")
    p.add_argument("--by-annotation-source", action="store_true")

    p = sub.add_parser("upload-dataset", help="Upload curated dataset folder to Hugging Face")
    p.add_argument("--dataset-dir", type=_path, required=True)
    p.add_argument("--repo-id")
    p.add_argument("--private", action="store_true")
    p.add_argument(
        "--pack",
        action="store_true",
        default=True,
        help="Pack image directories into tar shards before upload (default: true)",
    )
    p.add_argument(
        "--no-pack",
        action="store_false",
        dest="pack",
        help="Upload loose files without packing",
    )
    p.add_argument(
        "--replace-existing",
        action="store_true",
        default=True,
        help="Delete all previous Hub files before upload (default: true)",
    )
    p.add_argument(
        "--no-replace-existing",
        action="store_false",
        dest="replace_existing",
        help="Keep previous Hub files and upload additively",
    )
    p.add_argument(
        "--recreate-repo",
        action="store_true",
        help="Delete and recreate the Hub repo when replacing (resets download stats)",
    )
    p.add_argument("--max-files-per-shard", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=8)

    p = sub.add_parser("upload-model", help="Upload model artifact folder to Hugging Face")
    p.add_argument("--model-dir", type=_path, required=True)
    p.add_argument("--repo-id")
    p.add_argument("--private", action="store_true")

    p = sub.add_parser("submit-kaggle", help="Submit submission.csv to Kaggle")
    p.add_argument("--submission", type=_path, required=True)
    p.add_argument("--competition", default="handwritten-to-data")
    p.add_argument("--message", default="RUKOPYS HTR submission")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_env_file(args.env_file)
    config = load_yaml_config(args.config)

    if args.command == "download":
        path = download_dataset(
            output_dir=args.output,
            repo_id=args.repo_id or config.get("source_dataset", SOURCE_DATASET),
            max_workers=args.max_workers,
            allow_patterns=args.allow_patterns,
        )
        print(path)
        return 0

    if args.command == "download-curated":
        path, stats = download_curated_dataset(
            output_dir=args.output,
            repo_id=args.repo_id or _default_hf_dataset_id(config),
            unpack=not args.no_unpack,
            max_workers=args.max_workers,
            unpack_workers=args.unpack_workers,
        )
        print(json.dumps({"path": str(path), **stats}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "download-detector":
        path, checkpoint = download_detector_model(
            output_dir=args.output,
            repo_id=args.repo_id or _default_hf_detector_model_id(config),
            max_workers=args.max_workers,
        )
        print(
            json.dumps(
                {"path": str(path), "checkpoint": str(checkpoint)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "curate":
        curation = config.get("curation", {})
        stats = curate_dataset(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            include_silver=args.include_silver or bool(curation.get("include_silver")),
            max_silver=(
                args.max_silver if args.max_silver is not None else curation.get("max_silver")
            ),
            crop_images=args.crop_images or bool(curation.get("crop_images")),
            page_sft=not args.no_page_sft and curation.get("page_sft", True),
            val_fraction=args.val_fraction
            if args.val_fraction is not None
            else curation.get("val_fraction", 0.15),
            seed=args.seed if args.seed is not None else curation.get("seed", 42),
            num_workers=args.num_workers
            if args.num_workers is not None
            else curation.get("num_workers", 1),
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    if args.command == "pack-curated":
        stats = pack_curated(
            dataset_dir=args.dataset_dir,
            max_files_per_shard=args.max_files_per_shard,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    if args.command == "unpack-curated":
        stats = unpack_curated(
            dataset_dir=args.dataset_dir,
            keep_shards=args.keep_shards,
            num_workers=args.num_workers,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    if args.command == "train-detector":
        models = config.get("models", {})
        path = train_detector(
            data_yaml=args.data_yaml,
            model=args.model or models.get("detector", DEFAULT_DETECTOR_MODEL),
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
                base_model=_arg_or_config(
                    args,
                    preset,
                    config,
                    "base_model",
                    ("models", "vlm_base"),
                    preset.get("vlm_base", DEFAULT_VLM_BASE_MODEL),
                ),
                output_dir=args.output_dir,
                max_steps=_arg_or_config(
                    args, preset, config, "max_steps", ("training", "max_steps"), 200
                ),
                learning_rate=_arg_or_config(
                    args,
                    preset,
                    config,
                    "learning_rate",
                    ("training", "learning_rate"),
                    2e-4,
                ),
                batch_size=_arg_or_config(
                    args, preset, config, "batch_size", ("training", "batch_size"), 1
                ),
                grad_accum_steps=_arg_or_config(
                    args,
                    preset,
                    config,
                    "grad_accum_steps",
                    ("training", "grad_accum_steps"),
                    8,
                ),
                max_length=_arg_or_config(
                    args, preset, config, "max_length", ("training", "max_length"), 1024
                ),
                max_pixels=_arg_or_config(
                    args,
                    preset,
                    config,
                    "max_pixels",
                    ("training", "max_pixels"),
                    config.get("training", {}).get("max_pixels")
                    or config.get("inference", {}).get("max_pixels", DEFAULT_INFERENCE_MAX_PIXELS),
                ),
                lora_r=_arg_or_config(args, preset, config, "lora_r", ("training", "lora_r"), 16),
                lora_alpha=_arg_or_config(
                    args, preset, config, "lora_alpha", ("training", "lora_alpha"), 32
                ),
                sample_limit=args.sample_limit,
                min_quality_weight=args.min_quality_weight,
                use_weighted_sampling=not args.no_weighted_sampling,
                eval_fraction=args.eval_fraction
                if args.eval_fraction is not None
                else config.get("training", {}).get("eval_fraction", 0.1),
                eval_steps=args.eval_steps
                if args.eval_steps is not None
                else config.get("training", {}).get("eval_steps", 50),
                gradient_checkpointing=not args.no_gradient_checkpointing,
                push_to_hub=args.push_to_hub,
                hub_model_id=args.hub_model_id or _default_hf_model_id(),
            )
        )
        print(path)
        return 0

    if args.command == "infer":
        inference = config.get("inference", {})
        mode = args.mode or inference.get("mode", "detector-vlm")
        if mode == "detector_vlm":
            mode = "detector-vlm"

        detector = (
            YoloDetector(
                model_path=args.detector_model,
                confidence=args.detector_confidence
                if args.detector_confidence is not None
                else inference.get("detector_confidence", DEFAULT_DETECTOR_CONFIDENCE),
                iou=args.detector_iou
                if args.detector_iou is not None
                else inference.get("detector_iou", DEFAULT_DETECTOR_IOU),
            )
            if args.detector_model
            else EmptyDetector()
        )
        transcriber = EmptyTranscriber()
        page_detector = None
        page_text_recognizer = None
        load_in_4bit = not args.no_load_in_4bit
        region_max_new_tokens = (
            args.region_max_new_tokens
            if args.region_max_new_tokens is not None
            else inference.get("region_max_new_tokens", DEFAULT_REGION_MAX_NEW_TOKENS)
        )
        page_max_new_tokens = (
            args.page_max_new_tokens
            if args.page_max_new_tokens is not None
            else inference.get("page_max_new_tokens", DEFAULT_PAGE_MAX_NEW_TOKENS)
        )
        max_pixels = (
            args.max_pixels
            if args.max_pixels is not None
            else inference.get("max_pixels", DEFAULT_INFERENCE_MAX_PIXELS)
        )
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else inference.get("batch_size", 1)
        )

        if mode in {"detector-vlm", "ensemble"} and args.vlm_model:
            transcriber = VisionTextGenerationTranscriber(
                args.vlm_model,
                load_in_4bit=load_in_4bit,
                max_new_tokens=region_max_new_tokens,
                max_pixels=max_pixels,
            )
        if mode in {"page-vlm", "ensemble"}:
            if not args.vlm_model:
                raise ValueError(f"--mode {mode} requires --vlm-model")
            page_detector = VisionPageJsonDetector(
                args.vlm_model,
                max_new_tokens=page_max_new_tokens,
                load_in_4bit=load_in_4bit,
                max_pixels=max_pixels,
            )
        if mode in {"page-text-vlm", "detector-page-text"}:
            if not args.vlm_model:
                raise ValueError(f"--mode {mode} requires --vlm-model")
            page_text_recognizer = VisionPageTextRecognizer(
                args.vlm_model,
                max_new_tokens=page_max_new_tokens,
                load_in_4bit=load_in_4bit,
                max_pixels=max_pixels,
            )
        if mode == "detector-page-text" and not args.detector_model:
            raise ValueError("--mode detector-page-text requires --detector-model")
        if mode == "ensemble" and not args.detector_model:
            raise ValueError("--mode ensemble requires --detector-model")

        count = run_inference(
            test_dir=args.test_dir,
            output_jsonl=args.output_jsonl,
            detector=detector,
            transcriber=transcriber,
            page_detector=page_detector,
            page_text_recognizer=page_text_recognizer,
            ensemble=mode == "ensemble",
            ensemble_iou_threshold=args.ensemble_iou_threshold,
            batch_size=batch_size,
        )
        print(f"Wrote predictions for {count} images to {args.output_jsonl}")
        return 0

    if args.command == "make-submission":
        predictions = load_predictions_jsonl(args.predictions)
        write_submission(predictions, args.sample_submission, args.output)
        print(args.output)
        return 0

    if args.command == "evaluate":
        if args.split == "yolo_val":
            if not args.curated_dir:
                raise ValueError("--split yolo_val requires --curated-dir")
            metrics = evaluate_curated_val(
                args.curated_dir,
                args.predictions,
                iou_threshold=args.iou_threshold,
                by_source=args.by_source,
                by_annotation_source=args.by_annotation_source,
            )
        else:
            if not args.raw_dir:
                raise ValueError("--split train requires --raw-dir")
            metrics = evaluate_predictions(
                args.raw_dir,
                args.predictions,
                iou_threshold=args.iou_threshold,
                split=args.split,
                by_source=args.by_source,
                by_annotation_source=args.by_annotation_source,
            )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return 0

    if args.command == "upload-dataset":
        repo_id = _require_configured_dataset_id(args.repo_id or _default_hf_dataset_id(config))
        url = upload_folder_to_hub(
            local_dir=args.dataset_dir,
            repo_id=repo_id,
            repo_type="dataset",
            private=args.private or _env_private(),
            commit_message="Upload curated RUKOPYS MVP dataset",
            pack=args.pack,
            replace_existing=args.replace_existing,
            recreate_repo=args.recreate_repo,
            max_files_per_shard=args.max_files_per_shard,
            num_workers=args.num_workers,
        )
        print(url)
        return 0

    if args.command == "upload-model":
        repo_id = args.repo_id or _default_hf_model_id()
        if not repo_id:
            raise ValueError("Set HF_MODEL_ID in .env or pass --repo-id.")
        url = upload_folder_to_hub(
            local_dir=args.model_dir,
            repo_id=repo_id,
            repo_type="model",
            private=args.private or _env_private(),
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
