from __future__ import annotations

from pathlib import Path
from typing import Any


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML or run `pip install -e .` first.") from exc

    config_path = path or default_config_path()
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def nested_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_value(
    cli_value: Any,
    preset: dict[str, Any],
    config: dict[str, Any],
    preset_key: str,
    config_keys: tuple[str, ...],
    default: Any,
) -> Any:
    if cli_value is not None:
        return cli_value
    if preset_key in preset:
        return preset[preset_key]
    config_value = nested_get(config, *config_keys, default=None)
    if config_value is not None:
        return config_value
    return default
