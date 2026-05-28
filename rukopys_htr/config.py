from __future__ import annotations

import os
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


def load_env_file(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from a dotenv-style file into os.environ."""
    env_path = path or Path.cwd() / ".env"
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return loaded


def env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


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
