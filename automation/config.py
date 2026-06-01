"""Configuration loader — YAML config with dot-notation access."""

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def load(path: str | None = None) -> dict:
    path = path or "config.yaml"
    p = Path(path)
    if not p.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required for config loading")
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def get(cfg: dict, key: str, default: Any = None) -> Any:
    parts = key.split(".")
    current = cfg
    for part in parts:
        if not isinstance(current, dict):
            return default
        if part not in current:
            return default
        current = current[part]
    return current
