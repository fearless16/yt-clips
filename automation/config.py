"""config.py — Cached YAML config loader.

Reads config.yaml lazily with 10-minute TTL caching.
Supports dot-notation access via get("section.key").

Usage::

    from .config import get, load, log_path, log_level
    cfg = load()
    val = get("paths.input", default="data/")
"""

from pathlib import Path

from ._cache import CONFIG_CACHE


def _load_yaml(path: str) -> dict:
    import yaml
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r") as f:
        data = yaml.safe_load(f) or {}
    return data


def load(path: str = "config.yaml") -> dict:
    """Load and cache the config YAML file.

    Results cached 10 minutes in CONFIG_CACHE.
    Returns empty dict if file does not exist.
    """
    cached = CONFIG_CACHE.get(path)
    if cached is not None:
        return cached
    cfg = _load_yaml(path)
    CONFIG_CACHE.set(path, cfg)
    return cfg


def get(key: str, default=None):
    """Dot-notation access to config values.

    Example::

        get("paths.input")          # -> config["paths"]["input"]
        get("nonexistent.key", 42)  # -> 42
    """
    parts = key.split(".")
    cur = load()
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
            if cur is None:
                return default
        else:
            return default
    return cur


def log_path() -> str:
    """Return configured log file path, or 'logs/pipeline.log'."""
    return get("logging.log_file", "logs/pipeline.log")


def log_level() -> str:
    """Return configured log level, or 'INFO'."""
    return get("logging.level", "INFO")
