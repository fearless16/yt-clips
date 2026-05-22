"""config.py — Cached config loader. Lazy, fast, compact."""

from pathlib import Path
from functools import lru_cache

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
    cached = CONFIG_CACHE.get(path)
    if cached is not None:
        return cached
    cfg = _load_yaml(path)
    CONFIG_CACHE.set(path, cfg)
    return cfg


def get(key: str, default=None):
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
    return get("logging.log_file", "logs/pipeline.log")


def log_level() -> str:
    return get("logging.level", "INFO")
