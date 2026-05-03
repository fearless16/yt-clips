"""
utils/config.py — Load and expose config.yaml as a typed dict.
"""
from pathlib import Path
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load config.yaml from the given path and return as a nested dict."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with config_path.open("r") as f:
        return yaml.safe_load(f)
