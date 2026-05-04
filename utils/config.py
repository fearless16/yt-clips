"""
utils/config.py — Load and expose config.yaml as a typed dict.
"""
from pathlib import Path
import yaml


class Config(dict):
    """A dictionary subclass that allows for safe nested access with defaults."""
    def get_nested(self, keys: str, default=None):
        """Access nested keys using dot notation (e.g., 'paths.input')."""
        curr = self
        for key in keys.split("."):
            if isinstance(curr, dict) and key in curr:
                curr = curr[key]
            else:
                return default
        return curr

def load_config(path: str = "config.yaml") -> Config:
    """Load config.yaml from the given path and return as a Config object."""
    config_path = Path(path)
    if not config_path.exists():
        # Using print here as logger might not be initialized yet
        print(f"⚠️ Warning: Config file not found at {config_path}. Using empty defaults.")
        return Config({})
    with config_path.open("r") as f:
        data = yaml.safe_load(f) or {}
        return Config(data)
