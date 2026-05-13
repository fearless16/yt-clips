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

# Module-level cache — config is loaded once and reused across all imports
_config_cache: dict = {}

def load_config(path: str = "config.yaml") -> Config:
    """Load config.yaml from the given path and return as a Config object.
    
    Caches the result so repeated calls (15+ modules at import time)
    return the same object without re-parsing YAML.
    """
    if path in _config_cache:
        return _config_cache[path]
    
    config_path = Path(path)
    if not config_path.exists():
        # Using print here as logger might not be initialized yet
        print(f"⚠️ Warning: Config file not found at {config_path}. Using empty defaults.")
        result = Config({})
    else:
        with config_path.open("r") as f:
            data = yaml.safe_load(f) or {}
            result = Config(data)
    
    _config_cache[path] = result
    return result

