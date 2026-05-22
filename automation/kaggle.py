"""kaggle.py — Kaggle environment management. Minimal, reuses colab watcher."""

import os
import sys
import subprocess
from pathlib import Path

from .colab import start_watcher, kill_watcher  # noqa: F401


def is_kaggle() -> bool:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    return Path("/kaggle").exists() and Path("/kaggle").is_dir()


def setup() -> dict:
    status = {"status": "ok", "steps": []}
    for d in ["/kaggle/working/data", "/kaggle/working/output", "/kaggle/working/logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    status["steps"].append("dirs")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "yt-dlp", "youtube-transcript-api", "opencv-python-headless"],
            capture_output=True, timeout=120,
        )
        status["steps"].append("pip_deps")
    except Exception:
        status["steps"].append("pip_deps_failed")
    started = start_watcher()
    status["watcher"] = started
    return status
