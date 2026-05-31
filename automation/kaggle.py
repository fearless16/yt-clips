"""kaggle.py — Kaggle environment setup.

Minimal module: detects Kaggle, creates directories, installs deps,
and starts the watcher (reused from watcher.py).
"""

import sys
import subprocess
from pathlib import Path

from .env import is_kaggle  # noqa: F401
from .watcher import start_watcher, kill_watcher  # noqa: F401


def setup() -> dict:
    """Create directories and install Python deps for Kaggle.

    Also starts the watcher subprocess. Returns status dict.
    """
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
