"""kaggle.py — Kaggle environment setup.

Minimal module: detects Kaggle, creates directories, installs deps,
and starts the watcher (reused from watcher.py).
"""

import sys
import logging
import subprocess
from pathlib import Path

from utils.logger import get_logger
from .env import is_kaggle  # noqa: F401
from .watcher import start_watcher, kill_watcher  # noqa: F401

log = get_logger("kaggle")


def setup() -> dict:
    """Create directories and install Python deps for Kaggle.

    Also starts the watcher subprocess. Returns status dict.
    """
    status = {"status": "ok", "steps": []}
    log.info("Setting up Kaggle environment ...")
    for d in ["/kaggle/working/data", "/kaggle/working/output", "/kaggle/working/logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    status["steps"].append("dirs")
    log.info("Directories created")
    try:
        ret = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "yt-dlp", "youtube-transcript-api", "opencv-python-headless"],
            capture_output=True, timeout=120,
        )
        if ret.returncode == 0:
            log.info("Python deps installed")
            status["steps"].append("pip_deps")
        else:
            log.warning("pip install exited %d: %s", ret.returncode,
                        ret.stderr.decode()[:200] if ret.stderr else "")
            status["steps"].append("pip_deps_failed")
    except subprocess.TimeoutExpired:
        log.warning("pip install timed out after 120s")
        status["steps"].append("pip_deps_failed")
    except Exception as e:
        log.warning("pip install failed: %s", e)
        status["steps"].append("pip_deps_failed")
    log.info("Starting watcher ...")
    started = start_watcher()
    status["watcher"] = started
    log.info("Kaggle setup complete: watcher=%s steps=%s", started, status["steps"])
    return status
