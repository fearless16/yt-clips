"""env.py — Environment detection & GPU queries for Colab/Kaggle.

Detects whether the pipeline is running on Colab or Kaggle,
queries GPU info via nvidia-smi, and installs system deps.

All external calls (nvidia-smi) are cached via GPU_CACHE (TTL=30s).
"""

import os
import sys
import subprocess
from pathlib import Path

from ._cache import GPU_CACHE


def is_colab() -> bool:
    """Return True if running inside a Google Colab notebook."""
    if os.environ.get("COLAB_GPU"):
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        pass
    return Path("/content").exists()


def is_kaggle() -> bool:
    """Return True if running inside a Kaggle kernel."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    return Path("/kaggle").exists() and Path("/kaggle").is_dir()


def gpu_info() -> dict:
    """Return GPU name, memory_total_gb, memory_free_gb.

    Results cached 30s in GPU_CACHE. Returns unknown/defaults on failure.
    """
    cached = GPU_CACHE.get("gpu_info")
    if cached:
        return cached
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode or not r.stdout.strip():
            info = {"name": "unknown", "memory_total_gb": 0.0, "memory_free_gb": 0.0}
        else:
            parts = [p.strip() for p in r.stdout.strip().split("\n")[0].split(",")]
            info = {
                "name": parts[0] if len(parts) > 0 else "unknown",
                "memory_total_gb": round(float(parts[1]) / 1024, 2) if len(parts) > 1 else 0.0,
                "memory_free_gb": round(float(parts[2]) / 1024, 2) if len(parts) > 2 else 0.0,
            }
    except Exception:
        info = {"name": "unknown", "memory_total_gb": 0.0, "memory_free_gb": 0.0}
    GPU_CACHE.set("gpu_info", info)
    return info


def gpu_count() -> int:
    """Return number of GPUs available.

    Results cached 30s in GPU_CACHE. Returns 0 on failure.
    """
    cached = GPU_CACHE.get("gpu_count")
    if cached is not None:
        return cached
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        count = len(r.stdout.strip().split("\n")) if r.stdout.strip() else 0
    except Exception:
        count = 0
    GPU_CACHE.set("gpu_count", count)
    return count


def setup() -> dict:
    """Install system deps (ffmpeg, git) and Python packages on Colab.

    Creates data/output/logs directories. Returns status dict with:
    status, gpu info, list of completed steps.
    """
    import logging
    log = logging.getLogger("env.setup")
    status = {"status": "ok", "gpu": gpu_info(), "steps": []}
    log.info("Installing system packages (ffmpeg, git) ...")
    try:
        subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=120)
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "ffmpeg", "git"],
            capture_output=True, timeout=120,
        )
        status["steps"].append("apt")
        log.info("System packages installed")
    except Exception as e:
        log.warning("apt install skipped/errored: %s", e)
    log.info("Installing Python packages ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "yt-dlp", "youtube-transcript-api", "opencv-python-headless"],
            capture_output=True, timeout=120,
        )
        status["steps"].append("pip")
        log.info("Python packages installed")
    except Exception as e:
        log.warning("pip install skipped/errored: %s", e)
    for d in ["/content/data", "/content/output", "/content/logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    status["steps"].append("dirs")
    log.info("Setup complete: steps=%s gpu=%s", status["steps"], status["gpu"])
    return status
