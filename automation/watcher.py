"""watcher.py — Watcher process lifecycle.

Manages a subprocess running watcher.py (Flask health endpoint).
The watcher is required for tunnel heartbeat checks.

Port sourced from env PORT or defaults to 5000.
"""

import os
import sys
import time
import logging
import subprocess
import urllib.request
from pathlib import Path

from utils.logger import get_logger

log = get_logger("watcher_manager")

WATCHER_PORT = int(os.environ.get("PORT", "5000"))

_watcher_proc: subprocess.Popen | None = None


def start_watcher(port: int | None = None) -> bool:
    """Start the watcher subprocess if not already running.

    Polls /health endpoint up to 30s until ready. Returns True on success.
    """
    global _watcher_proc
    if _watcher_proc and _watcher_proc.poll() is None:
        log.info("Watcher already running on port %d", WATCHER_PORT)
        return True
    port = port or WATCHER_PORT
    base_dir = Path(__file__).resolve().parent.parent
    watcher_path = base_dir / "watcher.py"
    if not watcher_path.exists():
        log.error("Watcher script not found: %s", watcher_path)
        return False
    try:
        log.info("Starting watcher on port %d ...", port)
        _watcher_proc = subprocess.Popen(
            [sys.executable, str(watcher_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for attempt in range(30):
            try:
                r = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
                if r.status == 200:
                    log.info("Watcher started (pid=%d, port=%d)", _watcher_proc.pid, port)
                    return True
            except Exception:
                pass
            time.sleep(1)
        log.warning("Watcher failed to respond within 30s on port %d", port)
    except Exception as e:
        log.error("Failed to start watcher: %s", e)
    return False


def kill_watcher():
    """Terminate the watcher subprocess if running."""
    global _watcher_proc
    if _watcher_proc:
        log.info("Stopping watcher (pid=%d) ...", _watcher_proc.pid)
        try:
            _watcher_proc.terminate()
            _watcher_proc.wait(10)
            log.info("Watcher stopped")
        except Exception:
            try:
                _watcher_proc.kill()
            except Exception:
                pass
        _watcher_proc = None
