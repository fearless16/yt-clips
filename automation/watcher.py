"""watcher.py — Watcher process lifecycle.

Manages a subprocess running watcher.py (Flask health endpoint).
The watcher is required for tunnel heartbeat checks.

Port sourced from env PORT or defaults to 5000.
"""

import os
import sys
import time
import subprocess
import urllib.request
from pathlib import Path

WATCHER_PORT = int(os.environ.get("PORT", "5000"))

_watcher_proc: subprocess.Popen | None = None


def start_watcher(port: int | None = None) -> bool:
    """Start the watcher subprocess if not already running.

    Polls /health endpoint up to 30s until ready. Returns True on success.
    """
    global _watcher_proc
    if _watcher_proc and _watcher_proc.poll() is None:
        return True
    port = port or WATCHER_PORT
    watcher_path = Path("watcher.py")
    if not watcher_path.exists():
        return False
    try:
        _watcher_proc = subprocess.Popen(
            [sys.executable, str(watcher_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(30):
            try:
                r = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
                if r.status == 200:
                    return True
            except Exception:
                pass
            time.sleep(1)
    except Exception:
        pass
    return False


def kill_watcher():
    """Terminate the watcher subprocess if running."""
    global _watcher_proc
    if _watcher_proc:
        try:
            _watcher_proc.terminate()
            _watcher_proc.wait(10)
        except Exception:
            try:
                _watcher_proc.kill()
            except Exception:
                pass
        _watcher_proc = None
