"""tunnel.py — Always-up tunnel daemon for remote LLM interaction.

TunnelKeeper runs a background thread that keeps a public tunnel alive.
Heartbeat pings the watcher /health every 10s. Auto-reconnects after
3 consecutive heartbeat failures.

Fallback chain: serveo.net → localhost.run → localtunnel.
Port sourced from WATCHER_PORT (env PORT or 5000).
"""

import os
import re
import time
import json
import logging
import subprocess
import threading
import urllib.request
from pathlib import Path

from .watcher import WATCHER_PORT

log = logging.getLogger("tunnel")

TUNNEL_URL_FILE = Path("/content/colab_url.txt")


class TunnelKeeper:
    """Background daemon that keeps a public tunnel alive.

    Usage::

        keeper = TunnelKeeper(port=5000)
        url = keeper.start()        # blocks until tunnel ready (or None)
        keeper.status()             # -> dict with url, uptime, alive, fail_count
        keeper.stop()               # tear down
    """

    def __init__(self, port: int = WATCHER_PORT, url_file: Path = TUNNEL_URL_FILE):
        self._port = port
        self._url_file = url_file
        self._proc: subprocess.Popen | None = None
        self._url: str | None = None
        self._start_time: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fail_count = 0

    def start(self) -> str | None:
        """Start the tunnel daemon thread. Blocks up to 30s for first URL."""
        if self._thread and self._thread.is_alive():
            return self._url
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        for _ in range(30):
            if self._url:
                return self._url
            time.sleep(1)
        return None

    def stop(self):
        """Signal stop and join the daemon thread."""
        self._stop.set()
        self._kill_proc()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def url(self) -> str | None:
        """Current tunnel URL (None if not connected)."""
        with self._lock:
            return self._url

    @property
    def uptime(self) -> float:
        """Seconds since last successful tunnel connection."""
        if not self._start_time:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def alive(self) -> bool:
        """True if the tunnel process is running and URL is set."""
        return self._proc is not None and self._proc.poll() is None and self._url is not None

    def status(self) -> dict:
        """Return status dict: url, uptime, alive, fail_count, port."""
        return {
            "url": self._url,
            "uptime": round(self.uptime, 1),
            "alive": self.alive,
            "fail_count": self._fail_count,
            "port": self._port,
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _run(self):
        while not self._stop.is_set():
            if not self.alive:
                self._connect()
            if self.alive:
                self._heartbeat()
            for _ in range(10):
                if self._stop.is_set():
                    return
                time.sleep(1)

    def _connect(self):
        self._kill_proc()
        ngrok_token = os.environ.get("NGROK_AUTH_TOKEN")
        
        # If token exists, ONLY try ngrok to avoid unstable fallbacks
        if ngrok_token:
            url, proc = _tunnel_ngrok(self._port)
            if url and proc:
                with self._lock:
                    self._proc = proc
                    self._url = url
                    self._start_time = time.monotonic()
                    self._fail_count = 0
                self._url_file.parent.mkdir(parents=True, exist_ok=True)
                self._url_file.write_text(url)
                return

        # Fallback if no token or ngrok failed
        for method in [_tunnel_serveo, _tunnel_localhost_run, _tunnel_localtunnel]:
            url, proc = method(self._port)
            if url and proc:
                with self._lock:
                    self._proc = proc
                    self._url = url
                    self._start_time = time.monotonic()
                    self._fail_count = 0
                self._url_file.parent.mkdir(parents=True, exist_ok=True)
                self._url_file.write_text(url)
                return

    def _heartbeat(self):
        try:
            r = urllib.request.urlopen(f"http://localhost:{self._port}/health", timeout=5)
            if r.status == 200:
                self._fail_count = 0
                return
        except Exception as e:
            log.warning("Tunnel heartbeat failed on port %d: %s", self._port, e)
        self._fail_count += 1
        if self._fail_count >= 3:
            self._kill_proc()
            with self._lock:
                self._url = None
                self._start_time = 0.0

    def _kill_proc(self):
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None


# ── Singleton helpers ─────────────────────────────────────────────────────

_keeper: TunnelKeeper | None = None


def start_tunnel(port: int = WATCHER_PORT) -> str | None:
    """Start or reuse the module-level TunnelKeeper. Returns tunnel URL or None."""
    global _keeper
    if _keeper is None:
        _keeper = TunnelKeeper(port=port)
    return _keeper.start()


def tunnel_status() -> dict:
    """Return tunnel status.

    Priority:
    1. In-process TunnelKeeper singleton (if running).
    2. URL file written by automation.sh subprocess — probe /health via public URL.
    """
    global _keeper
    if _keeper is not None:
        return _keeper.status()

    # No in-process keeper — probe from the URL file written by automation.sh
    url = None
    alive = False
    for path in [TUNNEL_URL_FILE, Path("colab_url.txt")]:
        try:
            if path.exists():
                raw = path.read_text().strip()
                if raw:
                    url = raw
                    break
        except Exception:
            continue

    if url:
        try:
            r = urllib.request.urlopen(f"{url}/health", timeout=5)
            alive = r.status == 200
        except Exception:
            pass

    return {"url": url, "alive": alive, "uptime": 0.0, "fail_count": 0, "port": WATCHER_PORT}


def kill_tunnel():
    """Stop the module-level tunnel."""
    global _keeper
    if _keeper:
        _keeper.stop()
        _keeper = None


# ── Tunnel method implementations ─────────────────────────────────────────


def _tunnel_ngrok(port: int) -> tuple[str | None, subprocess.Popen | None]:
    """Create tunnel via ngrok using AUTH_TOKEN."""
    token = os.environ.get("NGROK_AUTH_TOKEN")
    if not token:
        return None, None
    try:
        bin_path = "/content/ngrok"
        if not os.path.exists(bin_path):
            subprocess.run(["curl", "-s", "https://bin.equinox.io/c/b34edqS6yS8/ngrok", "-o", bin_path], check=True)
            subprocess.run(["chmod", "+x", bin_path], check=True)
        
        subprocess.run([bin_path, "config", "add-authtoken", token], check=True)
        
        proc = subprocess.Popen(
            [bin_path, "http", str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        
        for _ in range(30):
            try:
                with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=2) as r:
                    data = json.loads(r.read().decode())
                    tunnels = data.get("tunnels", [])
                    if tunnels:
                        return tunnels[0]["public_url"], proc
            except Exception as e:
                log.warning("ngrok tunnel API poll failed: %s", e)
            time.sleep(1)
    except Exception as e:
        log.warning("ngrok tunnel failed on port %d: %s", port, e)
    return None, None

def _tunnel_serveo(port: int) -> tuple[str | None, subprocess.Popen | None]:
    """Create tunnel via serveo.net SSH reverse proxy."""
    try:
        proc = subprocess.Popen(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30",
             "-R", f"80:localhost:{port}", "serveo.net"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for _ in range(25):
            line = proc.stdout.readline() if proc.stdout else ""
            m = re.search(r"(https?://[a-zA-Z0-9.-]+\.serveo\.net)", line)
            if m:
                return m.group(1), proc
            time.sleep(1)
    except Exception as e:
        log.warning("serveo.net tunnel failed on port %d: %s", port, e)
    return None, None


def _tunnel_localhost_run(port: int) -> tuple[str | None, subprocess.Popen | None]:
    """Create tunnel via localhost.run SSH reverse proxy."""
    try:
        proc = subprocess.Popen(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30",
             "-R", f"80:localhost:{port}", "nokey@localhost.run"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for _ in range(25):
            line = proc.stdout.readline() if proc.stdout else ""
            m = re.search(
                r"(https?://[a-zA-Z0-9.-]+\.lhr\.life|https?://[a-zA-Z0-9.-]+\.localhost\.run)",
                line,
            )
            if m:
                return m.group(1), proc
            time.sleep(1)
    except Exception as e:
        log.warning("localhost.run tunnel failed on port %d: %s", port, e)
    return None, None


def _tunnel_localtunnel(port: int) -> tuple[str | None, subprocess.Popen | None]:
    """Create tunnel via localtunnel (npx)."""
    try:
        proc = subprocess.Popen(
            ["npx", "localtunnel", "--port", str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for _ in range(25):
            line = proc.stdout.readline() if proc.stdout else ""
            m = re.search(r"(https?://[a-zA-Z0-9.-]+\.loca\.lt)", line)
            if m:
                return m.group(1), proc
            time.sleep(1)
    except Exception as e:
        log.warning("localtunnel failed on port %d: %s", port, e)
    return None, None
