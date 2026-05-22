"""colab.py — Colab env detect, GPU queries, tunnel keeper (always up)."""

import os, re, sys, time, json, subprocess, threading, urllib.request
from pathlib import Path
from ._cache import GPU_CACHE

WATCHER_PORT = int(os.environ.get("PORT", "5000"))
TUNNEL_URL_FILE = Path("/content/colab_url.txt")


def is_colab() -> bool:
    if os.environ.get("COLAB_GPU"): return True
    try:
        import google.colab  # noqa
        return True
    except ImportError: pass
    return Path("/content").exists()


def gpu_info() -> dict:
    cached = GPU_CACHE.get("gpu_info")
    if cached: return cached
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        if r.returncode or not r.stdout.strip():
            info = {"name": "unknown", "memory_total_gb": 0.0, "memory_free_gb": 0.0}
        else:
            parts = [p.strip() for p in r.stdout.strip().split("\n")[0].split(",")]
            info = {"name": parts[0] if len(parts) > 0 else "unknown",
                    "memory_total_gb": round(float(parts[1]) / 1024, 2) if len(parts) > 1 else 0.0,
                    "memory_free_gb": round(float(parts[2]) / 1024, 2) if len(parts) > 2 else 0.0}
    except Exception:
        info = {"name": "unknown", "memory_total_gb": 0.0, "memory_free_gb": 0.0}
    GPU_CACHE.set("gpu_info", info)
    return info


def gpu_count() -> int:
    cached = GPU_CACHE.get("gpu_count")
    if cached is not None: return cached
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
        count = len(r.stdout.strip().split("\n")) if r.stdout.strip() else 0
    except Exception: count = 0
    GPU_CACHE.set("gpu_count", count)
    return count


def setup() -> dict:
    status = {"status": "ok", "gpu": gpu_info(), "steps": []}
    try:
        subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=120)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg", "git"], capture_output=True, timeout=120)
        status["steps"].append("apt")
    except Exception: pass
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "yt-dlp", "youtube-transcript-api", "opencv-python-headless"],
                       capture_output=True, timeout=120)
        status["steps"].append("pip")
    except Exception: pass
    for d in ["/content/data", "/content/output", "/content/logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    status["steps"].append("dirs")
    return status


# ─── Watcher ──────────────────────────────────────────────────────────────────

_watcher_proc: subprocess.Popen | None = None


def start_watcher() -> bool:
    global _watcher_proc
    if _watcher_proc and _watcher_proc.poll() is None: return True
    watcher_path = Path("watcher.py")
    if not watcher_path.exists(): return False
    try:
        _watcher_proc = subprocess.Popen(
            [sys.executable, str(watcher_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            try:
                r = urllib.request.urlopen(f"http://localhost:{WATCHER_PORT}/health", timeout=3)
                if r.status == 200: return True
            except Exception: pass
            time.sleep(1)
    except Exception: pass
    return False


def kill_watcher():
    global _watcher_proc
    if _watcher_proc:
        try: _watcher_proc.terminate(); _watcher_proc.wait(10)
        except Exception:
            try: _watcher_proc.kill()
            except Exception: pass
        _watcher_proc = None


# ─── Tunnel Keeper (always-up background daemon) ──────────────────────────────

class TunnelKeeper:
    """Background thread that keeps tunnel alive. Auto-reconnects on failure."""

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
        if self._thread and self._thread.is_alive():
            return self._url
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        for _ in range(30):
            if self._url: return self._url
            time.sleep(1)
        return None

    def stop(self):
        self._stop.set()
        self._kill_proc()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def url(self) -> str | None:
        with self._lock: return self._url

    @property
    def uptime(self) -> float:
        if not self._start_time: return 0.0
        return time.monotonic() - self._start_time

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None and self._url is not None

    def _run(self):
        while not self._stop.is_set():
            if not self.alive:
                self._connect()
            if self.alive:
                self._heartbeat()
            for _ in range(10):
                if self._stop.is_set(): return
                time.sleep(1)

    def _connect(self):
        self._kill_proc()
        for method in [_tunnel_serveo, _tunnel_localhost_run, _tunnel_localtunnel]:
            url = method(self._port)
            if url:
                with self._lock:
                    self._proc = _tunnel_proc
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
        except Exception: pass
        self._fail_count += 1
        if self._fail_count >= 3:
            self._kill_proc()
            with self._lock:
                self._url = None
                self._start_time = 0.0

    def _kill_proc(self):
        if self._proc:
            try: self._proc.terminate(); self._proc.wait(5)
            except Exception:
                try: self._proc.kill()
                except Exception: pass
            self._proc = None

    def status(self) -> dict:
        return {"url": self._url, "uptime": round(self.uptime, 1),
                "alive": self.alive, "fail_count": self._fail_count,
                "port": self._port}


# Module-level tunnel keeper (singleton)
_keeper: TunnelKeeper | None = None


def start_tunnel(port: int = WATCHER_PORT) -> str | None:
    global _keeper
    if _keeper is None:
        _keeper = TunnelKeeper(port=port)
    return _keeper.start()


def tunnel_status() -> dict:
    global _keeper
    if _keeper is None: return {"url": None, "alive": False, "uptime": 0.0, "fail_count": 0, "port": WATCHER_PORT}
    return _keeper.status()


def kill_tunnel():
    global _keeper
    if _keeper: _keeper.stop(); _keeper = None


# ─── Tunnel method implementations (used by TunnelKeeper._connect) ────────────

_tunnel_proc: subprocess.Popen | None = None

def _tunnel_serveo(port: int) -> str | None:
    global _tunnel_proc
    try:
        _tunnel_proc = subprocess.Popen(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30",
             "-R", f"80:localhost:{port}", "serveo.net"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(25):
            line = _tunnel_proc.stdout.readline() if _tunnel_proc.stdout else ""
            m = re.search(r"(https?://[a-zA-Z0-9.-]+\.serveo\.net)", line)
            if m: return m.group(1)
            time.sleep(1)
    except Exception: pass
    return None

def _tunnel_localhost_run(port: int) -> str | None:
    global _tunnel_proc
    try:
        _tunnel_proc = subprocess.Popen(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30",
             "-R", f"80:localhost:{port}", "nokey@localhost.run"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(25):
            line = _tunnel_proc.stdout.readline() if _tunnel_proc.stdout else ""
            m = re.search(r"(https?://[a-zA-Z0-9.-]+\.lhr\.life|https?://[a-zA-Z0-9.-]+\.localhost\.run)", line)
            if m: return m.group(1)
            time.sleep(1)
    except Exception: pass
    return None

def _tunnel_localtunnel(port: int) -> str | None:
    global _tunnel_proc
    try:
        _tunnel_proc = subprocess.Popen(
            ["npx", "localtunnel", "--port", str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(25):
            line = _tunnel_proc.stdout.readline() if _tunnel_proc.stdout else ""
            m = re.search(r"(https?://[a-zA-Z0-9.-]+\.loca\.lt)", line)
            if m: return m.group(1)
            time.sleep(1)
    except Exception: pass
    return None
