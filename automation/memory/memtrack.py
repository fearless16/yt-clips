"""memtrack.py — RAM/GPU tracking, backpressure, sparkline visualisation.

Reads /proc/meminfo on Linux. On macOS returns env="local" (all guards pass).

Usage::

    from .memtrack import memory_report, ensure_free, emit_graph

    report = memory_report()       # -> dict with GB + safe batch/worker sizes
    ok = ensure_free(2.0)          # block until >=2 GB free
    graph = emit_graph(last_n=30)  # -> "▃▄▅▆▇██▆▅▄▃" sparkline
"""

import time
import logging
from threading import Lock
from collections import deque

from automation._cache import MEMORY_CACHE

log = logging.getLogger("memory")


def _read_meminfo() -> tuple:
    try:
        with open("/proc/meminfo") as f:
            raw = f.read()
    except FileNotFoundError:
        return 0.0, 0.0, "local"
    lines = raw.strip().splitlines()
    total_kb = mem_free_kb = 0
    for line in lines:
        if line.startswith("MemTotal:"):
            total_kb = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            mem_free_kb = int(line.split()[1])
    if total_kb == 0:
        return 0.0, 0.0, "local"
    return total_kb / 1e6, mem_free_kb / 1e6, "linux"


class _RingBuffer:
    def __init__(self, maxlen: int = 60):
        self._buf = deque(maxlen=maxlen)
        self._lock = Lock()

    def append(self, val: float):
        with self._lock:
            self._buf.append(val)

    def snapshot(self) -> list:
        with self._lock:
            return list(self._buf)

    def clear(self):
        with self._lock:
            self._buf.clear()


_bars = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
_ring = _RingBuffer(60)
_lock = Lock()


def _sample():
    total, free, env = _read_meminfo()
    if total > 0:
        used_pct = (total - free) / total
    else:
        used_pct = 0.0
        total, free, env = 0.0, 0.0, "local"
    _ring.append(used_pct)
    return total, free, env


def emit_graph(last_n: int = 30) -> str:
    snap = _ring.snapshot()[-last_n:]
    if not snap:
        return ""
    mn, mx = min(snap), max(snap)
    span = mx - mn or 1.0
    indices = [int((v - mn) / span * 7) for v in snap]
    return "".join(_bars[i] for i in indices)


def ensure_free(gb: float = 2.0, poll_interval: float = 2.0, timeout: float = 120.0) -> bool:
    start = time.monotonic()
    while True:
        total, free, env = _sample()
        if total == 0 and env == "local":
            return True
        if total > 0 and free >= gb:
            return True
        if time.monotonic() - start > timeout:
            log.warning("ensure_free(%sGB) timed out after %ss", gb, timeout)
            return False
        time.sleep(poll_interval)


def safe_batch_size(default: int = 4, min_val: int = 1) -> int:
    total, free, env = _sample()
    if env == "local":
        return default
    if total > 0 and free < 2.0:
        return max(min_val, default // 2)
    return default


def safe_workers(default: int = 2, min_val: int = 1) -> int:
    total, free, env = _sample()
    if env == "local":
        return default
    if total > 0 and free < 3.0:
        return max(min_val, default // 2)
    return default


def memory_report() -> dict:
    cached = MEMORY_CACHE.get("report")
    if cached is not None:
        return cached
    total, free, env = _sample()
    used = total - free if total > 0 else 0.0
    report = dict(
        total_gb=round(total, 2),
        used_gb=round(used, 2),
        free_gb=round(free, 2),
        min_free_gb=2.0,
        safe_batch_size=safe_batch_size(),
        safe_parallel_workers=safe_workers(),
        environment=env,
    )
    MEMORY_CACHE.set("report", report)
    return report
