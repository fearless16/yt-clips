"""_cache.py — TTL + LRU query cache.

Thread-safe cache for all external queries (transcript, GPU info, config).
Keeps token usage low by avoiding repeated fetches.

Four global singletons with tuned TTLs:

    CONFIG_CACHE     maxsize=4   TTL=600s   (config YAML)
    TRANSCRIPT_CACHE maxsize=16  TTL=3600s  (YouTube transcripts)
    GPU_CACHE        maxsize=2   TTL=30s    (nvidia-smi queries)
    MEMORY_CACHE     maxsize=4   TTL=5s     (memory reports)

Usage::

    from ._cache import GPU_CACHE
    GPU_CACHE.set("gpu_info", {"name": "..."})
    info = GPU_CACHE.get("gpu_info")
"""

import time
from collections import OrderedDict
from threading import Lock


class TTLCache:
    """Thread-safe TTL cache with LRU eviction.

    Args:
        maxsize: Maximum number of entries before LRU eviction.
        ttl: Time-to-live in seconds for each entry.
    """

    def __init__(self, maxsize: int = 64, ttl: float = 300.0):
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict = OrderedDict()
        self._lock = Lock()

    def get(self, key: str):
        """Return cached value for *key*, or None if missing/expired."""
        with self._lock:
            if key not in self._store:
                return None
            val, expiry = self._store[key]
            if time.monotonic() > expiry:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return val

    def set(self, key: str, value):
        """Store *value* under *key* with the cache's TTL."""
        with self._lock:
            self._store[key] = (value, time.monotonic() + self._ttl)
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def clear(self):
        """Remove all entries."""
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        """Return number of non-expired entries."""
        with self._lock:
            self._prune()
            return len(self._store)

    def _prune(self):
        now = time.monotonic()
        stale = [k for k, (_, e) in self._store.items() if now > e]
        for k in stale:
            del self._store[k]

    def __contains__(self, key: str) -> bool:
        """Check if *key* is in the cache and not expired."""
        return self.get(key) is not None


# Global cache instances — import these from other modules
CONFIG_CACHE = TTLCache(maxsize=4, ttl=600)
TRANSCRIPT_CACHE = TTLCache(maxsize=16, ttl=3600)
GPU_CACHE = TTLCache(maxsize=2, ttl=30)
MEMORY_CACHE = TTLCache(maxsize=4, ttl=5)
