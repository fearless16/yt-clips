"""_cache.py — TTL + LRU query cache.
All external queries (transcript, GPU info, config) go through this.
Keeps token usage low — no repeated fetches."""

import time
from collections import OrderedDict
from threading import Lock


class TTLCache:
    """Thread-safe TTL cache with LRU eviction."""

    def __init__(self, maxsize: int = 64, ttl: float = 300.0):
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict = OrderedDict()
        self._lock = Lock()

    def get(self, key: str):
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
        with self._lock:
            self._store[key] = (value, time.monotonic() + self._ttl)
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def clear(self):
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            self._prune()
            return len(self._store)

    def _prune(self):
        now = time.monotonic()
        stale = [k for k, (_, e) in self._store.items() if now > e]
        for k in stale:
            del self._store[k]

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


# Global cache instances
CONFIG_CACHE = TTLCache(maxsize=4, ttl=600)
TRANSCRIPT_CACHE = TTLCache(maxsize=16, ttl=3600)
GPU_CACHE = TTLCache(maxsize=2, ttl=30)
MEMORY_CACHE = TTLCache(maxsize=4, ttl=5)
