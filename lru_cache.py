from typing import Any, Optional
import threading
from collections import OrderedDict
import time

class LRUCache:
    def __init__(self, capacity: int, ttl: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.ttl = ttl
        self.lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
            value, expiry = self.cache[key]
            if expiry < time.time():
                del self.cache[key]
                return None
            self.cache.move_to_end(key)
            return value

    def put(self, key: Any, value: Any) -> None:
        with self.lock:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = (value, time.time() + self.ttl)
            self.cache.move_to_end(key)

    def delete(self, key: Any) -> None:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
