import threading
import time

import pytest

from utils.lru_cache import LRUCache


class TestLRUBasic:
    def test_put_and_get(self):
        cache: LRUCache[str, int] = LRUCache(max_size=3)
        cache.put("a", 1)
        assert cache.get("a") == 1

    def test_get_missing_key_returns_none(self):
        cache: LRUCache[str, int] = LRUCache()
        assert cache.get("missing") is None

    def test_overwrite_existing_key(self):
        cache: LRUCache[str, int] = LRUCache(max_size=3)
        cache.put("a", 1)
        cache.put("a", 2)
        assert cache.get("a") == 2
        assert len(cache) == 1

    def test_evicts_lru_when_full(self):
        cache: LRUCache[str, int] = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_get_refreshes_order(self):
        cache: LRUCache[str, int] = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")
        cache.put("c", 3)
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_delete_existing(self):
        cache: LRUCache[str, int] = LRUCache()
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.get("a") is None

    def test_delete_missing_returns_false(self):
        cache: LRUCache[str, int] = LRUCache()
        assert cache.delete("nope") is False

    def test_contains(self):
        cache: LRUCache[str, int] = LRUCache()
        cache.put("a", 1)
        assert "a" in cache
        assert "b" not in cache

    def test_clear(self):
        cache: LRUCache[str, int] = LRUCache()
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert len(cache) == 0

    def test_max_size_must_be_positive(self):
        with pytest.raises(ValueError):
            LRUCache(max_size=0)


class TestLRUTTL:
    def test_expired_entry_returns_none(self):
        cache: LRUCache[str, int] = LRUCache(default_ttl=0.05)
        cache.put("a", 1)
        assert cache.get("a") == 1
        time.sleep(0.06)
        assert cache.get("a") is None

    def test_per_key_ttl_overrides_default(self):
        cache: LRUCache[str, int] = LRUCache(default_ttl=10)
        cache.put("a", 1, ttl=0.05)
        time.sleep(0.06)
        assert cache.get("a") is None

    def test_no_ttl_means_indefinite(self):
        cache: LRUCache[str, int] = LRUCache()
        cache.put("a", 1)
        assert cache.get("a") == 1

    def test_contains_skips_expired(self):
        cache: LRUCache[str, int] = LRUCache(default_ttl=0.05)
        cache.put("a", 1)
        time.sleep(0.06)
        assert "a" not in cache


class TestLRUThreadSafety:
    def test_concurrent_put_get(self):
        cache: LRUCache[int, int] = LRUCache(max_size=100)
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(start, start + 200):
                    cache.put(i, i * 2)
            except Exception as e:
                errors.append(e)

        def reader(start: int) -> None:
            try:
                for i in range(start, start + 200):
                    cache.get(i)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(200,)),
            threading.Thread(target=reader, args=(0,)),
            threading.Thread(target=reader, args=(200,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(cache) <= 100
