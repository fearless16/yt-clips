"""Tests for the automation module (caches, config, memory, transcript, worker, orchestrator, CLI)."""

import sys
import time
import json
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ── TTLCache ────────────────────────────────────────────────────────────────

class TestTTLCache:
    def test_get_set(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=64, ttl=60)
        c.set("k", "v")
        assert c.get("k") == "v"

    def test_expiry(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=64, ttl=60)
        c.set("k", "v")
        with c._lock:
            c._store["k"] = ("v", time.monotonic() - 1)
        assert c.get("k") is None

    def test_lru_eviction(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=3, ttl=60)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        c.get("a")
        c.set("d", 4)
        assert c.get("a") == 1
        assert c.get("b") is None
        assert c.get("c") == 3
        assert c.get("d") == 4

    def test_contains(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=64, ttl=60)
        c.set("k", "v")
        assert "k" in c
        assert "missing" not in c

    def test_clear(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=64, ttl=60)
        c.set("k", "v")
        c.clear()
        assert c.get("k") is None
        assert c.size() == 0

    def test_size_prune(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=64, ttl=60)
        c.set("a", 1)
        c.set("b", 2)
        with c._lock:
            c._store["a"] = (1, time.monotonic() - 1)
        assert c.size() == 1

    def test_empty_cache(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=64, ttl=60)
        assert c.get("missing") is None
        assert c.size() == 0

    def test_thread_safety(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=128, ttl=60)
        errors = []

        def worker(i):
            try:
                c.set(f"k{i}", i)
                v = c.get(f"k{i}")
                assert v == i
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ── Config ──────────────────────────────────────────────────────────────────

class TestConfig:
    def test_load(self, tmp_path):
        from automation.config import load
        from automation._cache import CONFIG_CACHE
        CONFIG_CACHE.clear()
        p = tmp_path / "test.yaml"
        p.write_text("key: val\nnested: {inner: 42}")
        cfg = load(str(p))
        assert cfg.get("key") == "val"

    def test_get(self, monkeypatch):
        from automation import config
        monkeypatch.setattr("automation.config.load", lambda p="config.yaml": {"key": "val"})
        assert config.get("key") == "val"

    def test_get_default(self):
        from automation.config import get
        assert get("nonexistent.key", 99) == 99

    def test_missing_file(self, tmp_path):
        from automation.config import load
        p = tmp_path / "missing.yaml"
        cfg = load(str(p))
        assert cfg == {}


# ── Memory ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_memory_cache():
    from automation._cache import MEMORY_CACHE
    MEMORY_CACHE.clear()


def _mock_read_meminfo(total_gb=8.0, free_gb=4.0, env="linux"):
    return lambda: (total_gb, free_gb, env)


class TestMemory:
    def test_memory_report_keys(self, monkeypatch):
        monkeypatch.setattr("automation.memory._read_meminfo", _mock_read_meminfo(8, 4, "linux"))
        from automation.memory import memory_report
        r = memory_report()
        assert "total_gb" in r
        assert "used_gb" in r
        assert "free_gb" in r
        assert "safe_batch" in r
        assert "safe_workers" in r
        assert "environment" in r

    def test_safe_batch_size_default(self, monkeypatch):
        monkeypatch.setattr("automation.memory._read_meminfo", _mock_read_meminfo(8, 4, "linux"))
        from automation.memory import safe_batch_size
        assert safe_batch_size(default=4) == 4

    def test_safe_batch_size_low_memory(self, monkeypatch):
        monkeypatch.setattr("automation.memory._read_meminfo", _mock_read_meminfo(4, 1, "linux"))
        from automation.memory import safe_batch_size
        assert safe_batch_size(default=4, min_val=2) == 2

    def test_safe_workers_default(self, monkeypatch):
        monkeypatch.setattr("automation.memory._read_meminfo", _mock_read_meminfo(8, 4, "linux"))
        from automation.memory import safe_workers
        assert safe_workers(default=2) == 2

    def test_safe_workers_low_memory(self, monkeypatch):
        monkeypatch.setattr("automation.memory._read_meminfo", _mock_read_meminfo(4, 1, "linux"))
        from automation.memory import safe_workers
        assert safe_workers(default=2, min_val=1) == 1

    def test_ensure_free_returns_bool(self, monkeypatch):
        monkeypatch.setattr("automation.memory._read_meminfo", _mock_read_meminfo(8, 4, "linux"))
        from automation.memory import ensure_free
        assert ensure_free(2.0, poll_interval=0.01, timeout=1.0) is True

    def test_ensure_free_timeout(self, monkeypatch):
        monkeypatch.setattr("automation.memory._read_meminfo", _mock_read_meminfo(8, 1, "linux"))
        from automation.memory import ensure_free
        assert ensure_free(4.0, poll_interval=0.01, timeout=0.1) is False


# ── Transcript ──────────────────────────────────────────────────────────────

class TestTranscript:
    def test_extract_video_id_standard(self):
        from automation.transcript import _extract_video_id
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_video_id_short(self):
        from automation.transcript import _extract_video_id
        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_video_id_shorts(self):
        from automation.transcript import _extract_video_id
        assert _extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_video_id_embed(self):
        from automation.transcript import _extract_video_id
        assert _extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_video_id_invalid(self):
        from automation.transcript import _extract_video_id
        assert _extract_video_id("not a url") is None
        assert _extract_video_id("") is None

    def test_vtt_timestamp_to_seconds(self):
        from automation.transcript import _vtt_timestamp_to_seconds
        assert _vtt_timestamp_to_seconds("00:01.234") == 1.234
        assert _vtt_timestamp_to_seconds("01:30.000") == 90.0
        assert _vtt_timestamp_to_seconds("00:00.000") == 0.0

    def test_vtt_timestamp_to_seconds_with_hours(self):
        from automation.transcript import _vtt_timestamp_to_seconds
        assert _vtt_timestamp_to_seconds("01:30:00.000") == 5400.0

    def test_parse_vtt_basic(self):
        from automation.transcript import _parse_vtt
        vtt = "00:01.234 --> 00:05.678\nHello world\n\n00:06.000 --> 00:10.500\nThis is a test"
        result = _parse_vtt(vtt)
        assert len(result) == 2
        assert result[0]["start"] == 1.234
        assert result[0]["end"] == 5.678
        assert result[0]["text"] == "Hello world"
        assert result[1]["start"] == 6.0
        assert result[1]["text"] == "This is a test"

    def test_parse_vtt_with_tags(self):
        from automation.transcript import _parse_vtt
        vtt = "00:00.000 --> 00:02.000\n<c.color:#ff0000>Hello</c> world"
        result = _parse_vtt(vtt)
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"

    def test_parse_vtt_empty(self):
        from automation.transcript import _parse_vtt
        assert _parse_vtt("") == []
        assert _parse_vtt("\n\n") == []

    def test_parse_vtt_multiline_text(self):
        from automation.transcript import _parse_vtt
        vtt = "00:00.000 --> 00:03.000\nLine one\nLine two\n\n00:03.000 --> 00:06.000\nFinal"
        result = _parse_vtt(vtt)
        assert len(result) == 2
        assert result[0]["text"] == "Line one Line two"
        assert result[1]["text"] == "Final"


# ── Colab ───────────────────────────────────────────────────────────────────

class TestColab:
    def test_is_colab_false_locally(self):
        from automation.colab import is_colab
        assert not is_colab()

    def test_gpu_info_dict_shape(self):
        from automation.colab import gpu_info
        from automation._cache import GPU_CACHE
        GPU_CACHE.clear()
        info = gpu_info()
        assert isinstance(info, dict)
        assert "name" in info
        assert "memory_total_gb" in info
        assert "memory_free_gb" in info

    def test_gpu_count_int(self):
        from automation.colab import gpu_count
        from automation._cache import GPU_CACHE
        GPU_CACHE.clear()
        count = gpu_count()
        assert isinstance(count, int)


# ── Kaggle ──────────────────────────────────────────────────────────────────

class TestKaggle:
    def test_is_kaggle_false_locally(self):
        from automation.kaggle import is_kaggle
        assert not is_kaggle()


# ── Worker ──────────────────────────────────────────────────────────────────

class TestWorker:
    def test_submit_get_result(self):
        from automation.worker import ParallelPool
        pool = ParallelPool(max_workers=2)
        future = pool.submit(lambda x: x * 2, 21)
        assert future.result() == 42
        pool.shutdown()

    def test_map(self):
        from automation.worker import ParallelPool
        pool = ParallelPool(max_workers=2)
        results = pool.map(lambda x: x + 1, [1, 2, 3])
        assert results == [2, 3, 4]
        pool.shutdown()

    def test_shutdown_reduces_active(self):
        from automation.worker import ParallelPool
        pool = ParallelPool(max_workers=2)
        pool.shutdown()
        assert pool.active_count() == 0

    def test_pool_size(self):
        from automation.worker import ParallelPool
        pool = ParallelPool(max_workers=4)
        assert pool.pool_size() == 4
        pool.shutdown()

    def test_active_count_starts_zero(self):
        from automation.worker import ParallelPool
        pool = ParallelPool(max_workers=2)
        assert pool.active_count() == 0
        pool.shutdown()

    def test_submit_after_shutdown_raises(self):
        from automation.worker import ParallelPool
        pool = ParallelPool(max_workers=2)
        pool.shutdown()
        with pytest.raises(RuntimeError, match="pool is shut down"):
            pool.submit(lambda: 1)


# ── Orchestrator ────────────────────────────────────────────────────────────

class TestOrchestrator:
    def test_pipeline_result_defaults(self):
        from automation.orchestrator import PipelineResult
        r = PipelineResult()
        assert r.exported == []
        assert r.uploaded_count == 0
        assert r.failures == []
        assert r.total_seconds == 0.0
        assert r.transcript_source == "none"


# ── CLI ─────────────────────────────────────────────────────────────────────

class TestCLI:
    def test_main_exists(self):
        from automation.cli import main
        assert callable(main)

    def test_main_help_does_not_crash(self):
        from automation.cli import main
        with pytest.raises(SystemExit):
            old = sys.argv[:]
            sys.argv = ["cli.py", "--help"]
            try:
                main()
            finally:
                sys.argv = old
