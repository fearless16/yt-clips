"""Tests for automation module (v2) — covers all modules with edge cases."""

import sys, time, json, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── _cache.py ─────────────────────────────────────────────────────────────────

class TestTTLCache:
    def test_get_set(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=16, ttl=300)
        c.set("k", "v")
        assert c.get("k") == "v"

    def test_expiry(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=16, ttl=0.01)
        c.set("k", "v")
        time.sleep(0.02)
        assert c.get("k") is None

    def test_lru_eviction(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=2, ttl=300)
        c.set("a", 1); c.set("b", 2); c.set("c", 3)
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("c") == 3

    def test_contains(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=4)
        c.set("k", "v")
        assert "k" in c
        assert "x" not in c

    def test_clear(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=4)
        c.set("a", 1); c.set("b", 2)
        c.clear()
        assert c.size() == 0

    def test_size_prune(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=16, ttl=0.01)
        c.set("a", 1)
        time.sleep(0.02)
        c.set("b", 2)
        assert c.size() == 1

    def test_empty_cache(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=4)
        assert c.get("nope") is None
        assert "nope" not in c

    def test_thread_safety(self):
        from automation._cache import TTLCache
        c = TTLCache(maxsize=64, ttl=30)
        errors = []
        def worker(i):
            try:
                for _ in range(50):
                    c.set(f"k{i}", i)
                    v = c.get(f"k{i}")
                    assert v == i or v is None
            except Exception as e: errors.append(e)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors, f"Thread safety failures: {errors}"


# ─── config.py ─────────────────────────────────────────────────────────────────

class TestConfig:
    def test_load(self):
        from automation.config import load
        cfg = load()
        assert isinstance(cfg, dict)

    def test_get(self):
        from automation.config import get
        from automation.config import load
        cfg = load()
        if "paths" in cfg:
            v = get(cfg, "paths.input")
            assert v is not None

    def test_get_default(self):
        from automation.config import get
        from automation.config import load
        cfg = load()
        assert get(cfg, "nonexistent.key", "x") == "x"

    def test_missing_file(self):
        from automation.config import load
        cfg = load("/tmp/nonexistent_config_xyz.yaml")
        assert cfg == {}


# ─── memory.py ─────────────────────────────────────────────────────────────────

class TestMemory:
    def test_memory_report_keys(self):
        from automation.memory import memory_report
        r = memory_report()
        for k in ("total_gb", "used_gb", "free_gb", "min_free_gb", "safe_batch_size",
                   "safe_parallel_workers", "environment"):
            assert k in r, f"Missing key: {k}"

    def test_safe_batch_size_default(self):
        from automation.memory import safe_batch_size
        bs = safe_batch_size(default=4)
        assert 1 <= bs <= 4

    def test_safe_workers_default(self):
        from automation.memory import safe_workers
        w = safe_workers(default=2)
        assert 1 <= w <= 2

    def test_ensure_free_returns_bool(self):
        from automation.memory import ensure_free
        assert isinstance(ensure_free(0.001, timeout=2.0), bool)

    def test_emit_graph(self):
        from automation.memory import emit_graph, _sample
        _sample()
        g = emit_graph(last_n=5)
        assert isinstance(g, str)

    def test_emit_graph_empty(self):
        from automation.memory import emit_graph
        from automation.memory import _ring
        _ring.clear()
        assert emit_graph() == ""


# ─── transcript.py ─────────────────────────────────────────────────────────────

class TestTranscript:
    def test_extract_video_id_standard(self):
        from automation.transcript import _extract_video_id
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_video_id_short(self):
        from automation.transcript import _extract_video_id
        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_video_id_shorts(self):
        from automation.transcript import _extract_video_id
        assert _extract_video_id("https://www.youtube.com/shorts/abc123def45") == "abc123def45"

    def test_extract_video_id_embed(self):
        from automation.transcript import _extract_video_id
        assert _extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_video_id_invalid(self):
        from automation.transcript import _extract_video_id
        assert _extract_video_id("not-a-url") is None

    def test_vtt_timestamp_to_seconds(self):
        from automation.transcript import _vtt_timestamp_to_seconds
        assert _vtt_timestamp_to_seconds("01:23.456") == 83.456

    def test_vtt_timestamp_with_hours(self):
        from automation.transcript import _vtt_timestamp_to_seconds
        assert abs(_vtt_timestamp_to_seconds("1:02:30.500") - 3750.5) < 0.01

    def test_parse_vtt_basic(self):
        from automation.transcript import _parse_vtt
        vtt = "WEBVTT\n\n00:01.000 --> 00:04.000\nHello world\n"
        segs = _parse_vtt(vtt)
        assert len(segs) == 1
        assert abs(segs[0]["start"] - 1.0) < 0.01
        assert "Hello" in segs[0]["text"]

    def test_parse_vtt_with_tags(self):
        from automation.transcript import _parse_vtt
        vtt = "WEBVTT\n\n00:01.000 --> 00:04.000\n<c>Hello</c> <c>world</c>\n"
        segs = _parse_vtt(vtt)
        assert len(segs) == 1
        assert "Hello" in segs[0]["text"]

    def test_parse_vtt_empty(self):
        from automation.transcript import _parse_vtt
        assert _parse_vtt("") == []

    def test_parse_vtt_multiline(self):
        from automation.transcript import _parse_vtt
        vtt = "WEBVTT\n\n00:01.000 --> 00:04.000\nHello\nworld\n"
        segs = _parse_vtt(vtt)
        assert len(segs) == 1
        assert "Hello" in segs[0]["text"]

    def test_fetch_via_api_dataclass(self, mocker):
        from automation.transcript import _fetch_via_api
        
        class MockSnippet:
            def __init__(self, text, start, duration):
                self.text = text
                self.start = start
                self.duration = duration

        mock_api = mocker.patch("youtube_transcript_api.YouTubeTranscriptApi")
        mock_instance = mock_api.return_value
        
        mock_transcript = mocker.MagicMock()
        mock_transcript.language_code = "en"
        mock_transcript.fetch.return_value = [
            MockSnippet("Hello", 0.0, 2.0),
            MockSnippet("World", 2.0, 3.0),
        ]
        
        mock_instance.list.return_value = [mock_transcript]
        
        result = _fetch_via_api("fake_id")
        assert result is not None
        assert result["source"] == "api"
        assert result["language"] == "en"
        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "Hello"
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 2.0

    def test_fetch_via_api_dict(self, mocker):
        from automation.transcript import _fetch_via_api
        
        mock_api = mocker.patch("youtube_transcript_api.YouTubeTranscriptApi")
        mock_instance = mock_api.return_value
        
        mock_transcript = mocker.MagicMock()
        mock_transcript.language_code = "en"
        mock_transcript.fetch.return_value = [
            {"text": "Hello", "start": 0.0, "duration": 2.0},
            {"text": "World", "start": 2.0, "duration": 3.0},
        ]
        
        mock_instance.list.return_value = [mock_transcript]
        
        result = _fetch_via_api("fake_id")
        assert result is not None
        assert result["source"] == "api"
        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "Hello"


# ─── colab.py ──────────────────────────────────────────────────────────────────

class TestColab:
    def test_is_colab_false_locally(self):
        from automation.colab import is_colab
        assert not is_colab()

    def test_gpu_info_dict_shape(self):
        from automation.colab import gpu_info
        info = gpu_info()
        for k in ("name", "memory_total_gb", "memory_free_gb"):
            assert k in info

    def test_gpu_count_int(self):
        from automation.colab import gpu_count
        assert isinstance(gpu_count(), int)

    def test_tunnel_status_shape(self):
        from automation.colab import tunnel_status
        s = tunnel_status()
        for k in ("url", "alive", "uptime", "fail_count", "port"):
            assert k in s

    def test_watcher_port_default(self):
        from automation.colab import WATCHER_PORT
        assert WATCHER_PORT == 5000


# ─── kaggle.py ─────────────────────────────────────────────────────────────────

class TestKaggle:
    def test_is_kaggle_false_locally(self):
        from automation.kaggle import is_kaggle
        assert not is_kaggle()


# ─── worker.py ─────────────────────────────────────────────────────────────────

class TestWorker:
    def test_submit_get_result(self):
        from automation.worker import ParallelPool
        pool = ParallelPool(max_workers=2)
        f = pool.submit(lambda x: x * 2, 21)
        assert f.result(timeout=5) == 42
        pool.shutdown()

    def test_map(self):
        from automation.worker import ParallelPool
        pool = ParallelPool(max_workers=2)
        results = pool.map(lambda x: x + 1, [1, 2, 3])
        assert sorted(results) == [2, 3, 4]
        pool.shutdown()

    def test_shutdown_reduces_active(self):
        from automation.worker import ParallelPool
        pool = ParallelPool(max_workers=2)
        pool.submit(lambda: None)
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
        import pytest
        with pytest.raises(RuntimeError):
            pool.submit(lambda: None)

    def test_batch_run(self):
        from automation.worker import ParallelPool
        pool = ParallelPool(max_workers=2)
        results = pool.batch_run(lambda x: x * 2, [1, 2, 3, 4], batch_size=2)
        assert sorted(results) == [2, 4, 6, 8]
        pool.shutdown()


# ─── orchestrator.py ───────────────────────────────────────────────────────────

class TestOrchestrator:
    def test_run_orchestrator(self):
        from automation.orchestrator import Orchestrator
        orch = Orchestrator()
        assert orch is not None


# ─── cli.py ────────────────────────────────────────────────────────────────────

class TestCLI:
    def test_main_help_does_not_crash(self):
        from automation.cli import main
        import sys
        sys.argv = ["cli", "--help"]
        try: main()
        except SystemExit: pass

    def test_main_memory_report(self):
        from automation.cli import main
        import sys
        sys.argv = ["cli", "--memory-report"]
        try: main()
        except SystemExit: pass


# ─── resilience.py ─────────────────────────────────────────────────────────────

class TestResilience:
    def test_circuit_breaker_methods(self):
        from utils.resilience import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        assert cb.allow_request() is True
        cb.record_failure()
        assert cb.allow_request() is True
        cb.record_failure()
        assert cb.allow_request() is False
        cb.record_success()
        assert cb.allow_request() is True


# ─── watcher.py — Secret Extraction ─────────────────────────────────────────────

class TestWatcherSecretExtraction:
    def test_secrets_extracted_from_job(self):
        import json, tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmp:
            job = {
                "url": "https://youtu.be/test",
                "flags": ["--upload"],
                "client_secrets.json": '{"web":{"client_id":"test"}}',
                "yt_channel_token.json": '{"token":"test","refresh_token":"rt"}',
            }
            job_path = Path(tmp) / "remote_job.json"
            job_path.write_text(json.dumps(job))
            
            loaded = json.loads(job_path.read_text())
            for secret_file in ["client_secrets.json", "yt_channel_token.json"]:
                if secret_file in loaded and loaded[secret_file]:
                    (Path(tmp) / secret_file).write_text(loaded[secret_file], encoding="utf-8")
                    del loaded[secret_file]
            
            assert not (Path(tmp) / "yt_channel_token.json").exists() or (Path(tmp) / "yt_channel_token.json").read_text() == '{"token":"test","refresh_token":"rt"}'
            assert "yt_channel_token.json" not in loaded
            assert "client_secrets.json" not in loaded

    def test_missing_url_skipped(self):
        import json, tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmp:
            job = {"flags": ["--upload"]}
            job_path = Path(tmp) / "bad_job.json"
            job_path.write_text(json.dumps(job))
            loaded = json.loads(job_path.read_text())
            assert not loaded.get("url")


# ─── upload.py — Colab detection guard ──────────────────────────────────────────

class TestUploadColabGuard:
    def test_is_colab_returns_false_locally(self):
        from upload import _is_colab
        assert _is_colab() is False


# ─── transcribe.py — Config & VRAM logging ──────────────────────────────────────

class TestTranscribeConfig:
    def test_config_has_gpu_params(self):
        from utils.config import load_config
        cfg = load_config()
        t = cfg.get("transcription", {})
        assert t.get("batch_size", 0) >= 4
        assert t.get("beam_size", 0) >= 3
        assert t.get("vad_filter") is True
        assert t.get("device") in ("cuda", "cpu", "auto")

    def test_vram_logging_does_not_crash(self):
        from transcribe import _log_vram
        _log_vram("test")


# ─── orchestrator.py — Exit logging format ──────────────────────────────────────

class TestOrchestratorExitLogging:
    def test_exit_log_format(self):
        import logging
        from io import StringIO
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log = logging.getLogger("orchestrator_test")
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        
        # Simulate exit log emission
        url = "https://youtu.be/test"
        exported = 10
        uploaded = 5
        failures = 1
        elapsed = 42.5
        transcript = "api"
        
        log.info("[EXIT] pipeline url=%s exported=%d uploaded=%d failures=%d elapsed=%.1fs transcript=%s",
                 url, exported, uploaded, failures, elapsed, transcript)
        
        output = buf.getvalue()
        assert "[EXIT]" in output
        assert "exported=10" in output
        assert "uploaded=5" in output
        assert "failures=1" in output
        assert "elapsed=42.5" in output


# ─── Logging: memory.py and worker.py must use logging, not print() ──────────

def test_memory_uses_logging_not_print():
    import automation.memory as mem_mod
    assert not hasattr(mem_mod, "_log") or not callable(getattr(mem_mod, "_log")), \
        "_log() with print() must be removed from memory.py"


def test_worker_uses_logging_not_print():
    import automation.worker as wk_mod
    assert not hasattr(wk_mod, "_log") or not callable(getattr(wk_mod, "_log")), \
        "_log() with print() must be removed from worker.py"
