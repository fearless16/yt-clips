import pytest
import logging
import json
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.logger import get_logger, JsonStreamHandler
from automation.orchestrator import run

def test_json_console_logger_instantiation():
    mock_config = {
        "logging": {
            "level": "INFO",
            "log_file": "logs/test_pipeline.log",
            "console_format": "json"
        }
    }
    
    test_logger = logging.getLogger("test_json_logger")
    test_logger.handlers.clear()
    
    with patch("utils.config.load_config", return_value=mock_config):
        logger = get_logger("test_json_logger", log_file="logs/test_pipeline.log")
        
        handlers = logger.handlers
        assert any(isinstance(h, JsonStreamHandler) for h in handlers)

def test_json_stream_handler_emit():
    stream = io.StringIO()
    handler = JsonStreamHandler(stream)
    
    logger = logging.getLogger("test_emit_logger")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    
    logger.info("Test message", extra={"stage": "test_stage", "duration_ms": 123})
    
    output = stream.getvalue()
    assert output
    
    data = json.loads(output.strip())
    assert data["msg"] == "Test message"
    assert data["stage"] == "test_stage"
    assert data["duration_ms"] == 123
    assert "ts" in data
    assert data["level"] == "INFO"

def test_orchestrator_stage_timing_logging():
    mock_config = {
        "paths": {"input": "test_input", "transcripts": "test_transcripts", "highlights": "test_highlights"},
        "download": {"output_filename": "video.mp4"},
    }
    
    with patch("automation.config.load", return_value=mock_config), \
         patch("automation.orchestrator.log.info") as mock_log_info, \
         patch("automation.transcript.fetch", return_value={"segments": [{"start": 0, "text": "hello"}], "source": "api"}), \
         patch("download.download"), \
         patch("transcribe.transcribe"), \
         patch("highlight.detect_highlights"), \
         patch("export.export_all", return_value=[Path("shorts/clip1.mp4")]), \
         patch("automation.seo.seo.process_all_seo"), \
         patch("thumbnail.process_all_thumbnails"), \
         patch("sync.sync_to_drive"), \
         patch("upload.upload_video"), \
         patch("automation.seo.analytics.generate_daily_insights"):
             
        run(
            url="https://youtu.be/test",
            skip_download=False,
            skip_transcribe=False,
            skip_highlight=False,
            skip_export=False,
            skip_sync=False,
            skip_seo=False,
            auto_sync=True,
            auto_upload=True
        )
        
        called_stages = []
        for args, kwargs in mock_log_info.call_args_list:
            extra = kwargs.get("extra")
            if extra and "stage" in extra:
                called_stages.append((extra["stage"], extra.get("duration_ms")))
                
        stages = [s[0] for s in called_stages]
        assert "transcript_fetch" in stages
        assert "download" in stages
        assert "highlight" in stages
        assert "export" in stages
        assert "seo" in stages
        assert "thumbnails" in stages
        assert "sync" in stages
        assert "upload" in stages
        assert "analytics" in stages
        assert "pipeline_total" in stages
        
        for stage, duration in called_stages:
            assert isinstance(duration, int)
            assert duration >= 0



# ─── run_phase context manager + richer schema (feat/observability) ───────────

import logging as _logging
from utils.logger import run_phase, new_run_id, _build_log_entry


def _capture_logger(name):
    """Return (logger, StringIO) wired to a JsonStreamHandler for assertions."""
    stream = io.StringIO()
    handler = JsonStreamHandler(stream)
    handler.setLevel(_logging.DEBUG)
    logger = _logging.getLogger(name)
    logger.handlers = [handler]
    logger.setLevel(_logging.DEBUG)
    logger.propagate = False
    return logger, stream


def _entries(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_new_run_id_is_short_and_unique():
    a, b = new_run_id(), new_run_id()
    assert isinstance(a, str) and 0 < len(a) <= 12
    assert a != b


def test_run_phase_success_emits_start_and_done():
    logger, stream = _capture_logger("test_phase_ok")
    rid = new_run_id()
    with run_phase(logger, "phase X Demo", "demo", run_id=rid,
                   phase_index=1, phase_total=3) as ph:
        ph.set(items=7)

    entries = _entries(stream)
    statuses = [e.get("status") for e in entries]
    assert "start" in statuses
    assert "ok" in statuses
    done = next(e for e in entries if e["status"] == "ok")
    assert done["stage"] == "demo"
    assert done["run_id"] == rid
    assert done["phase_index"] == 1 and done["phase_total"] == 3
    assert isinstance(done["duration_ms"], int) and done["duration_ms"] >= 0
    assert done["metadata"]["items"] == 7
    # Every record carrying a stage must also carry an int duration_ms (invariant).
    for e in entries:
        if "stage" in e:
            assert isinstance(e["duration_ms"], int) and e["duration_ms"] >= 0


def test_run_phase_failure_logs_at_site_with_exc_and_reraises():
    logger, stream = _capture_logger("test_phase_fail")
    rid = new_run_id()
    with pytest.raises(ValueError):
        with run_phase(logger, "phase Y Boom", "boom", run_id=rid):
            raise ValueError("kaboom")

    entries = _entries(stream)
    failed = [e for e in entries if e.get("status") == "failed"]
    assert len(failed) == 1
    rec = failed[0]
    assert rec["stage"] == "boom"
    assert rec["level"] == "ERROR"
    assert rec["error_type"] == "ValueError"
    assert rec["run_id"] == rid
    assert isinstance(rec["duration_ms"], int)
    # exc_info must be captured so logs answer "why did it fail".
    assert "exc" in rec and "ValueError" in rec["exc"]


def test_json_entry_includes_full_structured_schema():
    rec = _logging.LogRecord("t", _logging.INFO, __file__, 1, "msg", None, None)
    rec.stage = "seo"
    rec.status = "ok"
    rec.phase = "phase 5 SEO"
    rec.phase_index = 5
    rec.phase_total = 9
    rec.run_id = "abcd1234"
    rec.duration_ms = 42
    rec.error_type = None
    rec.metadata = {"clips": 3}
    handler = JsonStreamHandler(io.StringIO())
    entry = _build_log_entry(handler, rec)
    for key in ("stage", "status", "phase", "phase_index", "phase_total",
                "run_id", "duration_ms", "metadata"):
        assert key in entry, key
    assert entry["metadata"] == {"clips": 3}


def test_orchestrator_logs_failure_at_site_with_stage_and_error_type():
    """A throwing phase must emit an ERROR record with stage+status=failed+error_type,
    and still be recorded in result.failures (degrade-but-observe)."""
    mock_config = {
        "paths": {"input": "test_input", "transcripts": "test_transcripts", "highlights": "test_highlights"},
        "download": {"output_filename": "video.mp4"},
    }
    captured = []

    real_error = _logging.getLogger("orchestrator").error

    with patch("automation.config.load", return_value=mock_config), \
         patch("automation.transcript.fetch", side_effect=RuntimeError("transcript boom")), \
         patch("download.download", return_value=Path("input/video.mp4")), \
         patch("transcribe.transcribe"), \
         patch("highlight.detect_highlights"), \
         patch("export.export_all", return_value=[]), \
         patch("automation.orchestrator.log.error") as mock_err:

        from automation.orchestrator import run
        result = run(url="https://youtu.be/test", skip_download=True,
                     skip_highlight=True, skip_export=True)

        # Phase 0 raised -> an ERROR record with stage=transcript_fetch, status=failed
        found = False
        for args, kwargs in mock_err.call_args_list:
            extra = kwargs.get("extra") or {}
            if extra.get("stage") == "transcript_fetch" and extra.get("status") == "failed":
                assert extra.get("error_type") == "RuntimeError"
                assert kwargs.get("exc_info") is True
                found = True
        assert found, "failure was not logged at the site with structured context"
        assert any("phase0" in f for f in result.failures)
