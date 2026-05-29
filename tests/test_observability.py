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
