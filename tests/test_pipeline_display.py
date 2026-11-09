"""Presentation must never turn an empty/failed job into a green success."""

import io
import json
import logging

import pytest
from rich.console import Console


def render_record(**extra):
    from utils.pipeline_display import PipelineConsoleHandler

    output = io.StringIO()
    console = Console(file=output, width=110, force_terminal=True, color_system="truecolor")
    handler = PipelineConsoleHandler(console=console, show_time=False, show_path=False, show_level=False)
    record = logging.LogRecord("orchestrator", logging.INFO, "", 0, "fallback message", (), None)
    for key, value in extra.items():
        setattr(record, key, value)
    handler.handle(record)
    return output.getvalue()


def test_stage_success_has_a_green_tick_and_useful_counts():
    output = render_record(stage="export", phase="Export", status="ok", duration_ms=1250,
                           metadata={"exported": 2, "seo_generated": 2})
    assert "✓" in output
    assert "COMPLETE" in output
    assert "1.2s" in output
    assert "exported=2" in output
    assert "\x1b[" in output


@pytest.mark.parametrize("status,label", [
    ("failed", "FAILED"), ("partial", "PARTIAL"), ("review", "REVIEW"),
    ("skipped", "SKIPPED"), ("empty", "NO OUTPUT"),
])
def test_non_success_stages_never_get_a_tick(status, label):
    output = render_record(stage="export", phase="Export", status=status, duration_ms=1200)
    assert "✓" not in output
    assert label in output


def test_job_summary_includes_counts_paths_and_honest_release_notice():
    output = render_record(stage="pipeline", status="ok", run_id="abc12345", duration_ms=2500,
                           metadata={"exported": 2, "uploaded": 0, "selected": 2,
                                     "seo_generated": 2, "thumbnails_generated": 2,
                                     "thumbnails_preserved": 0, "transcript_source": "cached:api",
                                     "selection_status": "DRAFTS_SELECTED", "candidates": 27,
                                     "rejected": 3, "failures": [], "output_dirs": ["shorts/batch"],
                                     "log_file": "logs/pipeline.log", "review_required": True,
                                     "stages": []})
    assert "✓ JOB COMPLETE" in output
    assert "shorts/batch" in output
    assert "logs/pipeline.log" in output
    assert "review" in output.lower()
    assert "release approval" in output.lower()


def test_dynamic_text_is_not_interpreted_as_rich_markup():
    output = render_record(stage="pipeline", status="failed", run_id="abc12345", duration_ms=10,
                           metadata={"failures": ["[green]not approved[/green]"], "stages": []})
    assert "[green]not approved[/green]" in output
    assert "✓" not in output


def test_malformed_optional_display_data_cannot_crash_or_drop_the_message():
    output = render_record(stage="pipeline", status="partial", duration_ms=None, metadata="malformed")
    assert "fallback message" in output
    assert "✓" not in output


@pytest.mark.parametrize("status", ["partial", "review", "empty", "skipped"])
def test_phase_outcome_is_structured_without_raising(status):
    from utils.logger import JsonStreamHandler, run_phase

    buffer = io.StringIO()
    logger = logging.Logger("outcome-test", logging.DEBUG)
    logger.addHandler(JsonStreamHandler(buffer))
    with run_phase(logger, "Thumbnails", "thumbnails", run_id="test") as phase:
        phase.outcome(status, failures=2, reason="review needed")
    records = [json.loads(line) for line in buffer.getvalue().splitlines()]
    assert records[-1]["status"] == status
    assert records[-1]["metadata"]["failures"] == 2


def test_debug_details_stay_in_file_but_not_info_console(tmp_path, monkeypatch):
    import utils.logger as module

    output = io.StringIO()
    monkeypatch.setattr(module, "_CONSOLE", Console(file=output, width=100))
    logger = module.get_logger("test.console-noise-isolation", str(tmp_path / "run.jsonl"))
    try:
        logger.debug("Detailed encoder command")
        logger.info("Export completed")
        assert "Detailed encoder command" not in output.getvalue()
        assert "Export completed" in output.getvalue()
        assert "Detailed encoder command" in (tmp_path / "run.jsonl").read_text(encoding="utf-8")
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def test_json_console_is_machine_readable_without_decorations():
    from utils.logger import JsonStreamHandler

    stream = io.StringIO()
    handler = JsonStreamHandler(stream)
    record = logging.LogRecord("orchestrator", logging.INFO, "", 0, "job finished", (), None)
    record.stage, record.status, record.run_id = "pipeline", "partial", "abc12345"
    record.metadata = {"failures": ["thumbnail missing"]}
    handler.handle(record)
    parsed = json.loads(stream.getvalue())
    assert parsed["status"] == "partial"
    assert parsed["metadata"]["failures"] == ["thumbnail missing"]
    assert "\x1b" not in stream.getvalue()


def test_console_override_preserves_file_handler_and_debug_diagnostics(tmp_path, monkeypatch):
    import utils.logger as module

    output = io.StringIO()
    logger = logging.Logger("override-test", logging.DEBUG)
    file_handler = module.JsonFileHandler(str(tmp_path / "records.jsonl"))
    logger.addHandler(file_handler)
    logger.addHandler(module._console_handler("text", "INFO"))
    monkeypatch.setattr(logging.Logger.manager, "loggerDict", {logger.name: logger})
    monkeypatch.setattr(module.sys, "stdout", output)
    monkeypatch.setenv("YT_CLIPS_CONSOLE_FORMAT", "text")
    monkeypatch.setenv("YT_CLIPS_LOG_LEVEL", "INFO")
    try:
        module.configure_console(verbose=True, json_logs=True)
        assert file_handler in logger.handlers
        logger.debug("detailed diagnostic")
        assert json.loads(output.getvalue())["msg"] == "detailed diagnostic"
        assert json.loads((tmp_path / "records.jsonl").read_text(encoding="utf-8"))["msg"] == "detailed diagnostic"
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


@pytest.mark.parametrize("failures,exports,selection,upload_requested,uploaded,expected", [
    ([], 1, "DRAFTS_SELECTED", False, 0, "ok"),
    ([], 0, "NO_PUBLISHABLE_CANDIDATES", False, 0, "empty"),
    (["review needed"], 0, "REVIEW_REQUIRED", False, 0, "review"),
    (["caption failure"], 0, "DRAFTS_SELECTED", False, 0, "failed"),
    (["thumbnail failure"], 1, "DRAFTS_SELECTED", False, 0, "partial"),
    ([], 1, "NOT_RUN", True, 0, "partial"),
])
def test_job_status_cannot_claim_work_that_was_not_completed(failures, exports, selection,
                                                           upload_requested, uploaded, expected):
    from pathlib import Path
    from automation.orchestrator import PipelineResult, _result_status

    result = PipelineResult(failures=failures, exported=[Path("clip.mp4")] * exports,
                            selection_status=selection, uploaded_count=uploaded)
    assert _result_status(result, upload_requested=upload_requested) == expected
