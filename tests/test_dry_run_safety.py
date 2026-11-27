from pathlib import Path

import pytest

from dry_run import _fake_ffmpeg_output, _isolated_run


def test_dry_run_stubs_support_aurora_cover_without_hiding_composition_errors(tmp_path, monkeypatch):
    from dry_run import _api_intercept_context
    from automation.cover_design import compose_cover
    from PIL import Image
    monkeypatch.chdir(tmp_path)
    with _api_intercept_context(str(tmp_path / "source.mp4")):
        cover, evidence = compose_cover(Image.new("RGB", (1080, 1920)), None, "PEHLE OVER MEIN?")
        cover.save(tmp_path / "cover.jpg")
    assert (tmp_path / "cover.jpg").is_file()
    assert evidence["design"] == "aurora-v1"


@pytest.mark.parametrize("outcome", ["exception", "silent_missing"])
def test_dry_run_cannot_pass_when_thumbnail_generation_fails(monkeypatch, outcome):
    import dry_run
    import thumbnail

    def fail(*_):
        raise ValueError("fixture thumbnail failure")

    if outcome == "exception":
        monkeypatch.setattr(thumbnail.ThumbnailGenerator, "generate_for_clip", fail)
    else:
        # Verify acceptance independently of the batch's own error reporting.
        monkeypatch.setattr(thumbnail, "process_all_thumbnails", lambda *_: thumbnail.ThumbnailBatchResult())
    result = dry_run.dry_run("https://youtu.be/test", skip_download=True, skip_transcribe=True)
    assert result["exported"]
    assert "download" in result["skipped_stages"]
    assert "download" not in result["logged_stages"]
    if outcome == "exception":
        assert any("stage7b" in failure and "fixture thumbnail failure" in failure
                   for failure in result["failures"])
    else:
        assert any("acceptance: missing thumbnails" in failure for failure in result["failures"])
    assert result["validation"]["thumbnails_missing"] == len(result["exported"])
    assert not dry_run._print_validation_report(result)


def test_module_isolation_excludes_dependencies_and_face_os():
    from types import SimpleNamespace
    import dry_run
    root = Path(dry_run.__file__).resolve().parent
    assert dry_run._project_module(SimpleNamespace(__file__=str(root / "export.py")))
    assert not dry_run._project_module(SimpleNamespace(__file__=str(root / ".venv/Lib/site-packages/yaml/__init__.py")))
    assert not dry_run._project_module(SimpleNamespace(__file__=str(root / "face_os/pipeline.py")))


def test_isolation_preserves_live_runtime_stores(tmp_path, monkeypatch):
    import automation.orchestrator as orchestrator
    original_store, original_health = orchestrator._DECISION_STORE, orchestrator._PROVIDER_HEALTH
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("paths: {}\n")

    @_isolated_run
    def exercise():
        assert orchestrator._DECISION_STORE is not original_store
        assert orchestrator._PROVIDER_HEALTH is not original_health
        return {}

    exercise()
    assert orchestrator._DECISION_STORE is original_store
    assert orchestrator._PROVIDER_HEALTH is original_health


@pytest.mark.parametrize("output", ["-", "pipe:1", "NUL", "/dev/null"])
def test_analysis_command_never_selects_input(tmp_path, output):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original media")
    assert _fake_ffmpeg_output(["ffmpeg", "-i", str(source), "-f", "null", output], tmp_path) is None
    assert source.read_bytes() == b"original media"


def test_ffmpeg_refuses_input_overwrite_and_sandbox_escape(tmp_path):
    source = tmp_path / "source.mp4"
    with pytest.raises(ValueError, match="input"):
        _fake_ffmpeg_output(["ffmpeg", "-i", str(source), str(source)], tmp_path)
    with pytest.raises(ValueError, match="escaped"):
        _fake_ffmpeg_output(["ffmpeg", "-i", str(source), str(tmp_path.parent / "outside.mp4")], tmp_path)
    output = tmp_path / "output.mp4"
    assert _fake_ffmpeg_output(["ffmpeg", "-i", str(source), str(output)], tmp_path) == output


def test_isolation_restores_cwd_and_config_on_failure(tmp_path, monkeypatch):
    import utils.config as config

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("paths:\n  input: live-input\n")
    cached = config._config_cache.copy()

    @_isolated_run
    def crash():
        assert Path.cwd() != tmp_path
        assert not Path("live-input").exists()
        Path("input").mkdir()
        Path("input/video.mp4").write_bytes(b"fake")
        raise RuntimeError("fixture failure")

    with pytest.raises(RuntimeError, match="fixture failure"):
        crash()
    assert Path.cwd() == tmp_path
    assert not (tmp_path / "input").exists()
    assert config._config_cache == cached


@pytest.mark.parametrize("crash", [False, True])
def test_sandbox_is_removed_after_success_or_failure(tmp_path, monkeypatch, crash):
    monkeypatch.delenv("YT_CLIPS_KEEP_DRY_RUN", raising=False)
    captured = []

    @_isolated_run
    def exercise():
        captured.append(Path.cwd())
        Path("scratch.bin").write_bytes(b"fake")
        if crash:
            raise RuntimeError("expected")
        return {}

    if crash:
        with pytest.raises(RuntimeError, match="expected"):
            exercise()
    else:
        result = exercise()
        assert result["sandbox_retained"] is False
    assert not captured[0].exists()


def test_explicit_debug_retention_keeps_sandbox(monkeypatch):
    import shutil
    monkeypatch.setenv("YT_CLIPS_KEEP_DRY_RUN", "1")

    @_isolated_run
    def exercise():
        return {}

    result = exercise()
    sandbox = Path(result["sandbox"])
    try:
        assert result["sandbox_retained"] is True
        assert sandbox.is_dir()
    finally:
        shutil.rmtree(sandbox)


def test_sandbox_log_handles_close_without_closing_live_logs(tmp_path, monkeypatch):
    import logging
    from utils.logger import JsonFileHandler
    monkeypatch.delenv("YT_CLIPS_KEEP_DRY_RUN", raising=False)
    logger = logging.getLogger("dry-run-owned-handles-regression")
    live = logging.FileHandler(tmp_path / "live.log")
    logger.addHandler(live)
    owned = []

    @_isolated_run
    def exercise():
        handler = JsonFileHandler("logs/test.log")
        logger.addHandler(handler)
        owned.append(handler)
        return {}

    try:
        result = exercise()
        assert owned[0]._file.closed
        assert owned[0] not in logger.handlers
        assert live in logger.handlers
        assert not live.stream.closed
        assert not Path(result["sandbox"]).exists()
    finally:
        for handler in [live, *owned]:
            logger.removeHandler(handler)
            handler.close()


def test_first_import_dry_run_cannot_leave_fake_asr_in_live_module():
    """A fresh worker differs from pytest, which imports alignment at collection."""
    import subprocess
    import sys
    program = """
import dry_run
result = dry_run.dry_run('https://youtu.be/test', skip_download=True, skip_transcribe=True)
assert not result['failures'], result['failures']
from utils import whisper_cpp
assert whisper_cpp.run_whisper_cli.__module__ == 'utils.whisper_cpp', 'fake whisper.cpp escaped sandbox isolation'
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=90)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
