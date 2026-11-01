from pathlib import Path

import pytest

from dry_run import _fake_ffmpeg_output, _isolated_run


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
