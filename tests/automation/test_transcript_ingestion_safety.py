"""Remote transcript failures must not consume unrelated scratch or stale text."""
import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from automation import transcript


@pytest.mark.parametrize("output", ["valid", "empty", "none", "failure"])
def test_vtt_fetch_owns_files_and_returns_none_for_empty_output(tmp_path, monkeypatch, output):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    unrelated = tmp_path / "another-video.en.vtt"
    unrelated.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nPRIVATE REVIEW TEXT\n")
    captured = []

    def download(cmd, **kwargs):
        base = Path(cmd[cmd.index("-o") + 1])
        captured.append(base)
        if output in {"valid", "empty"}:
            base.with_suffix(".en.vtt").write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nKohli waits.\n" if output == "valid" else "")
        if output == "failure":
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", download)
    result = transcript._fetch_via_ytdlp("AbC123xyz98")
    assert unrelated.exists() and "PRIVATE" in unrelated.read_text()
    assert captured and captured[0].parent != tmp_path
    assert not captured[0].parent.exists()
    if output == "valid":
        assert result["segments"][0]["text"] == "Kohli waits."
    else:
        assert result is None


def test_empty_remote_result_falls_back_to_local_whisper(tmp_path, monkeypatch):
    transcript.TRANSCRIPT_CACHE.clear()
    for method in ("_fetch_via_youtube_data_api", "_fetch_via_api", "_fetch_via_ytdlp"):
        monkeypatch.setattr(transcript, method, lambda *_: {"source": "vtt", "segments": []})

    def local(video_path, output_path):
        Path(output_path).write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "Actual fixture"}]}))

    monkeypatch.setattr("transcribe.transcribe", local)
    try:
        result = transcript.fetch("https://youtu.be/AbC123xyz98", video_path="source.mp4")
        assert result["source"] == "local_whisper"
        assert result["segments"][0]["text"] == "Actual fixture"
    finally:
        transcript.TRANSCRIPT_CACHE.clear()


def test_failed_local_fallback_never_reads_previous_output(tmp_path, monkeypatch):
    transcript.TRANSCRIPT_CACHE.clear()
    for method in ("_fetch_via_youtube_data_api", "_fetch_via_api", "_fetch_via_ytdlp"):
        monkeypatch.setattr(transcript, method, lambda *_: None)
    previous = tmp_path / "reviewed.json"
    previous.write_text(json.dumps({"segments": [{"text": "PREVIOUS SOURCE"}]}))
    monkeypatch.setattr("transcribe.transcribe", lambda *_: None)
    try:
        result = transcript.fetch("https://youtu.be/AbC123xyz98", str(previous), "source.mp4")
        assert not result["segments"]
        assert "PREVIOUS SOURCE" in previous.read_text()
    finally:
        transcript.TRANSCRIPT_CACHE.clear()


def test_dry_run_transcript_api_uses_upstream_duration_contract(tmp_path, monkeypatch):
    from dry_run import _api_intercept_context
    monkeypatch.chdir(tmp_path)
    with _api_intercept_context(str(tmp_path / "source.mp4")):
        result = transcript._fetch_via_api("AbC123xyz98")
    assert result and result["segments"]
    assert all(s["end"] > s["start"] for s in result["segments"])


@pytest.mark.parametrize("outcome", ["ok", "raise"])
def test_whisper_scratch_is_unique_and_cleaned_on_all_exits(tmp_path, monkeypatch, outcome):
    import transcribe
    from utils import whisper_cpp

    raw = tmp_path / "whisper_rocm_out.json"
    raw.write_text("preserved raw transcript evidence")
    owned = []

    def extract(video, wav):
        owned.append(Path(wav).parent)
        Path(wav).write_bytes(b"partial audio")

    def run(wav, **kwargs):
        (kwargs["scratch"] / "whisper_rocm_new.json").write_text("new raw output")
        if outcome == "raise":
            raise whisper_cpp.WhisperCppError("fixture")
        return {"segments": [{"start": 0, "end": 1, "text": "x", "words": []}],
                "language": "hi", "backend": "rocm", "model": "m"}

    monkeypatch.setattr(transcribe, "_extract_audio", extract)
    monkeypatch.setattr(whisper_cpp, "run_whisper_cli", run)
    monkeypatch.setattr(transcribe, "correct_segments_with_llm", lambda x: x)
    monkeypatch.setattr(transcribe, "cfg", {"paths": {"temp": str(tmp_path)}, "transcription": {"model": "small"}})
    out = tmp_path / "out" / "t.json"
    if outcome == "raise":
        with pytest.raises(whisper_cpp.WhisperCppError):
            transcribe.transcribe("source", str(out))
    else:
        transcribe.transcribe("source", str(out))
        assert out.exists()
    assert owned and not owned[0].exists()
    assert raw.read_text() == "preserved raw transcript evidence"
