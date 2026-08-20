import json

from dry_run import _ensure_skipped_transcript_fixture


def test_skip_transcribe_dry_run_creates_and_returns_temporary_fixture(tmp_path):
    transcript = _ensure_skipped_transcript_fixture(
        {"transcripts": str(tmp_path / "transcripts")},
        str(tmp_path / "input" / "video.mp4"),
        True,
    )

    assert transcript is not None
    payload = json.loads(transcript.read_text(encoding="utf-8"))
    assert payload["source"] == "api"
    assert any("Kohli" in segment["text"] or "coaly" in segment["text"] for segment in payload["segments"])


def test_skip_transcribe_dry_run_never_overwrites_existing_transcript(tmp_path):
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    existing = transcript_dir / "video.json"
    existing.write_text('{"segments": [{"text": "real"}]}', encoding="utf-8")

    created = _ensure_skipped_transcript_fixture(
        {"transcripts": str(transcript_dir)},
        str(tmp_path / "input" / "video.mp4"),
        True,
    )

    assert created is None
    assert "real" in existing.read_text(encoding="utf-8")
