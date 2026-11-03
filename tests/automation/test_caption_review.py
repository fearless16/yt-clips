"""Listening packets prepare evidence, never certify ASR or release a clip."""

import json
import subprocess
from pathlib import Path

import pytest

from automation import caption_review


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source evidence")
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps([
        {"start": 8, "end": 14, "text": "<script>untrusted cue</script>"},
        {"start": 40, "end": 45, "text": "unrelated"},
    ]))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, '{"format":{"duration":"60"}}', "")
        Path(command[-1]).write_bytes(b"audio evidence")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(caption_review.subprocess, "run", run)
    return source, transcript, tmp_path / "packet", calls


def test_packet_starts_unreviewed_preserves_context_and_has_no_release_effect(inputs):
    source, transcript, output, calls = inputs
    caption_review.create_packet(source, transcript, [{"id": "scores", "start": 10, "end": 20}], output)
    payload = json.loads((output / "listening.json").read_text(encoding="utf-8"))
    case = payload["cases"][0]
    assert payload["status"] == "NOT CHECKED"
    assert case["corrected_text"] == ""
    assert case["reviewer"] == ""
    assert case["source_range"] == [10, 20]
    assert case["listening_range"] == [8, 22]
    assert len(case["cue_context"]) == 1
    assert "words" not in case  # No proportional alignment of corrected text.
    assert payload["source"]["sha256"] == caption_review.sha256(source)
    assert case["audio"]["sha256"] == caption_review.sha256(output / "scores.wav")
    html = (output / "index.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;" in html and "<script>untrusted" not in html
    assert "unrelated" not in html
    assert len(calls) == 2
    assert source.read_bytes() == b"source evidence"


@pytest.mark.parametrize("case", [
    {"id": "../escape", "start": 10, "end": 20},
    {"id": "bad", "start": True, "end": 20},
    {"id": "bad", "start": 10, "end": float("nan")},
    {"id": "bad", "start": 10, "end": 61},
    {"id": "bad", "start": 20, "end": 10},
])
def test_invalid_manifest_never_extracts_audio(inputs, case):
    source, transcript, output, calls = inputs
    with pytest.raises(ValueError):
        caption_review.create_packet(source, transcript, [case], output)
    assert not output.exists()
    assert all(command[0] == "ffprobe" for command in calls)


def test_duplicate_ids_rejected_before_extraction(inputs):
    source, transcript, output, calls = inputs
    with pytest.raises(ValueError, match="Duplicate"):
        caption_review.create_packet(source, transcript, [
            {"id": "same", "start": 10, "end": 20},
            {"id": "same", "start": 30, "end": 40}], output)
    assert all(command[0] == "ffprobe" for command in calls)


def test_extraction_failure_cleans_only_owned_scratch(inputs, monkeypatch):
    source, transcript, output, _ = inputs
    original = caption_review.subprocess.run

    def fail(command, **kwargs):
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"partial")
            raise subprocess.CalledProcessError(1, command)
        return original(command, **kwargs)

    monkeypatch.setattr(caption_review.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        caption_review.create_packet(source, transcript, [{"id": "x", "start": 10, "end": 20}], output)
    assert not output.exists()
    assert sorted(p.name for p in output.parent.iterdir()) == ["source.mp4", "transcript.json"]


def test_existing_review_work_is_never_overwritten(inputs):
    source, transcript, output, calls = inputs
    output.mkdir()
    (output / "corrections.json").write_text("owner work")
    with pytest.raises(FileExistsError):
        caption_review.create_packet(source, transcript, [], output)
    assert (output / "corrections.json").read_text() == "owner work"
    assert calls == []


def test_context_is_clamped_and_authoritative_cue_identity_is_not_spoofable(inputs):
    source, transcript, output, _ = inputs
    transcript.write_text(json.dumps([{"start": 0, "end": 4, "text": "speech", "cue_id": "spoofed"}]))
    payload = caption_review.create_packet(source, transcript, [
        {"id": "opening", "start": 0, "end": 3},
        {"id": "ending", "start": 58, "end": 60}], output)
    assert payload["cases"][0]["listening_range"] == [0, 5]
    assert payload["cases"][1]["listening_range"] == [56, 60]
    assert payload["cases"][0]["cue_context"][0]["cue_id"] == 0


def test_invalid_cues_are_disclosed_not_silently_fixed_or_used(inputs):
    source, transcript, output, _ = inputs
    transcript.write_text(json.dumps([
        {"start": 8, "end": 14, "text": "retained"},
        {"start": 12, "end": 12, "text": "zero-duration"},
        {"start": 15, "end": 13, "text": "backwards"},
    ]))
    payload = caption_review.create_packet(source, transcript, [{"id": "x", "start": 10, "end": 20}], output)
    assert payload["invalid_cue_ids"] == [1, 2]
    assert len(payload["cases"][0]["cue_context"]) == 1
    assert "Invalid cue IDs excluded" in (output / "index.html").read_text(encoding="utf-8")
