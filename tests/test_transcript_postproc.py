"""Tests for utils.transcript_postproc (feat/transcription)."""
import pytest

from utils.transcript_postproc import (
    correct_text, correct_segments, validate_and_apply_llm_corrections,
)


def test_correct_text_fixes_clear_mishearings():
    out, n = correct_text("coaly and bumra played at wankede")
    assert "Kohli" in out and "Bumrah" in out and "Wankhede" in out
    assert n >= 3


def test_correct_text_guards_false_positives():
    # Ordinary English words must NOT be corrupted.
    for word in ["sky", "stark", "head", "root", "hope", "salt", "young"]:
        out, n = correct_text(f"the ball went over the {word} today")
        assert word in out, f"{word} should be left intact"


def test_correct_text_is_idempotent():
    once, _ = correct_text("coaly hit a six")
    twice, n2 = correct_text(once)
    assert once == twice
    assert n2 == 0


def test_correct_segments_counts_and_mutates():
    segs = [{"text": "coaly six"}, {"text": "normal ball"}, {"text": "bumra yorker"}]
    out, total = correct_segments(segs)
    assert out[0]["text"] == "Kohli six"
    assert out[2]["text"] == "Bumrah yorker"
    assert total == 2


def test_validate_llm_corrections_applies_valid_only():
    segs = [{"text": "coaly six"}, {"text": "good ball"}]
    corrected = {0: "Kohli six", 1: ""}  # idx1 empty -> rejected
    segs, applied, rejected = validate_and_apply_llm_corrections(segs, corrected)
    assert segs[0]["text"] == "Kohli six"
    assert segs[1]["text"] == "good ball"  # unchanged
    assert applied == 1 and rejected == 1


def test_validate_llm_corrections_rejects_out_of_range_and_bloat():
    segs = [{"text": "six"}]
    corrected = {
        5: "nonexistent index",                 # out of range
        0: "six " + "x" * 500,                   # absurd growth -> rejected
    }
    segs, applied, rejected = validate_and_apply_llm_corrections(segs, corrected)
    assert segs[0]["text"] == "six"
    assert applied == 0 and rejected == 2


def test_remote_fetch_corrects_api_segments(monkeypatch, tmp_path):
    """fetch() must apply spelling correction to api/vtt sources too."""
    import automation.transcript as tr

    def fake_api(video_id):
        return {"segments": [{"start": 0, "end": 1, "text": "coaly smashed bumra"}],
                "language": "en", "source": "api"}

    monkeypatch.setattr(tr, "_fetch_via_api", fake_api)
    tr.TRANSCRIPT_CACHE.clear() if hasattr(tr.TRANSCRIPT_CACHE, "clear") else None
    res = tr.fetch("https://www.youtube.com/watch?v=abcdef12345")
    assert res["source"] == "api"
    assert res["segments"][0]["text"] == "Kohli smashed Bumrah"
