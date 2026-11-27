"""Source-bound editing regressions; synthetic evidence is not audience validation."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from automation.clip_selection import pipeline


def words():
    return [
        {"start": 1.0, "end": 1.2, "word": "India", "probability": .99},
        {"start": 1.3, "end": 4.9, "word": "win!", "probability": .98},
        {"start": 5.0, "end": 5.2, "word": "Bumrah", "probability": .97},
        {"start": 5.3, "end": 6.0, "word": "brilliant.", "probability": .96},
    ]


def test_pause_split_preserves_word_evidence_and_never_bisects_word():
    evidence = words()
    result = pipeline._split_segments_on_pauses(
        [{"start": 0, "end": 7, "text": "India win! Bumrah brilliant.", "words": evidence}],
        max_block_seconds=3)
    assert [word for part in result for word in part["words"]] == evidence
    for part in result:
        assert part["start"] <= part["words"][0]["start"]
        assert part["end"] >= part["words"][-1]["end"]


def test_sentence_split_follows_word_punctuation_not_segment_token_ratios():
    evidence = words()
    # Text intentionally disagrees in token count: timing must follow word evidence.
    result = pipeline._prepare_complete_thoughts([{
        "start": 0, "end": 7, "text": "India win by a huge margin! Bumrah brilliant.",
        "words": evidence,
    }])
    assert result[0]["text"] == "India win!"
    assert result[0]["words"] == evidence[:2]
    assert result[0]["end"] <= result[1]["start"]
    assert result[0]["end"] >= 4.9
    assert result[1]["start"] <= 5.0
    assert all(part["boundary_method"] == "source_words" for part in result)


def test_pause_boundaries_are_not_merged_away():
    evidence = [
        {"start": 0, "end": 1, "word": "India"},
        {"start": 3, "end": 4, "word": "Bumrah"},
    ]
    split = pipeline._split_segments_on_pauses([
        {"start": 0, "end": 4, "text": "India Bumrah", "words": evidence}])
    prepared = pipeline._prepare_complete_thoughts(split)
    assert len(prepared) == 2
    assert all(part["boundary_method"] == "source_words" for part in prepared)


@pytest.mark.parametrize("limit", [0, -1])
def test_zero_discovery_budget_returns_nothing(limit):
    assert pipeline._select_diverse_windows([
        {"start": 0, "end": 10, "text": "India win", "score": 10}], limit) == []


def test_discovery_budget_keeps_quiet_late_alternative():
    windows = [{"start": i * 25, "end": i * 25 + 10, "text": str(i), "score": 100-i}
               for i in range(12)]
    windows.append({"start": 3600, "end": 3620, "text": "quiet analysis", "score": 1})
    result = pipeline._select_diverse_windows(windows, 4)
    assert windows[-1] in result
    assert len(result) == 4


def candidate(start, score=60, text="India win the cricket match!"):
    return {"start": start, "end": start+10, "text": text, "final_score": score, "agent_scores": {}}












@pytest.mark.parametrize("failure", ["extract", "read", "success"])
def test_rms_scratch_is_unique_and_removed_on_every_exit(tmp_path, monkeypatch, failure):
    monkeypatch.setitem(pipeline.cfg["paths"], "temp", str(tmp_path))
    unrelated = tmp_path / "audio_analysis.wav"
    unrelated.write_bytes(b"existing user scratch")
    captured = []

    def extract(cmd, **kwargs):
        path = Path(cmd[-1])
        captured.append(path)
        if failure == "success":
            import wave
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\0\0" * 16000)
        else:
            path.write_bytes(b"partial")
        return subprocess.CompletedProcess(cmd, 1 if failure == "extract" else 0, "", "failed")

    monkeypatch.setattr(subprocess, "run", extract)
    pipeline._extract_audio_rms("source.mp4")
    assert captured[0] != unrelated
    assert not captured[0].exists()
    assert unrelated.read_bytes() == b"existing user scratch"
    assert list(tmp_path.iterdir()) == [unrelated]


def test_candidate_window_retains_cue_identity_and_word_provenance():
    thoughts = [{"start": 0, "end": 12, "text": "India score six runs in the cricket match.",
                 "score": 4, "words": words(), "source_cue_ids": [8], "boundary_method": "source_words"}]
    result = pipeline._build_livestream_windows(thoughts, 10, 30)
    assert result[0]["source_cue_ids"] == [8]
    assert result[0]["boundary_method"] == "source_words"
    assert result[0]["words"] == words()
    assert result[0]["complete_thought"] is None


def test_merging_cues_preserves_all_source_ids():
    result = pipeline._merge_caption_fragments([
        {"start": 0, "end": 2, "text": "India", "source_cue_ids": [1]},
        {"start": 2, "end": 4, "text": "win!", "source_cue_ids": [2]},
    ])
    assert result[0]["source_cue_ids"] == [1, 2]


@pytest.mark.parametrize("bounds", [(float("nan"), 4), (1, float("inf")), (-1, 4), (3, 2)])
def test_invalid_word_evidence_is_not_silently_replaced_by_partial_evidence(bounds):
    evidence = words() + [{"start": bounds[0], "end": bounds[1], "word": "invalid"}]
    with pytest.raises(ValueError, match="word"):
        pipeline._split_segments_on_pauses([
            {"start": 0, "end": 10, "text": "India win", "words": evidence}])


def test_actual_word_overlap_is_never_hidden_by_midpoint_cut():
    evidence = [{"start": 0, "end": 3, "word": "India!"},
                {"start": 2, "end": 4, "word": "win!"}]
    with pytest.raises(ValueError, match="word"):
        pipeline._prepare_complete_thoughts([
            {"start": 0, "end": 4, "text": "India! win!", "words": evidence}])






def test_evidence_retains_boundary_identity_and_ai_rejection(tmp_path):
    from automation.editorial_evidence import write_discovery_evidence
    transcript = tmp_path / "speech.json"
    transcript.write_text("[]")
    proposals = [{**candidate(0), "source_cue_ids": [7], "boundary_method": "source_words"}, candidate(20)]
    decision = {"considered_windows": [[0, 10], [20, 30]],
                "response": {"selected": [], "rejected": [{"candidate_id": 1, "reason": "payoff outside cut"}]}}
    path = write_discovery_evidence(str(tmp_path / "clips.yaml"), {"duration": 30}, str(transcript),
                                    [], proposals, [], decision)
    evidence = json.loads(path.read_text())
    assert evidence["candidates"][0]["source_cue_ids"] == [7]
    assert evidence["candidates"][0]["boundary_method"] == "source_words"
    assert evidence["candidates"][0]["editorial_rejection"] == "payoff outside cut"


def test_words_outside_parent_cue_are_not_clamped_into_clipped_speech():
    with pytest.raises(ValueError, match="word"):
        pipeline._split_segments_on_pauses([{
            "start": 2, "end": 5, "text": "India win", "words": words()}])








