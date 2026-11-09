"""Source-bound editing regressions; synthetic evidence is not audience validation."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from automation.clip_selection import arbiter, pipeline


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


def run_arbiter(monkeypatch, candidates, selected, context=None):
    ai = Mock()
    ai.generate_json.return_value = json.dumps({"selected": [
        {"complete_thought": True, **selection} for selection in selected]})
    monkeypatch.setattr(arbiter, "_get_ai", lambda: ai)
    context = context if context is not None else {"transcript_segments": []}
    result = arbiter.llm_arbiter_refine(candidates, context, max_selected=1)
    return result, ai.generate_json.call_args.args[0], context


def test_arbiter_compares_late_candidate_outside_old_top_k(monkeypatch):
    candidates = [candidate(i*100, 80-i) for i in range(12)]
    result, _, context = run_arbiter(monkeypatch, candidates,
        [{"candidate_id": 12, "score": 90, "reason": "strongest payoff"}])
    assert result[0]["start"] == 1100
    assert context["editorial_decision"]["not_considered_count"] == 0


def test_every_candidate_gets_local_before_after_context(monkeypatch):
    transcript = [{"start": i*10, "end": i*10+9, "text": f"context-{i}"} for i in range(106)]
    _, prompt, _ = run_arbiter(monkeypatch, [candidate(1000)], [],
        {"transcript_segments": transcript})
    assert "context-99" in prompt
    assert "context-101" in prompt
    assert "context-0 " not in prompt


def test_truncated_script_cannot_be_selected_as_complete(monkeypatch):
    result, _, context = run_arbiter(monkeypatch, [candidate(0, text="India " * 1000)],
        [{"candidate_id": 1, "score": 90, "reason": "looks complete"}])
    assert result == []
    assert context["editorial_decision"]["status"] == "REVIEW"


def test_final_selection_cannot_repeat_overlapping_moment(monkeypatch):
    ai = Mock()
    ai.generate_json.return_value = json.dumps({"selected": [
        {"candidate_id": 1, "score": 90, "complete_thought": True, "reason": "payoff"},
        {"candidate_id": 2, "score": 80, "complete_thought": True, "reason": "same payoff"}]})
    monkeypatch.setattr(arbiter, "_get_ai", lambda: ai)
    result = arbiter.llm_arbiter_refine([candidate(0), candidate(1)], {}, max_selected=2)
    assert len(result) == 1


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


def test_non_llm_selection_also_deduplicates_overlapping_events():
    from automation.clip_selection.selector import ClipSelector
    result = ClipSelector(use_llm_arbiter=False).select([
        {**candidate(0), "complete_thought": True},
        {**candidate(1), "complete_thought": True}], {}, max_selected=2)
    assert len(result) == 1


@pytest.mark.parametrize("limit", [0, -1])
def test_selector_zero_or_negative_release_limit_never_calls_ai(monkeypatch, limit):
    from automation.clip_selection.selector import ClipSelector
    ai = Mock(side_effect=AssertionError("AI not needed"))
    monkeypatch.setattr("automation.clip_selection.selector.llm_arbiter_refine", ai)
    assert ClipSelector().select([candidate(0), candidate(20)], {}, max_selected=limit) == []
    ai.assert_not_called()


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


def test_real_discovery_preserves_original_script_and_provenance(tmp_path, monkeypatch):
    """Exercise normalization -> windows -> evidence -> YAML without external APIs."""
    import yaml
    from automation.clip_selection.selector import ClipSelector
    import automation.editorial_evidence as evidence
    monkeypatch.setitem(pipeline.cfg["clip_selection"], "editorial", {"enabled": False})

    transcript = tmp_path / "speech.json"
    original = [{"start": 0, "end": 15, "text": "इंडिया क्रिकेट मैच में छह रन बनाता है।", "words": [
        {"start": i*1.8, "end": i*1.8+1.5, "word": text, "probability": .99}
        for i, text in enumerate("इंडिया क्रिकेट मैच में छह रन बनाता है।".split())]}]
    transcript.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(pipeline, "_extract_audio_rms", lambda _: [])
    monkeypatch.setattr(pipeline.TopicSegmenter, "segment", lambda *args: [])
    monkeypatch.setattr(pipeline, "_load_source_context", lambda _: "India cricket match")
    monkeypatch.setattr(pipeline, "_load_source_title", lambda _: "India cricket")
    monkeypatch.setattr("shorts_intelligence.bridge.apply_selection_policy", lambda *args: 0)
    monkeypatch.setattr(ClipSelector, "score_candidates", lambda self, candidates, context: [
        {**c, "final_score": 80} for c in candidates])
    def select_reviewed(self, candidates, *args, **kwargs):
        from shorts_intelligence.completeness import set_completeness
        set_completeness(candidates[0], "PASS", reason="retained setup and payoff", method="text_only")
        candidates[0]["semantic_review"] = "text_only"
        return candidates[:1]
    monkeypatch.setattr(ClipSelector, "select", select_reviewed)
    monkeypatch.setattr(evidence, "source_identity", lambda _: {"duration": 15})
    monkeypatch.setitem(pipeline.cfg, "release_review", {"required": True})
    output = tmp_path / "highlights.yaml"
    result = pipeline.detect_highlights(str(transcript), "source.mp4", str(output), match_key="test")
    assert result
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))["clip1"]
    assert payload["source_cue_ids"] == [0]
    assert payload["boundary_method"] == "source_words"
    assert payload["complete_thought"] is True
    assert payload["completeness"]["method"] == "text_only"
    # Original text must remain available to compare against selection-normalized script.
    stored = json.loads(output.with_suffix(".editorial.json").read_text(encoding="utf-8"))
    assert stored["candidates"][0]["source_script"] == original[0]["text"]
    assert json.loads(transcript.read_text(encoding="utf-8")) == original


@pytest.mark.parametrize("selection", [
    {"candidate_id": 1, "score": 80},
    {"candidate_id": 1, "score": 80, "reason": {"invented": "object"}},
    {"candidate_id": 1, "score": True, "reason": "not a numeric score"},
])
def test_unusable_selection_cannot_become_text_reviewed(monkeypatch, selection):
    result, _, context = run_arbiter(monkeypatch, [candidate(0)], [selection])
    assert result == []
    assert context["editorial_decision"]["status"] == "REVIEW"


def test_omitted_alternative_is_disclosed_not_claimed_reviewed(monkeypatch):
    _, _, context = run_arbiter(monkeypatch, [candidate(0), candidate(30)],
        [{"candidate_id": 1, "score": 80, "reason": "complete payoff"}])
    assert context["editorial_decision"]["unexplained_candidate_ids"] == [2]
    assert context["editorial_decision"]["status"] == "REVIEW"


def test_comparison_has_explicit_cost_bound_and_discloses_omissions(monkeypatch):
    proposals = [candidate(i*20) for i in range(100)]
    _, _, context = run_arbiter(monkeypatch, proposals, [])
    decision = context["editorial_decision"]
    assert len(decision["considered_windows"]) <= 60
    assert decision["not_considered_count"] == 40
    assert decision["status"] == "REVIEW"
