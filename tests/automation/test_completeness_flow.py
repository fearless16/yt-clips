"""Real policy -> selector -> arbiter regressions; only external work is stubbed."""

import copy
import json
from unittest.mock import Mock

import pytest
import yaml

from automation.clip_selection import arbiter, pipeline
from automation.clip_selection.selector import ClipSelector
from shorts_intelligence.bridge import apply_selection_policy, record_exported
from shorts_intelligence.completeness import completeness_status, completeness_value, set_completeness
from shorts_intelligence.policy import SelectionPolicy
from shorts_intelligence.store import ShortsStore


@pytest.fixture
def config(tmp_path):
    cfg = copy.deepcopy(pipeline.cfg)
    cfg["paths"] = {name: str(tmp_path / name) for name in
                    ("input", "temp", "transcripts", "highlights", "shorts")}
    cfg["shorts_intelligence"].update(enabled=True, db_path=str(tmp_path / "learner.sqlite"))
    cfg["release_review"] = {"required": True}
    # This suite exercises the still-supported legacy selector/arbiter route.
    # The transcript-first route has real integration coverage in dry_run and
    # test_transcript_curation; do not let a config default switch this subject.
    cfg["clip_selection"]["editorial"] = {"enabled": False}
    return cfg


def candidate():
    return {"start": 10.0, "end": 25.0, "text": "What a six! Kohli finishes the cricket match!",
            "final_score": 65.0, "agent_scores": {}, "should_reject": False}


def response(complete=True):
    return {"selected": [{"candidate_id": 1, "score": 85, "complete_thought": complete,
                          "reason": "Kohli needs six to win; the retained six finishes the match."}],
            "rejected": []}


@pytest.mark.parametrize("shadow", [False, True])
def test_unknown_reaches_real_review_and_policy_only_rewards_reviewed_candidate(config, monkeypatch, shadow):
    config["shorts_intelligence"]["shadow_mode"] = shadow
    item = candidate()
    set_completeness(item, "UNVERIFIED")
    monkeypatch.setattr(ShortsStore, "active_recommendations", lambda _: [{
        "feature_name": "duration_bucket", "feature_value": "15_24", "effect": .10}])
    assert apply_selection_policy(config, [item]) == 0
    assert item["should_reject"] is False
    assert item["final_score"] == 65
    assert item["intelligence_policy_reason"] == "completeness_unverified"
    ai = Mock()
    ai.generate_json.return_value = json.dumps(response())
    monkeypatch.setattr(arbiter, "_get_ai", lambda: ai)
    context = {"transcript_segments": []}
    selected = ClipSelector().select([item], context, min_quality=38, max_selected=3)
    ai.generate_json.assert_called_once()
    assert len(selected) == 1
    assert completeness_status(selected[0]) == "PASS"
    assert selected[0]["completeness"]["method"] == "text_only"
    assert context["selection_summary"]["status"] == "DRAFTS_SELECTED"
    assert apply_selection_policy(config, selected) == 1
    expected = 65 if shadow else 68
    assert selected[0]["final_score"] == expected
    apply_selection_policy(config, selected)
    assert selected[0]["final_score"] == expected  # No compounded learning bonus.
    assert [selected[0]["start"], selected[0]["end"]] == [10, 25]


@pytest.mark.parametrize("value", [None, "true", "false", 0, 1, [], {}])
def test_unknown_or_truthy_garbage_never_earns_a_learning_bonus(value):
    result = SelectionPolicy([], shadow_mode=False).evaluate(duration_seconds=15, complete_thought=value)
    assert result.eligible is False
    assert result.reason == "completeness_unverified"
    assert result.applied_adjustment == 0


@pytest.mark.parametrize("shadow", [False, True])
def test_confirmed_failure_remains_a_hard_gate(config, monkeypatch, shadow):
    config["shorts_intelligence"]["shadow_mode"] = shadow
    item = candidate()
    set_completeness(item, "FAIL", method="text_only", reason="Payoff occurs after the cut")
    apply_selection_policy(config, [item])
    assert item["should_reject"] is True
    assert item["rejection_reasons"] == ["incomplete_thought"]
    ai = Mock(side_effect=AssertionError("Known failures must not reach AI"))
    monkeypatch.setattr(arbiter, "_get_ai", ai)
    assert ClipSelector().select([item], {}, min_quality=38) == []
    ai.assert_not_called()


@pytest.mark.parametrize("complete", [None, False, "true", 1])
def test_arbiter_cannot_pass_missing_false_or_non_boolean_attestation(monkeypatch, complete):
    payload = response(complete)
    if complete is None:
        del payload["selected"][0]["complete_thought"]
    ai = Mock()
    ai.generate_json.return_value = json.dumps(payload)
    monkeypatch.setattr(arbiter, "_get_ai", lambda: ai)
    context = {}
    assert ClipSelector().select([candidate()], context, min_quality=38) == []
    assert context["selection_summary"]["status"] == "REVIEW_REQUIRED"
    assert context["editorial_decision"]["held_selections"]


def test_failed_arbiter_is_a_review_block_not_an_editorial_empty_selection(monkeypatch):
    ai = Mock()
    ai.generate_json.side_effect = RuntimeError("provider unavailable")
    monkeypatch.setattr(arbiter, "_get_ai", lambda: ai)
    context = {}
    assert ClipSelector().select([candidate()], context, min_quality=38) == []
    assert context["selection_summary"]["status"] == "REVIEW_REQUIRED"
    assert ai.generate_json.call_count == 2


def test_structured_retry_gets_feedback_and_retains_attempt_evidence(monkeypatch):
    ai = Mock()
    ai.generate_json.side_effect = ['{"selected": [', json.dumps(response())]
    monkeypatch.setattr(arbiter, "_get_ai", lambda: ai)
    context = {}
    selected = ClipSelector().select([candidate()], context, min_quality=38)
    assert len(selected) == 1
    first_prompt, second_prompt = [call.args[0] for call in ai.generate_json.call_args_list]
    example = first_prompt.split("Return JSON:\n", 1)[1]
    assert isinstance(json.loads(example), dict)
    assert "previous response failed validation" in second_prompt
    attempts = context["editorial_decision"]["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["response"] == '{"selected": ['


def test_explicit_editorial_rejection_is_a_valid_empty_selection(monkeypatch):
    ai = Mock()
    ai.generate_json.return_value = json.dumps({"selected": [], "rejected": [
        {"candidate_id": 1, "reason": "No self-contained payoff"}]})
    monkeypatch.setattr(arbiter, "_get_ai", lambda: ai)
    context = {}
    assert ClipSelector().select([candidate()], context, min_quality=38) == []
    assert context["selection_summary"]["status"] == "NO_PUBLISHABLE_CANDIDATES"
    assert context["selection_summary"]["editorial_rejected"] == 1


def test_disabling_arbiter_does_not_bypass_completeness():
    context = {}
    assert ClipSelector(use_llm_arbiter=False).select([candidate()], context) == []
    assert context["selection_summary"]["status"] == "REVIEW_REQUIRED"


def test_review_is_invalid_after_range_change_and_malformed_evidence_does_not_fallback():
    item = candidate()
    set_completeness(item, "PASS", method="text_only", reason="Retained setup and payoff")
    assert completeness_value(item) is True
    item["end"] += 1
    assert completeness_status(item) == "NEEDS_REVIEW"
    assert completeness_value(item) is None
    item["completeness"] = None
    assert completeness_value(item) is None


def test_yaml_edit_points_do_not_round_away_reviewed_boundaries():
    entry = pipeline._build_clip_yaml_entry(10.123, 25.987, 65, "cricket", 25, 1.0)
    assert entry["start_sec"] == 10.123
    assert entry["end_sec"] == 25.987


@pytest.mark.parametrize("status,expected", [("UNVERIFIED", "unknown"), ("PASS", "true"), ("FAIL", "false")])
def test_export_records_actual_review_not_export_success(config, tmp_path, status, expected):
    item = {**candidate(), "id": "clip1"}
    set_completeness(item, status, method="text_only", reason="Source-bound review")
    item["semantic_review"] = "text_only" if status != "UNVERIFIED" else "NOT CHECKED"
    clip = tmp_path / "batch" / "clip1.mp4"
    clip.parent.mkdir()
    assert record_exported(config, [item], [clip]) == 1
    with ShortsStore(config["shorts_intelligence"]["db_path"], channel_id=config["channel"]["id"]) as store:
        features = store.get_production("batch/clip1")["features"]
    assert features["complete_thought"] == expected
    assert features["completeness_status"] == status
    assert features["semantic_review"] == item["semantic_review"]


@pytest.mark.parametrize("shadow", [False, True])
def test_discovery_through_real_agents_policy_selector_and_arbiter(config, tmp_path, monkeypatch, shadow):
    """Regression for the exact zero-output bug: do NOT mock the policy or selector."""
    config["shorts_intelligence"]["shadow_mode"] = shadow
    monkeypatch.setattr(pipeline, "cfg", config)
    monkeypatch.setattr(pipeline, "_extract_audio_rms", lambda _: [(float(t), .08) for t in range(30)])
    monkeypatch.setattr(pipeline.TopicSegmenter, "segment", lambda *_: [])
    # Source identity invokes ffprobe; the unit test has no real video.
    monkeypatch.setattr("automation.editorial_evidence.source_identity", lambda _: {"duration": 30})
    from pathlib import Path
    source_dir = Path(config["paths"]["input"])
    source_dir.mkdir()
    (source_dir / "video_metadata.json").write_text(json.dumps({"title": "India vs New Zealand cricket"}))
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({"source": "fixture", "segments": [{"start": 0, "end": 15,
        "text": "What a six! Kohli smashes an unbelievable six! India win the cricket match!"}]}))
    ai = Mock()
    ai.generate_json.return_value = json.dumps(response())
    monkeypatch.setattr(arbiter, "_get_ai", lambda: ai)
    output = tmp_path / "highlights.yaml"
    diagnostics = {}
    selected = pipeline.detect_highlights(str(transcript), "fixture.mp4", str(output),
                                          match_key="test", diagnostics=diagnostics)
    assert selected, diagnostics
    ai.generate_json.assert_called_once()
    assert diagnostics["review_candidates"] >= 1
    assert diagnostics["status"] == "DRAFTS_SELECTED"
    saved = yaml.safe_load(output.read_text(encoding="utf-8"))["clip1"]
    assert saved["complete_thought"] is True
    assert saved["completeness"] == selected[0]["completeness"]
    assert saved["completeness"]["source_range"] == [selected[0]["start"], selected[0]["end"]]
    evidence = json.loads(output.with_suffix(".editorial.json").read_text(encoding="utf-8"))
    assert evidence["selected"][0]["completeness"] == saved["completeness"]
    assert evidence["release_status"] == "REVIEW"  # Text approval never authorizes publication.
