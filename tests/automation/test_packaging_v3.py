"""Phase-2 packaging contract: Full-English metadata, anti-AI-generic gates,
emoji-mandatory titles, right-sized descriptions, trends→selection bridge,
and natural-pace duration targets."""

import json
from pathlib import Path

import pytest
import yaml


# ── A: Quality gate hardens ─────────────────────────────────────────────────


def _long_body() -> str:
    """v4 budget: algorithm-facing copy must clear 3000 chars naturally."""
    sentence = (
        "Jasprit Bumrah cleans up the tail with unplayable yorkers in the "
        "final over of the session while India protect a slim lead and "
        "England's lower order survives against reverse swing under lights. "
    )
    return sentence * 22  # ~3300 chars


def _base_item(**overrides):
    item = {
        "title": "Bumrah destroys stumps with a searing yorker 🏏",
        "description": (
            "Jasprit Bumrah cleans up the tail with two unplayable yorkers "
            "in the final over of the session. India were protecting a slim "
            "lead and those two wickets flipped the session completely. "
            "England now need their lower order to survive against a "
            "reverse-swinging ball under lights, and the new pair looked "
            "uncomfortable from the very first delivery they faced. "
            + _long_body()
        ),
        "hashtags": ["#Shorts", "#Bumrah"],
    }
    item.update(overrides)
    return item


def test_quality_gate_rejects_devanagari_in_description():
    from automation.seo.seo import _validate_seo_quality

    item = _base_item(description="Bumrah strikes! क्रिकेट का महायुद्ध देखें।")
    assert _validate_seo_quality(item) is False


def test_quality_gate_rejects_devanagari_in_hashtags():
    from automation.seo.seo import _validate_seo_quality

    item = _base_item(hashtags=["#Shorts", "#क्रिकेट"])
    assert _validate_seo_quality(item) is False


@pytest.mark.parametrize("phrase", [
    "stay tuned",
    "is video mein",
    "dekhte hain",
    "welcome back",
    "cricket lovers",
    "dil jhoom",
    "dhamakedaar",
    "aapko pasand aayega",
    "toh chaliye",
    "aaj ke match mein",
    "hello guys",
    "in this video we will",
])
def test_quality_gate_rejects_ai_slop_phrases(phrase):
    from automation.seo.seo import _validate_seo_quality

    padded = phrase + " filler text that keeps the description long enough to pass other checks"
    item = _base_item(
        description="Bumrah strikes twice. " + padded + ". India seals the session."
    )
    assert _validate_seo_quality(item) is False, f"slop phrase not caught: {phrase}"


def test_quality_gate_accepts_clean_english_short_description():
    from automation.seo.seo import _validate_seo_quality

    assert _validate_seo_quality(_base_item()) is True


def test_enforce_limits_appends_emoji_when_missing():
    from automation.seo.seo import _enforce_limits

    out = _enforce_limits(
        {"title": "Bumrah destroys stumps with a searing final-over yorker"},
        is_shorts=True,
    )
    assert out["title"].count("🏏") == 1


def test_enforce_limits_keeps_existing_emoji_title():
    from automation.seo.seo import _enforce_limits

    original = "Bumrah destroys stumps with a yorker 🔥"
    out = _enforce_limits({"title": original}, is_shorts=True)
    assert out["title"] == original


def test_enforce_limits_caps_title_at_configured_limit():
    from utils.config import load_config

    cap = int(load_config().get("seo", {}).get("title_max_chars", 60))
    from automation.seo.seo import _enforce_limits

    long_title = "Siraj runs through the tail with a hostile seven over spell of pace"
    out = _enforce_limits({"title": long_title}, is_shorts=True)
    assert len(out["title"]) <= cap


def test_description_budget_is_right_sized_for_shorts():
    """v4 long-tail policy: algorithm-facing copy, 150-300 chars enforced."""
    from utils.config import load_config
    from automation.seo.seo import (
        _description_min_chars,
        _description_max_chars,
    )

    seo_cfg = load_config().get("seo", {})
    assert int(seo_cfg.get("description_target_chars", 250)) >= 150
    assert _description_min_chars() >= 100
    assert _description_max_chars() <= 4800
    long_desc = "word " * 200  # ~1000 chars, must clamp to max budget
    from automation.seo.seo import _enforce_limits

    out = _enforce_limits({"title": "Fine title 🏏", "description": long_desc}, is_shorts=True)
    assert len(out["description"]) <= _description_max_chars()


# ── B: Trends → selection bridge ────────────────────────────────────────────


def test_arbiter_prompt_carries_trend_demand(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    captured = {}

    class FakeAI:
        def generate_text(self, prompt, system_instruction=None):
            captured["prompt"] = prompt
            captured["system"] = system_instruction
            return '{"selected":[{"candidate_id":1,"score":80,"reason":"hot topic"}]}'

    monkeypatch.setattr(arbiter, "_get_ai", lambda: FakeAI())
    candidates = [
        {"start": 1.0, "end": 14.0, "text": "Gill drives through covers for four", "final_score": 60, "agent_scores": {}},
        {"start": 20.0, "end": 30.0, "text": "weather chat continues", "final_score": 59, "agent_scores": {}},
    ]
    context = {
        "transcript_segments": [],
        "trend_topics": ["bumrah return date", "ind vs eng highlights today"],
    }

    result = arbiter.llm_arbiter_refine(candidates, context, max_selected=2)

    assert [item["text"] for item in result]
    assert "bumrah return date" in captured["prompt"]
    assert "search demand" in (captured["system"] or "").lower()


def test_content_type_classifier_labels_core_angles():
    from automation.clip_selection.content_type import classify_content_type

    assert classify_content_type(
        "What a six! Kohli smashes it over long on, massive hit!"
    ) == "moment"
    assert classify_content_type(
        "That was so funny, he slipped again, comedy scenes in the dugout meme material"
    ) == "comedy"
    assert classify_content_type(
        "I think they should drop him, in my opinion this debate is settled, greatest of all time argument"
    ) == "debate"
    assert classify_content_type(
        "Rohit Sharma returns to the squad, selectors announced the fifteen members today"
    ) == "news"


def test_arbiter_candidate_lines_carry_content_type(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    captured = {}

    class FakeAI:
        def generate_text(self, prompt, system_instruction=None):
            captured["prompt"] = prompt
            return '{"selected":[{"candidate_id":1,"score":70,"reason":"ok"}]}'

    monkeypatch.setattr(arbiter, "_get_ai", lambda: FakeAI())
    candidates = [
        {"start": 1.0, "end": 12.0, "text": "clean bowled!", "final_score": 55,
         "agent_scores": {}, "content_type": "moment"},
        {"start": 20.0, "end": 30.0, "text": "more chatter", "final_score": 54,
         "agent_scores": {}, "content_type": "debate"},
    ]

    arbiter.llm_arbiter_refine(candidates, {"transcript_segments": []}, max_selected=2)

    assert "type=moment" in captured["prompt"]
    assert "type=debate" in captured["prompt"]


def test_bridge_records_content_type_feature(tmp_path, monkeypatch):
    from shorts_intelligence import bridge

    db_path = tmp_path / "bridge_test.db"
    root_config = {
        "shorts_intelligence": {
            "enabled": True,
            "db_path": str(db_path),
            "channel_id": "test-channel",
        }
    }
    clip_dir = tmp_path / "batch"
    clip_dir.mkdir()
    exported = clip_dir / "clip1.mp4"
    exported.write_bytes(b"x")
    highlights = [{
        "id": "clip1",
        "start": 0.0,
        "end": 28.0,
        "speed_factor": 1.0,
        "final_score": 62.0,
        "text": "Kohli finishes the chase with a six",
        "content_type": "moment",
    }]

    recorded = bridge.record_exported(root_config, highlights, [exported])
    assert recorded == 1

    import sqlite3
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT features_json FROM production_features").fetchone()
    conn.close()
    features = json.loads(row[0])
    assert features.get("content_type") == "moment"


def test_pipeline_attaches_content_type_to_highlights(tmp_path, monkeypatch):
    """detect_highlights must stamp every selected clip with its angle."""
    import automation.clip_selection.pipeline as cs_pipeline

    input_dir = tmp_path / "input"
    transcripts_dir = tmp_path / "transcripts"
    highlights_dir = tmp_path / "highlights"
    for d in (input_dir, transcripts_dir, highlights_dir):
        d.mkdir()

    segments = [
        {"start": float(i * 10), "end": float(i * 10 + 8),
         "text": t}
        for i, t in enumerate([
            "What a six that is, huge hit into the stands!",
            "I personally think the debate about the best format is boring",
            "Bumrah takes the wicket, brilliant catch taken!",
        ])
    ]
    (transcripts_dir / "video.json").write_text(json.dumps(segments), encoding="utf-8")
    (input_dir / "video_metadata.json").write_text(
        json.dumps({"title": "India vs England Test"}), encoding="utf-8")

    monkeypatch.setattr(cs_pipeline, "_extract_audio_rms", lambda p: [(float(t), 0.03) for t in range(60)])
    monkeypatch.setattr(cs_pipeline, "_filter_source_match_candidates",
                        lambda c, t, minimum_matches=3: c)
    monkeypatch.setattr(cs_pipeline.TopicSegmenter, "segment", lambda self, segs: [])

    class FakeSelector:
        def __init__(self, **kwargs):
            pass

        def score_candidates(self, candidates, context):
            for c in candidates:
                c["final_score"] = 60.0
                c["agent_scores"] = {}
                c["rejection_reasons"] = []
                c["should_reject"] = False
                c.setdefault("complete_thought", True)
            return candidates

        def select(self, scored, context, max_selected=3, min_quality=45.0):
            return [dict(c) for c in scored[:max_selected]]

    monkeypatch.setattr(cs_pipeline, "ClipSelector", FakeSelector)
    monkeypatch.setattr("shorts_intelligence.bridge.apply_selection_policy", lambda cfg, cands: 0)

    output_path = highlights_dir / "video.yaml"
    highlights = cs_pipeline.detect_highlights(
        transcript_path=str(transcripts_dir / "video.json"),
        video_path="nonexistent.mp4",
        output_path=str(output_path),
        match_key="testmatch",
    )

    assert highlights, "expected at least one selected clip"
    types = {h.get("content_type") for h in highlights}
    assert types and all(types), f"content_type missing on highlights: {types}"
    assert "moment" in types


# ── C: Natural-pace duration config ────────────────────────────────────────


def test_config_targets_natural_pace_shorts():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    hl = cfg["highlight"]
    exp = cfg["export"]
    sel = cfg["clip_selection"]
    seo = cfg["seo"]

    assert int(hl["target_duration"]) == 15
    assert int(hl.get("preferred_duration_min", 0)) >= 10
    assert int(hl.get("preferred_duration_max", 99)) <= 40
    assert exp.get("variable_speed_aggressive") is False
    assert float(sel["min_quality"]) >= 38.0
    assert int(sel["max_selected"]) == 3
    assert int(seo["title_max_chars"]) == 60
