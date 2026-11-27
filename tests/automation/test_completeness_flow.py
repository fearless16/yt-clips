"""Completeness evidence regressions for discovery, YAML and learning records."""

import copy
import json
from unittest.mock import Mock

import pytest
import yaml

from automation.clip_selection import pipeline
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
    return cfg


def candidate():
    return {"start": 10.0, "end": 25.0, "text": "What a six! Kohli finishes the cricket match!",
            "final_score": 65.0, "agent_scores": {}, "should_reject": False}


def response(complete=True):
    return {"selected": [{"candidate_id": 1, "score": 85, "complete_thought": complete,
                          "reason": "Kohli needs six to win; the retained six finishes the match."}],
            "rejected": []}




@pytest.mark.parametrize("value", [None, "true", "false", 0, 1, [], {}])
def test_unknown_or_truthy_garbage_never_earns_a_learning_bonus(value):
    result = SelectionPolicy([], shadow_mode=False).evaluate(duration_seconds=15, complete_thought=value)
    assert result.eligible is False
    assert result.reason == "completeness_unverified"
    assert result.applied_adjustment == 0














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


