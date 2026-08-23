from pathlib import Path
import json

from shorts_intelligence.bridge import (
    apply_selection_policy,
    record_exported,
    record_upload,
    shadow_status,
)
from shorts_intelligence.store import ShortsStore


CHANNEL_ID = "UCQtCMKPc41MHd7hujVuGm5g"


def _config(db_path: Path) -> dict:
    return {
        "channel": {"id": CHANNEL_ID},
        "shorts_intelligence": {
            "enabled": True,
            "shadow_mode": True,
            "db_path": str(db_path),
            "handle": "@channel",
        },
    }


def test_bridge_records_only_exported_highlights_and_pending_upload(tmp_path):
    db = tmp_path / "db.sqlite"
    export_dir = tmp_path / "batch"
    exported = [export_dir / "clip1.mp4"]
    export_dir.mkdir()
    (export_dir / "clip1_metadata.json").write_text(json.dumps({
        "title": "A" * 50,
        "description": "D" * 3000,
        "tags": [f"tag-{index}" for index in range(22)],
        "hashtags": [f"#h{index}" for index in range(12)],
        "provider": "groq",
        "model": "model-x",
        "packaging_version": "promise_v2",
        "promise_alignment_score": 0.82,
        "primary_search_terms": ["player opinion", "cricket analysis"],
    }), encoding="utf-8")
    highlights = [
        {
            "id": "clip1",
            "start": 10.0,
            "end": 22.0,
            "text": "A complete cricket thought",
            "final_score": 72.0,
            "speed_factor": 1.2,
            "agent_scores": {"hook_expert": {"reasoning": "instant_payoff"}},
        },
        {"id": "clip2", "start": 30.0, "end": 50.0, "text": "not exported"},
    ]

    assert record_exported(_config(db), highlights, exported) == 1
    assert record_upload(_config(db), "batch/clip1", "yt-id") is True

    with ShortsStore(db, channel_id=CHANNEL_ID) as store:
        production = store.get_production("batch/clip1")
        assert production["features"]["duration_bucket"] == "under_15"
        assert production["features"]["complete_thought"] == "true"
        assert production["features"]["hook_type"] == "instant_payoff"
        assert production["features"]["seo_title_length_bucket"] == "50_59"
        assert production["features"]["seo_description_length_bucket"] == "900_plus"
        assert production["features"]["seo_provider"] == "groq"
        assert production["features"]["seo_packaging_version"] == "promise_v2"
        assert production["features"]["seo_promise_alignment_bucket"] == "75_89"
        assert production["features"]["seo_primary_query_count_bucket"] == "2_2"

    status = shadow_status(_config(db))
    assert status["mode"] == "shadow"
    assert status["production_records"] == 1


def test_disabled_bridge_is_a_noop(tmp_path):
    config = _config(tmp_path / "db.sqlite")
    config["shorts_intelligence"]["enabled"] = False

    assert record_exported(config, [], []) == 0
    assert record_upload(config, "batch/clip", "id") is False


def test_bridge_applies_active_policy_without_changing_clip_boundaries(tmp_path, monkeypatch):
    config = _config(tmp_path / "db.sqlite")
    config["shorts_intelligence"]["shadow_mode"] = False
    candidate = {
        "start": 10.0,
        "end": 22.0,
        "complete_thought": True,
        "final_score": 50.0,
    }
    recommendation = {
        "feature_name": "duration_bucket",
        "feature_value": "under_15",
        "effect": 0.04,
        "effect_lower_bound": 0.01,
        "actionable": True,
    }
    monkeypatch.setattr(ShortsStore, "active_recommendations", lambda _self: [recommendation])

    assert apply_selection_policy(config, [candidate]) == 1
    assert candidate["final_score"] == 53.0
    assert candidate["start"] == 10.0
    assert candidate["end"] == 22.0
