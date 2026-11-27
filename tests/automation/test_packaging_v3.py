"""Phase-2 packaging contract: Full-English metadata, anti-AI-generic gates,
optional-emoji titles, right-sized descriptions, trends→selection bridge,
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


















# ── B: Trends → selection bridge ────────────────────────────────────────────




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




# ── C: Natural-pace duration config ────────────────────────────────────────


