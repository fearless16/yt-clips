"""Integration tests: full feedback loop verification.

Uses proper pytest fixtures instead of standalone runner.
"""

import json
import pytest
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))


@pytest.fixture
def populated_db(tmp_path):
    """Fixture: populate clip_learner.db with realistic performance data."""
    from automation.clip_selection.clip_learner import ClipLearner

    db = tmp_path / "test_integration.db"

    learner = ClipLearner(db_path=db)

    # Pattern A: strong hook → high views (10 clips)
    for i in range(10):
        agent_scores = {
            "hook_expert": {"score": 75 + i, "reasoning": "reaction_opener"},
            "emotion_expert": {"score": 40 + i, "reasoning": "ok"},
            "cricket_context": {"score": 30 + i, "reasoning": "ok"},
            "viral_potential": {"score": 20 + i, "reasoning": "ok"},
            "viewer_psychology": {"score": 25 + i, "reasoning": "ok"},
            "retention_expert": {"score": 45 + i, "reasoning": "ok"},
            "brutal_rejection": {"score": 10, "should_reject": False},
        }
        learner.save_clip_selection(f"hook_clip_{i}", 1, 70.0 + i, agent_scores, "Test")
        learner.update_performance(f"hook_clip_{i}", f"yt{i}", 500 + i * 100, 0.7, 0.7)

    # Pattern B: strong cricket, weak hook → medium views (5 clips)
    for i in range(5):
        agent_scores = {
            "hook_expert": {"score": 20 + i, "reasoning": "no_strong_hook"},
            "emotion_expert": {"score": 30 + i, "reasoning": "ok"},
            "cricket_context": {"score": 80 + i, "reasoning": "players=5"},
            "viral_potential": {"score": 15 + i, "reasoning": "ok"},
            "viewer_psychology": {"score": 20 + i, "reasoning": "ok"},
            "retention_expert": {"score": 35 + i, "reasoning": "ok"},
            "brutal_rejection": {"score": 15, "should_reject": False},
        }
        learner.save_clip_selection(f"cricket_clip_{i}", 2, 55.0 + i, agent_scores, "Test")
        learner.update_performance(f"cricket_clip_{i}", f"yt_b{i}", 200 + i * 50, 0.55, 0.55)

    # Pattern C: high retention but weak everything → low views (5 clips)
    for i in range(5):
        agent_scores = {
            "hook_expert": {"score": 10 + i, "reasoning": "no_strong_hook"},
            "emotion_expert": {"score": 15 + i, "reasoning": "low_emotion"},
            "cricket_context": {"score": 10 + i, "reasoning": "no_cricket"},
            "viral_potential": {"score": 5 + i, "reasoning": "low_viral"},
            "viewer_psychology": {"score": 10 + i, "reasoning": "weak"},
            "retention_expert": {"score": 80 + i, "reasoning": "sweet_spot"},
            "brutal_rejection": {"score": 50, "should_reject": True},
        }
        learner.save_clip_selection(f"ret_clip_{i}", 3, 25.0 + i, agent_scores, "Test")
        learner.update_performance(f"ret_clip_{i}", f"yt_c{i}", 30 + i * 10, 0.3, 0.3)

    learner.close()
    return db


class TestIntegration:
    def test_weight_learner_adaptive(self, populated_db):
        from automation.clip_selection.weight_learner import (
            recalibrate_weights, load_performance_data,
        )
        data = load_performance_data(populated_db)
        assert len(data) == 20

        weights = recalibrate_weights(populated_db, min_clips=1, drift_rate=0.8)
        assert weights["hook_expert"] > weights["retention_expert"], \
            "Hook expert should outrank retention"

    def test_agents_exciting_vs_boring(self):
        from automation.clip_selection.agents import ALL_AGENTS
        from automation.clip_selection.arbiter import compute_weighted_score

        rms_map = {t: 0.5 for t in range(0, 200)}
        rms_map[130] = 0.9
        context = {
            "rms_map": rms_map, "avg_rms": 0.45, "max_rms": 1.0,
            "transcript_segments": [], "match_context": {}, "entity_bias": {},
        }

        exciting = {
            "start": 120.0, "end": 145.0,
            "text": "Oh wow! Kohli ne maara chhakka! Six! India dominates!",
        }
        boring = {
            "start": 500.0, "end": 508.0,
            "text": "and he takes a single to rotate the strike",
        }

        as_exciting = {a.name: a.score(exciting, context) for a in ALL_AGENTS}
        as_boring = {a.name: a.score(boring, context) for a in ALL_AGENTS}

        score_e = compute_weighted_score(as_exciting)["final_score"]
        score_b = compute_weighted_score(as_boring)["final_score"]
        assert score_e > score_b

    def test_entity_bias_boost_and_penalty(self):
        from automation.clip_selection.agents import CricketContextExpert

        agent = CricketContextExpert()
        context = {
            "rms_map": {}, "avg_rms": 0.5, "max_rms": 1.0,
            "match_context": {},
            "entity_bias": {
                "top_players": [("virat kohli", 0.95)],
                "avoid_players": ["lowperf player"],
                "top_teams": [("india", 0.9)],
                "avoid_teams": [],
            },
        }

        result_high = agent.score(
            {"start": 0, "end": 10, "text": "Virat Kohli hits a six!"}, context)
        result_low = agent.score(
            {"start": 0, "end": 10, "text": "lowperf player takes single"}, context)
        assert result_high["score"] > result_low["score"]

    def test_clip_selector_pipeline(self):
        from automation.clip_selection.selector import ClipSelector

        rms_map = {}
        for t in range(0, 300):
            rms_map[t] = 0.7 if 120 <= t <= 150 else 0.3

        context = {
            "rms_map": rms_map, "avg_rms": 0.4, "max_rms": 1.0,
            "transcript_segments": [], "match_context": {},
            "entity_bias": {"top_players": [("kohli", 0.9)], "avoid_players": [],
                            "top_teams": [("india", 0.85)], "avoid_teams": []},
        }

        candidates = [
            {"start": 120.0, "end": 145.0,
             "text": "Oh wow! Kohli ne maara chhakka! India wins! What a shot!"},
            {"start": 330.0, "end": 338.0,
             "text": "and he takes a single to rotate the strike"},
            {"start": 200.0, "end": 220.0,
             "text": "Bumrah bowling brilliantly here, tight line and length"},
            {"start": 80.0, "end": 100.0,
             "text": "Rohit Sharma hits it high... AND SIX!"},
        ]

        selector = ClipSelector(use_llm_arbiter=False)
        scored = selector.score_candidates(candidates, context)
        final = selector.select(scored, context, max_selected=3, min_quality=0)
        assert len(final) >= 1
        assert final[0]["final_score"] >= final[-1]["final_score"]

    def test_hook_auditor_reliable(self):
        from automation.clip_selection.hook_auditor import analyze_clip_hook

        r1 = analyze_clip_hook("/f.mp4",
                               "OUT! Bowled him first ball! What a delivery!", 0, 20)
        assert r1["hook_score"] > 15
        assert r1["swipe_risk"] == "low"

        r2 = analyze_clip_hook("/f.mp4", "", 0, 8)
        assert r2["swipe_risk"] == "high"
