import itertools
import json
import pytest
from unittest.mock import patch, MagicMock

from automation.clip_selection.agent_base import Agent
from automation.clip_selection.agents import (
    HookExpert,
    EmotionExpert,
    CricketContextExpert,
    ViralPotentialExpert,
    ViewerPsychologyExpert,
    RetentionExpert,
    BrutalRejectionAgent,
    ALL_AGENTS,
    _words,
    _count_overlap,
    _rms_range,
    _avg_rms,
)
from automation.clip_selection.arbiter import (
    compute_weighted_score,
    AGENT_WEIGHTS,
    fmt_ts,
)
from automation.clip_selection.selector import ClipSelector
from automation.clip_selection.hook_auditor import (
    analyze_clip_hook,
    HookAuditor,
    HOOK_TYPES,
)
from automation.clip_selection.clip_learner import ClipLearner


@pytest.fixture
def sample_candidate():
    return {
        "start": 120.0,
        "end": 145.0,
        "text": "Oh wow! Kohli ne maara chhakka! What a shot! Crowd goes wild. "
                "Unbelievable six from the king! India putting on a masterclass "
                "here at the world cup. Brilliant batting display.",
    }


@pytest.fixture
def dull_candidate():
    return {
        "start": 330.0,
        "end": 338.0,
        "text": "and he takes a single to rotate the strike",
    }


@pytest.fixture
def silent_candidate():
    return {
        "start": 500.0,
        "end": 520.0,
        "text": "",
    }


@pytest.fixture
def rms_context():
    rms_map = {t: 0.5 + abs(t - 130) * 0.02 for t in range(110, 160)}
    rms_map[130] = 0.95
    rms_map[131] = 0.92
    rms_map[132] = 0.88
    rms_map[133] = 0.85
    for t in range(500, 520):
        rms_map[t] = 0.1
    return {
        "rms_map": rms_map,
        "avg_rms": 0.45,
        "max_rms": 1.0,
        "transcript_segments": [],
        "match_context": {},
    }


@pytest.fixture
def match_context():
    return {
        "tournament": "World Cup 2024",
        "match": "India vs Pakistan",
        "teams": ["India", "Pakistan"],
        "result": "India won by 6 wickets",
        "key_players": {"Virat Kohli": "82*", "Rohit Sharma": "45"},
        "highlights": ["Kohli century", "Bumrah hat-trick"],
        "highlight_timestamps": [120.0, 180.0],
        "venue": "MCG, Melbourne",
    }


# ── Helpers ────────────────────────────────────────────────────────

class TestHelpers:
    def test_words(self):
        result = _words("Hello World! HELLO")
        assert "hello" in result
        assert "world" in result
        assert len(result) == 2

    def test_count_overlap(self):
        assert _count_overlap({"hello", "world"}, {"hello", "foo"}) == 1

    def test_rms_range(self):
        m = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        result = _rms_range(m, 0.5, 2.5)
        assert len(result) == 3  # keys 1, 2
        assert 0.2 in result

    def test_avg_rms(self):
        m = {0: 0.1, 1: 0.2, 2: 0.3}
        avg = _avg_rms(m, 0.0, 2.0)
        assert 0.1 <= avg <= 0.3


# ── Agent 1: Hook Expert ───────────────────────────────────────────

class TestHookExpert:
    def test_exciting_hook(self, sample_candidate, rms_context):
        agent = HookExpert()
        result = agent.score(sample_candidate, rms_context)
        assert result["score"] > 60
        assert "reaction_opener" in result["reasoning"]

    def test_dull_hook(self, dull_candidate, rms_context):
        agent = HookExpert()
        result = agent.score(dull_candidate, rms_context)
        assert result["score"] < 50

    def test_returns_required_fields(self, sample_candidate, rms_context):
        agent = HookExpert()
        result = agent.score(sample_candidate, rms_context)
        assert "score" in result
        assert "reasoning" in result
        assert "hook_rms_ratio" in result

    def test_score_in_range(self, sample_candidate, rms_context):
        agent = HookExpert()
        result = agent.score(sample_candidate, rms_context)
        assert 0 <= result["score"] <= 100


# ── Agent 2: Emotion Expert ────────────────────────────────────────

class TestEmotionExpert:
    def test_emotional_content(self, sample_candidate, rms_context):
        agent = EmotionExpert()
        result = agent.score(sample_candidate, rms_context)
        assert result["score"] > 40

    def test_dull_silent(self, silent_candidate, rms_context):
        agent = EmotionExpert()
        result = agent.score(silent_candidate, rms_context)
        assert result["score"] < 40

    def test_score_range(self, sample_candidate, rms_context):
        agent = EmotionExpert()
        result = agent.score(sample_candidate, rms_context)
        assert 0 <= result["score"] <= 100


# ── Agent 3: Cricket Context Expert ────────────────────────────────

class TestCricketContextExpert:
    def test_cricket_content(self, sample_candidate, rms_context, match_context):
        rms_context["match_context"] = match_context
        agent = CricketContextExpert()
        result = agent.score(sample_candidate, rms_context)
        assert result["score"] > 30  # Kohli + chhakka + world cup

    def test_no_cricket(self, dull_candidate, rms_context):
        agent = CricketContextExpert()
        result = agent.score(dull_candidate, rms_context)
        assert result["score"] < 40

    def test_rivalry_boost(self, rms_context, match_context):
        rms_context["match_context"] = match_context
        agent = CricketContextExpert()
        c = {"start": 0, "end": 10, "text": "India vs Pakistan rivalry at its finest!"}
        result = agent.score(c, rms_context)
        assert "rivalry" in result["reasoning"]

    def test_known_highlight_match(self, rms_context, match_context):
        rms_context["match_context"] = match_context
        agent = CricketContextExpert()
        c = {"start": 120.0, "end": 130.0, "text": "highlight moment here"}
        result = agent.score(c, rms_context)
        assert result["score"] > 30
        assert "matches_known_highlight" in result["reasoning"]


# ── Agent 4: Viral Potential Expert ────────────────────────────────

class TestViralPotentialExpert:
    def test_viral_potential(self, sample_candidate, rms_context):
        agent = ViralPotentialExpert()
        result = agent.score(sample_candidate, rms_context)
        assert result["score"] > 20

    def test_rare_event(self, rms_context):
        agent = ViralPotentialExpert()
        c = {"start": 0, "end": 10, "text": "HATTRICK! This is history! Record broken!"}
        result = agent.score(c, rms_context)
        assert result["score"] >= 40  # hatTrick + Record


# ── Agent 5: Viewer Psychology Expert ──────────────────────────────

class TestViewerPsychologyExpert:
    def test_cliffhanger(self, rms_context):
        agent = ViewerPsychologyExpert()
        c = {"start": 0, "end": 15, "text": "Can India chase this total? Lets find out what happens!"}
        result = agent.score(c, rms_context)
        assert result["score"] > 30

    def test_identity_word(self, rms_context):
        agent = ViewerPsychologyExpert()
        c = {"start": 0, "end": 15, "text": "The king of cricket delivers again! Our champion!"}
        result = agent.score(c, rms_context)
        assert "identity_trigger" in result["reasoning"]


# ── Agent 6: Retention Expert ──────────────────────────────────────

class TestRetentionExpert:
    def test_good_retention(self, sample_candidate, rms_context):
        agent = RetentionExpert()
        result = agent.score(sample_candidate, rms_context)
        assert result["score"] > 30

    def test_short_clip_retention(self, rms_context):
        agent = RetentionExpert()
        c = {"start": 0, "end": 6, "text": "quick clip"}
        result = agent.score(c, rms_context)
        # Should be low but not rejected (no sweet spot hit)
        assert result["score"] < 60


# ── Agent 7: Brutal Rejection Agent ────────────────────────────────

class TestBrutalRejectionAgent:
    def test_rejects_dull(self, dull_candidate, rms_context):
        agent = BrutalRejectionAgent()
        result = agent.score(dull_candidate, rms_context)
        assert result["should_reject"]

    def test_pass_exciting(self, sample_candidate, rms_context):
        agent = BrutalRejectionAgent()
        result = agent.score(sample_candidate, rms_context)
        assert not result["should_reject"]

    def test_rejects_silent(self, silent_candidate, rms_context):
        agent = BrutalRejectionAgent()
        result = agent.score(silent_candidate, rms_context)
        assert result["should_reject"]

    def test_rejects_too_short(self, rms_context):
        agent = BrutalRejectionAgent()
        c = {"start": 0, "end": 3, "text": "hi"}
        result = agent.score(c, rms_context)
        assert result["should_reject"]


# ── All Agents ─────────────────────────────────────────────────────

class TestAllAgents:
    def test_all_agents_registered(self):
        assert len(ALL_AGENTS) == 7
        names = {a.name for a in ALL_AGENTS}
        assert names == {
            "hook_expert", "emotion_expert", "cricket_context",
            "viral_potential", "viewer_psychology",
            "retention_expert", "brutal_rejection",
        }

    def test_weights_sum_is_reasonable(self):
        total = sum(a.weight for a in ALL_AGENTS if a.name != "brutal_rejection")
        assert 0.95 <= total <= 1.05  # ~1.0 excluding rejection (penalty only)


# ── Arbiter ────────────────────────────────────────────────────────

class TestArbiter:
    def test_compute_weighted_score(self):
        agent_scores = {
            "hook_expert": {"score": 60, "reasoning": "ok"},
            "emotion_expert": {"score": 50, "reasoning": "ok"},
            "cricket_context": {"score": 40, "reasoning": "ok"},
            "viral_potential": {"score": 30, "reasoning": "ok"},
            "viewer_psychology": {"score": 20, "reasoning": "ok"},
            "retention_expert": {"score": 70, "reasoning": "ok"},
            "brutal_rejection": {"score": 10, "reasoning": "pass", "should_reject": False},
        }
        result = compute_weighted_score(agent_scores)
        assert "final_score" in result
        assert "breakdown" in result
        assert 0 <= result["final_score"] <= 100
        assert not result["should_reject"]

    def test_rejection_penalty(self):
        agent_scores = {
            "hook_expert": {"score": 60, "reasoning": "ok"},
            "emotion_expert": {"score": 50, "reasoning": "ok"},
            "cricket_context": {"score": 40, "reasoning": "ok"},
            "viral_potential": {"score": 30, "reasoning": "ok"},
            "viewer_psychology": {"score": 20, "reasoning": "ok"},
            "retention_expert": {"score": 70, "reasoning": "ok"},
            "brutal_rejection": {
                "score": 50,
                "reasoning": "dull; boring",
                "should_reject": True,
            },
        }
        result = compute_weighted_score(agent_scores)
        assert result["final_score"] < 50
        assert result["rejection_reasons"] == ["dull; boring"]
        # Only 1 agent flagged — needs 2+ for should_reject=True

    def test_weights_match_agents(self):
        for agent_name, weight in AGENT_WEIGHTS.items():
            assert 0 <= weight <= 1
        assert abs(sum(AGENT_WEIGHTS.values()) - 1.0) < 0.001

    def test_fmt_ts(self):
        assert fmt_ts(0) == "00:00:00"
        assert fmt_ts(65) == "00:01:05"
        assert fmt_ts(3661) == "01:01:01"


# ── ClipSelector ───────────────────────────────────────────────────

class TestClipSelector:
    def test_score_candidates(self, sample_candidate, dull_candidate, rms_context):
        selector = ClipSelector(use_llm_arbiter=False)
        candidates = [sample_candidate, dull_candidate]
        scored = selector.score_candidates(candidates, rms_context)
        assert len(scored) == 2
        assert scored[0]["final_score"] >= scored[1]["final_score"]
        for c in scored:
            assert "agent_scores" in c
            assert "final_score" in c
            assert "score_breakdown" in c

    def test_select_filters_rejected(self, sample_candidate, dull_candidate,
                                      rms_context):
        selector = ClipSelector(use_llm_arbiter=False)
        candidates = [sample_candidate, dull_candidate]
        scored = selector.score_candidates(candidates, rms_context)
        final = selector.select(scored, rms_context, max_selected=2, min_quality=0)
        assert len(final) <= 2

    def test_select_respects_max(self, sample_candidate, rms_context):
        selector = ClipSelector(use_llm_arbiter=False)
        candidates = [dict(sample_candidate) for _ in range(5)]
        scored = selector.score_candidates(candidates, rms_context)
        final = selector.select(scored, rms_context, max_selected=3, min_quality=0)
        assert len(final) <= 3


# ── Hook Auditor ───────────────────────────────────────────────────

class TestHookAuditor:
    def test_wicket_hook(self):
        result = analyze_clip_hook(
            clip_path="/fake/path.mp4",
            transcript_text="OUT! Bowled him! That is a peach of a delivery!",
            start_sec=0,
            clip_duration=20,
        )
        assert result["hook_score"] >= 15
        assert "wicket" in result.get("hook_types_found", [])

    def test_six_hook(self):
        result = analyze_clip_hook(
            clip_path="/fake/path.mp4",
            transcript_text="SIX! What a chhakka! Massive hit into the stands!",
            start_sec=0,
            clip_duration=20,
        )
        assert result["hook_score"] >= 15
        assert "six" in result.get("hook_types_found", [])

    def test_reaction_opener(self):
        result = analyze_clip_hook(
            clip_path="/fake/path.mp4",
            transcript_text="Oh wow! What a catch! Absolutely brilliant!",
            start_sec=0,
            clip_duration=20,
        )
        assert result["hook_score"] >= 25
        assert "reaction_face" in result.get("hook_types_found", [])

    def test_dull_opening(self):
        result = analyze_clip_hook(
            clip_path="/fake/path.mp4",
            transcript_text="and he takes a single to rotate the strike...",
            start_sec=0,
            clip_duration=20,
        )
        assert result["hook_score"] < 20
        assert result["swipe_risk"] in ("high", "medium")

    def test_silent_opening(self):
        result = analyze_clip_hook(
            clip_path="/fake/path.mp4",
            transcript_text="",
            start_sec=0,
            clip_duration=20,
        )
        assert result["hook_score"] == 0
        assert "silent_opening" in result.get("swipe_risks", [])

    def test_hook_types_present(self):
        for ht in HOOK_TYPES:
            assert ht in (
                "wicket", "six", "four", "crowd_eruption",
                "commentator_scream", "reaction_face", "controversy",
                "milestone", "drama", "instant_payoff",
            )

    def test_class_wrapper(self):
        auditor = HookAuditor()
        result = auditor.analyze_clip_hook(
            clip_path="/fake/path.mp4",
            transcript_text="SIX!",
            start_sec=0,
            clip_duration=15,
        )
        assert result["hook_score"] > 0


# ── ClipLearner ────────────────────────────────────────────────────

class TestClipLearner:
    def test_save_and_retrieve_selection(self, tmp_path):
        db = tmp_path / "test_learner.db"
        learner = ClipLearner(db_path=db)

        learner.save_clip_selection(
            clip_id="clip-001",
            selected_rank=1,
            final_score=75.0,
            agent_scores={"hook_expert": {"score": 80, "reasoning": "great"}},
            video_title="Test Match",
        )

        clips = learner.get_all_clips()
        assert len(clips) == 1
        assert clips[0]["clip_id"] == "clip-001"
        assert clips[0]["final_score"] == 75.0
        agent_json = json.loads(clips[0]["agent_scores_json"])
        assert "hook_expert" in agent_json

        learner.close()

    def test_update_performance(self, tmp_path):
        db = tmp_path / "test_learner.db"
        learner = ClipLearner(db_path=db)

        learner.save_clip_selection(
            clip_id="clip-001",
            selected_rank=1,
            final_score=80.0,
            agent_scores={},
        )
        learner.update_performance(
            clip_id="clip-001",
            youtube_video_id="abc123",
            views=1500,
            estimated_retention=0.65,
            completion_rate=0.72,
        )

        clips = learner.get_all_clips()
        assert clips[0]["youtube_video_id"] == "abc123"
        assert clips[0]["views"] == 1500

        learner.close()

    def test_get_performance_summary(self, tmp_path):
        db = tmp_path / "test_learner.db"
        learner = ClipLearner(db_path=db)

        learner.save_clip_selection("c1", 1, 70.0, {})
        learner.save_clip_selection("c2", 2, 60.0, {})
        learner.update_performance("c1", "abc", 100, 0.5, 0.5)
        learner.update_performance("c2", "def", 200, 0.6, 0.6)

        summary = learner.get_performance_summary()
        assert summary["total_clips"] == 2
        assert summary["clips_with_performance"] == 2
        assert summary["avg_views"] == 150.0

        learner.close()

    def test_hook_analysis(self, tmp_path):
        db = tmp_path / "test_learner.db"
        learner = ClipLearner(db_path=db)

        learner.save_hook_analysis(
            clip_id="clip-001",
            hook_score=75,
            hook_type="six",
            swipe_risk="low",
            swipe_risks=[],
            first_20_words="SIX! What a shot from",
            agent_hook_score=80.0,
        )

        learner.close()

    def test_get_top_performers(self, tmp_path):
        db = tmp_path / "test_learner.db"
        learner = ClipLearner(db_path=db)

        for i in range(5):
            learner.save_clip_selection(f"c{i}", i + 1, 60.0 + i, {})
            learner.update_performance(f"c{i}", f"yt{i}", 100 * (i + 1), 0.5, 0.5)

        top = learner.get_top_performers(min_views=50, limit=3)
        assert len(top) >= 2
        assert top[0]["views"] >= top[1]["views"]

        learner.close()


# ── Agent Base ─────────────────────────────────────────────────────

class TestAgentBase:
    def test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Agent().score({}, {})

    def test_weight_default(self):
        assert Agent().weight == 0.0

    def test_repr(self):
        agent = HookExpert()
        assert "hook_expert" in repr(agent)
        assert str(agent.weight) in repr(agent)


# ── Weight Learner (feedback loop) ──────────────────────────────────

class TestWeightLearner:
    def test_load_empty_db(self, tmp_path):
        """Empty DB should return empty list."""
        from automation.clip_selection.weight_learner import load_performance_data
        db = tmp_path / "empty.db"
        result = load_performance_data(db)
        assert result == []

    def test_compute_with_insufficient_data(self):
        """Fewer than min_clips should return defaults."""
        from automation.clip_selection.weight_learner import (
            compute_adaptive_weights,
            DEFAULT_WEIGHTS,
        )
        result = compute_adaptive_weights([], min_clips=10)
        assert result == DEFAULT_WEIGHTS

    def test_compute_with_enough_data(self):
        """With enough samples, should return blended weights."""
        from automation.clip_selection.weight_learner import compute_adaptive_weights

        perf_data = []
        for i in range(15):
            perf_data.append({
                "clip_id": f"c{i}",
                "final_score": 50.0 + i,
                "views": 100 + i * 50,
                "agent_scores": {
                    "hook_expert": {"score": 40 + i * 2, "reasoning": "ok"},
                    "emotion_expert": {"score": 30 + i, "reasoning": "ok"},
                    "cricket_context": {"score": 20 + i, "reasoning": "ok"},
                    "viral_potential": {"score": 10 + i, "reasoning": "ok"},
                    "viewer_psychology": {"score": 15 + i, "reasoning": "ok"},
                    "retention_expert": {"score": 50 + i, "reasoning": "ok"},
                    "brutal_rejection": {"score": 0, "should_reject": False},
                },
            })

        result = compute_adaptive_weights(perf_data, min_clips=10, drift_rate=0.3)
        assert "hook_expert" in result
        assert "technical_quality" in result
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_recalibrate_weights_convenience(self):
        """recalibrate_weights should handle missing DB gracefully."""
        from automation.clip_selection.weight_learner import recalibrate_weights
        result = recalibrate_weights(db_path="/nonexistent/db.db")
        assert abs(sum(result.values()) - 1.0) < 0.01


# ── Arbiter with Custom Weights ─────────────────────────────────────

class TestArbiterCustomWeights:
    def test_custom_weights_overrides_defaults(self):
        from automation.clip_selection.arbiter import compute_weighted_score

        agent_scores = {
            "hook_expert": {"score": 80, "reasoning": "great"},
            "emotion_expert": {"score": 50, "reasoning": "ok"},
            "cricket_context": {"score": 30, "reasoning": "ok"},
            "viral_potential": {"score": 20, "reasoning": "ok"},
            "viewer_psychology": {"score": 20, "reasoning": "ok"},
            "retention_expert": {"score": 50, "reasoning": "ok"},
            "brutal_rejection": {"score": 10, "reasoning": "pass", "should_reject": False},
        }

        default_result = compute_weighted_score(agent_scores)

        custom_weights = {
            "hook_expert": 0.50,
            "emotion_expert": 0.10,
            "viral_potential": 0.05,
            "cricket_context": 0.05,
            "viewer_psychology": 0.05,
            "retention_expert": 0.20,
            "technical_quality": 0.05,
        }
        custom_result = compute_weighted_score(agent_scores, weights=custom_weights)

        # Higher hook weight should increase final score when hook is strong
        assert custom_result["final_score"] > default_result["final_score"]


# ── CricketContextExpert Entity Bias ─────────────────────────────────

class TestEntityBias:
    def test_high_perf_players_get_boost(self, rms_context):
        agent = CricketContextExpert()
        rms_context["entity_bias"] = {
            "top_players": [("virat kohli", 0.9), ("rohit sharma", 0.8)],
            "avoid_players": [],
            "top_teams": [],
            "avoid_teams": [],
        }

        c = {"start": 0, "end": 10, "text": "Virat Kohli scores a century!"}
        result = agent.score(c, rms_context)
        assert "high_perf_players" in result["reasoning"]

    def test_low_perf_players_get_penalty(self, rms_context):
        agent = CricketContextExpert()
        rms_context["entity_bias"] = {
            "top_players": [],
            "avoid_players": ["lowperf player"],
            "top_teams": [],
            "avoid_teams": [],
        }

        c = {"start": 0, "end": 10, "text": "lowperf player takes a single"}
        result = agent.score(c, rms_context)
        assert result["score"] < 25

    def test_entity_bias_optional(self, rms_context):
        """No entity_bias should still work (backward compat)."""
        agent = CricketContextExpert()
        rms_context.pop("entity_bias", None)
        c = {"start": 0, "end": 10, "text": "Kohli hits a six"}
        result = agent.score(c, rms_context)
        assert result["score"] > 20
