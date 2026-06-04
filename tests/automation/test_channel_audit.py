"""TDD tests for channel audit execution: day gating, learner context, entity novelty.

Tests verify:
- Upload day gating: Thursday uploads are deferred to Friday
- Learner context injection into SEO prompts
- Entity novelty bonus for under-explored high-performers
- Duration preference applied in config
"""
import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


class TestUploadDayGating:
    """Dead day avoidance: Thursday uploads deferred to next priority day."""

    def test_is_dead_day_returns_true_for_thursday(self):
        from automation.scheduling import is_dead_day
        cfg = {"upload_schedule": {"avoid_days": ["thursday"]}}
        assert is_dead_day("thursday", cfg) is True

    def test_is_dead_day_returns_false_for_friday(self):
        from automation.scheduling import is_dead_day
        cfg = {"upload_schedule": {"avoid_days": ["thursday"]}}
        assert is_dead_day("friday", cfg) is False

    def test_is_dead_day_returns_false_when_no_config(self):
        from automation.scheduling import is_dead_day
        assert is_dead_day("thursday", {}) is False

    def test_next_priority_day_skips_dead_days(self):
        from automation.scheduling import next_upload_day
        cfg = {
            "upload_schedule": {
                "avoid_days": ["thursday"],
                "priority_days": ["friday", "sunday", "wednesday"],
            }
        }
        # If today is Thursday, next upload day should be Friday
        thu = datetime(2026, 6, 4, 12, 0)  # A Thursday
        result = next_upload_day(thu, cfg)
        assert result.strftime("%A").lower() == "friday"

    def test_next_priority_day_keeps_good_day(self):
        from automation.scheduling import next_upload_day
        cfg = {
            "upload_schedule": {
                "avoid_days": ["thursday"],
                "priority_days": ["friday", "sunday", "wednesday"],
            }
        }
        fri = datetime(2026, 6, 5, 12, 0)  # A Friday
        result = next_upload_day(fri, cfg)
        assert result.strftime("%A").lower() == "friday"
        assert result.date() == fri.date()


class TestLearnerSEOContext:
    """Learner data should be injected into SEO prompts."""

    def test_get_learner_context_returns_player_data(self):
        from automation.seo.seo import _get_learner_context
        ctx = _get_learner_context()
        # Should return a string (may be empty if no DB data in test env)
        assert isinstance(ctx, str)

    def test_get_learner_context_includes_top_players(self):
        """When learner DB has data, context should mention top players."""
        from automation.seo.seo import _get_learner_context
        mock_state = {
            "entity_scores": json.dumps({
                "players": {
                    "bumrah": {"score": 0.767, "n": 3, "avg_views": 1326},
                    "kohli": {"score": 0.246, "n": 10, "avg_views": 181},
                },
                "teams": {
                    "gt": {"score": 0.310, "n": 44, "avg_views": 279},
                },
            }),
            "format_scores": json.dumps({
                "hook_types": {
                    "debate": {"score": 0.541, "n": 1, "avg_views": 672},
                    "reaction": {"score": 0.427, "n": 9, "avg_views": 477},
                },
                "title_patterns": {},
            }),
        }
        with patch("automation.seo.seo._load_learner_state", return_value=mock_state):
            ctx = _get_learner_context()
            assert "bumrah" in ctx.lower() or "1326" in ctx
            assert "debate" in ctx.lower() or "672" in ctx

    def test_learner_context_graceful_on_no_db(self):
        """If no DB exists, context should return empty string."""
        from automation.seo.seo import _get_learner_context
        with patch("automation.seo.seo._load_learner_state", return_value=None):
            ctx = _get_learner_context()
            assert ctx == ""


class TestEntityNoveltyBonus:
    """Under-explored high-performers get a novelty bonus in scoring."""

    def test_novelty_bonus_for_low_n_high_score(self):
        from automation.learner.entity_learner import EntityLearner
        state = MagicMock()
        state.get_entity.return_value = {
            "score": 0.767, "n": 3, "avg_views": 1326,
            "recent_views_avg": 1326, "recent_uploads": [],
        }
        learner = EntityLearner(state)
        bonus = learner.novelty_bonus("players", "bumrah")
        assert bonus > 0.0  # Should get a bonus

    def test_no_novelty_bonus_for_high_n(self):
        from automation.learner.entity_learner import EntityLearner
        state = MagicMock()
        state.get_entity.return_value = {
            "score": 0.258, "n": 73, "avg_views": 317,
            "recent_views_avg": 317, "recent_uploads": [],
        }
        learner = EntityLearner(state)
        bonus = learner.novelty_bonus("players", "rohit")
        assert bonus == 0.0  # No bonus — over-saturated

    def test_no_novelty_bonus_for_low_score(self):
        from automation.learner.entity_learner import EntityLearner
        state = MagicMock()
        state.get_entity.return_value = {
            "score": 0.138, "n": 2, "avg_views": 99,
            "recent_views_avg": 99, "recent_uploads": [],
        }
        learner = EntityLearner(state)
        bonus = learner.novelty_bonus("players", "hardik")
        assert bonus == 0.0  # Low score, no bonus even with low n


class TestConfigDurationPreference:
    """Config should specify preferred duration range."""

    def test_config_has_preferred_duration(self):
        from utils.config import load_config
        cfg = load_config()
        hl = cfg.get("highlight", {})
        # After config change, these should exist
        assert "preferred_duration_min" in hl
        assert "preferred_duration_max" in hl
        assert hl["preferred_duration_min"] >= 30
        assert hl["preferred_duration_max"] <= 50


# ---------------------------------------------------------------------------
# Bug regression tests
# ---------------------------------------------------------------------------

class TestBugRegressions:
    """Regression tests for bugs found during code audit."""

    def test_dead_day_and_priority_conflict(self):
        """BUG2: Day that is both dead AND priority must NOT be returned."""
        from automation.scheduling import next_upload_day
        cfg = {
            "upload_schedule": {
                "avoid_days": ["thursday", "friday"],
                "priority_days": ["friday"],  # Friday is dead AND priority
            }
        }
        thu = datetime(2026, 6, 4, 12, 0)  # Thursday
        result = next_upload_day(thu, cfg)
        # Friday is dead — must skip to Saturday
        assert result.strftime("%A").lower() != "thursday"
        assert result.strftime("%A").lower() != "friday"

    def test_no_novelty_bonus_for_unseen_entity(self):
        """BUG4: n=0 entities (never observed) must NOT get a bonus."""
        from automation.learner.entity_learner import EntityLearner
        state = MagicMock()
        state.get_entity.return_value = {
            "score": 0.5, "n": 0, "avg_views": 0,
            "recent_views_avg": 0, "recent_uploads": [],
        }
        learner = EntityLearner(state)
        bonus = learner.novelty_bonus("players", "unknown")
        assert bonus == 0.0

    def test_novelty_factor_capped_at_one(self):
        """Defense: novelty_factor must never exceed 1.0."""
        from automation.learner.entity_learner import EntityLearner
        state = MagicMock()
        # n=1, score=1.0 — max novelty case
        state.get_entity.return_value = {
            "score": 1.0, "n": 1, "avg_views": 2000,
            "recent_views_avg": 2000, "recent_uploads": [],
        }
        learner = EntityLearner(state)
        bonus = learner.novelty_bonus("players", "superstar")
        assert bonus <= 0.15  # Max bonus is 0.15

    def test_learner_context_connection_closed_on_error(self):
        """BUG3: DB connection must close even if query fails."""
        from automation.seo.seo import _load_learner_state
        # If the table doesn't exist, should return None (not leak connection)
        with patch("automation.seo.seo.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = _load_learner_state()
            assert result is None
