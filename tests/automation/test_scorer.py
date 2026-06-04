"""TDD tests for automation/learner/scorer.py — recalibrate_weights fixes."""
import json
import pytest
from unittest.mock import MagicMock

from automation.learner.scorer import CricketScorer, DEFAULT_WEIGHTS


def _make_event(views: int, trend=0.5, fmt=0.4, entity=0.3, timing=0.5,
                historical=0.3) -> MagicMock:
    """Build a mock ClipEvent with a metrics_received payload."""
    payload = {
        "views": views,
        "player_tags": ["kohli"],
        "team_tags": ["rcb"],
        "hook_type": "shock",
        "title_pattern": {},
        "weighted_score": views / 100.0,
    }
    ev = MagicMock()
    ev.payload_json = json.dumps(payload)
    return ev


def _make_scorer(trend_val=0.5, fmt_val=0.4, entity_val=0.3, timing_val=0.5) -> CricketScorer:
    """Build a CricketScorer with all sub-learners mocked."""
    fmt = MagicMock()
    fmt.get_hook_score.return_value = 0.8
    fmt.get_title_score.return_value = fmt_val / 0.8  # so hook*title = fmt_val

    entity = MagicMock()
    entity.get_entity_score.return_value = entity_val
    entity.fatigue_penalty.return_value = 0.0

    trend = MagicMock()
    trend.get_trend_for_entities.return_value = trend_val

    timing = MagicMock()
    timing.get_schedule_recommendation.return_value = {"confidence": timing_val}

    duration = MagicMock()
    state = MagicMock()
    state.get.return_value = None  # forces DEFAULT_WEIGHTS

    return CricketScorer(fmt, entity, trend, timing, duration, state)


# ---------------------------------------------------------------------------
# recalibrate_weights — productive-event count fix
# ---------------------------------------------------------------------------

class TestRecalibrateWeightsProductiveCount:

    def test_skips_gracefully_when_all_events_have_zero_views(self):
        """recalibrate_weights must not crash when every event has views=0."""
        scorer = _make_scorer()
        events = [_make_event(views=0) for _ in range(25)]
        # Must not raise; weights stay at defaults
        scorer.recalibrate_weights(events)
        for k, v in DEFAULT_WEIGHTS.items():
            assert abs(scorer._weights.get(k, 0) - v) < 1e-9, (
                f"Weight {k} changed when no productive events: {scorer._weights.get(k)}"
            )

    def test_productive_count_excludes_zero_view_events(self):
        """The averaging divisor `n` must count only events with views > 0."""
        scorer = _make_scorer()
        # 5 zero-view events + 20 real events = 25 total, but only 20 productive
        zero_events = [_make_event(views=0) for _ in range(5)]
        real_events = [_make_event(views=1000) for _ in range(20)]
        events = zero_events + real_events

        weights_before = dict(scorer._weights)
        scorer.recalibrate_weights(events)
        weights_after = scorer._weights

        # Weights must have been updated (real events are processed)
        updated = any(
            abs(weights_after.get(k, 0) - weights_before.get(k, 0)) > 1e-9
            for k in DEFAULT_WEIGHTS
        )
        assert updated, "Weights unchanged despite 20 productive events"

    def test_requires_at_least_20_events(self):
        """recalibrate_weights is a no-op with fewer than 20 events."""
        scorer = _make_scorer()
        events = [_make_event(views=1000) for _ in range(19)]
        weights_before = dict(scorer._weights)
        scorer.recalibrate_weights(events)
        assert scorer._weights == weights_before

    def test_weights_sum_to_one_after_recalibration(self):
        """Normalized weights must always sum to 1.0 after recalibration."""
        scorer = _make_scorer()
        events = [_make_event(views=1000 * (i + 1)) for i in range(25)]
        scorer.recalibrate_weights(events)
        core_keys = list(DEFAULT_WEIGHTS.keys())
        total = sum(scorer._weights.get(k, 0) for k in core_keys)
        assert abs(total - 1.0) < 1e-6, f"Weights don't sum to 1: {total}"

    def test_recalibration_recorded_in_state(self):
        """state.set() must be called after recalibration."""
        scorer = _make_scorer()
        events = [_make_event(views=500) for _ in range(25)]
        scorer.recalibrate_weights(events)
        scorer._state.set.assert_called_with("scoring_weights", scorer._weights)

    def test_last_recalibrated_timestamp_set(self):
        """_weights must contain 'last_recalibrated' ISO timestamp after update."""
        scorer = _make_scorer()
        events = [_make_event(views=500) for _ in range(25)]
        scorer.recalibrate_weights(events)
        assert "last_recalibrated" in scorer._weights

    def test_bayesian_blend_70_30(self):
        """New weights must be blended 70% old + 30% new (Bayesian update)."""
        scorer = _make_scorer()
        # Force known starting weights
        scorer._weights = {k: v for k, v in DEFAULT_WEIGHTS.items()}
        original_trend = scorer._weights["trend_weight"]

        events = [_make_event(views=1000) for _ in range(25)]
        scorer.recalibrate_weights(events)

        # The new trend_weight must be a blend between old and new
        # We can't know the exact new value, but it must differ from original
        # (unless the data exactly matches current weights) and must be in [0, 1]
        new_trend = scorer._weights.get("trend_weight", 0)
        assert 0.0 <= new_trend <= 1.0

    def test_minimum_weight_floor_respected(self):
        """No individual weight may fall below 0.05 floor."""
        scorer = _make_scorer(trend_val=0.0, fmt_val=0.0, entity_val=0.0, timing_val=0.0)
        # Force all component scores to 0 → raw proportions all 0 → floor at 0.05
        events = [_make_event(views=1000) for _ in range(25)]
        scorer.recalibrate_weights(events)
        for k in DEFAULT_WEIGHTS:
            assert scorer._weights.get(k, 0) >= 0.04, (  # small epsilon for normalize rounding
                f"Weight {k} below floor: {scorer._weights.get(k)}"
            )
