from datetime import datetime, timedelta, timezone

from shorts_intelligence.learner import BayesianSegmentLearner, LearnerConfig


NOW = datetime(2026, 8, 20, tzinfo=timezone.utc)


def _row(outcome: float, value: str, *, days_ago: int = 1) -> dict:
    return {
        "video_id": f"{value}-{outcome}-{days_ago}",
        "outcome_score": outcome,
        "captured_at": (NOW - timedelta(days=days_ago)).isoformat(),
        "features": {"duration_bucket": value},
    }


def test_small_samples_are_shrunk_and_not_actionable():
    learner = BayesianSegmentLearner(LearnerConfig(
        prior_strength=10.0,
        min_segment_samples=5,
    ))
    observations = [_row(0.5, "baseline") for _ in range(20)]
    observations.append(_row(1.0, "tiny"))

    model = learner.fit(observations, now=NOW)
    tiny = model.segment("duration_bucket", "tiny")

    assert 0.5 < tiny.posterior_mean < 0.6
    assert tiny.actionable is False
    assert tiny.confidence < 0.5


def test_supported_signal_becomes_actionable():
    learner = BayesianSegmentLearner(LearnerConfig(
        prior_strength=5.0,
        min_segment_samples=5,
    ))
    observations = [_row(0.35, "weak") for _ in range(12)]
    observations += [_row(0.85, "strong") for _ in range(12)]

    model = learner.fit(observations, now=NOW)

    strong = model.segment("duration_bucket", "strong")
    weak = model.segment("duration_bucket", "weak")
    assert strong.actionable is True
    assert strong.posterior_mean > weak.posterior_mean
    assert strong.lower_bound > weak.lower_bound
    assert strong.effect_lower_bound > 0
    assert weak.effect_upper_bound < 0


def test_recent_evidence_outweighs_stale_evidence():
    learner = BayesianSegmentLearner(LearnerConfig(
        prior_strength=1.0,
        min_segment_samples=2,
        half_life_days=10.0,
    ))
    rows = [_row(0.1, "candidate", days_ago=100) for _ in range(10)]
    rows += [_row(0.9, "candidate", days_ago=1) for _ in range(3)]

    model = learner.fit(rows, now=NOW)

    assert model.segment("duration_bucket", "candidate").raw_mean > 0.7


def test_publish_evidence_time_overrides_snapshot_capture_time():
    learner = BayesianSegmentLearner(LearnerConfig(half_life_days=10.0, prior_strength=1.0))
    recent = _row(0.9, "candidate", days_ago=1)
    stale = _row(0.1, "candidate", days_ago=1)
    stale["evidence_at"] = (NOW - timedelta(days=100)).isoformat()
    recent["evidence_at"] = (NOW - timedelta(days=1)).isoformat()

    model = learner.fit([stale, recent], now=NOW)

    assert model.segment("duration_bucket", "candidate").raw_mean > 0.8


def test_recommendations_require_effect_interval_above_baseline():
    learner = BayesianSegmentLearner(LearnerConfig(prior_strength=5.0, min_segment_samples=5))
    rows = [_row(0.50, "baseline") for _ in range(30)]
    rows += [_row(0.51, "noise") for _ in range(20)]
    rows += [_row(0.90, "winner") for _ in range(20)]

    model = learner.fit(rows, now=NOW)
    recommended = {item.feature_value for item in model.recommendations()}

    assert "winner" in recommended
    assert "noise" not in recommended
