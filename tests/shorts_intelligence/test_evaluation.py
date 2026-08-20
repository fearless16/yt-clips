from datetime import datetime, timedelta, timezone

from shorts_intelligence.evaluation import temporal_holdout
from shorts_intelligence.learner import BayesianSegmentLearner, LearnerConfig


def test_temporal_holdout_is_leak_free_and_compares_naive_baselines():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(40):
        strong = index % 2 == 0
        rows.append({
            "video_id": str(index),
            "captured_at": (start + timedelta(days=index)).isoformat(),
            "evidence_at": (start + timedelta(days=index)).isoformat(),
            "outcome_score": 0.8 if strong else 0.2,
            "features": {"hook_type": "strong" if strong else "weak"},
        })

    result = temporal_holdout(
        rows,
        BayesianSegmentLearner(LearnerConfig(
            prior_strength=1,
            min_segment_samples=2,
            half_life_days=10_000,
        )),
        train_fraction=0.75,
    )

    assert result["train"] == 30
    assert result["test"] == 10
    assert result["model_rmse"] < result["baseline_rmse"]
    assert result["model_rmse"] < result["legacy_zero_rmse"]
    assert result["beats_baseline"] is True
