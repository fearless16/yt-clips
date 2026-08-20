"""Leak-free temporal replay for promotion and regression gates."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable

from shorts_intelligence.learner import BayesianSegmentLearner, LearnerModel


def temporal_holdout(
    observations: Iterable[dict[str, Any]],
    learner: BayesianSegmentLearner,
    *,
    train_fraction: float = 0.8,
) -> dict[str, Any]:
    """Fit only on older Shorts, then compare unseen newer outcomes."""
    if not 0.5 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9")
    rows = sorted(list(observations), key=_evidence_at)
    if len(rows) < 10:
        return {"status": "insufficient_data", "observations": len(rows)}
    split = max(1, min(len(rows) - 1, int(len(rows) * train_fraction)))
    train, test = rows[:split], rows[split:]
    model = learner.fit(train, now=_parse_time(_evidence_at(train[-1])))
    actual = [float(row["outcome_score"]) for row in test]
    predicted = [_predict(model, row, learner.config.min_segment_samples) for row in test]
    baseline = [model.baseline_mean] * len(test)
    legacy_zero = [0.0] * len(test)
    model_rmse = _rmse(actual, predicted)
    baseline_rmse = _rmse(actual, baseline)
    legacy_rmse = _rmse(actual, legacy_zero)
    return {
        "status": "ok",
        "observations": len(rows),
        "train": len(train),
        "test": len(test),
        "model_rmse": round(model_rmse, 6),
        "baseline_rmse": round(baseline_rmse, 6),
        "legacy_zero_rmse": round(legacy_rmse, 6),
        "beats_baseline": model_rmse < baseline_rmse,
        "beats_legacy": model_rmse < legacy_rmse,
    }


def _predict(model: LearnerModel, row: dict[str, Any], minimum: int) -> float:
    values = [
        model.segments[(str(name), str(value))].posterior_mean
        for name, value in row.get("features", {}).items()
        if (str(name), str(value)) in model.segments
        and model.segments[(str(name), str(value))].sample_count >= minimum
    ]
    return sum(values) / len(values) if values else model.baseline_mean


def _evidence_at(row: dict[str, Any]) -> str:
    return str(row.get("evidence_at") or row["captured_at"])


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _rmse(actual: list[float], predicted: list[float]) -> float:
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))
