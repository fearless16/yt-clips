"""Recency-aware Bayesian segment learning with calibrated uncertainty."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    """Controls shrinkage, evidence thresholds, and time decay."""

    prior_strength: float = 12.0
    min_segment_samples: int = 6
    half_life_days: float = 120.0
    confidence_z: float = 1.64


@dataclass(frozen=True, slots=True)
class SegmentStats:
    """Posterior statistics for one feature value."""

    feature_name: str
    feature_value: str
    sample_count: int
    effective_samples: float
    raw_mean: float
    posterior_mean: float
    effect: float
    effect_lower_bound: float
    effect_upper_bound: float
    uncertainty: float
    lower_bound: float
    upper_bound: float
    confidence: float
    actionable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LearnerModel:
    """Immutable fitted model with auditable segment results."""

    baseline_mean: float
    baseline_variance: float
    observations: int
    fitted_at: str
    segments: dict[tuple[str, str], SegmentStats]

    def segment(self, feature_name: str, feature_value: str) -> SegmentStats:
        """Get one fitted segment or raise KeyError if unseen."""
        return self.segments[(feature_name, feature_value)]

    def recommendations(self, limit: int = 10) -> list[SegmentStats]:
        """Return supported positive effects ordered conservatively."""
        candidates = [
            segment for segment in self.segments.values()
            if segment.actionable and segment.effect_lower_bound > 0
        ]
        return sorted(
            candidates,
            key=lambda item: (item.effect_lower_bound, item.posterior_mean),
            reverse=True,
        )[:limit]


class BayesianSegmentLearner:
    """Fit explainable feature effects without pretending tiny samples are truth."""

    def __init__(self, config: LearnerConfig | None = None) -> None:
        self.config = config or LearnerConfig()

    def fit(
        self,
        observations: Iterable[dict[str, Any]],
        *,
        now: datetime | None = None,
    ) -> LearnerModel:
        """Fit a recency-weighted empirical-Bayes segment model."""
        rows = list(observations)
        fitted_at = now or datetime.now(timezone.utc)
        if fitted_at.tzinfo is None:
            fitted_at = fitted_at.replace(tzinfo=timezone.utc)
        weighted = [(row, self._weight(row, fitted_at)) for row in rows]
        total_weight = sum(weight for _, weight in weighted)
        if total_weight <= 0:
            return LearnerModel(0.0, 0.0, len(rows), fitted_at.isoformat(), {})

        baseline = sum(float(row["outcome_score"]) * weight for row, weight in weighted) / total_weight
        variance = sum(
            weight * (float(row["outcome_score"]) - baseline) ** 2
            for row, weight in weighted
        ) / total_weight
        baseline_uncertainty = math.sqrt(max(variance, 1e-6) / total_weight)

        grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for row, weight in weighted:
            outcome = max(0.0, min(1.0, float(row["outcome_score"])))
            for name, value in row.get("features", {}).items():
                grouped.setdefault((str(name), str(value)), []).append((outcome, weight))

        segments: dict[tuple[str, str], SegmentStats] = {}
        for (name, value), samples in grouped.items():
            effective = sum(weight for _, weight in samples)
            raw_mean = sum(outcome * weight for outcome, weight in samples) / effective
            posterior = (
                raw_mean * effective + baseline * self.config.prior_strength
            ) / (effective + self.config.prior_strength)
            posterior_n = effective + self.config.prior_strength
            uncertainty = math.sqrt(max(variance, 1e-6) / posterior_n)
            effect_uncertainty = math.sqrt(uncertainty ** 2 + baseline_uncertainty ** 2)
            effect = posterior - baseline
            lower = max(0.0, posterior - self.config.confidence_z * uncertainty)
            upper = min(1.0, posterior + self.config.confidence_z * uncertainty)
            confidence = 1.0 - math.exp(-effective / max(1.0, self.config.min_segment_samples))
            segments[(name, value)] = SegmentStats(
                feature_name=name,
                feature_value=value,
                sample_count=len(samples),
                effective_samples=effective,
                raw_mean=raw_mean,
                posterior_mean=posterior,
                effect=effect,
                effect_lower_bound=effect - self.config.confidence_z * effect_uncertainty,
                effect_upper_bound=effect + self.config.confidence_z * effect_uncertainty,
                uncertainty=uncertainty,
                lower_bound=lower,
                upper_bound=upper,
                confidence=confidence,
                actionable=len(samples) >= self.config.min_segment_samples,
            )
        return LearnerModel(
            baseline_mean=baseline,
            baseline_variance=variance,
            observations=len(rows),
            fitted_at=fitted_at.isoformat(),
            segments=segments,
        )

    def _weight(self, row: dict[str, Any], now: datetime) -> float:
        captured = datetime.fromisoformat(
            str(row.get("evidence_at") or row["captured_at"]).replace("Z", "+00:00")
        )
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - captured).total_seconds() / 86_400.0)
        return 2.0 ** (-age_days / max(0.001, self.config.half_life_days))
