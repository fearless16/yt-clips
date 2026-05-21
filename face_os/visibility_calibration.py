"""Visibility Calibration Module.

Validates that visibility metrics correlate with actual quality:
- Metric-truth correlation
- Confidence calibration
- Latent consistency validation
- Visibility robustness under ambiguity

Purpose:
    Visibility exists (huge improvement).
    But visible ≠ correct.
    This module adds calibration to ensure metrics are meaningful.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class CalibrationConfig:
    """Configuration for visibility calibration."""

    # Minimum correlation threshold for valid metric
    min_correlation: float = 0.5

    # Minimum confidence calibration score
    min_calibration_score: float = 0.7

    # Maximum allowed metric drift
    max_metric_drift: float = 0.1

    # Window size for drift detection
    drift_window: int = 30


@dataclass
class CalibrationResult:
    """Result of metric calibration."""

    # Metric name
    metric_name: str

    # Correlation with ground truth (if available)
    correlation: float

    # Calibration score (how well confidence predicts accuracy)
    calibration_score: float

    # Drift detection (is metric drifting?)
    is_drifting: bool
    drift_magnitude: float

    # Is metric valid?
    is_valid: bool

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "correlation": self.correlation,
            "calibration_score": self.calibration_score,
            "is_drifting": self.is_drifting,
            "drift_magnitude": self.drift_magnitude,
            "is_valid": self.is_valid,
        }


@dataclass
class CalibrationReport:
    """Report for all calibrated metrics."""

    # Individual metric results
    metrics: Dict[str, CalibrationResult]

    # Overall calibration quality
    overall_quality: float

    # Number of valid metrics
    valid_count: int
    total_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "metrics": {name: r.to_dict() for name, r in self.metrics.items()},
            "overall_quality": self.overall_quality,
            "valid_count": self.valid_count,
            "total_count": self.total_count,
        }


class VisibilityCalibrator:
    """Calibrates visibility metrics to ensure they correlate with quality.

    Tracks:
    - Metric values over time
    - Confidence values
    - Ground truth (if available)
    - Drift detection
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        """Initialize calibrator.

        Args:
            config: Calibration configuration
        """
        self.config = config or CalibrationConfig()

        # Metric history
        self._metric_history: Dict[str, list] = {}
        self._confidence_history: Dict[str, list] = {}
        self._ground_truth_history: Dict[str, list] = {}

        # Drift detection
        self._metric_baseline: Dict[str, float] = {}

    def update(
        self,
        metric_name: str,
        metric_value: float,
        confidence: float,
        ground_truth: Optional[float] = None,
    ) -> CalibrationResult:
        """Update calibration with new metric value.

        Args:
            metric_name: Name of metric
            metric_value: Metric value
            confidence: Confidence in metric [0, 1]
            ground_truth: Ground truth value (if available)

        Returns:
            CalibrationResult for this metric
        """
        # Initialize history if needed
        if metric_name not in self._metric_history:
            self._metric_history[metric_name] = []
            self._confidence_history[metric_name] = []
            self._ground_truth_history[metric_name] = []
            self._metric_baseline[metric_name] = metric_value

        # Update history
        self._metric_history[metric_name].append(metric_value)
        self._confidence_history[metric_name].append(confidence)
        if ground_truth is not None:
            self._ground_truth_history[metric_name].append(ground_truth)

        # Compute calibration
        correlation = self._compute_correlation(metric_name)
        calibration_score = self._compute_calibration_score(metric_name)
        is_drifting, drift_magnitude = self._detect_drift(metric_name)

        # Determine validity
        is_valid = (
            correlation >= self.config.min_correlation
            and calibration_score >= self.config.min_calibration_score
            and not is_drifting
        )

        return CalibrationResult(
            metric_name=metric_name,
            correlation=correlation,
            calibration_score=calibration_score,
            is_drifting=is_drifting,
            drift_magnitude=drift_magnitude,
            is_valid=is_valid,
        )

    def _compute_correlation(self, metric_name: str) -> float:
        """Compute correlation between metric and ground truth.

        Args:
            metric_name: Name of metric

        Returns:
            Correlation coefficient [-1, 1]
        """
        metrics = self._metric_history.get(metric_name, [])
        truths = self._ground_truth_history.get(metric_name, [])

        if len(metrics) < 2 or len(truths) < 2:
            return 0.0  # Not enough data

        # Use same length
        n = min(len(metrics), len(truths))
        metrics = metrics[-n:]
        truths = truths[-n:]

        # Compute correlation
        correlation = np.corrcoef(metrics, truths)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    def _compute_calibration_score(self, metric_name: str) -> float:
        """Compute calibration score.

        How well does confidence predict metric accuracy?

        Args:
            metric_name: Name of metric

        Returns:
            Calibration score [0, 1]
        """
        metrics = self._metric_history.get(metric_name, [])
        confidences = self._confidence_history.get(metric_name, [])

        if len(metrics) < 2 or len(confidences) < 2:
            return 0.5  # Default

        # Use same length
        n = min(len(metrics), len(confidences))
        metrics = metrics[-n:]
        confidences = confidences[-n:]

        # Compute calibration: high confidence should correlate with stability
        metric_stability = 1.0 - np.std(metrics) / (np.mean(np.abs(metrics)) + 1e-8)
        confidence_mean = np.mean(confidences)

        # Calibration = how well confidence predicts stability
        calibration = 1.0 - abs(metric_stability - confidence_mean)
        return float(np.clip(calibration, 0, 1))

    def _detect_drift(self, metric_name: str) -> tuple:
        """Detect if metric is drifting.

        Args:
            metric_name: Name of metric

        Returns:
            (is_drifting, drift_magnitude)
        """
        metrics = self._metric_history.get(metric_name, [])

        if len(metrics) < self.config.drift_window:
            return False, 0.0

        # Compare recent window to baseline
        recent = metrics[-self.config.drift_window:]
        baseline = self._metric_baseline.get(metric_name, metrics[0])

        drift = abs(np.mean(recent) - baseline)
        is_drifting = drift > self.config.max_metric_drift

        return is_drifting, float(drift)

    def get_report(self) -> CalibrationReport:
        """Get calibration report for all metrics.

        Returns:
            CalibrationReport with all metric results
        """
        metrics = {}
        valid_count = 0

        for metric_name in self._metric_history.keys():
            result = self.update(
                metric_name=metric_name,
                metric_value=self._metric_history[metric_name][-1],
                confidence=self._confidence_history[metric_name][-1],
                ground_truth=self._ground_truth_history[metric_name][-1] if self._ground_truth_history[metric_name] else None,
            )
            metrics[metric_name] = result
            if result.is_valid:
                valid_count += 1

        overall_quality = valid_count / max(len(metrics), 1)

        return CalibrationReport(
            metrics=metrics,
            overall_quality=overall_quality,
            valid_count=valid_count,
            total_count=len(metrics),
        )

    def reset(self) -> None:
        """Reset calibration state."""
        self._metric_history.clear()
        self._confidence_history.clear()
        self._ground_truth_history.clear()
        self._metric_baseline.clear()
