"""Visibility Calibration Module.

BEAST MODE FIXES:
- Nuked the get_report() side-effect that duplicated history data on every call.
- Fixed the First-Frame Baseline Trap (now uses warmup window mean).
- Fixed negative stability math in calibration score using bounded CV.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class CalibrationConfig:
    min_correlation: float = 0.5
    min_calibration_score: float = 0.7
    max_metric_drift: float = 0.1
    drift_window: int = 30


@dataclass
class CalibrationResult:
    metric_name: str
    correlation: float
    calibration_score: float
    is_drifting: bool
    drift_magnitude: float
    is_valid: bool

    def to_dict(self) -> dict:
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
    metrics: Dict[str, CalibrationResult]
    overall_quality: float
    valid_count: int
    total_count: int

    def to_dict(self) -> dict:
        return {
            "metrics": {name: r.to_dict() for name, r in self.metrics.items()},
            "overall_quality": self.overall_quality,
            "valid_count": self.valid_count,
            "total_count": self.total_count,
        }


class VisibilityCalibrator:
    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self._metric_history: Dict[str, list] = {}
        self._confidence_history: Dict[str, list] = {}
        self._ground_truth_history: Dict[str, list] = {}
        self._metric_baseline: Dict[str, float] = {}

    def update(
        self,
        metric_name: str,
        metric_value: float,
        confidence: float,
        ground_truth: Optional[float] = None,
    ) -> CalibrationResult:
        if metric_name not in self._metric_history:
            self._metric_history[metric_name] = []
            self._confidence_history[metric_name] = []
            self._ground_truth_history[metric_name] = []

        self._metric_history[metric_name].append(metric_value)
        self._confidence_history[metric_name].append(confidence)
        if ground_truth is not None:
            self._ground_truth_history[metric_name].append(ground_truth)

        correlation = self._compute_correlation(metric_name)
        calibration_score = self._compute_calibration_score(metric_name)
        is_drifting, drift_magnitude = self._detect_drift(metric_name)

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
        metrics = self._metric_history.get(metric_name, [])
        truths = self._ground_truth_history.get(metric_name, [])

        if len(metrics) < 2 or len(truths) < 2:
            return 0.0

        n = min(len(metrics), len(truths))
        metrics = metrics[-n:]
        truths = truths[-n:]

        with np.errstate(invalid='ignore'):
            correlation = np.corrcoef(metrics, truths)[0, 1]
        if np.isnan(correlation):
            return 0.0
        return float(correlation)

    def _compute_calibration_score(self, metric_name: str) -> float:
        metrics = self._metric_history.get(metric_name, [])
        confidences = self._confidence_history.get(metric_name, [])

        if len(metrics) < 2 or len(confidences) < 2:
            return 0.5

        n = min(len(metrics), len(confidences))
        metrics = metrics[-n:]
        confidences = confidences[-n:]

        # BEAST MODE FIX: Bounded stability using 1 / (1 + CV).
        # Prevents negative stability when std > mean.
        cv = np.std(metrics) / (np.mean(np.abs(metrics)) + 1e-8)
        metric_stability = 1.0 / (1.0 + cv)
        
        confidence_mean = np.mean(confidences)
        calibration = 1.0 - abs(metric_stability - confidence_mean)
        return float(np.clip(calibration, 0, 1))

    def _detect_drift(self, metric_name: str) -> tuple:
        metrics = self._metric_history.get(metric_name, [])

        if len(metrics) < self.config.drift_window:
            return False, 0.0

        recent = metrics[-self.config.drift_window:]
        
        # BEAST MODE FIX: Baseline is the mean of the first N (warmup) frames, not just frame 0.
        # Prevents permanent drift hallucination if frame 0 was an outlier.
        warmup = min(self.config.drift_window, len(metrics))
        baseline = np.mean(metrics[:warmup])
        self._metric_baseline[metric_name] = baseline

        drift = abs(np.mean(recent) - baseline)
        is_drifting = drift > self.config.max_metric_drift

        return is_drifting, float(drift)

    def get_report(self) -> CalibrationReport:
        metrics = {}
        valid_count = 0

        for metric_name in self._metric_history.keys():
            # BEAST MODE FIX: DO NOT call self.update() here! 
            # Calling update() mutates state and duplicates the last data point in history.
            correlation = self._compute_correlation(metric_name)
            calibration_score = self._compute_calibration_score(metric_name)
            is_drifting, drift_magnitude = self._detect_drift(metric_name)
            
            is_valid = (
                correlation >= self.config.min_correlation
                and calibration_score >= self.config.min_calibration_score
                and not is_drifting
            )

            result = CalibrationResult(
                metric_name=metric_name,
                correlation=correlation,
                calibration_score=calibration_score,
                is_drifting=is_drifting,
                drift_magnitude=drift_magnitude,
                is_valid=is_valid,
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
        self._metric_history.clear()
        self._confidence_history.clear()
        self._ground_truth_history.clear()
        self._metric_baseline.clear()