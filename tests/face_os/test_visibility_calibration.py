"""Tests for Visibility Calibration Module.

Tests for:
- CalibrationResult
- CalibrationReport
- VisibilityCalibrator
- Correlation computation
- Calibration score
- Drift detection

Target: 15 tests
"""

import numpy as np
import pytest

from face_os.visibility_calibration import (
    CalibrationConfig,
    CalibrationResult,
    CalibrationReport,
    VisibilityCalibrator,
)


class TestCalibrationConfig:
    """Test CalibrationConfig."""

    def test_default_config(self):
        """Default config must have reasonable values."""
        config = CalibrationConfig()
        assert config.min_correlation > 0
        assert config.min_calibration_score > 0
        assert config.max_metric_drift > 0
        assert config.drift_window > 0


class TestCalibrationResult:
    """Test CalibrationResult."""

    def test_result_to_dict(self):
        """Result must convert to dict."""
        result = CalibrationResult(
            metric_name="test",
            correlation=0.8,
            calibration_score=0.9,
            is_drifting=False,
            drift_magnitude=0.01,
            is_valid=True,
        )
        d = result.to_dict()
        assert d["metric_name"] == "test"
        assert d["correlation"] == 0.8
        assert d["is_valid"] is True

    def test_result_has_all_fields(self):
        """Result must have all required fields."""
        result = CalibrationResult(
            metric_name="test",
            correlation=0.8,
            calibration_score=0.9,
            is_drifting=False,
            drift_magnitude=0.01,
            is_valid=True,
        )
        assert hasattr(result, "metric_name")
        assert hasattr(result, "correlation")
        assert hasattr(result, "calibration_score")
        assert hasattr(result, "is_drifting")
        assert hasattr(result, "drift_magnitude")
        assert hasattr(result, "is_valid")


class TestCalibrationReport:
    """Test CalibrationReport."""

    def test_report_to_dict(self):
        """Report must convert to dict."""
        metrics = {
            "test": CalibrationResult(
                metric_name="test",
                correlation=0.8,
                calibration_score=0.9,
                is_drifting=False,
                drift_magnitude=0.01,
                is_valid=True,
            )
        }
        report = CalibrationReport(
            metrics=metrics,
            overall_quality=1.0,
            valid_count=1,
            total_count=1,
        )
        d = report.to_dict()
        assert d["overall_quality"] == 1.0
        assert d["valid_count"] == 1

    def test_report_has_all_fields(self):
        """Report must have all required fields."""
        report = CalibrationReport(
            metrics={},
            overall_quality=0.0,
            valid_count=0,
            total_count=0,
        )
        assert hasattr(report, "metrics")
        assert hasattr(report, "overall_quality")
        assert hasattr(report, "valid_count")
        assert hasattr(report, "total_count")


class TestVisibilityCalibrator:
    """Test VisibilityCalibrator."""

    def test_initialization(self):
        """Calibrator must initialize."""
        calibrator = VisibilityCalibrator()
        assert calibrator.config is not None

    def test_update_with_metric(self):
        """Update must work with metric value."""
        calibrator = VisibilityCalibrator()
        result = calibrator.update(
            metric_name="test",
            metric_value=0.8,
            confidence=0.9,
        )
        assert result.metric_name == "test"
        assert result.correlation == 0.0  # No ground truth yet

    def test_update_with_ground_truth(self):
        """Update must work with ground truth."""
        calibrator = VisibilityCalibrator()
        
        # Add some data
        for i in range(10):
            result = calibrator.update(
                metric_name="test",
                metric_value=0.8 + i * 0.01,
                confidence=0.9,
                ground_truth=0.8 + i * 0.01,
            )
        
        assert result.correlation > 0.5  # Should correlate

    def test_calibration_score(self):
        """Calibration score must be computed."""
        calibrator = VisibilityCalibrator()
        
        # Add consistent data
        for _ in range(10):
            result = calibrator.update(
                metric_name="test",
                metric_value=0.8,
                confidence=0.9,
            )
        
        assert result.calibration_score > 0.5

    def test_drift_detection(self):
        """Drift detection must work."""
        calibrator = VisibilityCalibrator()
        
        # Add stable data
        for _ in range(10):
            calibrator.update(
                metric_name="test",
                metric_value=0.8,
                confidence=0.9,
            )
        
        # Add drifting data
        for i in range(30):
            result = calibrator.update(
                metric_name="test",
                metric_value=0.8 + i * 0.01,
                confidence=0.9,
            )
        
        # Should detect drift
        assert result.is_drifting or result.drift_magnitude > 0

    def test_get_report(self):
        """Report must be generated."""
        calibrator = VisibilityCalibrator()
        
        # Add some data
        for _ in range(10):
            calibrator.update(
                metric_name="test",
                metric_value=0.8,
                confidence=0.9,
            )
        
        report = calibrator.get_report()
        assert report.total_count == 1
        assert report.overall_quality >= 0

    def test_reset(self):
        """Reset must clear state."""
        calibrator = VisibilityCalibrator()
        
        # Add some data
        calibrator.update(
            metric_name="test",
            metric_value=0.8,
            confidence=0.9,
        )
        
        calibrator.reset()
        
        # Should be empty
        assert len(calibrator._metric_history) == 0

    def test_multiple_metrics(self):
        """Multiple metrics must work."""
        calibrator = VisibilityCalibrator()
        
        calibrator.update(metric_name="metric1", metric_value=0.8, confidence=0.9)
        calibrator.update(metric_name="metric2", metric_value=0.7, confidence=0.8)
        
        report = calibrator.get_report()
        assert report.total_count == 2

    def test_correlation_with_perfect_data(self):
        """Perfect correlation must be detected."""
        calibrator = VisibilityCalibrator()
        
        # Perfect correlation: metric = ground_truth
        for i in range(20):
            value = 0.5 + i * 0.02
            result = calibrator.update(
                metric_name="test",
                metric_value=value,
                confidence=0.9,
                ground_truth=value,
            )
        
        assert result.correlation > 0.9

    def test_correlation_with_noisy_data(self):
        """Noisy correlation must be lower."""
        calibrator = VisibilityCalibrator()
        
        # Noisy correlation
        for i in range(20):
            value = 0.5 + i * 0.02
            noise = np.random.randn() * 0.1
            result = calibrator.update(
                metric_name="test",
                metric_value=value + noise,
                confidence=0.9,
                ground_truth=value,
            )
        
        # Correlation should be positive but not perfect
        assert 0.5 < result.correlation < 0.99

    def test_validity_criteria(self):
        """Validity must check all criteria."""
        config = CalibrationConfig(
            min_correlation=0.5,
            min_calibration_score=0.7,
            max_metric_drift=0.1,
        )
        calibrator = VisibilityCalibrator(config=config)
        
        # Add good data
        for i in range(20):
            result = calibrator.update(
                metric_name="test",
                metric_value=0.8,
                confidence=0.9,
                ground_truth=0.8,
            )
        
        # Should be valid
        assert result.is_valid
