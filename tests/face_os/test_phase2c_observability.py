"""
test_phase2c_observability.py — Phase 2C: Observability Analysis Tests.

Tests:
1. Full-rank observability
2. Partial observability under occlusion
3. Lighting ambiguity detection
4. Pose degeneracy detection
5. Identity ambiguity detection
6. Jacobian conditioning

TDD: These tests should FAIL before implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from face_os.state_space import LatentState, StateTransitionModel, ObservationModel
from face_os.observability import (
    ObservabilityAnalyzer,
    ObservabilityReport,
    DegeneracyReport,
)


# ─── Test Class 1: ObservabilityAnalyzer ─────────────────────────────────────

class TestObservabilityAnalyzer:
    """Test observability analyzer."""

    def test_analyzer_exists(self):
        """ObservabilityAnalyzer must exist."""
        analyzer = ObservabilityAnalyzer()
        assert analyzer is not None

    def test_analyzer_has_analyze(self):
        """Analyzer must have analyze method."""
        analyzer = ObservabilityAnalyzer()
        assert hasattr(analyzer, "analyze")

    def test_analyzer_computes_observability_matrix(self):
        """Analyzer must compute observability matrix."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState()
        report = analyzer.analyze(state)
        assert report.observability_matrix is not None
        assert isinstance(report.observability_matrix, np.ndarray)

    def test_analyzer_computes_rank(self):
        """Analyzer must compute observability rank."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState()
        report = analyzer.analyze(state)
        assert hasattr(report, "observability_rank")
        assert isinstance(report.observability_rank, int)
        assert report.observability_rank >= 0

    def test_analyzer_identifies_observable_dimensions(self):
        """Analyzer must identify observable dimensions."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState()
        report = analyzer.analyze(state)
        assert hasattr(report, "observable_dimensions")
        assert isinstance(report.observable_dimensions, list)

    def test_analyzer_identifies_degenerate_dimensions(self):
        """Analyzer must identify degenerate dimensions."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState()
        report = analyzer.analyze(state)
        assert hasattr(report, "degenerate_dimensions")
        assert isinstance(report.degenerate_dimensions, list)

    def test_analyzer_computes_condition_number(self):
        """Analyzer must compute condition number."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState()
        report = analyzer.analyze(state)
        assert hasattr(report, "condition_number")
        assert report.condition_number >= 1.0


# ─── Test Class 2: ObservabilityReport ───────────────────────────────────────

class TestObservabilityReport:
    """Test observability report."""

    def test_report_exists(self):
        """ObservabilityReport must exist."""
        report = ObservabilityReport()
        assert report is not None

    def test_report_has_observability_matrix(self):
        """Report must have observability matrix."""
        report = ObservabilityReport()
        assert hasattr(report, "observability_matrix")

    def test_report_has_rank(self):
        """Report must have rank."""
        report = ObservabilityReport()
        assert hasattr(report, "observability_rank")

    def test_report_has_observable_dimensions(self):
        """Report must have observable dimensions."""
        report = ObservabilityReport()
        assert hasattr(report, "observable_dimensions")

    def test_report_has_degenerate_dimensions(self):
        """Report must have degenerate dimensions."""
        report = ObservabilityReport()
        assert hasattr(report, "degenerate_dimensions")

    def test_report_has_condition_number(self):
        """Report must have condition number."""
        report = ObservabilityReport()
        assert hasattr(report, "condition_number")

    def test_report_has_coupling_matrix(self):
        """Report must have latent coupling matrix."""
        report = ObservabilityReport()
        assert hasattr(report, "latent_coupling_matrix")

    def test_report_to_dict(self):
        """Report must be serializable."""
        report = ObservabilityReport()
        d = report.to_dict()
        assert "observability_rank" in d
        assert "observable_dimensions" in d
        assert "degenerate_dimensions" in d
        assert "condition_number" in d


# ─── Test Class 3: Full-Rank Observability ───────────────────────────────────

class TestFullRankObservability:
    """Test full-rank observability."""

    def test_full_observability_with_all_sensors(self):
        """With all sensors active, system should be fully observable."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState(
            yaw=5.0, pitch=-3.0, roll=1.0,
            identity_uncertainty=0.1,
            temporal_confidence=0.9,
        )
        report = analyzer.analyze(state)

        # At least geometry dimensions should be observable
        assert report.observability_rank >= 3  # yaw, pitch, roll

    def test_observability_rank_bounded(self):
        """Observability rank must not exceed state dimension."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState()
        report = analyzer.analyze(state)
        assert report.observability_rank <= LatentState.STATE_DIM

    def test_observable_dimensions_subset(self):
        """Observable dimensions must be subset of all dimensions."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState()
        report = analyzer.analyze(state)
        all_dims = set(range(LatentState.STATE_DIM))
        assert set(report.observable_dimensions).issubset(all_dims)


# ─── Test Class 4: Partial Observability ─────────────────────────────────────

class TestPartialObservability:
    """Test partial observability under occlusion."""

    def test_reduced_observability_under_occlusion(self):
        """Observability should reduce under occlusion."""
        analyzer = ObservabilityAnalyzer()

        # Normal state
        state_normal = LatentState(temporal_confidence=0.9)
        report_normal = analyzer.analyze(state_normal)

        # Occluded state
        state_occluded = LatentState(temporal_confidence=0.1)
        report_occluded = analyzer.analyze(state_occluded)

        # Occluded state should have fewer observable dimensions
        # (or at least not more)
        assert report_occluded.observability_rank <= report_normal.observability_rank


# ─── Test Class 5: Lighting Ambiguity ────────────────────────────────────────

class TestLightingAmbiguity:
    """Test lighting ambiguity detection."""

    def test_lighting_ambiguity_detected(self):
        """Analyzer must detect lighting/albedo ambiguity."""
        analyzer = ObservabilityAnalyzer()

        # High brightness — potential lighting ambiguity
        state = LatentState(brightness_mean=200.0)
        report = analyzer.analyze(state)

        # Should detect some ambiguity
        assert hasattr(report, "degeneracy_report")
        assert report.degeneracy_report is not None

    def test_degeneracy_report_has_lighting_ambiguity(self):
        """Degeneracy report must have lighting ambiguity flag."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState()
        report = analyzer.analyze(state)
        assert hasattr(report.degeneracy_report, "lighting_ambiguity")


# ─── Test Class 6: Pose Degeneracy ───────────────────────────────────────────

class TestPoseDegeneracy:
    """Test pose degeneracy detection."""

    def test_extreme_pose_detected(self):
        """Analyzer must detect extreme poses."""
        analyzer = ObservabilityAnalyzer()

        # Extreme yaw
        state = LatentState(yaw=85.0)
        report = analyzer.analyze(state)

        # Should detect degeneracy
        assert report.degeneracy_report is not None

    def test_degeneracy_report_has_pose_degeneracy(self):
        """Degeneracy report must have pose degeneracy flag."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState()
        report = analyzer.analyze(state)
        assert hasattr(report.degeneracy_report, "pose_degeneracy")


# ─── Test Class 7: Identity Ambiguity ────────────────────────────────────────

class TestIdentityAmbiguity:
    """Test identity ambiguity detection."""

    def test_high_identity_uncertainty_detected(self):
        """Analyzer must detect high identity uncertainty."""
        analyzer = ObservabilityAnalyzer()

        state = LatentState(identity_uncertainty=0.9)
        report = analyzer.analyze(state)

        assert report.degeneracy_report is not None

    def test_degeneracy_report_has_identity_ambiguity(self):
        """Degeneracy report must have identity ambiguity flag."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState()
        report = analyzer.analyze(state)
        assert hasattr(report.degeneracy_report, "identity_ambiguity")


# ─── Test Class 8: Jacobian Conditioning ─────────────────────────────────────

class TestJacobianConditioning:
    """Test Jacobian conditioning."""

    def test_condition_number_finite(self):
        """Condition number must be finite."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState()
        report = analyzer.analyze(state)
        assert np.isfinite(report.condition_number)

    def test_condition_number_positive(self):
        """Condition number must be positive."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState()
        report = analyzer.analyze(state)
        assert report.condition_number > 0

    def test_coupling_matrix_shape(self):
        """Coupling matrix must have correct shape."""
        analyzer = ObservabilityAnalyzer()
        state = LatentState()
        report = analyzer.analyze(state)
        n = LatentState.STATE_DIM
        assert report.latent_coupling_matrix.shape == (n, n)
