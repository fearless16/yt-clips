"""
test_phase1_energy.py — Phase 1: Energy Function Reformulation Tests.

Tests:
1. Energy term existence — all 5 terms must exist and be floats
2. Energy term numeric range — all terms must be non-negative and bounded
3. Energy delta regression — energy must decrease under stable input
4. Energy monotonicity — energy must converge under repeated stable input
5. EnergyReport per frame — pipeline must emit energy reports
6. Energy terms from each subsystem — each subsystem contributes its own energy
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import List

from face_os.types import (
    EnergyTerms,
    EnergyReport,
    GeometryState,
    IdentityState,
    TemporalState,
    GeometryMetrics,
    IdentityMetrics,
    TemporalMetrics,
    RendererMetrics,
    PassReport,
)
from face_os.energy import EnergyComputer


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_stable_geometry(confidence: float = 0.9) -> GeometryState:
    """Create a stable geometry state."""
    return GeometryState(
        pose=(0.0, 0.0, 0.0),
        geometry_confidence=confidence,
        mask=np.ones((256, 256), dtype=np.float32) * 0.5,
        canonical_transform=np.eye(3, dtype=np.float32),
        landmarks_478=np.random.rand(478, 3).astype(np.float32) * 100,
    )


def _make_stable_identity(uncertainty: float = 0.1) -> IdentityState:
    """Create a stable identity state."""
    return IdentityState(
        identity_uncertainty=uncertainty,
        region_confidence={"skin": 0.9, "left_eye": 0.8, "right_eye": 0.8},
        appearance_latent=np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8),
        anchor_weights=[1.0],
        initialized=True,
    )


def _make_stable_temporal(confidence: float = 0.9) -> TemporalState:
    """Create a stable temporal state."""
    return TemporalState(
        temporal_confidence=confidence,
        drift_score=0.05,
        continuity_score=0.95,
        motion_field=np.zeros((100, 100, 2), dtype=np.float32),
    )


# ─── Test Class 1: Energy Term Existence ─────────────────────────────────────

class TestEnergyTermExistence:
    """All 5 energy terms must exist and be measurable floats."""

    def test_E_geom_exists(self):
        """E_geom must exist and be a float."""
        computer = EnergyComputer()
        report = computer.compute(
            0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
        )
        assert hasattr(report.terms, "E_geom")
        assert isinstance(report.terms.E_geom, float)

    def test_E_identity_exists(self):
        """E_identity must exist and be a float."""
        computer = EnergyComputer()
        report = computer.compute(
            0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
        )
        assert hasattr(report.terms, "E_identity")
        assert isinstance(report.terms.E_identity, float)

    def test_E_temporal_exists(self):
        """E_temporal must exist and be a float."""
        computer = EnergyComputer()
        report = computer.compute(
            0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
        )
        assert hasattr(report.terms, "E_temporal")
        assert isinstance(report.terms.E_temporal, float)

    def test_E_photometric_exists(self):
        """E_photometric must exist and be a float."""
        computer = EnergyComputer()
        report = computer.compute(
            0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
        )
        assert hasattr(report.terms, "E_photometric")
        assert isinstance(report.terms.E_photometric, float)

    def test_E_smoothness_exists(self):
        """E_smoothness must exist and be a float."""
        computer = EnergyComputer()
        report = computer.compute(
            0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
        )
        assert hasattr(report.terms, "E_smoothness")
        assert isinstance(report.terms.E_smoothness, float)

    def test_E_total_exists(self):
        """E_total must exist and be the sum of all terms."""
        computer = EnergyComputer()
        report = computer.compute(
            0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
        )
        assert hasattr(report.terms, "E_total")
        expected = (
            report.terms.E_geom
            + report.terms.E_identity
            + report.terms.E_temporal
            + report.terms.E_photometric
            + report.terms.E_smoothness
        )
        assert abs(report.terms.E_total - expected) < 1e-6

    def test_all_terms_to_dict(self):
        """All energy terms must be serializable to dict."""
        computer = EnergyComputer()
        report = computer.compute(
            0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
        )
        d = report.terms.to_dict()
        assert "E_geom" in d
        assert "E_identity" in d
        assert "E_temporal" in d
        assert "E_photometric" in d
        assert "E_smoothness" in d
        assert "E_total" in d


# ─── Test Class 2: Energy Term Numeric Range ─────────────────────────────────

class TestEnergyTermNumericRange:
    """All energy terms must be non-negative and bounded."""

    def test_E_geom_non_negative(self):
        """E_geom must be >= 0."""
        computer = EnergyComputer()
        for _ in range(10):
            report = computer.compute(
                0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
            )
            assert report.terms.E_geom >= 0.0, f"E_geom = {report.terms.E_geom}"

    def test_E_identity_non_negative(self):
        """E_identity must be >= 0."""
        computer = EnergyComputer()
        for _ in range(10):
            report = computer.compute(
                0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
            )
            assert report.terms.E_identity >= 0.0, f"E_identity = {report.terms.E_identity}"

    def test_E_temporal_non_negative(self):
        """E_temporal must be >= 0."""
        computer = EnergyComputer()
        for _ in range(10):
            report = computer.compute(
                0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
            )
            assert report.terms.E_temporal >= 0.0, f"E_temporal = {report.terms.E_temporal}"

    def test_E_photometric_non_negative(self):
        """E_photometric must be >= 0."""
        computer = EnergyComputer()
        for _ in range(10):
            report = computer.compute(
                0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
            )
            assert report.terms.E_photometric >= 0.0, f"E_photometric = {report.terms.E_photometric}"

    def test_E_smoothness_non_negative(self):
        """E_smoothness must be >= 0."""
        computer = EnergyComputer()
        for _ in range(10):
            report = computer.compute(
                0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
            )
            assert report.terms.E_smoothness >= 0.0, f"E_smoothness = {report.terms.E_smoothness}"

    def test_E_total_non_negative(self):
        """E_total must be >= 0."""
        computer = EnergyComputer()
        for _ in range(10):
            report = computer.compute(
                0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
            )
            assert report.terms.E_total >= 0.0, f"E_total = {report.terms.E_total}"

    def test_energy_bounded_under_stable_input(self):
        """Energy must be bounded (< 100) under stable input."""
        computer = EnergyComputer()
        for _ in range(50):
            report = computer.compute(
                0, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
            )
            assert report.terms.E_total < 100.0, f"E_total = {report.terms.E_total}"

    def test_energy_bounded_under_varied_confidence(self):
        """Energy must be bounded across different confidence levels."""
        computer = EnergyComputer()
        for conf in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            geo = _make_stable_geometry(confidence=conf)
            id_state = _make_stable_identity(uncertainty=1.0 - conf)
            temp = _make_stable_temporal(confidence=conf)
            report = computer.compute(0, geo, id_state, temp)
            assert report.terms.E_total < 200.0, f"E_total = {report.terms.E_total} at conf={conf}"


# ─── Test Class 3: Energy Delta Regression ───────────────────────────────────

class TestEnergyDeltaRegression:
    """Energy must decrease when state improves."""

    def test_energy_decreases_with_higher_geometry_confidence(self):
        """Higher geometry confidence should give lower E_geom."""
        computer = EnergyComputer()

        geo_low = _make_stable_geometry(confidence=0.3)
        geo_high = _make_stable_geometry(confidence=0.9)
        id_state = _make_stable_identity()
        temp = _make_stable_temporal()

        report_low = computer.compute(0, geo_low, id_state, temp)
        report_high = computer.compute(0, geo_high, id_state, temp)

        assert report_high.terms.E_geom <= report_low.terms.E_geom

    def test_energy_decreases_with_lower_identity_uncertainty(self):
        """Lower identity uncertainty should give lower E_identity."""
        computer = EnergyComputer()

        geo = _make_stable_geometry()
        id_low = _make_stable_identity(uncertainty=0.8)
        id_high = _make_stable_identity(uncertainty=0.1)
        temp = _make_stable_temporal()

        report_low = computer.compute(0, geo, id_low, temp)
        report_high = computer.compute(0, geo, id_high, temp)

        assert report_high.terms.E_identity <= report_low.terms.E_identity

    def test_energy_decreases_with_higher_temporal_confidence(self):
        """Higher temporal confidence should give lower E_temporal."""
        computer = EnergyComputer()

        geo = _make_stable_geometry()
        id_state = _make_stable_identity()
        temp_low = _make_stable_temporal(confidence=0.3)
        temp_high = _make_stable_temporal(confidence=0.9)

        report_low = computer.compute(0, geo, id_state, temp_low)
        report_high = computer.compute(0, geo, id_state, temp_high)

        assert report_high.terms.E_temporal <= report_low.terms.E_temporal

    def test_energy_delta_computable(self):
        """Energy delta must be computable between two states."""
        computer = EnergyComputer()

        geo = _make_stable_geometry()
        id_state = _make_stable_identity()
        temp = _make_stable_temporal()

        report1 = computer.compute(0, geo, id_state, temp)
        report2 = computer.compute(1, geo, id_state, temp)

        delta = report2.terms.E_total - report1.terms.E_total
        # Under identical input, delta should be near zero
        assert abs(delta) < 1e-6

    def test_pass_report_delta(self):
        """PassReport must compute delta correctly."""
        report = PassReport(
            pass_id="test",
            frame_id=0,
            before={"E_total": 10.0, "E_geom": 3.0},
            after={"E_total": 8.0, "E_geom": 2.0},
        )
        report.compute_delta()
        assert abs(report.delta["E_total"] - (-2.0)) < 1e-6
        assert abs(report.delta["E_geom"] - (-1.0)) < 1e-6


# ─── Test Class 4: Energy Monotonicity ───────────────────────────────────────

class TestEnergyMonotonicity:
    """Energy must converge under repeated stable input."""

    def test_energy_converges_under_repeated_input(self):
        """Energy should converge (not oscillate) under repeated identical input."""
        computer = EnergyComputer()
        anchor = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        computer.anchor_face = anchor

        geo = _make_stable_geometry()
        id_state = _make_stable_identity()
        temp = _make_stable_temporal()

        energies: List[float] = []
        for i in range(100):
            report = computer.compute(i, geo, id_state, temp)
            energies.append(report.terms.E_total)

        # Energy should be stable (not diverging)
        first_10 = np.mean(energies[:10])
        last_10 = np.mean(energies[-10:])
        # Under identical input, energy should be identical
        assert abs(first_10 - last_10) < 1e-6

    def test_energy_does_not_diverge(self):
        """Energy must not diverge over many frames."""
        computer = EnergyComputer()

        energies: List[float] = []
        for i in range(200):
            geo = _make_stable_geometry()
            id_state = _make_stable_identity()
            temp = _make_stable_temporal()
            report = computer.compute(i, geo, id_state, temp)
            energies.append(report.terms.E_total)

        # Energy must remain bounded
        assert max(energies) < 100.0, f"Max energy: {max(energies)}"
        assert min(energies) >= 0.0, f"Min energy: {min(energies)}"

    def test_energy_stable_across_frames(self):
        """Energy variance must be low under stable input."""
        computer = EnergyComputer()

        energies: List[float] = []
        for i in range(50):
            report = computer.compute(
                i, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
            )
            energies.append(report.terms.E_total)

        variance = np.var(energies)
        assert variance < 1e-10, f"Energy variance: {variance}"


# ─── Test Class 5: EnergyReport Per Frame ────────────────────────────────────

class TestEnergyReportPerFrame:
    """EnergyReport must contain all required sections."""

    def test_energy_report_has_geometry_metrics(self):
        """EnergyReport must have geometry metrics."""
        computer = EnergyComputer()
        report = computer.compute(
            42, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
        )
        assert isinstance(report.geometry, GeometryMetrics)
        assert report.geometry.yaw == 0.0
        assert report.geometry.pitch == 0.0
        assert report.geometry.roll == 0.0

    def test_energy_report_has_identity_metrics(self):
        """EnergyReport must have identity metrics."""
        computer = EnergyComputer()
        report = computer.compute(
            42, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
        )
        assert isinstance(report.identity, IdentityMetrics)
        assert report.identity.uncertainty == 0.1

    def test_energy_report_has_temporal_metrics(self):
        """EnergyReport must have temporal metrics."""
        computer = EnergyComputer()
        report = computer.compute(
            42, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
        )
        assert isinstance(report.temporal, TemporalMetrics)
        assert report.temporal.temporal_confidence == 0.9

    def test_energy_report_has_renderer_metrics(self):
        """EnergyReport must have renderer metrics."""
        computer = EnergyComputer()
        report = computer.compute(
            42, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
        )
        assert isinstance(report.renderer, RendererMetrics)

    def test_energy_report_to_dict(self):
        """EnergyReport must be serializable to dict."""
        computer = EnergyComputer()
        report = computer.compute(
            42, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
        )
        d = report.to_dict()
        assert "frame_idx" in d
        assert "energy_terms" in d
        assert "geometry" in d
        assert "identity" in d
        assert "temporal" in d
        assert "renderer" in d

    def test_energy_report_frame_idx(self):
        """EnergyReport must have correct frame_idx."""
        computer = EnergyComputer()
        for idx in [0, 42, 100]:
            report = computer.compute(
                idx, _make_stable_geometry(), _make_stable_identity(), _make_stable_temporal()
            )
            assert report.frame_idx == idx


# ─── Test Class 6: Energy Terms From Each Subsystem ──────────────────────────

class TestEnergyFromSubsystems:
    """Each subsystem must contribute its own energy term."""

    def test_E_geom_from_geometry_state(self):
        """E_geom must depend on geometry state."""
        computer = EnergyComputer()

        geo_good = _make_stable_geometry(confidence=0.9)
        geo_bad = _make_stable_geometry(confidence=0.1)
        id_state = _make_stable_identity()
        temp = _make_stable_temporal()

        report_good = computer.compute(0, geo_good, id_state, temp)
        report_bad = computer.compute(0, geo_bad, id_state, temp)

        # Different geometry states must produce different E_geom
        assert report_good.terms.E_geom != report_bad.terms.E_geom

    def test_E_identity_from_identity_state(self):
        """E_identity must depend on identity state."""
        computer = EnergyComputer()

        geo = _make_stable_geometry()
        id_good = _make_stable_identity(uncertainty=0.1)
        id_bad = _make_stable_identity(uncertainty=0.9)
        temp = _make_stable_temporal()

        report_good = computer.compute(0, geo, id_good, temp)
        report_bad = computer.compute(0, geo, id_bad, temp)

        # Different identity states must produce different E_identity
        assert report_good.terms.E_identity != report_bad.terms.E_identity

    def test_E_temporal_from_temporal_state(self):
        """E_temporal must depend on temporal state."""
        computer = EnergyComputer()

        geo = _make_stable_geometry()
        id_state = _make_stable_identity()
        temp_good = _make_stable_temporal(confidence=0.9)
        temp_bad = _make_stable_temporal(confidence=0.1)

        report_good = computer.compute(0, geo, id_state, temp_good)
        report_bad = computer.compute(0, geo, id_state, temp_bad)

        # Different temporal states must produce different E_temporal
        assert report_good.terms.E_temporal != report_bad.terms.E_temporal

    def test_E_smoothness_depends_on_mask(self):
        """E_smoothness must depend on mask smoothness."""
        computer = EnergyComputer()

        # Smooth mask
        geo_smooth = _make_stable_geometry()
        geo_smooth.mask = np.ones((256, 256), dtype=np.float32) * 0.5
        cv2.GaussianBlur(geo_smooth.mask, (21, 21), 10, geo_smooth.mask)

        # Rough mask (random)
        geo_rough = _make_stable_geometry()
        geo_rough.mask = np.random.rand(256, 256).astype(np.float32)

        id_state = _make_stable_identity()
        temp = _make_stable_temporal()

        report_smooth = computer.compute(0, geo_smooth, id_state, temp)
        report_rough = computer.compute(0, geo_rough, id_state, temp)

        # Smooth mask should have lower E_smoothness
        assert report_smooth.terms.E_smoothness <= report_rough.terms.E_smoothness

    def test_geometry_metrics_extracted(self):
        """Geometry metrics must be extracted from geometry state."""
        computer = EnergyComputer()

        geo = _make_stable_geometry()
        geo.pose = (5.0, -3.0, 1.0)
        geo.geometry_confidence = 0.85

        report = computer.compute(0, geo, _make_stable_identity(), _make_stable_temporal())

        assert report.geometry.yaw == 5.0
        assert report.geometry.pitch == -3.0
        assert report.geometry.roll == 1.0
        assert report.geometry.geometry_confidence == 0.85

    def test_identity_metrics_extracted(self):
        """Identity metrics must be extracted from identity state."""
        computer = EnergyComputer()

        id_state = _make_stable_identity()
        id_state.identity_uncertainty = 0.25
        id_state.appearance_latent = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        report = computer.compute(
            0, _make_stable_geometry(), id_state, _make_stable_temporal()
        )

        assert report.identity.uncertainty == 0.25
        assert report.identity.appearance_latent_norm > 0

    def test_temporal_metrics_extracted(self):
        """Temporal metrics must be extracted from temporal state."""
        computer = EnergyComputer()

        temp = _make_stable_temporal()
        temp.temporal_confidence = 0.88
        temp.drift_score = 0.12
        temp.continuity_score = 0.95

        report = computer.compute(
            0, _make_stable_geometry(), _make_stable_identity(), temp
        )

        assert report.temporal.temporal_confidence == 0.88
        assert report.temporal.drift_score == 0.12
        assert report.temporal.continuity_score == 0.95


# ─── Need cv2 for one test ──────────────────────────────────────────────────
import cv2
