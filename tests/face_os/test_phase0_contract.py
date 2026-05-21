"""
test_phase0_contract.py — Phase 0: Contract Lockdown Tests.

Tests the frame contract, energy report, renderer report, and visibility logging.
Every output frame must satisfy the FrameContract invariants.
Every change must have a PassReport with before/after/delta metrics.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from face_os.types import (
    FrameContract,
    EnergyTerms,
    EnergyReport,
    GeometryMetrics,
    IdentityMetrics,
    TemporalMetrics,
    RendererMetrics,
    RendererReport,
    PassReport,
    GeometryState,
    IdentityState,
    TemporalState,
)
from face_os.energy import EnergyComputer
from face_os.visibility import VisibilityLogger, validate_pass_report


# ─── FrameContract Tests ─────────────────────────────────────────────────────

class TestFrameContract:
    """Test frame contract validation."""

    def test_valid_frame_passes(self):
        """A valid 1920x1080x3 uint8 frame should pass."""
        contract = FrameContract()
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        passed, reason = contract.validate(frame)
        assert passed, f"Valid frame failed: {reason}"

    def test_wrong_shape_fails(self):
        """Wrong shape should fail."""
        contract = FrameContract()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        passed, reason = contract.validate(frame)
        assert not passed
        assert "shape_mismatch" in reason

    def test_wrong_dtype_fails(self):
        """Wrong dtype should fail."""
        contract = FrameContract()
        frame = np.zeros((1920, 1080, 3), dtype=np.float32)
        passed, reason = contract.validate(frame)
        assert not passed
        assert "dtype_mismatch" in reason

    def test_nan_fails(self):
        """NaN values should fail."""
        contract = FrameContract()
        frame = np.zeros((1920, 1080, 3), dtype=np.float32)
        frame[0, 0, 0] = np.nan
        # NaN becomes 0 in uint8, so test with float contract
        contract_float = FrameContract(expected_dtype="float32", allow_nan=False)
        passed, reason = contract_float.validate(frame)
        assert not passed
        assert "nan_detected" in reason

    def test_inf_fails(self):
        """Inf values should fail."""
        contract = FrameContract(expected_dtype="float32")
        frame = np.zeros((1920, 1080, 3), dtype=np.float32)
        frame[0, 0, 0] = np.inf
        passed, reason = contract.validate(frame)
        assert not passed
        assert "inf_detected" in reason

    def test_value_range_fails(self):
        """Values outside [0, 255] should fail for uint8."""
        contract = FrameContract()
        # uint8 can't have values outside [0, 255], so test with int16
        contract_int = FrameContract(expected_dtype="int16", min_value=0, max_value=255)
        frame = np.full((1920, 1080, 3), 300, dtype=np.int16)
        passed, reason = contract_int.validate(frame)
        assert not passed
        assert "value_range" in reason

    def test_custom_contract(self):
        """Custom contract should validate against custom parameters."""
        contract = FrameContract(
            expected_height=720,
            expected_width=1280,
            expected_channels=3,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        passed, reason = contract.validate(frame)
        assert passed, f"Custom contract failed: {reason}"


# ─── EnergyTerms Tests ────────────────────────────────────────────────────────

class TestEnergyTerms:
    """Test energy terms."""

    def test_energy_terms_defaults(self):
        """Energy terms should default to 0.0."""
        terms = EnergyTerms()
        assert terms.E_geom == 0.0
        assert terms.E_identity == 0.0
        assert terms.E_temporal == 0.0
        assert terms.E_photometric == 0.0
        assert terms.E_smoothness == 0.0
        assert terms.E_total == 0.0

    def test_energy_terms_to_dict(self):
        """Energy terms should convert to dict."""
        terms = EnergyTerms(
            E_geom=1.0,
            E_identity=2.0,
            E_temporal=3.0,
            E_photometric=4.0,
            E_smoothness=5.0,
            E_total=15.0,
        )
        d = terms.to_dict()
        assert d["E_geom"] == 1.0
        assert d["E_total"] == 15.0
        assert len(d) == 8  # 6 original + E_total_raw + normalized

    def test_energy_terms_are_floats(self):
        """All energy terms must be floats."""
        terms = EnergyTerms()
        assert isinstance(terms.E_geom, float)
        assert isinstance(terms.E_identity, float)
        assert isinstance(terms.E_temporal, float)
        assert isinstance(terms.E_photometric, float)
        assert isinstance(terms.E_smoothness, float)
        assert isinstance(terms.E_total, float)


# ─── EnergyReport Tests ───────────────────────────────────────────────────────

class TestEnergyReport:
    """Test energy report."""

    def test_energy_report_defaults(self):
        """Energy report should have sensible defaults."""
        report = EnergyReport()
        assert report.frame_idx == 0
        assert report.status == "pending"
        assert isinstance(report.terms, EnergyTerms)
        assert isinstance(report.geometry, GeometryMetrics)
        assert isinstance(report.identity, IdentityMetrics)
        assert isinstance(report.temporal, TemporalMetrics)
        assert isinstance(report.renderer, RendererMetrics)

    def test_energy_report_to_dict(self):
        """Energy report should convert to dict with all sections."""
        report = EnergyReport(frame_idx=42, status="computed")
        d = report.to_dict()
        assert d["frame_idx"] == 42
        assert d["status"] == "computed"
        assert "energy_terms" in d
        assert "geometry" in d
        assert "identity" in d
        assert "temporal" in d
        assert "renderer" in d


# ─── RendererReport Tests ─────────────────────────────────────────────────────

class TestRendererReport:
    """Test renderer report."""

    def test_renderer_report_defaults(self):
        """Renderer report should have sensible defaults."""
        report = RendererReport()
        assert report.frame_idx == 0
        assert report.output_shape == (1920, 1080, 3)
        assert report.output_dtype == "uint8"
        assert report.nan_count == 0
        assert report.inf_count == 0
        assert report.contract_passed is True

    def test_renderer_report_to_dict(self):
        """Renderer report should convert to dict."""
        report = RendererReport(frame_idx=10, contract_passed=True)
        d = report.to_dict()
        assert d["frame_idx"] == 10
        assert d["contract_passed"] is True
        assert "output_shape" in d
        assert "value_range" in d


# ─── PassReport Tests ─────────────────────────────────────────────────────────

class TestPassReport:
    """Test pass report with before/after/delta."""

    def test_pass_report_defaults(self):
        """Pass report should have sensible defaults."""
        report = PassReport()
        assert report.pass_id == ""
        assert report.frame_id == 0
        assert report.status == "pending"

    def test_pass_report_compute_delta(self):
        """Pass report should compute delta from before and after."""
        report = PassReport(
            pass_id="test",
            frame_id=1,
            before={"det_A": 0.99, "coverage": 60.0},
            after={"det_A": 1.0, "coverage": 61.0},
        )
        report.compute_delta()
        assert abs(report.delta["det_A"] - 0.01) < 1e-6
        assert abs(report.delta["coverage"] - 1.0) < 1e-6

    def test_pass_report_to_dict(self):
        """Pass report should convert to dict with all fields."""
        report = PassReport(
            pass_id="phase2_transform",
            frame_id=42,
            before={"det_A": 0.99},
            after={"det_A": 1.0},
            status="accepted",
        )
        report.compute_delta()
        d = report.to_dict()
        assert d["pass_id"] == "phase2_transform"
        assert d["frame_id"] == 42
        assert d["status"] == "accepted"
        assert "before" in d
        assert "after" in d
        assert "delta" in d

    def test_pass_report_missing_before_after_is_valid(self):
        """Pass report with only metrics should be valid."""
        report = PassReport(
            pass_id="test",
            frame_id=1,
            metrics={"key": "value"},
        )
        assert validate_pass_report(report)

    def test_pass_report_empty_is_invalid(self):
        """Pass report with no data should be invalid."""
        report = PassReport()
        assert not validate_pass_report(report)


# ─── EnergyComputer Tests ─────────────────────────────────────────────────────

class TestEnergyComputer:
    """Test energy computation."""

    def test_energy_computer_creates_report(self):
        """EnergyComputer should create a valid EnergyReport."""
        computer = EnergyComputer()

        geo = GeometryState(
            pose=(0.0, 0.0, 0.0),
            geometry_confidence=0.9,
            mask=np.ones((256, 256), dtype=np.float32) * 0.5,
        )
        id_state = IdentityState(
            identity_uncertainty=0.2,
            region_confidence={"skin": 0.8, "eye": 0.7},
        )
        temp = TemporalState(
            temporal_confidence=0.9,
            drift_score=0.1,
            continuity_score=0.95,
        )

        report = computer.compute(42, geo, id_state, temp)

        assert report.frame_idx == 42
        assert report.status == "computed"
        assert isinstance(report.terms, EnergyTerms)
        assert report.terms.E_total >= 0.0

    def test_energy_terms_are_non_negative(self):
        """Energy terms should be non-negative (lower = better)."""
        computer = EnergyComputer()

        geo = GeometryState(geometry_confidence=0.9)
        id_state = IdentityState(identity_uncertainty=0.2)
        temp = TemporalState(temporal_confidence=0.9)

        report = computer.compute(0, geo, id_state, temp)

        assert report.terms.E_geom >= 0.0
        assert report.terms.E_identity >= 0.0
        assert report.terms.E_temporal >= 0.0
        assert report.terms.E_photometric >= 0.0
        assert report.terms.E_smoothness >= 0.0
        assert report.terms.E_total >= 0.0

    def test_energy_decreases_with_better_confidence(self):
        """Energy should decrease when confidence increases."""
        computer = EnergyComputer()

        geo_low = GeometryState(geometry_confidence=0.3)
        geo_high = GeometryState(geometry_confidence=0.9)
        id_state = IdentityState(identity_uncertainty=0.5)
        temp = TemporalState(temporal_confidence=0.5)

        report_low = computer.compute(0, geo_low, id_state, temp)
        report_high = computer.compute(0, geo_high, id_state, temp)

        # Higher confidence should give lower energy
        assert report_high.terms.E_geom <= report_low.terms.E_geom

    def test_energy_report_has_all_metrics(self):
        """Energy report should have all metric sections."""
        computer = EnergyComputer()

        geo = GeometryState(
            pose=(5.0, -2.0, 1.0),
            geometry_confidence=0.85,
            mask=np.ones((256, 256), dtype=np.float32) * 0.6,
            landmarks_478=np.random.rand(478, 3).astype(np.float32),
        )
        id_state = IdentityState(
            identity_uncertainty=0.15,
            region_confidence={"skin": 0.9},
            appearance_latent=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        )
        temp = TemporalState(
            temporal_confidence=0.92,
            drift_score=0.05,
            continuity_score=0.98,
        )

        report = computer.compute(10, geo, id_state, temp)

        # Check geometry metrics
        assert report.geometry.yaw == 5.0
        assert report.geometry.pitch == -2.0
        assert report.geometry.roll == 1.0
        assert report.geometry.landmark_count == 478

        # Check identity metrics
        assert report.identity.uncertainty == 0.15
        assert "skin" in report.identity.region_confidence

        # Check temporal metrics
        assert report.temporal.temporal_confidence == 0.92
        assert report.temporal.drift_score == 0.05

        # Check renderer metrics
        assert report.renderer.M_mean == pytest.approx(0.6, abs=0.01)


# ─── VisibilityLogger Tests ───────────────────────────────────────────────────

class TestVisibilityLogger:
    """Test visibility logging."""

    def test_logger_creates_reports(self):
        """Logger should create pass reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = VisibilityLogger(output_dir=tmpdir)

            report = logger.log_pass(
                pass_id="test_pass",
                frame_id=42,
                before={"det_A": 0.99},
                after={"det_A": 1.0},
                metrics={"key": "value"},
                status="accepted",
            )

            assert report.pass_id == "test_pass"
            assert report.frame_id == 42
            assert abs(report.delta["det_A"] - 0.01) < 1e-6

    def test_logger_saves_to_json(self):
        """Logger should save reports to JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = VisibilityLogger(output_dir=tmpdir)

            logger.log_pass(
                pass_id="test",
                frame_id=1,
                before={"val": 1.0},
                after={"val": 2.0},
                metrics={},
            )

            summary_file = logger.save_all(prefix="test")
            assert os.path.exists(summary_file)

            with open(summary_file) as f:
                data = json.load(f)
            assert data["total_passes"] == 1

    def test_logger_energy_reports(self):
        """Logger should save energy reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = VisibilityLogger(output_dir=tmpdir)

            report = EnergyReport(frame_idx=10, status="computed")
            logger.log_energy(report)

            summary_file = logger.save_all(prefix="energy")
            assert os.path.exists(summary_file)

    def test_logger_renderer_reports(self):
        """Logger should save renderer reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = VisibilityLogger(output_dir=tmpdir)

            report = RendererReport(frame_idx=10, contract_passed=True)
            logger.log_renderer(report)

            summary_file = logger.save_all(prefix="renderer")
            assert os.path.exists(summary_file)

    def test_validate_pass_report(self):
        """validate_pass_report should check required fields."""
        valid = PassReport(pass_id="test", frame_id=1, metrics={"k": "v"})
        assert validate_pass_report(valid)

        invalid = PassReport()
        assert not validate_pass_report(invalid)
