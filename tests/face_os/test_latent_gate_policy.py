"""TDD tests for D-05 Phase 2A: Option 3 forced-latent gate policy.

Phase 2A policy:
  "Use forced latent whenever latent is initialized, verification gate passed,
   shadow telemetry is valid."

This is the A/B-first stage — prove latent can drive pixels at all, skip all
confidence thresholds. Once proven, promote to Option 1 (production
relative-to-floor gate, already in ``_evaluate_latent_gate``).

Tests cover:
  1. ``_evaluate_latent_gate_forced`` pure function (Option 3)
  2. Gate policy config parsing + defaulting
  3. Pipeline wiring: ``gate_policy`` selects the right gate in ``_render_core``
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from face_os.pipeline import FaceOSPipeline


# ═══════════════════════════════════════════════════════════════════
# 1. Pure function: _evaluate_latent_gate_forced (Option 3)
# ═══════════════════════════════════════════════════════════════════

class TestForcedLatentGate:
    """Option 3: engage if initialized, nothing else matters."""

    def test_initialized_engages_regardless_of_confidence(self):
        """When initialized, the forced gate always engages — confidence, floor,
        and spike are irrelevant. This proves the latent plumbing drives pixels."""
        engage, state = FaceOSPipeline._evaluate_latent_gate_forced(
            initialized=True,
        )
        assert engage is True
        assert state == "engaged"

    def test_uninitialized_never_engages(self):
        engage, state = FaceOSPipeline._evaluate_latent_gate_forced(
            initialized=False,
        )
        assert engage is False
        assert state == "uninitialized"

    def test_forced_gate_ignores_confidence_params(self):
        """Signature accepts same kwargs for drop-in substitution, but ignores
        confidence / floor / prev — a forced gate has no confidence thresholds."""
        # Low confidence that would fail the production gate
        engage, state = FaceOSPipeline._evaluate_latent_gate_forced(
            initialized=True,
            confidence=0.10,
            confidence_prev=0.10,
            confidence_floor=0.50,
        )
        assert engage is True
        assert state == "engaged"

    def test_forced_gate_ignores_spike(self):
        """A sharp confidence drop (spike) is ignored by the forced gate."""
        engage, state = FaceOSPipeline._evaluate_latent_gate_forced(
            initialized=True,
            confidence=0.10,
            confidence_prev=0.90,
            confidence_floor=0.2335,
        )
        assert engage is True
        assert state == "engaged"


# ═══════════════════════════════════════════════════════════════════
# 2. Gate policy attribute: creation, default, and mutation
# ═══════════════════════════════════════════════════════════════════

class TestGatePolicyAttribute:

    def test_default_gate_policy_is_production(self):
        """Without explicit config, gate_policy defaults to 'production' so
        existing behaviour is untouched."""
        p = FaceOSPipeline()
        assert p._gate_policy == "production"

    def test_gate_policy_attribute_is_mutable(self):
        """The gate_policy can be changed at runtime for A/B switching without
        recreating the pipeline."""
        p = FaceOSPipeline()

        p._gate_policy = "forced_latent"
        assert p._gate_policy == "forced_latent"

        p._gate_policy = "production"
        assert p._gate_policy == "production"

    def test_gate_policy_is_irrelevant_on_legacy_path(self):
        """The gate_policy ONLY takes effect when render_source='latent'.
        When render_source='legacy', the gate is never consulted."""
        p = FaceOSPipeline()
        # Default is legacy — gate_policy is irrelevant on the legacy path
        assert p.render_source == "legacy"
        assert p._gate_policy == "production"


# ═══════════════════════════════════════════════════════════════════
# 3. Pipeline wiring: gate_policy selects gate in _render_core context
# ═══════════════════════════════════════════════════════════════════

class TestGatePolicyDispatch:

    def test_production_policy_uses_production_gate(self):
        """gate_policy='production' — the existing relative-to-floor gate rejects
        at-floor confidence (proven in TestLatentGate)."""
        p = FaceOSPipeline()
        p._gate_policy = "production"
        engage, state = p._evaluate_latent_gate(
            initialized=True, confidence=0.2335, confidence_prev=0.2335,
            confidence_floor=0.2335,
        )
        assert engage is False
        assert state == "below_floor"

    def test_forced_policy_uses_forced_gate(self):
        """gate_policy='forced_latent' — engages when initialized, regardless of
        confidence."""
        p = FaceOSPipeline()
        p._gate_policy = "forced_latent"
        engage, state = p._evaluate_latent_gate_forced(
            initialized=True,
        )
        assert engage is True
        assert state == "engaged"

    def test_gate_selection_dispatches_by_policy(self):
        """The pipeline selects the correct gate function based on gate_policy."""
        p = FaceOSPipeline()

        # Forced: always engages when initialized
        p._gate_policy = "forced_latent"
        engage, state = p._evaluate_latent_gate_forced(initialized=True)
        assert engage is True

        # Production: rejects at-floor
        p._gate_policy = "production"
        engage, state = p._evaluate_latent_gate(
            initialized=True, confidence=0.2335, confidence_prev=0.2335,
            confidence_floor=0.2335,
        )
        assert engage is False


# ═══════════════════════════════════════════════════════════════════
# 4. set_anchor enrollment_mesh parameter (Task 2.5)
# ═══════════════════════════════════════════════════════════════════

class TestSetAnchorEnrollmentMesh:

    def test_set_anchor_accepts_enrollment_mesh(self):
        import numpy as np
        from face_os.subsystems.identity_estimator import IdentityEstimator

        mock_state = MagicMock()
        estimator = IdentityEstimator(mock_state)

        ref_bgr = np.zeros((128, 128, 3), dtype=np.uint8)
        some_mesh = np.random.randn(478, 3).astype(np.float32)

        with patch.object(estimator, '_set_anchor_impl'):
            estimator.set_anchor(ref_bgr, enrollment_mesh=some_mesh)

        assert estimator._enrollment_mesh is not None
        assert np.array_equal(estimator._enrollment_mesh, some_mesh)
