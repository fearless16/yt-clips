"""Tests for §16.8 Reconstruction Confidence C_recon (D-05 Phase-2B prerequisite).

arch §16.8:
    C_recon = C_obs · Coverage_pose · Coverage_light · Visibility
    Invariant: C_recon ≤ C_obs always (coverage/visibility can only reduce trust)
    Required test: with high C_obs but low coverage, the gate does NOT engage
        the latent to drive the face.

C_recon is the COMPOSITE confidence the Phase-2B production gate MUST consume
(arch §19). It is the product of raw observation confidence and all three
coverage/visibility factors:
  - C_obs: latent.mean_confidence() (the raw per-frame confidence)
  - Coverage_pose: |observed pose bins| / |total pose bins| (§16.7)
  - Coverage_light: |observed lighting bins| / |total lighting bins| (§16.7)
  - Visibility: mean_visibility (§16.6 scalar proxy for V(u,v,t))

This is an observable telemetry signal AND the Phase-2B gate input (once all
factors exist). Currently NOT folded into the live gate — that's the Phase-2B
default-flip decision, which requires all factors to be real (they now are).

Determinism: fixed synthetic inputs, no randomness (arch §3).
"""
from __future__ import annotations

import numpy as np
import pytest

from face_os.reconstruction_confidence import compute_reconstruction_confidence


class TestComputeReconstructionConfidence:
    def test_all_factors_one_leaves_confidence_unchanged(self):
        assert compute_reconstruction_confidence(0.75, 1.0, 1.0, 1.0) == pytest.approx(0.75)

    def test_zero_pose_coverage_zeros(self):
        assert compute_reconstruction_confidence(0.8, 0.0, 1.0, 1.0) == 0.0

    def test_zero_lighting_coverage_zeros(self):
        assert compute_reconstruction_confidence(0.8, 1.0, 0.0, 1.0) == 0.0

    def test_zero_visibility_zeros(self):
        assert compute_reconstruction_confidence(0.8, 1.0, 1.0, 0.0) == 0.0

    def test_never_exceeds_c_obs(self):
        """§16.8 invariant: C_recon ≤ C_obs (coverage/visibility can only
        REDUCE trust, never add it)."""
        for c_obs in (0.0, 0.3, 0.5, 0.8, 1.0):
            for cp in (0.0, 0.1, 0.5, 1.0):
                for cl in (0.0, 0.3, 0.8, 1.0):
                    for mv in (0.0, 0.5, 0.9, 1.0):
                        result = compute_reconstruction_confidence(c_obs, cp, cl, mv)
                        assert result <= c_obs + 1e-12, (
                            f"C_recon={result} > C_obs={c_obs} for ({cp},{cl},{mv})"
                        )

    def test_clamps_inputs_to_unit_interval(self):
        """Out-of-range inputs clamp, never explode."""
        assert compute_reconstruction_confidence(1.5, 1.0, 1.0, 1.0) == pytest.approx(1.0)
        assert compute_reconstruction_confidence(0.5, 1.5, 1.0, 1.0) == pytest.approx(0.5)
        assert compute_reconstruction_confidence(0.5, 1.0, -0.1, 1.0) == 0.0
        assert compute_reconstruction_confidence(0.5, 1.0, 1.0, 2.0) == pytest.approx(0.5)

    def test_low_coverage_caps_below_high_coverage_ceiling(self):
        """arch §16.8 required test: with high C_obs but low coverage, the
        gate must NOT engage the latent (C_recon < C_obs)."""
        c_obs = 0.95
        # Low coverage: frontal only, one lighting, partial visibility
        c_recon_low = compute_reconstruction_confidence(c_obs, 1.0/37, 1.0/18, 0.85)
        # High coverage: many poses, many lights, full visibility
        c_recon_high = compute_reconstruction_confidence(c_obs, 0.5, 0.5, 1.0)
        assert c_recon_low < c_recon_high
        assert c_recon_low < c_obs
        assert c_recon_high < c_obs

    def test_low_c_obs_low_coverage_is_very_low(self):
        """A face barely observed under few conditions gives near-zero C_recon."""
        c_recon = compute_reconstruction_confidence(0.1, 0.05, 0.05, 0.5)
        assert c_recon < 0.01
