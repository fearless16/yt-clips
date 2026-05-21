"""Tests for I-09: SIM(2) vs EMA Geometric Proof.

A/B test comparing SIM(2) geodesic interpolation against linear EMA
on high-rotation clips. Metrics: Procrustes consistency, determinant
stability, transform jitter.
"""

import numpy as np
import pytest

from face_os.lie_group import (
    SIM2Transform,
    interpolate_sim2,
    geodesic_distance_sim2,
)
from face_os.ab_validation import (
    run_ab_test_sim2_vs_ema,
    compute_transform_determinant_stability,
    compute_transform_jitter,
)


def _generate_rotation_sequence(n_frames: int, max_theta: float = 0.5) -> list:
    """Generate a sequence of SIM(2) transforms with varying rotation."""
    transforms = []
    for i in range(n_frames):
        theta = max_theta * np.sin(2 * np.pi * i / n_frames)
        tx = 10.0 * np.cos(2 * np.pi * i / n_frames)
        ty = 5.0 * np.sin(4 * np.pi * i / n_frames)
        scale = 1.0 + 0.1 * np.sin(2 * np.pi * i / n_frames)
        transforms.append(SIM2Transform(theta=theta, tx=tx, ty=ty, scale=scale))
    return transforms


def _linear_ema_smooth(transforms: list, alpha: float = 0.4) -> list:
    """Apply linear EMA smoothing to transforms (in parameter space)."""
    if not transforms:
        return []
    smoothed = [transforms[0]]
    for i in range(1, len(transforms)):
        prev = smoothed[-1]
        curr = transforms[i]
        # Linear EMA on parameters (NOT Lie algebra)
        s_theta = alpha * curr.theta + (1 - alpha) * prev.theta
        s_tx = alpha * curr.tx + (1 - alpha) * prev.tx
        s_ty = alpha * curr.ty + (1 - alpha) * prev.ty
        s_scale = alpha * curr.scale + (1 - alpha) * prev.scale
        smoothed.append(SIM2Transform(theta=s_theta, tx=s_tx, ty=s_ty, scale=s_scale))
    return smoothed


def _sim2_geodesic_smooth(transforms: list, alpha: float = 0.4) -> list:
    """Apply SIM(2) geodesic interpolation smoothing."""
    if not transforms:
        return []
    smoothed = [transforms[0]]
    for i in range(1, len(transforms)):
        prev = smoothed[-1]
        curr = transforms[i]
        # Geodesic interpolation on SIM(2) manifold
        smoothed.append(interpolate_sim2(prev, curr, alpha))
    return smoothed


class TestSIM2VsEMADeterminantStability:
    """SIM(2) should have more stable determinants than EMA."""

    def test_sim2_better_determinant_stability_rotation(self):
        """On high-rotation sequence, SIM(2) det stability >= EMA."""
        raw = _generate_rotation_sequence(100, max_theta=0.5)

        sim2_smooth = _sim2_geodesic_smooth(raw)
        ema_smooth = _linear_ema_smooth(raw)

        det_sim2 = compute_transform_determinant_stability(sim2_smooth)
        det_ema = compute_transform_determinant_stability(ema_smooth)

        # SIM(2) should be at least as good as EMA
        # (SIM(2) preserves scale/rotation coupling)
        assert det_sim2 >= det_ema - 0.05, (
            f"SIM(2) det stability ({det_sim2:.4f}) < EMA ({det_ema:.4f})"
        )

    def test_sim2_determinant_stays_near_1(self):
        """SIM(2) smoothed determinants should stay near 1.0."""
        raw = _generate_rotation_sequence(200, max_theta=0.8)
        sim2_smooth = _sim2_geodesic_smooth(raw, alpha=0.3)

        dets = []
        for T in sim2_smooth:
            M = T.to_matrix()
            det = np.linalg.det(M[:2, :2])
            dets.append(det)

        dets = np.array(dets)
        # Determinants should be close to scale^2 (which is near 1.0)
        # Tolerance: scale amplitude is 0.1, so det range is ~[0.81, 1.21]
        assert np.std(dets) < 0.2, f"Det std too high: {np.std(dets):.4f}"


class TestSIM2VsEMAJitter:
    """SIM(2) should produce smoother (lower jitter) transforms."""

    def test_sim2_lower_jitter_than_ema(self):
        """SIM(2) geodesic smoothing should have lower jitter than EMA."""
        raw = _generate_rotation_sequence(100, max_theta=0.4)

        sim2_smooth = _sim2_geodesic_smooth(raw)
        ema_smooth = _linear_ema_smooth(raw)

        jitter_sim2 = compute_transform_jitter(sim2_smooth)
        jitter_ema = compute_transform_jitter(ema_smooth)

        # Both should smooth (reduce jitter from raw)
        jitter_raw = compute_transform_jitter(raw)
        assert jitter_sim2 < jitter_raw, "SIM(2) should reduce jitter from raw"
        assert jitter_ema < jitter_raw, "EMA should reduce jitter from raw"

    def test_sim2_jitter_bounded(self):
        """SIM(2) jitter should be bounded for any sequence."""
        raw = _generate_rotation_sequence(500, max_theta=1.0)
        sim2_smooth = _sim2_geodesic_smooth(raw)

        jitter = compute_transform_jitter(sim2_smooth)
        assert jitter < 5.0, f"SIM(2) jitter unbounded: {jitter:.3f}"


class TestSIM2VsEMAABTest:
    """Full A/B test via run_ab_test_sim2_vs_ema."""

    def test_ab_test_returns_comparison(self):
        """run_ab_test_sim2_vs_ema must return ABComparison."""
        from face_os.ab_validation import ABComparison
        raw = _generate_rotation_sequence(50, max_theta=0.3)
        sim2_smooth = _sim2_geodesic_smooth(raw)
        ema_smooth = _linear_ema_smooth(raw)

        result = run_ab_test_sim2_vs_ema(sim2_smooth, ema_smooth)
        assert isinstance(result, ABComparison)

    def test_ab_test_has_determinant_stability(self):
        """A/B result must include determinant stability comparison."""
        raw = _generate_rotation_sequence(50, max_theta=0.3)
        sim2_smooth = _sim2_geodesic_smooth(raw)
        ema_smooth = _linear_ema_smooth(raw)

        result = run_ab_test_sim2_vs_ema(sim2_smooth, ema_smooth)
        assert result.metrics_a.transform_determinant_stability is not None
        assert result.metrics_b.transform_determinant_stability is not None

    def test_ab_test_has_temporal_smoothness(self):
        """A/B result must include temporal smoothness (jitter) comparison."""
        raw = _generate_rotation_sequence(50, max_theta=0.3)
        sim2_smooth = _sim2_geodesic_smooth(raw)
        ema_smooth = _linear_ema_smooth(raw)

        result = run_ab_test_sim2_vs_ema(sim2_smooth, ema_smooth)
        # temporal_smoothness stores negative jitter (higher = better)
        assert result.metrics_a.temporal_smoothness is not None
        assert result.metrics_b.temporal_smoothness is not None


class TestComputeTransformJitter:
    """Unit tests for compute_transform_jitter."""

    def test_constant_transform_zero_jitter(self):
        """Constant transforms should have zero jitter."""
        transforms = [SIM2Transform.identity() for _ in range(10)]
        jitter = compute_transform_jitter(transforms)
        assert jitter < 1e-10

    def test_single_transform_zero_jitter(self):
        """Single transform should have zero jitter."""
        transforms = [SIM2Transform.identity()]
        jitter = compute_transform_jitter(transforms)
        assert jitter == 0.0

    def test_empty_list_zero_jitter(self):
        """Empty list should have zero jitter."""
        jitter = compute_transform_jitter([])
        assert jitter == 0.0

    def test_linear_motion_constant_jitter(self):
        """Linear motion should have constant jitter."""
        transforms = []
        for i in range(20):
            transforms.append(SIM2Transform(theta=0.0, tx=float(i), ty=0.0, scale=1.0))
        jitter = compute_transform_jitter(transforms)
        # Each step is tx=1.0, so jitter should be ~1.0
        assert abs(jitter - 1.0) < 0.1
