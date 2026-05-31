"""Layer 1: Lie Group math foundation tests.

Validates SE(2) and SIM(2) group operations that underpin the
temporal state estimation system (D-06).

Mathematical invariants tested:
  - Group axioms: closure, associativity, identity, inverse
  - Exponential map / logarithmic map roundtrip
  - Interpolation endpoint consistency
  - Geodesic distance metric properties (symmetry, triangle inequality)
"""
import numpy as np
import pytest

from face_os.lie_group import (
    SE2Transform,
    SIM2Transform,
    interpolate_se2,
    interpolate_sim2,
    geodesic_distance_se2,
    geodesic_distance_sim2,
)


# ═══════════════════════════════════════════════════════════════════
# SE(2) Group Axioms
# ═══════════════════════════════════════════════════════════════════

class TestSE2GroupAxioms:
    """SE(2) must satisfy group axioms: identity, inverse, associativity."""

    def test_identity_roundtrip(self):
        """SE2.identity() → to_matrix() → from_matrix() = identity."""
        I = SE2Transform.identity()
        M = I.to_matrix()
        recovered = SE2Transform.from_matrix(M)
        np.testing.assert_allclose(recovered.to_matrix(), np.eye(3), atol=1e-12)

    def test_inverse_compose_identity(self):
        """T ∘ T⁻¹ = I for arbitrary transform."""
        T = SE2Transform(tx=3.5, ty=-1.2, theta=0.7)
        T_inv = T.inverse()
        composed = T.compose(T_inv)
        np.testing.assert_allclose(composed.to_matrix(), np.eye(3), atol=1e-10)

    def test_compose_associative(self):
        """(A ∘ B) ∘ C = A ∘ (B ∘ C)."""
        A = SE2Transform(tx=1.0, ty=2.0, theta=0.3)
        B = SE2Transform(tx=-0.5, ty=1.5, theta=-0.2)
        C = SE2Transform(tx=0.8, ty=-0.3, theta=0.5)

        left = A.compose(B).compose(C)
        right = A.compose(B.compose(C))
        np.testing.assert_allclose(left.to_matrix(), right.to_matrix(), atol=1e-10)

    def test_transform_point_identity(self):
        """Identity transform leaves point unchanged."""
        I = SE2Transform.identity()
        p = np.array([3.0, 7.0])
        result = I.transform_point(p)
        np.testing.assert_allclose(result, p, atol=1e-12)

    def test_transform_point_translation(self):
        """Pure translation moves point by (tx, ty)."""
        T = SE2Transform(tx=5.0, ty=-3.0, theta=0.0)
        p = np.array([1.0, 2.0])
        result = T.transform_point(p)
        np.testing.assert_allclose(result, [6.0, -1.0], atol=1e-12)


# ═══════════════════════════════════════════════════════════════════
# SE(2) Log / Exp Map
# ═══════════════════════════════════════════════════════════════════

class TestSE2LogExp:
    """exp(log(T)) must equal T — the fundamental Lie group property."""

    @pytest.mark.parametrize("tx,ty,theta", [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.5),
        (3.5, -1.2, 0.7),
        (-2.0, 4.0, -1.0),
        (0.1, 0.1, np.pi * 0.9),
    ])
    def test_log_exp_roundtrip(self, tx, ty, theta):
        """exp(log(T)) = T for various transforms."""
        T = SE2Transform(tx=tx, ty=ty, theta=theta)
        v = T.log()
        T_recovered = SE2Transform.exp(v)
        np.testing.assert_allclose(
            T_recovered.to_matrix(), T.to_matrix(), atol=1e-8,
            err_msg=f"Roundtrip failed for tx={tx}, ty={ty}, theta={theta}"
        )


# ═══════════════════════════════════════════════════════════════════
# SIM(2) Group Tests
# ═══════════════════════════════════════════════════════════════════

class TestSIM2GroupAxioms:
    """SIM(2) — similarity transforms (rotation + translation + scale)."""

    def test_identity_roundtrip(self):
        """SIM2.identity() → to_matrix() → from_matrix() = identity."""
        I = SIM2Transform.identity()
        M = I.to_matrix()
        recovered = SIM2Transform.from_matrix(M)
        np.testing.assert_allclose(recovered.to_matrix(), np.eye(3), atol=1e-12)

    def test_inverse_compose_identity(self):
        """T ∘ T⁻¹ = I."""
        T = SIM2Transform(tx=2.0, ty=-1.0, theta=0.4, scale=1.5)
        T_inv = T.inverse()
        composed = T.compose(T_inv)
        np.testing.assert_allclose(composed.to_matrix(), np.eye(3), atol=1e-8)

    def test_compose_associative(self):
        """(A ∘ B) ∘ C = A ∘ (B ∘ C)."""
        A = SIM2Transform(tx=1.0, ty=0.5, theta=0.2, scale=1.2)
        B = SIM2Transform(tx=-0.3, ty=1.0, theta=-0.1, scale=0.9)
        C = SIM2Transform(tx=0.5, ty=-0.5, theta=0.3, scale=1.1)

        left = A.compose(B).compose(C)
        right = A.compose(B.compose(C))
        np.testing.assert_allclose(left.to_matrix(), right.to_matrix(), atol=1e-8)

    def test_det_positive(self):
        """det(T.to_matrix()) > 0 for all valid transforms (orientation-preserving)."""
        for _ in range(20):
            T = SIM2Transform(
                tx=np.random.uniform(-10, 10),
                ty=np.random.uniform(-10, 10),
                theta=np.random.uniform(-np.pi, np.pi),
                scale=np.random.uniform(0.1, 5.0),
            )
            det = np.linalg.det(T.to_matrix())
            assert det > 0, f"Negative determinant: {det} for T={T}"

    def test_scale_preserves_orientation(self):
        """Scale changes don't flip handedness (det remains positive)."""
        for s in [0.01, 0.5, 1.0, 2.0, 10.0]:
            T = SIM2Transform(tx=1.0, ty=1.0, theta=0.5, scale=s)
            det = np.linalg.det(T.to_matrix())
            assert det > 0, f"Scale={s} produced negative det={det}"


class TestSIM2LogExp:
    """SIM(2) exponential map roundtrip."""

    @pytest.mark.parametrize("tx,ty,theta,scale", [
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.5, 1.0),
        (0.0, 0.0, 0.0, 2.0),
        (2.0, -1.0, 0.4, 1.5),
        (-1.0, 3.0, -0.6, 0.7),
    ])
    def test_log_exp_roundtrip(self, tx, ty, theta, scale):
        """exp(log(T)) = T for SIM(2) transforms."""
        T = SIM2Transform(tx=tx, ty=ty, theta=theta, scale=scale)
        v = T.log()
        T_recovered = SIM2Transform.exp(v)
        np.testing.assert_allclose(
            T_recovered.to_matrix(), T.to_matrix(), atol=1e-6,
            err_msg=f"SIM2 roundtrip failed for tx={tx}, ty={ty}, θ={theta}, s={scale}"
        )


# ═══════════════════════════════════════════════════════════════════
# Interpolation
# ═══════════════════════════════════════════════════════════════════

class TestInterpolation:
    """Interpolation between transforms must hit endpoints exactly."""

    def test_se2_interpolation_endpoints(self):
        """interp(T1, T2, 0) = T1 and interp(T1, T2, 1) = T2."""
        T1 = SE2Transform(tx=0.0, ty=0.0, theta=0.0)
        T2 = SE2Transform(tx=4.0, ty=2.0, theta=1.0)
        at_0 = interpolate_se2(T1, T2, 0.0)
        at_1 = interpolate_se2(T1, T2, 1.0)
        np.testing.assert_allclose(at_0.to_matrix(), T1.to_matrix(), atol=1e-10)
        np.testing.assert_allclose(at_1.to_matrix(), T2.to_matrix(), atol=1e-10)

    def test_sim2_interpolation_endpoints(self):
        """interp(T1, T2, 0) = T1 and interp(T1, T2, 1) = T2."""
        T1 = SIM2Transform(tx=0.0, ty=0.0, theta=0.0, scale=1.0)
        T2 = SIM2Transform(tx=3.0, ty=1.0, theta=0.5, scale=2.0)
        at_0 = interpolate_sim2(T1, T2, 0.0)
        at_1 = interpolate_sim2(T1, T2, 1.0)
        np.testing.assert_allclose(at_0.to_matrix(), T1.to_matrix(), atol=1e-8)
        np.testing.assert_allclose(at_1.to_matrix(), T2.to_matrix(), atol=1e-8)

    def test_se2_midpoint_sane(self):
        """Midpoint translation should be between T1 and T2 translations."""
        T1 = SE2Transform(tx=0.0, ty=0.0, theta=0.0)
        T2 = SE2Transform(tx=10.0, ty=6.0, theta=0.0)
        mid = interpolate_se2(T1, T2, 0.5)
        assert 3.0 < mid.tx < 7.0, f"Midpoint tx={mid.tx} not between 0 and 10"
        assert 1.0 < mid.ty < 5.0, f"Midpoint ty={mid.ty} not between 0 and 6"


# ═══════════════════════════════════════════════════════════════════
# Geodesic Distance
# ═══════════════════════════════════════════════════════════════════

class TestGeodesicDistance:
    """Geodesic distance must be a proper metric (symmetric, triangle inequality)."""

    def test_se2_distance_symmetric(self):
        """d(A, B) = d(B, A)."""
        A = SE2Transform(tx=1.0, ty=2.0, theta=0.3)
        B = SE2Transform(tx=-1.0, ty=0.5, theta=-0.5)
        d_ab = geodesic_distance_se2(A, B)
        d_ba = geodesic_distance_se2(B, A)
        assert abs(d_ab - d_ba) < 1e-10, f"Asymmetric: d(A,B)={d_ab}, d(B,A)={d_ba}"

    def test_se2_distance_zero_for_same(self):
        """d(A, A) = 0."""
        A = SE2Transform(tx=1.0, ty=2.0, theta=0.3)
        d = geodesic_distance_se2(A, A)
        assert abs(d) < 1e-10, f"Self-distance not zero: {d}"

    def test_se2_triangle_inequality(self):
        """d(A, C) ≤ d(A, B) + d(B, C)."""
        A = SE2Transform(tx=0.0, ty=0.0, theta=0.0)
        B = SE2Transform(tx=1.0, ty=1.0, theta=0.2)
        C = SE2Transform(tx=3.0, ty=2.0, theta=0.5)
        d_ac = geodesic_distance_se2(A, C)
        d_ab = geodesic_distance_se2(A, B)
        d_bc = geodesic_distance_se2(B, C)
        assert d_ac <= d_ab + d_bc + 1e-10, (
            f"Triangle inequality violated: d(A,C)={d_ac:.4f} > "
            f"d(A,B)+d(B,C)={d_ab + d_bc:.4f}"
        )

    def test_sim2_distance_symmetric(self):
        """SIM(2) distance is symmetric."""
        A = SIM2Transform(tx=1.0, ty=0.5, theta=0.2, scale=1.2)
        B = SIM2Transform(tx=-0.5, ty=2.0, theta=-0.3, scale=0.8)
        d_ab = geodesic_distance_sim2(A, B)
        d_ba = geodesic_distance_sim2(B, A)
        assert abs(d_ab - d_ba) < 1e-10, f"Asymmetric: d(A,B)={d_ab}, d(B,A)={d_ba}"
