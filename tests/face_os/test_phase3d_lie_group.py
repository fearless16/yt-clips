"""Phase 3D: Lie-Group Transforms Tests.

Tests for Lie-group geometric transforms:
- SE(2): Special Euclidean group (rotation + translation)
- SIM(2): Similarity group (rotation + translation + scale)
- Exponential/logarithmic maps
- Geodesic interpolation
- Group closure and properties

Target: 15 tests
"""

import numpy as np
import pytest

from face_os.lie_group import (
    SE2Transform,
    SIM2Transform,
    interpolate_se2,
    interpolate_sim2,
    se2_log,
    se2_exp,
    sim2_log,
    sim2_exp,
    geodesic_distance_se2,
    geodesic_distance_sim2,
)


class TestSE2Transform:
    """Test SE(2) transform (rotation + translation)."""

    def test_identity_transform(self):
        """Identity transform must be [0, 0, 0]."""
        T = SE2Transform.identity()
        assert T.theta == 0.0
        assert T.tx == 0.0
        assert T.ty == 0.0

    def test_to_matrix_shape(self):
        """Matrix must be 3x3."""
        T = SE2Transform(theta=0.5, tx=1.0, ty=2.0)
        M = T.to_matrix()
        assert M.shape == (3, 3)

    def test_to_matrix_determinant(self):
        """Matrix determinant must be 1 (rigid transform)."""
        T = SE2Transform(theta=0.5, tx=1.0, ty=2.0)
        M = T.to_matrix()
        assert abs(np.linalg.det(M) - 1.0) < 1e-10

    def test_to_matrix_orthogonal(self):
        """Rotation part must be orthogonal."""
        T = SE2Transform(theta=0.5, tx=1.0, ty=2.0)
        M = T.to_matrix()
        R = M[:2, :2]
        assert np.allclose(R @ R.T, np.eye(2), atol=1e-10)

    def test_from_matrix_roundtrip(self):
        """from_matrix(to_matrix(T)) must equal T."""
        T1 = SE2Transform(theta=0.5, tx=1.0, ty=2.0)
        M = T1.to_matrix()
        T2 = SE2Transform.from_matrix(M)
        assert abs(T1.theta - T2.theta) < 1e-10
        assert abs(T1.tx - T2.tx) < 1e-10
        assert abs(T1.ty - T2.ty) < 1e-10

    def test_exp_log_roundtrip(self):
        """exp(log(T)) must equal T."""
        T1 = SE2Transform(theta=0.5, tx=1.0, ty=2.0)
        v = T1.log()
        T2 = SE2Transform.exp(v)
        assert abs(T1.theta - T2.theta) < 1e-10
        assert abs(T1.tx - T2.tx) < 1e-10
        assert abs(T1.ty - T2.ty) < 1e-10

    def test_compose(self):
        """Composition must be group operation."""
        T1 = SE2Transform(theta=0.3, tx=1.0, ty=0.0)
        T2 = SE2Transform(theta=0.2, tx=0.0, ty=1.0)
        T3 = T1.compose(T2)
        
        # Check via matrices
        M1 = T1.to_matrix()
        M2 = T2.to_matrix()
        M3 = T3.to_matrix()
        np.testing.assert_allclose(M3, M1 @ M2, atol=1e-10)

    def test_inverse(self):
        """Inverse must satisfy T * T^-1 = I."""
        T = SE2Transform(theta=0.5, tx=1.0, ty=2.0)
        T_inv = T.inverse()
        I = T.compose(T_inv)
        assert abs(I.theta) < 1e-10
        assert abs(I.tx) < 1e-10
        assert abs(I.ty) < 1e-10

    def test_transform_point(self):
        """Transform must correctly transform points."""
        T = SE2Transform(theta=np.pi / 2, tx=1.0, ty=0.0)
        point = np.array([1.0, 0.0])
        transformed = T.transform_point(point)
        
        # Rotation by 90° then translate: (0, 1) + (1, 0) = (1, 1)
        expected = np.array([1.0, 1.0])
        np.testing.assert_allclose(transformed, expected, atol=1e-10)

    def test_no_skew_or_flip(self):
        """Transform must not introduce skew or flip."""
        T = SE2Transform(theta=0.5, tx=1.0, ty=2.0)
        M = T.to_matrix()
        
        # Check no skew: R^T R = I
        R = M[:2, :2]
        assert np.allclose(R.T @ R, np.eye(2), atol=1e-10)
        
        # Check no flip: det(R) = 1
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


class TestSIM2Transform:
    """Test SIM(2) transform (rotation + translation + scale)."""

    def test_identity_transform(self):
        """Identity transform must be [0, 0, 0, 1]."""
        T = SIM2Transform.identity()
        assert T.theta == 0.0
        assert T.tx == 0.0
        assert T.ty == 0.0
        assert T.scale == 1.0

    def test_to_matrix_shape(self):
        """Matrix must be 3x3."""
        T = SIM2Transform(theta=0.5, tx=1.0, ty=2.0, scale=1.5)
        M = T.to_matrix()
        assert M.shape == (3, 3)

    def test_to_matrix_determinant(self):
        """Matrix determinant must be scale^2."""
        T = SIM2Transform(theta=0.5, tx=1.0, ty=2.0, scale=1.5)
        M = T.to_matrix()
        expected_det = T.scale ** 2
        assert abs(np.linalg.det(M) - expected_det) < 1e-10

    def test_from_matrix_roundtrip(self):
        """from_matrix(to_matrix(T)) must equal T."""
        T1 = SIM2Transform(theta=0.5, tx=1.0, ty=2.0, scale=1.5)
        M = T1.to_matrix()
        T2 = SIM2Transform.from_matrix(M)
        assert abs(T1.theta - T2.theta) < 1e-10
        assert abs(T1.tx - T2.tx) < 1e-10
        assert abs(T1.ty - T2.ty) < 1e-10
        assert abs(T1.scale - T2.scale) < 1e-10

    def test_exp_log_roundtrip(self):
        """exp(log(T)) must equal T."""
        T1 = SIM2Transform(theta=0.5, tx=1.0, ty=2.0, scale=1.5)
        v = T1.log()
        T2 = SIM2Transform.exp(v)
        assert abs(T1.theta - T2.theta) < 1e-10
        assert abs(T1.tx - T2.tx) < 1e-10
        assert abs(T1.ty - T2.ty) < 1e-10
        assert abs(T1.scale - T2.scale) < 1e-10

    def test_compose(self):
        """Composition must be group operation."""
        T1 = SIM2Transform(theta=0.3, tx=1.0, ty=0.0, scale=1.5)
        T2 = SIM2Transform(theta=0.2, tx=0.0, ty=1.0, scale=0.8)
        T3 = T1.compose(T2)
        
        # Check via matrices
        M1 = T1.to_matrix()
        M2 = T2.to_matrix()
        M3 = T3.to_matrix()
        np.testing.assert_allclose(M3, M1 @ M2, atol=1e-10)


class TestInterpolation:
    """Test geodesic interpolation."""

    def test_interpolate_se2_identity(self):
        """Interpolation at t=0 and t=1 must give endpoints."""
        T1 = SE2Transform(theta=0.0, tx=0.0, ty=0.0)
        T2 = SE2Transform(theta=1.0, tx=1.0, ty=1.0)
        
        T_start = interpolate_se2(T1, T2, 0.0)
        T_end = interpolate_se2(T1, T2, 1.0)
        
        assert abs(T_start.theta - T1.theta) < 1e-10
        assert abs(T_end.theta - T2.theta) < 1e-10

    def test_interpolate_se2_midpoint(self):
        """Interpolation at t=0.5 must give midpoint."""
        T1 = SE2Transform(theta=0.0, tx=0.0, ty=0.0)
        T2 = SE2Transform(theta=1.0, tx=2.0, ty=2.0)
        
        T_mid = interpolate_se2(T1, T2, 0.5)
        
        assert abs(T_mid.theta - 0.5) < 1e-10
        assert abs(T_mid.tx - 1.0) < 1e-10
        assert abs(T_mid.ty - 1.0) < 1e-10

    def test_interpolate_sim2_midpoint(self):
        """Interpolation at t=0.5 must give midpoint (geometric for scale)."""
        T1 = SIM2Transform(theta=0.0, tx=0.0, ty=0.0, scale=1.0)
        T2 = SIM2Transform(theta=1.0, tx=2.0, ty=2.0, scale=2.0)
        
        T_mid = interpolate_sim2(T1, T2, 0.5)
        
        assert abs(T_mid.theta - 0.5) < 1e-10
        assert abs(T_mid.tx - 1.0) < 1e-10
        assert abs(T_mid.ty - 1.0) < 1e-10
        # Scale uses geometric mean: sqrt(1.0 * 2.0) = sqrt(2) ≈ 1.414
        expected_scale = np.sqrt(1.0 * 2.0)
        assert abs(T_mid.scale - expected_scale) < 1e-10

    def test_interpolate_smoothness(self):
        """Interpolation must be smooth (no discontinuities)."""
        T1 = SE2Transform(theta=0.0, tx=0.0, ty=0.0)
        T2 = SE2Transform(theta=1.0, tx=1.0, ty=1.0)
        
        # Check smoothness via finite differences
        thetas = []
        for t in np.linspace(0, 1, 100):
            T = interpolate_se2(T1, T2, t)
            thetas.append(T.theta)
        
        # Second derivative should be small (smooth)
        thetas = np.array(thetas)
        second_deriv = np.diff(thetas, 2)
        assert np.max(np.abs(second_deriv)) < 0.01

    def test_geodesic_distance_se2(self):
        """Geodesic distance must be symmetric."""
        T1 = SE2Transform(theta=0.3, tx=1.0, ty=0.0)
        T2 = SE2Transform(theta=0.7, tx=0.0, ty=1.0)
        
        d1 = geodesic_distance_se2(T1, T2)
        d2 = geodesic_distance_se2(T2, T1)
        
        assert abs(d1 - d2) < 1e-10

    def test_geodesic_distance_zero(self):
        """Geodesic distance to self must be zero."""
        T = SE2Transform(theta=0.5, tx=1.0, ty=2.0)
        d = geodesic_distance_se2(T, T)
        assert abs(d) < 1e-10

    def test_geodesic_distance_triangle(self):
        """Triangle inequality must hold."""
        T1 = SE2Transform(theta=0.0, tx=0.0, ty=0.0)
        T2 = SE2Transform(theta=0.5, tx=1.0, ty=0.0)
        T3 = SE2Transform(theta=1.0, tx=1.0, ty=1.0)
        
        d12 = geodesic_distance_se2(T1, T2)
        d23 = geodesic_distance_se2(T2, T3)
        d13 = geodesic_distance_se2(T1, T3)
        
        assert d13 <= d12 + d23 + 1e-10
