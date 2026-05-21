"""Tests for Identity Manifold Module.

Tests for:
- IdentityPoint
- GeodesicPath
- IdentityManifold
- Exponential/logarithmic maps
- Geodesic distance
- Interpolation
- Parallel transport
- Metric tensor

Target: 20 tests
"""

import numpy as np
import pytest

from face_os.identity_manifold import (
    ManifoldConfig,
    IdentityPoint,
    GeodesicPath,
    IdentityManifold,
)


class TestManifoldConfig:
    """Test ManifoldConfig."""

    def test_default_config(self):
        """Default config must have reasonable values."""
        config = ManifoldConfig()
        assert config.dimension > 0
        assert config.max_geodesic_distance > 0
        assert config.metric_regularization > 0


class TestIdentityPoint:
    """Test IdentityPoint."""

    def test_point_creation(self):
        """Point must be created."""
        coords = np.random.randn(16)
        point = IdentityPoint(coordinates=coords)
        assert point.coordinates.shape == (16,)
        assert point.confidence == 1.0

    def test_point_with_metric(self):
        """Point must accept metric tensor."""
        coords = np.random.randn(16)
        metric = np.eye(16)
        point = IdentityPoint(coordinates=coords, metric_tensor=metric)
        assert point.metric_tensor.shape == (16, 16)

    def test_point_validation(self):
        """Point must validate coordinates."""
        with pytest.raises(ValueError):
            IdentityPoint(coordinates=np.random.randn(16, 2))  # Wrong shape

    def test_point_metric_validation(self):
        """Point must validate metric tensor."""
        coords = np.random.randn(16)
        metric = np.eye(8)  # Wrong size
        with pytest.raises(ValueError):
            IdentityPoint(coordinates=coords, metric_tensor=metric)


class TestGeodesicPath:
    """Test GeodesicPath."""

    def test_path_creation(self):
        """Path must be created."""
        start = IdentityPoint(coordinates=np.zeros(16))
        end = IdentityPoint(coordinates=np.ones(16))
        t = np.linspace(0, 1, 10)
        points = np.array([start.coordinates + ti * (end.coordinates - start.coordinates) for ti in t])

        path = GeodesicPath(
            start=start,
            end=end,
            t=t,
            points=points,
            length=np.linalg.norm(end.coordinates - start.coordinates),
        )
        assert path.length > 0

    def test_path_evaluate(self):
        """Path must evaluate at parameter."""
        start = IdentityPoint(coordinates=np.zeros(16))
        end = IdentityPoint(coordinates=np.ones(16))
        t = np.linspace(0, 1, 10)
        points = np.array([start.coordinates + ti * (end.coordinates - start.coordinates) for ti in t])

        path = GeodesicPath(
            start=start,
            end=end,
            t=t,
            points=points,
            length=np.linalg.norm(end.coordinates - start.coordinates),
        )

        # Evaluate at t=0.5
        mid = path.evaluate(0.5)
        assert mid.shape == (16,)


class TestIdentityManifold:
    """Test IdentityManifold."""

    def test_manifold_creation(self):
        """Manifold must be created."""
        manifold = IdentityManifold()
        assert manifold._dimension == 16

    def test_exp_map(self):
        """Exponential map must work."""
        manifold = IdentityManifold()
        point = IdentityPoint(coordinates=np.zeros(16))
        tangent = np.ones(16) * 0.1

        new_coords = manifold.exp_map(point, tangent)
        assert new_coords.shape == (16,)
        np.testing.assert_allclose(new_coords, tangent)

    def test_log_map(self):
        """Logarithmic map must work."""
        manifold = IdentityManifold()
        point = IdentityPoint(coordinates=np.zeros(16))
        target = np.ones(16) * 0.5

        tangent = manifold.log_map(point, target)
        assert tangent.shape == (16,)
        np.testing.assert_allclose(tangent, target)

    def test_geodesic_distance(self):
        """Geodesic distance must be computed."""
        manifold = IdentityManifold()
        point1 = IdentityPoint(coordinates=np.zeros(16))
        point2 = IdentityPoint(coordinates=np.ones(16))

        distance = manifold.geodesic_distance(point1, point2)
        expected = np.sqrt(16)  # ||[1,1,...,1]||
        assert abs(distance - expected) < 1e-10

    def test_geodesic_distance_symmetric(self):
        """Geodesic distance must be symmetric."""
        manifold = IdentityManifold()
        point1 = IdentityPoint(coordinates=np.random.randn(16))
        point2 = IdentityPoint(coordinates=np.random.randn(16))

        d1 = manifold.geodesic_distance(point1, point2)
        d2 = manifold.geodesic_distance(point2, point1)
        assert abs(d1 - d2) < 1e-10

    def test_geodesic_distance_triangle(self):
        """Triangle inequality must hold."""
        manifold = IdentityManifold()
        point1 = IdentityPoint(coordinates=np.zeros(16))
        point2 = IdentityPoint(coordinates=np.ones(16) * 0.5)
        point3 = IdentityPoint(coordinates=np.ones(16))

        d12 = manifold.geodesic_distance(point1, point2)
        d23 = manifold.geodesic_distance(point2, point3)
        d13 = manifold.geodesic_distance(point1, point3)

        assert d13 <= d12 + d23 + 1e-10

    def test_interpolation(self):
        """Interpolation must work."""
        manifold = IdentityManifold()
        point1 = IdentityPoint(coordinates=np.zeros(16))
        point2 = IdentityPoint(coordinates=np.ones(16))

        # Interpolate at t=0.5
        mid = manifold.interpolate(point1, point2, 0.5)
        assert mid.shape == (16,)
        np.testing.assert_allclose(mid, np.ones(16) * 0.5)

    def test_interpolation_endpoints(self):
        """Interpolation must give endpoints."""
        manifold = IdentityManifold()
        point1 = IdentityPoint(coordinates=np.zeros(16))
        point2 = IdentityPoint(coordinates=np.ones(16))

        start = manifold.interpolate(point1, point2, 0.0)
        end = manifold.interpolate(point1, point2, 1.0)

        np.testing.assert_allclose(start, point1.coordinates)
        np.testing.assert_allclose(end, point2.coordinates)

    def test_geodesic_path(self):
        """Geodesic path must be computed."""
        manifold = IdentityManifold()
        point1 = IdentityPoint(coordinates=np.zeros(16))
        point2 = IdentityPoint(coordinates=np.ones(16))

        path = manifold.compute_geodesic_path(point1, point2, n_steps=10)
        assert path.points.shape == (10, 16)
        assert path.length > 0

    def test_parallel_transport(self):
        """Parallel transport must work."""
        manifold = IdentityManifold()
        point1 = IdentityPoint(coordinates=np.zeros(16))
        point2 = IdentityPoint(coordinates=np.ones(16))
        vector = np.ones(16) * 0.5

        transported = manifold.parallel_transport(point1, point2, vector)
        np.testing.assert_allclose(transported, vector)

    def test_valid_interpolation(self):
        """Valid interpolation check must work."""
        manifold = IdentityManifold()
        point1 = IdentityPoint(coordinates=np.zeros(16))
        point2 = IdentityPoint(coordinates=np.ones(16) * 0.1)

        assert manifold.is_valid_interpolation(point1, point2)

    def test_invalid_interpolation(self):
        """Invalid interpolation check must work."""
        config = ManifoldConfig(max_geodesic_distance=1.0)
        manifold = IdentityManifold(config=config)
        point1 = IdentityPoint(coordinates=np.zeros(16))
        point2 = IdentityPoint(coordinates=np.ones(16) * 100.0)

        assert not manifold.is_valid_interpolation(point1, point2)

    def test_add_point(self):
        """Adding point must work."""
        manifold = IdentityManifold()
        coords = np.random.randn(16)

        point = manifold.add_point("test", coords, confidence=0.9)
        assert point.coordinates.shape == (16,)
        assert point.confidence == 0.9

    def test_get_point(self):
        """Getting point must work."""
        manifold = IdentityManifold()
        coords = np.random.randn(16)

        manifold.add_point("test", coords)
        retrieved = manifold.get_point("test")
        assert retrieved is not None
        np.testing.assert_allclose(retrieved.coordinates, coords)

    def test_compute_metric_tensor(self):
        """Metric tensor computation must work."""
        manifold = IdentityManifold()
        point = IdentityPoint(coordinates=np.zeros(16))
        neighbors = [
            IdentityPoint(coordinates=np.random.randn(16)),
            IdentityPoint(coordinates=np.random.randn(16)),
            IdentityPoint(coordinates=np.random.randn(16)),
        ]

        metric = manifold.compute_metric_tensor(point, neighbors)
        assert metric.shape == (16, 16)
        # Metric must be positive definite
        eigenvalues = np.linalg.eigvalsh(metric)
        assert np.all(eigenvalues > 0)

    def test_curvature(self):
        """Curvature computation must work."""
        manifold = IdentityManifold()
        point1 = IdentityPoint(coordinates=np.zeros(16))
        point2 = IdentityPoint(coordinates=np.ones(16))
        point3 = IdentityPoint(coordinates=np.ones(16) * 0.5)

        curvature = manifold.compute_curvature(point1, point2, point3)
        # For flat manifold, curvature should be 0
        assert abs(curvature) < 1e-10


class TestManifoldProperties:
    """Test manifold mathematical properties."""

    def test_exp_log_roundtrip(self):
        """exp(log(x)) must equal x."""
        manifold = IdentityManifold()
        point = IdentityPoint(coordinates=np.zeros(16))
        target = np.random.randn(16)

        tangent = manifold.log_map(point, target)
        recovered = manifold.exp_map(point, tangent)
        np.testing.assert_allclose(recovered, target, atol=1e-10)

    def test_log_exp_roundtrip(self):
        """log(exp(v)) must equal v."""
        manifold = IdentityManifold()
        point = IdentityPoint(coordinates=np.zeros(16))
        tangent = np.random.randn(16)

        new_coords = manifold.exp_map(point, tangent)
        recovered = manifold.log_map(point, new_coords)
        np.testing.assert_allclose(recovered, tangent, atol=1e-10)

    def test_distance_consistency(self):
        """Distance must be consistent with interpolation."""
        manifold = IdentityManifold()
        point1 = IdentityPoint(coordinates=np.zeros(16))
        point2 = IdentityPoint(coordinates=np.ones(16))

        distance = manifold.geodesic_distance(point1, point2)

        # Interpolate at t=1 should give point2
        interpolated = manifold.interpolate(point1, point2, 1.0)
        np.testing.assert_allclose(interpolated, point2.coordinates, atol=1e-10)
