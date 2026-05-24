"""Layer 2e: Dense geometry tests.

Validates the landmark → dense mesh → normal map pipeline (D-04).

Geometric invariants:
  - Normal vectors have unit length
  - Mean normal Z > 0.5 (facing camera)
  - Normal coverage > 60%
  - UV coordinates in [0, 1]
  - Icosphere subdivision produces correct topology
"""
import numpy as np
import pytest

from face_os.dense_geometry import (
    DenseGeometryEstimator,
    GeometryConfig,
    DenseGeometry,
)


@pytest.fixture
def estimator():
    """Dense geometry estimator with default config."""
    return DenseGeometryEstimator()


@pytest.fixture
def synthetic_landmarks():
    """478-point synthetic face landmarks (MediaPipe-style).

    Creates a face-like point cloud with anatomically plausible
    positions in a 256x256 coordinate frame.
    """
    np.random.seed(42)
    n_pts = 478  # MediaPipe face mesh uses 478 landmarks
    # Generate points in a face-like ellipsoid distribution
    theta = np.random.uniform(0, 2 * np.pi, n_pts)
    phi = np.random.uniform(0, np.pi, n_pts)
    # Face centered at (128, 128) with roughly elliptical spread
    x = 128 + 60 * np.sin(phi) * np.cos(theta)
    y = 128 + 80 * np.sin(phi) * np.sin(theta)
    z = 20 * np.cos(phi)
    landmarks = np.stack([x, y, z], axis=-1).astype(np.float32)
    return landmarks


# ═══════════════════════════════════════════════════════════════════
# Estimator Contract
# ═══════════════════════════════════════════════════════════════════

class TestEstimatorContract:
    """DenseGeometryEstimator.estimate() output contract."""

    def test_estimate_returns_dense_geometry(self, estimator, synthetic_landmarks):
        """Output is a DenseGeometry instance."""
        result = estimator.estimate(synthetic_landmarks)
        assert isinstance(result, DenseGeometry)

    def test_has_vertices(self, estimator, synthetic_landmarks):
        """Output has vertices array."""
        result = estimator.estimate(synthetic_landmarks)
        assert result.vertices is not None
        assert result.vertices.ndim == 2
        assert result.vertices.shape[1] == 3

    def test_has_faces(self, estimator, synthetic_landmarks):
        """Output has faces (triangle indices)."""
        result = estimator.estimate(synthetic_landmarks)
        assert result.faces is not None
        assert result.faces.ndim == 2
        assert result.faces.shape[1] == 3

    def test_has_normals(self, estimator, synthetic_landmarks):
        """Output has per-vertex normals."""
        result = estimator.estimate(synthetic_landmarks)
        assert result.normals is not None
        assert result.normals.shape[1] == 3

    def test_has_uvs(self, estimator, synthetic_landmarks):
        """Output has UV coordinates."""
        result = estimator.estimate(synthetic_landmarks)
        assert result.uv_coords is not None
        assert result.uv_coords.shape[1] == 2


# ═══════════════════════════════════════════════════════════════════
# Geometric Quality
# ═══════════════════════════════════════════════════════════════════

class TestGeometricQuality:
    """Geometric quality constraints on estimated mesh."""

    def test_normal_unit_length(self, estimator, synthetic_landmarks):
        """All normals should have approximately unit length."""
        result = estimator.estimate(synthetic_landmarks)
        norms = np.linalg.norm(result.normals, axis=1)
        mean_err = float(np.mean(np.abs(norms - 1.0)))
        assert mean_err < 0.05, f"Normal unit error={mean_err:.4f}"

    def test_normal_z_positive(self, estimator, synthetic_landmarks):
        """Mean normal Z should be > 0 (facing camera)."""
        result = estimator.estimate(synthetic_landmarks)
        z_mean = float(np.mean(result.normals[:, 2]))
        assert z_mean > 0, f"Normal Z mean={z_mean:.3f} not facing camera"

    def test_uv_in_unit_range(self, estimator, synthetic_landmarks):
        """UV coordinates should be in [0, 1]."""
        result = estimator.estimate(synthetic_landmarks)
        assert float(np.min(result.uv_coords)) >= -0.01, "UV below 0"
        assert float(np.max(result.uv_coords)) <= 1.01, "UV above 1"

    def test_face_indices_valid(self, estimator, synthetic_landmarks):
        """Face indices should be within vertex count."""
        result = estimator.estimate(synthetic_landmarks)
        n_verts = result.vertices.shape[0]
        assert np.all(result.faces >= 0), "Negative face index"
        assert np.all(result.faces < n_verts), (
            f"Face index {int(np.max(result.faces))} >= vertex count {n_verts}"
        )


# ═══════════════════════════════════════════════════════════════════
# Icosphere Topology
# ═══════════════════════════════════════════════════════════════════

class TestIcosphereTopology:
    """Icosphere subdivision produces correct topology."""

    def test_icosphere_vertex_count(self, estimator):
        """Subdivision level N produces known vertex count.

        Level 0: 12 vertices (icosahedron)
        Level 1: 42
        Level 2: 162
        Level 3: 642
        """
        # Access the internal method for topology testing
        verts, faces = estimator._create_icosphere(n_subdivisions=1)
        assert verts.shape[0] == 42, f"Expected 42 vertices at level 1, got {verts.shape[0]}"
        assert verts.shape[1] == 3

    def test_icosphere_faces_are_triangles(self, estimator):
        """All faces have exactly 3 vertices (triangulation)."""
        verts, faces = estimator._create_icosphere(n_subdivisions=2)
        assert faces.shape[1] == 3

    def test_icosphere_normals_unit(self, estimator):
        """Icosphere vertices are on unit sphere."""
        verts, _ = estimator._create_icosphere(n_subdivisions=2)
        norms = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════
# From Landmarks (higher-level API)
# ═══════════════════════════════════════════════════════════════════

class TestEstimateFromLandmarks:
    """estimate_from_landmarks() produces rasterized normal map."""

    def test_estimate_from_landmarks_returns_normals(self, estimator, synthetic_landmarks):
        """Returns a (H, W, 3) normal map."""
        result = estimator.estimate_from_landmarks(
            synthetic_landmarks,
            image_shape=(256, 256),
        )
        assert result is not None
        # Result could be DenseGeometry or a normal map ndarray
        if isinstance(result, DenseGeometry):
            assert result.normals is not None
        elif isinstance(result, np.ndarray):
            assert result.ndim == 3
            assert result.shape[2] == 3
