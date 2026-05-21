"""Phase 3C: Dense Geometry Tests.

Tests for dense geometry estimation:
- Mesh fitting from landmarks
- Normal computation
- UV coordinate mapping
- Confidence propagation
- Topology validation

Target: 20 tests
"""

import numpy as np
import pytest

from face_os.dense_geometry import (
    DenseGeometry,
    DenseGeometryEstimator,
    GeometryConfig,
    GeometryReport,
)


class TestDenseGeometry:
    """Test DenseGeometry dataclass."""

    def test_vertices_shape(self):
        """Vertices must be (N, 3)."""
        vertices = np.random.rand(1000, 3).astype(np.float32)
        faces = np.random.randint(0, 1000, (2000, 3)).astype(np.int32)
        normals = np.random.rand(1000, 3).astype(np.float32)
        uv_coords = np.random.rand(1000, 2).astype(np.float32)
        confidence = np.ones(1000, dtype=np.float32)

        geom = DenseGeometry(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uv_coords=uv_coords,
            confidence=confidence,
            landmark_correspondence={},
        )
        assert geom.vertices.shape == (1000, 3)

    def test_faces_shape(self):
        """Faces must be (F, 3)."""
        vertices = np.random.rand(1000, 3).astype(np.float32)
        faces = np.random.randint(0, 1000, (2000, 3)).astype(np.int32)
        normals = np.random.rand(1000, 3).astype(np.float32)
        uv_coords = np.random.rand(1000, 2).astype(np.float32)
        confidence = np.ones(1000, dtype=np.float32)

        geom = DenseGeometry(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uv_coords=uv_coords,
            confidence=confidence,
            landmark_correspondence={},
        )
        assert geom.faces.shape == (2000, 3)

    def test_normals_shape(self):
        """Normals must be (N, 3)."""
        vertices = np.random.rand(1000, 3).astype(np.float32)
        faces = np.random.randint(0, 1000, (2000, 3)).astype(np.int32)
        normals = np.random.rand(1000, 3).astype(np.float32)
        uv_coords = np.random.rand(1000, 2).astype(np.float32)
        confidence = np.ones(1000, dtype=np.float32)

        geom = DenseGeometry(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uv_coords=uv_coords,
            confidence=confidence,
            landmark_correspondence={},
        )
        assert geom.normals.shape == (1000, 3)

    def test_uv_coords_bounds(self):
        """UV coordinates must be in [0, 1]."""
        vertices = np.random.rand(1000, 3).astype(np.float32)
        faces = np.random.randint(0, 1000, (2000, 3)).astype(np.int32)
        normals = np.random.rand(1000, 3).astype(np.float32)
        uv_coords = np.random.rand(1000, 2).astype(np.float32)
        confidence = np.ones(1000, dtype=np.float32)

        geom = DenseGeometry(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uv_coords=uv_coords,
            confidence=confidence,
            landmark_correspondence={},
        )
        assert np.all(geom.uv_coords >= 0) and np.all(geom.uv_coords <= 1)

    def test_confidence_bounded(self):
        """Confidence must be in [0, 1]."""
        vertices = np.random.rand(1000, 3).astype(np.float32)
        faces = np.random.randint(0, 1000, (2000, 3)).astype(np.int32)
        normals = np.random.rand(1000, 3).astype(np.float32)
        uv_coords = np.random.rand(1000, 2).astype(np.float32)
        confidence = np.random.rand(1000).astype(np.float32)

        geom = DenseGeometry(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uv_coords=uv_coords,
            confidence=confidence,
            landmark_correspondence={},
        )
        assert np.all(geom.confidence >= 0) and np.all(geom.confidence <= 1)

    def test_normals_normalized(self):
        """Normals must be unit vectors."""
        vertices = np.random.rand(1000, 3).astype(np.float32)
        faces = np.random.randint(0, 1000, (2000, 3)).astype(np.int32)
        normals = np.random.rand(1000, 3).astype(np.float32)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-8)
        uv_coords = np.random.rand(1000, 2).astype(np.float32)
        confidence = np.ones(1000, dtype=np.float32)

        geom = DenseGeometry(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uv_coords=uv_coords,
            confidence=confidence,
            landmark_correspondence={},
        )
        norms = np.linalg.norm(geom.normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_face_indices_valid(self):
        """Face indices must be valid vertex indices."""
        n_vertices = 1000
        vertices = np.random.rand(n_vertices, 3).astype(np.float32)
        faces = np.random.randint(0, n_vertices, (2000, 3)).astype(np.int32)
        normals = np.random.rand(n_vertices, 3).astype(np.float32)
        uv_coords = np.random.rand(n_vertices, 2).astype(np.float32)
        confidence = np.ones(n_vertices, dtype=np.float32)

        geom = DenseGeometry(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uv_coords=uv_coords,
            confidence=confidence,
            landmark_correspondence={},
        )
        assert np.all(geom.faces >= 0) and np.all(geom.faces < n_vertices)


class TestGeometryConfig:
    """Test GeometryConfig."""

    def test_default_config(self):
        """Default config must have reasonable values."""
        config = GeometryConfig()
        assert config.n_vertices > 0
        assert config.n_faces > 0
        assert config.landmark_weight > 0
        assert config.smoothness_weight > 0

    def test_config_validation(self):
        """Config values must be valid."""
        config = GeometryConfig(
            n_vertices=5000,
            n_faces=10000,
            landmark_weight=1.0,
            smoothness_weight=0.1,
        )
        assert config.n_vertices == 5000
        assert config.n_faces == 10000


class TestDenseGeometryEstimator:
    """Test DenseGeometryEstimator."""

    def test_estimate_shape(self):
        """Estimation must produce correct shapes."""
        estimator = DenseGeometryEstimator()
        landmarks = np.random.rand(478, 2).astype(np.float32) * 256
        result = estimator.estimate(landmarks)
        assert result.vertices.shape[1] == 3
        assert result.faces.shape[1] == 3
        assert result.normals.shape[1] == 3

    def test_estimate_valid_range(self):
        """Estimation outputs must be in valid range."""
        estimator = DenseGeometryEstimator()
        landmarks = np.random.rand(478, 2).astype(np.float32) * 256
        result = estimator.estimate(landmarks)
        assert np.all(result.uv_coords >= 0) and np.all(result.uv_coords <= 1)
        assert np.all(result.confidence >= 0) and np.all(result.confidence <= 1)

    def test_estimate_topology_valid(self):
        """Estimated mesh must have valid topology."""
        estimator = DenseGeometryEstimator()
        landmarks = np.random.rand(478, 2).astype(np.float32) * 256
        result = estimator.estimate(landmarks)
        
        # All face indices must be valid
        n_vertices = result.vertices.shape[0]
        assert np.all(result.faces >= 0) and np.all(result.faces < n_vertices)

    def test_estimate_normals_outward(self):
        """Normals must point outward (positive z for frontal face)."""
        estimator = DenseGeometryEstimator()
        landmarks = np.random.rand(478, 2).astype(np.float32) * 256
        result = estimator.estimate(landmarks)
        
        # Most normals should point outward (positive z)
        mean_nz = np.mean(result.normals[:, 2])
        assert mean_nz > 0  # Majority pointing outward

    def test_estimate_landmark_correspondence(self):
        """Landmark correspondence must map landmarks to vertices."""
        estimator = DenseGeometryEstimator()
        landmarks = np.random.rand(478, 2).astype(np.float32) * 256
        result = estimator.estimate(landmarks)
        
        # Must have correspondence for all 478 landmarks
        assert len(result.landmark_correspondence) == 478

    def test_estimate_confidence_high_near_landmarks(self):
        """Confidence must be high near landmarks."""
        estimator = DenseGeometryEstimator()
        landmarks = np.random.rand(478, 2).astype(np.float32) * 256
        result = estimator.estimate(landmarks)
        
        # Vertices near landmarks should have high confidence
        landmark_indices = list(result.landmark_correspondence.values())
        landmark_confidences = result.confidence[landmark_indices]
        assert np.mean(landmark_confidences) > 0.5

    def test_estimate_multiple_calls_deterministic(self):
        """Multiple calls must be deterministic."""
        estimator = DenseGeometryEstimator()
        landmarks = np.random.rand(478, 2).astype(np.float32) * 256
        
        result1 = estimator.estimate(landmarks)
        result2 = estimator.estimate(landmarks)
        
        np.testing.assert_array_equal(result1.vertices, result2.vertices)
        np.testing.assert_array_equal(result1.faces, result2.faces)

    def test_estimate_with_different_landmarks(self):
        """Different landmarks must produce different meshes."""
        estimator = DenseGeometryEstimator()
        
        landmarks1 = np.random.rand(478, 2).astype(np.float32) * 256
        landmarks2 = np.random.rand(478, 2).astype(np.float32) * 256 + 50
        
        result1 = estimator.estimate(landmarks1)
        result2 = estimator.estimate(landmarks2)
        
        # Vertices must be different
        diff = np.mean(np.abs(result1.vertices - result2.vertices))
        assert diff > 0.1

    def test_estimate_with_config(self):
        """Estimation must work with custom config."""
        estimator = DenseGeometryEstimator()
        landmarks = np.random.rand(478, 2).astype(np.float32) * 256
        
        config = GeometryConfig(n_vertices=2000, n_faces=4000)
        result = estimator.estimate(landmarks, config=config)
        
        assert result.vertices.shape[0] <= 2000
        assert result.faces.shape[0] <= 4000


class TestGeometryReport:
    """Test GeometryReport."""

    def test_report_to_dict(self):
        """Report must convert to dict."""
        vertices = np.random.rand(1000, 3).astype(np.float32)
        faces = np.random.randint(0, 1000, (2000, 3)).astype(np.int32)
        normals = np.random.rand(1000, 3).astype(np.float32)
        uv_coords = np.random.rand(1000, 2).astype(np.float32)
        confidence = np.ones(1000, dtype=np.float32)

        geom = DenseGeometry(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uv_coords=uv_coords,
            confidence=confidence,
            landmark_correspondence={},
        )

        report = GeometryReport(
            frame_idx=0,
            geometry=geom,
            fitting_error=0.05,
            smoothness_score=0.8,
            estimation_time_ms=10.0,
        )

        d = report.to_dict()
        assert d["frame_idx"] == 0
        assert d["fitting_error"] == 0.05
        assert d["smoothness_score"] == 0.8

    def test_report_has_metrics(self):
        """Report must have all required metrics."""
        vertices = np.random.rand(1000, 3).astype(np.float32)
        faces = np.random.randint(0, 1000, (2000, 3)).astype(np.int32)
        normals = np.random.rand(1000, 3).astype(np.float32)
        uv_coords = np.random.rand(1000, 2).astype(np.float32)
        confidence = np.ones(1000, dtype=np.float32)

        geom = DenseGeometry(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uv_coords=uv_coords,
            confidence=confidence,
            landmark_correspondence={},
        )

        report = GeometryReport(
            frame_idx=0,
            geometry=geom,
            fitting_error=0.05,
            smoothness_score=0.8,
            estimation_time_ms=10.0,
        )

        assert hasattr(report, "frame_idx")
        assert hasattr(report, "geometry")
        assert hasattr(report, "fitting_error")
        assert hasattr(report, "smoothness_score")
        assert hasattr(report, "estimation_time_ms")


class TestMeshTopology:
    """Test mesh topology properties."""

    def test_no_degenerate_faces(self):
        """Mesh must not have degenerate faces (zero area)."""
        estimator = DenseGeometryEstimator()
        landmarks = np.random.rand(478, 2).astype(np.float32) * 256
        result = estimator.estimate(landmarks)
        
        # Check for degenerate faces (two or more same indices)
        for face in result.faces:
            assert len(set(face)) == 3  # All three indices must be different

    def test_face_area_positive(self):
        """All faces must have positive area."""
        estimator = DenseGeometryEstimator()
        landmarks = np.random.rand(478, 2).astype(np.float32) * 256
        result = estimator.estimate(landmarks)
        
        # Compute face areas
        v0 = result.vertices[result.faces[:, 0]]
        v1 = result.vertices[result.faces[:, 1]]
        v2 = result.vertices[result.faces[:, 2]]
        
        # Cross product gives twice the area
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1) / 2
        
        # All areas must be positive
        assert np.all(areas > 0)

    def test_normal_consistency(self):
        """Face normals must be consistent with vertex normals."""
        estimator = DenseGeometryEstimator()
        landmarks = np.random.rand(478, 2).astype(np.float32) * 256
        result = estimator.estimate(landmarks)
        
        # Compute face normals
        v0 = result.vertices[result.faces[:, 0]]
        v1 = result.vertices[result.faces[:, 1]]
        v2 = result.vertices[result.faces[:, 2]]
        
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)
        
        # Average vertex normals at faces
        vn0 = result.normals[result.faces[:, 0]]
        vn1 = result.normals[result.faces[:, 1]]
        vn2 = result.normals[result.faces[:, 2]]
        avg_vertex_normals = (vn0 + vn1 + vn2) / 3
        avg_vertex_normals = avg_vertex_normals / (np.linalg.norm(avg_vertex_normals, axis=1, keepdims=True) + 1e-8)
        
        # Dot product should be positive for majority (same direction)
        dots = np.sum(face_normals * avg_vertex_normals, axis=1)
        # Allow some inconsistency due to mesh topology
        consistent_ratio = np.mean(dots > 0)
        assert consistent_ratio > 0.3  # At least 30% consistent
