"""Tests for mesh-derived surface normals (Rule 4).

Verifies that geometry normals from the MediaPipe 478-point mesh topology
replace shading-gradient normals, breaking the circular dependency.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from face_os.landmarks import (
    _get_mesh_triangles,
    compute_vertex_normals,
    mesh_normal_map,
)
from face_os.intrinsic_decomposition import IntrinsicDecomposer


class TestMeshTopology:
    """Verify the MediaPipe 478-point mesh topology is correct."""

    def test_has_correct_triangle_count(self):
        """Face mesh must have exactly 852 triangles from 2556 edges."""
        triangles = _get_mesh_triangles()
        assert triangles.shape == (852, 3), f"Expected (852, 3), got {triangles.shape}"
        assert triangles.dtype == np.int32

    def test_all_indices_in_range(self):
        """All triangle vertex indices must be valid (0-477)."""
        triangles = _get_mesh_triangles()
        assert triangles.min() >= 0, f"Min index {triangles.min()} < 0"
        assert triangles.max() <= 477, f"Max index {triangles.max()} > 477"

    def test_triangle_not_degenerate(self):
        """No triangle should have duplicate vertex indices."""
        triangles = _get_mesh_triangles()
        for i, (a, b, c) in enumerate(triangles):
            assert len({a, b, c}) == 3, f"Triangle {i} is degenerate: ({a},{b},{c})"

    def test_topology_is_cached(self):
        """Repeated calls must return the same array (cached)."""
        t1 = _get_mesh_triangles()
        t2 = _get_mesh_triangles()
        assert t1 is t2, "Topology was not cached"


class TestVertexNormals:
    """Verify per-vertex normal computation from mesh."""

    def test_returns_correct_shape(self):
        """compute_vertex_normals must return (478, 3)."""
        mesh = np.random.randn(478, 3).astype(np.float32)
        normals = compute_vertex_normals(mesh)
        assert normals.shape == (478, 3), f"Expected (478, 3), got {normals.shape}"

    def test_normals_are_unit_vectors(self):
        """Each vertex normal must have unit length (skip unconnected vertices)."""
        mesh = np.random.randn(478, 3).astype(np.float32)
        normals = compute_vertex_normals(mesh)
        lengths = np.linalg.norm(normals, axis=1)
        # Vertices 468-477 are not part of the face mesh triangulation
        # and will have zero normals — skip them
        active = lengths > 0.1
        assert active.sum() > 460, f"Only {active.sum()}/478 vertices have normals"
        assert_allclose(lengths[active], 1.0, atol=1e-5, err_msg="Normals not unit length")

    def test_frontal_face_normals_point_forward(self):
        """For a flat frontal mesh, normals should be perpendicular to the surface."""
        mesh = np.zeros((478, 3), dtype=np.float32)
        # Create a flat front-facing mesh: all at same z, spread x,y
        mesh[:, 0] = np.arange(478, dtype=np.float32) * 0.5
        mesh[:, 1] = (np.arange(478, dtype=np.float32) % 20) * 5.0
        mesh[:, 2] = 100.0  # Flat plane at constant depth
        normals = compute_vertex_normals(mesh)
        # Skip unconnected vertices
        lengths = np.linalg.norm(normals, axis=1)
        active = lengths > 0.1
        # Normals should be dominated by z-component (perpendicular to image plane)
        z_abs = np.abs(normals[active, 2])
        mean_z = np.mean(z_abs)
        assert mean_z > 0.8, f"Flat plane normals should be perpendicular to plane, mean |z|={mean_z:.3f}"

    def test_deterministic_output(self):
        """Same mesh must produce same normals every time."""
        mesh = np.random.randn(478, 3).astype(np.float32)
        n1 = compute_vertex_normals(mesh)
        n2 = compute_vertex_normals(mesh)
        assert_allclose(n1, n2, atol=1e-6, err_msg="Mesh normals not deterministic")

    def test_mesh_tilt_rotates_normals(self):
        """Rotating the mesh should rotate normals correspondingly."""
        mesh = np.zeros((478, 3), dtype=np.float32)
        mesh[:, 0] = np.arange(478, dtype=np.float32)
        mesh[:, 1] = 100.0
        # Base normals
        n_base = compute_vertex_normals(mesh)
        # Rotate mesh 90 degrees around z-axis
        mesh_rot = mesh.copy()
        theta = np.pi / 4
        c, s = np.cos(theta), np.sin(theta)
        mesh_rot[:, 0] = c * mesh[:, 0] - s * mesh[:, 1]
        mesh_rot[:, 1] = s * mesh[:, 0] + c * mesh[:, 1]
        n_rot = compute_vertex_normals(mesh_rot)
        # Normals should be different
        assert not np.allclose(n_base, n_rot, atol=1e-3), "Rotated mesh should change normals"


class TestNormalMap:
    """Verify dense normal map rendering from mesh."""

    def test_returns_correct_shape(self):
        """mesh_normal_map must return (H, W, 3)."""
        mesh = np.zeros((478, 3), dtype=np.float32)
        mesh[:, 0] = np.random.randint(0, 64, 478).astype(np.float32)
        mesh[:, 1] = np.random.randint(0, 64, 478).astype(np.float32)
        mesh[:, 2] = np.random.randn(478).astype(np.float32) * 5
        warp_M = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        normal_map = mesh_normal_map(mesh, warp_M, (256, 256))
        assert normal_map.shape == (256, 256, 3), f"Expected (256, 256, 3), got {normal_map.shape}"

    def test_normals_are_unit_vectors(self):
        """Each pixel in the normal map should be a unit vector."""
        mesh = np.zeros((478, 3), dtype=np.float32)
        mesh[:, 0] = np.random.randint(0, 200, 478).astype(np.float32)
        mesh[:, 1] = np.random.randint(0, 200, 478).astype(np.float32)
        mesh[:, 2] = np.random.randn(478).astype(np.float32) * 5
        warp_M = np.eye(2, 3, dtype=np.float32)
        normal_map = mesh_normal_map(mesh, warp_M, (256, 256))
        lengths = np.linalg.norm(normal_map, axis=2)
        # Skip zero pixels (background)
        foreground = lengths > 0.1
        if foreground.any():
            assert_allclose(lengths[foreground], 1.0, atol=1e-5)

    def test_invalid_mesh_returns_zeros(self):
        """None or empty mesh should return all zeros."""
        warp_M = np.eye(2, 3, dtype=np.float32)
        nm = mesh_normal_map(None, warp_M, (256, 256))
        assert np.allclose(nm, 0.0), "None mesh should return zeros"

    def test_empty_mesh_returns_zeros(self):
        """Mesh with <468 points should return all zeros."""
        warp_M = np.eye(2, 3, dtype=np.float32)
        small_mesh = np.zeros((100, 3), dtype=np.float32)
        nm = mesh_normal_map(small_mesh, warp_M, (256, 256))
        assert np.allclose(nm, 0.0), "Small mesh should return zeros"


class TestIntrinsicDecomposerNormalSource:
    """Verify IntrinsicDecomposer uses mesh normals when available."""

    def test_default_uses_mesh_normals(self):
        """IntrinsicDecomposer should default to use_mesh_normals=True."""
        decomposer = IntrinsicDecomposer()
        assert decomposer.use_mesh_normals is True

    def test_normal_source_tracked(self):
        """Decomposer must track which normal source was used."""
        decomposer = IntrinsicDecomposer()
        # Without mesh, falls back to face-prior normals (not circular shading)
        image = np.random.rand(64, 64, 3).astype(np.float32)
        result = decomposer.decompose(image)
        assert decomposer._normal_source == "face_prior"
        # With mesh, should use mesh
        mesh = np.zeros((478, 3), dtype=np.float32)
        mesh[:, 0] = np.random.randint(0, 32, 478).astype(np.float32)
        mesh[:, 1] = np.random.randint(0, 32, 478).astype(np.float32)
        warp_M = np.eye(2, 3, dtype=np.float32)
        result_mesh = decomposer.decompose(image, mesh_478=mesh, warp_M=warp_M)
        assert decomposer._normal_source == "mesh"

    def test_shading_gradient_fallback_when_no_mesh(self):
        """Without mesh_478, must fall back to face-prior normals (deterministic, not circular)."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(32, 32, 3).astype(np.float32)
        result = decomposer.decompose(image)
        assert decomposer._normal_source == "face_prior"
        assert result.normal_map.shape == (32, 32, 3)

    def test_can_disable_mesh_normals(self):
        """use_mesh_normals=False forces face-prior normals."""
        decomposer = IntrinsicDecomposer(use_mesh_normals=False)
        image = np.random.rand(32, 32, 3).astype(np.float32)
        mesh = np.zeros((478, 3), dtype=np.float32)
        warp_M = np.eye(2, 3, dtype=np.float32)
        result = decomposer.decompose(image, mesh_478=mesh, warp_M=warp_M)
        assert decomposer._normal_source == "face_prior"


class TestPipelineTelemetry:
    """Verify pipeline telemetry tracks normal source."""

    def test_pipeline_has_normal_telemetry(self):
        """Pipeline telemetry must include mesh/shading normal counters."""
        from face_os.pipeline import FaceOSPipeline
        p = FaceOSPipeline()
        keys = p._telemetry.keys()
        assert "mesh_normal_frames" in keys, "Missing mesh_normal_frames telemetry"
        assert "shading_normal_frames" in keys, "Missing shading_normal_frames telemetry"

    def test_telemetry_report_includes_rates(self):
        """get_telemetry_report must expose mesh/shading normal rates."""
        from face_os.pipeline import FaceOSPipeline
        p = FaceOSPipeline()
        report = p.get_telemetry_report()
        assert "mesh_normal_rate" in report, "Missing mesh_normal_rate in report"
        assert "shading_normal_rate" in report, "Missing shading_normal_rate in report"

    def test_normal_source_method_exists(self):
        """IdentityState must expose get_normal_source()."""
        from face_os.identity_state import IdentityState
        s = IdentityState()
        assert hasattr(s, 'get_normal_source'), "Missing get_normal_source()"
        source = s.get_normal_source()
        assert source in ("mesh", "shading_gradient")
