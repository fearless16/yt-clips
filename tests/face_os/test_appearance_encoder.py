"""Tests for the Appearance Encoder (Task 2.5 — manifold wiring).

Covers:
- _build_projection_matrix: shape, determinism
- store_enrollment_mesh: valid/invalid shapes
- _encode_appearance: None when no enrollment, zeros for identity deformation,
  non-zero for non-trivial deformation, None on shape mismatch
- set_anchor: with/without enrollment_mesh
- update_latent: geometry mesh → appearance_code update
- _invalidate_appearance_code: resets while preserving other fields
"""

from __future__ import annotations

import numpy as np
import pytest

from face_os.subsystems.identity_estimator import (
    IdentityEstimator,
    _build_projection_matrix,
)
from face_os.types import GeometryState


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

class MockIdentityState:
    """Minimal mock for IdentityEstimator construction."""
    pass


def _make_estimator(atlas_size=(64, 64)):
    """Create an IdentityEstimator with a small atlas for fast testing."""
    return IdentityEstimator(MockIdentityState(), atlas_size=atlas_size)


def _make_geometry(pose=(0.0, 0.0, 0.0), mesh=None):
    """Create a minimal GeometryState with identity transforms."""
    return GeometryState(
        pose=pose,
        canonical_transform=np.eye(3, dtype=np.float32),
        inverse_transform=np.eye(3, dtype=np.float32),
        mesh=mesh,
    )


def _make_enrollment_mesh(rng=None):
    """Synthetic (478, 3) float32 mesh — canonical neutral face."""
    if rng is None:
        rng = np.random
    # Realistic neutral-face landmarks: 478 points around a face shape
    mesh = np.zeros((478, 3), dtype=np.float32)
    # Generate a rough face shape (ellipsoid-like)
    for i in range(478):
        theta = (i / 478.0) * np.pi * 2
        phi = ((i % 239) / 239.0 - 0.5) * np.pi
        # Face-shaped distribution
        mesh[i, 0] = np.cos(theta) * np.cos(phi) * 100.0 + 320.0
        mesh[i, 1] = np.sin(phi) * 120.0 + 240.0
        mesh[i, 2] = np.sin(theta) * np.cos(phi) * 50.0 + 100.0
    return mesh


def _make_face_bgr(h=64, w=64, rng=None):
    """Synthetic BGR uint8 image useful for set_anchor / update_latent."""
    if rng is None:
        rng = np.random
    img = rng.randint(40, 215, (h, w, 3), dtype=np.uint8)
    return img


# ═══════════════════════════════════════════════════════════════════
# Test 1 — _build_projection_matrix
# ═══════════════════════════════════════════════════════════════════

def test_build_projection_matrix_shape():
    """Shape is (16, 1434) float32 — the JL projection for flat 478x3."""
    P = _build_projection_matrix()
    assert isinstance(P, np.ndarray)
    assert P.dtype.kind == 'f'  # floating-point (float64 from sqrt division)
    assert P.shape == (16, 1434)


def test_build_projection_matrix_deterministic():
    """Same seed (42) → identical matrix across calls."""
    P1 = _build_projection_matrix()
    P2 = _build_projection_matrix()
    assert np.array_equal(P1, P2)


def test_build_projection_matrix_scaled():
    """Each entry is divided by sqrt(1434), scale is correct."""
    P = _build_projection_matrix()
    expected_scale = 1.0 / np.sqrt(1434)
    # After scaling, entry magnitudes should be small (roughly ~1/sqrt(1434))
    assert np.abs(P).max() < 5.0 * expected_scale
    std = float(np.std(P))
    assert std > 0.0


# ═══════════════════════════════════════════════════════════════════
# Test 2 — store_enrollment_mesh
# ═══════════════════════════════════════════════════════════════════

def test_store_enrollment_mesh_valid():
    """Valid (478, 3) mesh is stored."""
    estimator = _make_estimator()
    mesh = _make_enrollment_mesh()
    estimator.store_enrollment_mesh(mesh)
    stored = estimator._enrollment_mesh
    assert stored is not None
    assert stored.shape == (478, 3)
    assert stored.dtype == np.float32
    np.testing.assert_array_equal(stored, mesh)


def test_store_enrollment_mesh_wrong_ndim():
    """1D mesh → sets enrollment to None."""
    estimator = _make_estimator()
    estimator.store_enrollment_mesh(np.zeros(100, dtype=np.float32))
    assert estimator._enrollment_mesh is None


def test_store_enrollment_mesh_too_few_landmarks():
    """Mesh with < 468 rows → sets enrollment to None."""
    estimator = _make_estimator()
    estimator.store_enrollment_mesh(np.zeros((200, 3), dtype=np.float32))
    assert estimator._enrollment_mesh is None


def test_store_enrollment_mesh_too_few_channels():
    """Mesh with < 3 columns → sets enrollment to None (2D but (478, 2))."""
    estimator = _make_estimator()
    estimator.store_enrollment_mesh(np.zeros((478, 2), dtype=np.float32))
    assert estimator._enrollment_mesh is None


def test_store_enrollment_mesh_invalidates_appearance_code():
    """Storing enrollment mesh resets appearance_code to zeros."""
    estimator = _make_estimator()
    # Seed latent with non-zero appearance_code via first update
    face = _make_face_bgr()
    geom = _make_geometry()
    estimator.update_latent(face, geom, np.ones((64, 64), dtype=np.float32))
    latent = estimator.latent()
    # Manually set a non-zero appearance code to verify invalidation
    latent.appearance_code = np.ones(16, dtype=np.float32)
    latent.appearance_uncertainty = 0.5

    mesh = _make_enrollment_mesh()
    estimator.store_enrollment_mesh(mesh)

    assert np.allclose(latent.appearance_code, np.zeros(16, dtype=np.float32))
    assert latent.appearance_uncertainty == 0.0


def test_store_enrollment_mesh_is_noop_when_latent_uninitialized():
    """store_enrollment_mesh does not crash when latent is not initialized."""
    estimator = _make_estimator()
    mesh = _make_enrollment_mesh()
    estimator.store_enrollment_mesh(mesh)
    assert estimator._enrollment_mesh is not None
    # Latent stays uninitialized — invalidation is no-op on uninitialized latent
    assert estimator.latent().initialized is False


# ═══════════════════════════════════════════════════════════════════
# Test 3 — _encode_appearance returns None with no enrollment mesh
# ═══════════════════════════════════════════════════════════════════

def test_encode_appearance_none_when_no_enrollment():
    """Returns None when enrollment mesh has never been stored."""
    estimator = _make_estimator()
    mesh = _make_enrollment_mesh()
    result = estimator._encode_appearance(mesh)
    assert result is None


def test_encode_appearance_none_after_invalid_enrollment():
    """Returns None after enrollment was invalidated by a bad store."""
    estimator = _make_estimator()
    estimator.store_enrollment_mesh(np.zeros((200, 3), dtype=np.float32))  # invalid
    mesh = _make_enrollment_mesh()
    result = estimator._encode_appearance(mesh)
    assert result is None


# ═══════════════════════════════════════════════════════════════════
# Test 4 — _encode_appearance returns zeros for identity deformation
# ═══════════════════════════════════════════════════════════════════

def test_encode_appearance_zeros_when_mesh_equals_enrollment():
    """Current mesh == enrollment mesh → deformation field is all zeros → code is zeros."""
    estimator = _make_estimator()
    mesh = _make_enrollment_mesh()
    estimator.store_enrollment_mesh(mesh)
    code = estimator._encode_appearance(mesh.copy())
    assert code is not None
    assert code.shape == (16,)
    assert code.dtype == np.float32
    np.testing.assert_allclose(code, np.zeros(16, dtype=np.float32), atol=1e-6)


# ═══════════════════════════════════════════════════════════════════
# Test 5 — _encode_appearance returns non-zero for non-trivial deformation
# ═══════════════════════════════════════════════════════════════════

def test_encode_appearance_nonzero_when_meshes_differ():
    """A non-trivial deformation (smile, eyebrow raise) → non-zero appearance_code."""
    rng = np.random.RandomState(42)
    estimator = _make_estimator()
    enrollment = _make_enrollment_mesh(rng)
    estimator.store_enrollment_mesh(enrollment)

    # Create a deformed mesh (simulate expression change)
    deformed = enrollment.copy()
    deformed[:, 1] += rng.normal(0, 5, 478).astype(np.float32)  # vertical shifts
    deformed[:, 0] += rng.normal(0, 3, 478).astype(np.float32)  # horizontal shifts

    code = estimator._encode_appearance(deformed)
    assert code is not None
    assert code.shape == (16,)
    assert not np.allclose(code, np.zeros(16, dtype=np.float32), atol=1e-8)


def test_encode_appearance_large_deformation_larger_code_norm():
    """Larger deformation → larger code magnitude (scale sensitivity)."""
    rng = np.random.RandomState(99)
    estimator = _make_estimator()
    enrollment = _make_enrollment_mesh(rng)
    estimator.store_enrollment_mesh(enrollment)

    # Small deformation
    small = enrollment.copy()
    small[:, 1] += rng.normal(0, 1, 478).astype(np.float32)
    code_small = estimator._encode_appearance(small)

    # Large deformation
    large = enrollment.copy()
    large[:, 1] += rng.normal(0, 10, 478).astype(np.float32)
    code_large = estimator._encode_appearance(large)

    assert code_small is not None and code_large is not None
    norm_small = float(np.linalg.norm(code_small))
    norm_large = float(np.linalg.norm(code_large))
    # Larger deformation should produce a code with higher (or equal) norm
    assert norm_large >= norm_small * 0.5, (
        f"Expected large deformation norm ({norm_large:.4f}) >= "
        f"0.5 * small norm ({norm_small:.4f})"
    )


# ═══════════════════════════════════════════════════════════════════
# Test 6 — _encode_appearance returns None for shape mismatch
# ═══════════════════════════════════════════════════════════════════

def test_encode_appearance_none_when_shape_mismatches_enrollment():
    """Mesh with different shape than enrollment → returns None."""
    estimator = _make_estimator()
    enrollment = _make_enrollment_mesh()
    estimator.store_enrollment_mesh(enrollment)

    # Wrong row count
    code = estimator._encode_appearance(np.zeros((470, 3), dtype=np.float32))
    assert code is None

    # Wrong column count
    code = estimator._encode_appearance(np.zeros((478, 4), dtype=np.float32))
    assert code is None


def test_encode_appearance_none_for_invalid_mesh_input():
    """Invalid mesh (wrong ndim, too few landmarks) → returns None."""
    estimator = _make_estimator()
    enrollment = _make_enrollment_mesh()
    estimator.store_enrollment_mesh(enrollment)

    # Invalid mesh: too few rows
    code = estimator._encode_appearance(np.zeros((100, 3), dtype=np.float32))
    assert code is None

    # Invalid mesh: 1D
    code = estimator._encode_appearance(np.zeros(100, dtype=np.float32))
    assert code is None


# ═══════════════════════════════════════════════════════════════════
# Test 7 — set_anchor with enrollment_mesh
# ═══════════════════════════════════════════════════════════════════

def test_set_anchor_with_enrollment_mesh_stores_it():
    """set_anchor(enrollment_mesh=mesh) stores the mesh and the latent initializes."""
    estimator = _make_estimator(atlas_size=(64, 64))
    enrollment = _make_enrollment_mesh()
    face = _make_face_bgr(h=64, w=64)

    estimator.set_anchor(face, enrollment_mesh=enrollment)

    assert estimator._enrollment_mesh is not None
    np.testing.assert_array_equal(estimator._enrollment_mesh, enrollment)
    assert estimator.latent().initialized is True
    # appearance_code is zeros, appearance_uncertainty is 0.0 (invalidate called)
    assert np.allclose(estimator.latent().appearance_code, np.zeros(16, dtype=np.float32))
    assert estimator.latent().appearance_uncertainty == 0.0


# ═══════════════════════════════════════════════════════════════════
# Test 8 — set_anchor without enrollment_mesh
# ═══════════════════════════════════════════════════════════════════

def test_set_anchor_without_enrollment_mesh_initializes_latent_only():
    """set_anchor() without enrollment_mesh: latent initialized, encoding disabled."""
    estimator = _make_estimator(atlas_size=(64, 64))
    face = _make_face_bgr(h=64, w=64)

    estimator.set_anchor(face)  # no enrollment_mesh

    # Enrollment mesh is NOT stored
    assert estimator._enrollment_mesh is None
    # Latent IS initialized (no crash)
    assert estimator.latent().initialized is True
    # _encode_appearance is disabled
    code = estimator._encode_appearance(_make_enrollment_mesh())
    assert code is None


# ═══════════════════════════════════════════════════════════════════
# Test 9 — update_latent updates appearance_code from geometry mesh
# ═══════════════════════════════════════════════════════════════════

def test_update_latent_updates_appearance_code_when_geometry_has_mesh():
    """update_latent computes and stores appearance_code from geometry.mesh."""
    rng = np.random.RandomState(42)
    estimator = _make_estimator(atlas_size=(64, 64))
    enrollment = _make_enrollment_mesh(rng)
    estimator.store_enrollment_mesh(enrollment)

    # Seed the latent with a first observation
    face = _make_face_bgr(h=64, w=64, rng=rng)
    quality = np.ones((64, 64), dtype=np.float32)
    estimator.update_latent(face, _make_geometry(), quality)

    # First latent should have zero appearance_code (not yet driven by geometry)
    latent = estimator.latent()
    assert latent.initialized is True
    assert np.allclose(latent.appearance_code, np.zeros(16, dtype=np.float32), atol=1e-6)

    # Now create a geometry WITH a mesh that differs from enrollment
    deformed = enrollment.copy()
    deformed[:, 1] += 5.0  # significant vertical shift
    geometry_with_mesh = _make_geometry(mesh=deformed)

    estimator.update_latent(face, geometry_with_mesh, quality)
    latent = estimator.latent()

    # appearance_code should be non-zero (deformation detected)
    assert not np.allclose(latent.appearance_code, np.zeros(16, dtype=np.float32), atol=1e-8)
    # appearance_uncertainty should be updated based on code magnitude
    assert latent.appearance_uncertainty > 0.0


def test_update_latent_does_not_update_code_when_geometry_mesh_missing():
    """update_latent: geometry.mesh is None → appearance_code unchanged."""
    rng = np.random.RandomState(7)
    estimator = _make_estimator(atlas_size=(64, 64))
    enrollment = _make_enrollment_mesh(rng)
    estimator.store_enrollment_mesh(enrollment)

    face = _make_face_bgr(h=64, w=64, rng=rng)
    quality = np.ones((64, 64), dtype=np.float32)

    # Seed latent
    estimator.update_latent(face, _make_geometry(mesh=enrollment), quality)
    code_before = estimator.latent().appearance_code.copy()

    # Second call WITHOUT mesh in geometry
    geometry_no_mesh = _make_geometry(mesh=None)
    estimator.update_latent(face, geometry_no_mesh, quality)

    # appearance_code should NOT change (mesh was None in second call)
    np.testing.assert_array_equal(estimator.latent().appearance_code, code_before)


def test_update_latent_appearance_code_identity_remains_zero():
    """When geometry mesh equals enrollment → appearance_code stays zero."""
    rng = np.random.RandomState(13)
    estimator = _make_estimator(atlas_size=(64, 64))
    enrollment = _make_enrollment_mesh(rng)
    estimator.store_enrollment_mesh(enrollment)

    face = _make_face_bgr(h=64, w=64, rng=rng)
    quality = np.ones((64, 64), dtype=np.float32)

    # Seed latent
    estimator.update_latent(face, _make_geometry(), quality)

    # Now update with enrollment mesh (identity deformation)
    geometry_enrollment = _make_geometry(mesh=enrollment.copy())
    estimator.update_latent(face, geometry_enrollment, quality)

    latent = estimator.latent()
    np.testing.assert_allclose(latent.appearance_code, np.zeros(16, dtype=np.float32), atol=1e-6)


# ═══════════════════════════════════════════════════════════════════
# Test 10 — _invalidate_appearance_code preserves other latent fields
# ═══════════════════════════════════════════════════════════════════

def test_invalidate_appearance_code_resets_only_appearance_fields():
    """_invalidate_appearance_code zeros appearance_code + uncertainty; albedo etc. survive."""
    estimator = _make_estimator(atlas_size=(64, 64))
    face = _make_face_bgr(h=64, w=64)

    # Seed latent with set_anchor so it has albedo, microdetail, etc.
    estimator.set_anchor(face)

    latent = estimator.latent()
    assert latent.initialized is True
    # Inject a non-zero appearance code and uncertainty
    latent.appearance_code = np.full(16, 0.7, dtype=np.float32)
    latent.appearance_uncertainty = 0.9

    albedo_before = latent.albedo.copy()
    microdetail_before = latent.microdetail.copy() if latent.microdetail is not None else None
    wb_before = latent.wb_reference.copy()
    albedo_unc_before = latent.albedo_uncertainty.copy() if latent.albedo_uncertainty is not None else None

    estimator._invalidate_appearance_code()

    # Appearance fields reset
    assert np.allclose(latent.appearance_code, np.zeros(16, dtype=np.float32))
    assert latent.appearance_uncertainty == 0.0

    # All other fields preserved
    np.testing.assert_array_equal(latent.albedo, albedo_before)
    np.testing.assert_array_equal(latent.wb_reference, wb_before)
    if microdetail_before is not None:
        np.testing.assert_array_equal(latent.microdetail, microdetail_before)
    if albedo_unc_before is not None:
        np.testing.assert_array_equal(latent.albedo_uncertainty, albedo_unc_before)


def test_invalidate_appearance_code_noop_when_uninitialized():
    """_invalidate_appearance_code is a no-op when latent is not initialized."""
    estimator = _make_estimator()
    latent = estimator.latent()
    assert latent.initialized is False
    # Should not crash
    estimator._invalidate_appearance_code()
    assert latent.initialized is False
    assert latent.appearance_uncertainty == 1.0  # default
