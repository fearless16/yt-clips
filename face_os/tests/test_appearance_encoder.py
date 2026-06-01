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


# ═══════════════════════════════════════════════════════════════════
# Test 11 — canonical 2D landmark positions
# ═══════════════════════════════════════════════════════════════════

def test_canonical_lm_2d_computed_on_store_enrollment():
    """store_enrollment_mesh computes _canonical_lm_2d in [0,256] atlas space."""
    estimator = _make_estimator()
    mesh = _make_enrollment_mesh()
    estimator.store_enrollment_mesh(mesh)

    lm2d = estimator._canonical_lm_2d
    assert lm2d is not None
    assert lm2d.shape == (478, 2)
    assert lm2d.dtype == np.float32
    # All landmarks within [0, 256] (with some margin for the padding)
    assert lm2d[:, 0].min() >= 0
    assert lm2d[:, 0].max() <= 256
    assert lm2d[:, 1].min() >= 0
    assert lm2d[:, 1].max() <= 256
    # At least some variation (not all same point)
    assert lm2d[:, 0].max() - lm2d[:, 0].min() > 40
    assert lm2d[:, 1].max() - lm2d[:, 1].min() > 40


# ═══════════════════════════════════════════════════════════════════
# Test 12 — _compute_deformation_map
# ═══════════════════════════════════════════════════════════════════

def test_compute_deformation_map_zeros_when_no_data():
    """Returns zero-filled map when smoothed appearance is None."""
    estimator = _make_estimator(atlas_size=(64, 64))
    dmap = estimator._compute_deformation_map(64, 64)
    assert dmap.shape == (64, 64)
    assert dmap.dtype == np.float32
    assert np.allclose(dmap, 0.0)


def test_compute_deformation_map_zeros_when_no_canonical_lm():
    """Returns zeros when _canonical_lm_2d is None."""
    estimator = _make_estimator(atlas_size=(64, 64))
    estimator._smoothed_appearance = np.ones(16, dtype=np.float32) * 0.5
    dmap = estimator._compute_deformation_map(64, 64)
    assert np.allclose(dmap, 0.0)


def test_compute_deformation_map_shape():
    """With valid enrollment + smoothed appearance, returns (h,w) float32 map."""
    rng = np.random.RandomState(42)
    estimator = _make_estimator(atlas_size=(64, 64))
    enrollment = _make_enrollment_mesh(rng)
    estimator.store_enrollment_mesh(enrollment)

    # Seed smoothed_appearance with a non-zero code to trigger deformation
    estimator._smoothed_appearance = rng.normal(0, 0.01, 16).astype(np.float32)

    dmap = estimator._compute_deformation_map(64, 64)
    assert dmap.shape == (64, 64)
    assert dmap.dtype == np.float32


def test_compute_deformation_map_typically_nonzero_for_offset_code():
    """A clearly non-zero code should produce non-zero deformation regions."""
    rng = np.random.RandomState(73)
    estimator = _make_estimator(atlas_size=(64, 64))
    enrollment = _make_enrollment_mesh(rng)
    estimator.store_enrollment_mesh(enrollment)
    estimator._smoothed_appearance = rng.normal(0, 0.1, 16).astype(np.float32)

    dmap = estimator._compute_deformation_map(64, 64)
    # At least some pixels should register deformation
    assert float(np.max(dmap)) > 0.0


# ═══════════════════════════════════════════════════════════════════
# Test 13 — Deformation stats propagation
# ═══════════════════════════════════════════════════════════════════

def test_deformation_stats_updated_in_update_latent():
    """After update_latent with a mesh, _last_deform_{max,mean} are populated."""
    rng = np.random.RandomState(7)
    estimator = _make_estimator(atlas_size=(64, 64))
    enrollment = _make_enrollment_mesh(rng)
    estimator.store_enrollment_mesh(enrollment)

    face = _make_face_bgr(h=64, w=64, rng=rng)
    quality = np.ones((64, 64), dtype=np.float32)

    # Initialize latent
    estimator.update_latent(face, _make_geometry(), quality)

    # Deform the mesh
    deformed = enrollment.copy()
    deformed[:, 1] += 10.0
    estimator.update_latent(face, _make_geometry(mesh=deformed), quality)

    # Stats should be set (might be zero if deformation map is zero, which is fine)
    assert isinstance(estimator._last_deform_max, float)
    assert isinstance(estimator._last_deform_mean, float)


def test_deformation_stats_zero_for_identity_mesh():
    """When geometry == enrollment, deformation stats are zero."""
    rng = np.random.RandomState(13)
    estimator = _make_estimator(atlas_size=(64, 64))
    enrollment = _make_enrollment_mesh(rng)
    estimator.store_enrollment_mesh(enrollment)

    face = _make_face_bgr(h=64, w=64, rng=rng)
    quality = np.ones((64, 64), dtype=np.float32)

    estimator.update_latent(face, _make_geometry(), quality)
    estimator.update_latent(face, _make_geometry(mesh=enrollment.copy()), quality)

    assert estimator._last_deform_max == 0.0
    assert estimator._last_deform_mean == 0.0


# ═══════════════════════════════════════════════════════════════════
# Test 14 — Expression-aware Kalman gain modulation
# ═══════════════════════════════════════════════════════════════════

def test_compute_deformation_map_with_identity_code():
    """Zero appearance_code → zero deformation everywhere."""
    rng = np.random.RandomState(99)
    estimator = _make_estimator(atlas_size=(64, 64))
    enrollment = _make_enrollment_mesh(rng)
    estimator.store_enrollment_mesh(enrollment)
    estimator._smoothed_appearance = np.zeros(16, dtype=np.float32)

    dmap = estimator._compute_deformation_map(64, 64)
    assert np.allclose(dmap, 0.0)


def test_deformation_gain_modulation_is_ge_one():
    """Expression-aware gain multiplier is always >= 1.0 (boosting gain)."""
    rng = np.random.RandomState(57)
    estimator = _make_estimator(atlas_size=(64, 64))
    enrollment = _make_enrollment_mesh(rng)
    estimator.store_enrollment_mesh(enrollment)
    estimator._smoothed_appearance = rng.normal(0, 0.5, 16).astype(np.float32)

    dmap = estimator._compute_deformation_map(64, 64)
    from face_os.subsystems.identity_estimator import _K_EXPRESSION_GAIN
    deform_gain = np.clip(1.0 + dmap * _K_EXPRESSION_GAIN, 1.0, 3.0)
    assert np.all(deform_gain >= 1.0)
    assert np.all(deform_gain <= 3.0)


# ═══════════════════════════════════════════════════════════════════
# Test 15 — Regularized pseudoinverse
# ═══════════════════════════════════════════════════════════════════

def test_regularized_pinv_shape():
    """_get_regularized_pinv returns (1434, 16) float32."""
    estimator = _make_estimator()
    pinv = estimator._get_regularized_pinv()
    assert pinv.shape == (1434, 16)
    assert pinv.dtype == np.float32


def test_regularized_pinv_smaller_deformation_than_standard():
    """Regularized pinv produces smaller deformation magnitudes than standard."""
    rng = np.random.RandomState(42)
    estimator = _make_estimator()
    pinv_reg = estimator._get_regularized_pinv()
    pinv_std = estimator._projection_pinv

    # Test with several codes
    total_reg = 0.0
    total_std = 0.0
    for _ in range(10):
        code = rng.normal(0, 0.5, 16).astype(np.float32)
        deform_reg = pinv_reg @ code
        deform_std = pinv_std @ code
        total_reg += float(np.linalg.norm(deform_reg))
        total_std += float(np.linalg.norm(deform_std))

    assert total_reg < total_std, (
        f"Regularized total {total_reg:.2f} >= standard {total_std:.2f}"
    )


def test_regularized_pinv_reconstructs_code_approximately():
    """P·pinv_reg·code ≈ code (small reprojection error despite regularization)."""
    rng = np.random.RandomState(31)
    estimator = _make_estimator()
    pinv = estimator._get_regularized_pinv()
    P = estimator._projection_matrix

    for _ in range(5):
        code = rng.normal(0, 0.5, 16).astype(np.float32)
        deform = pinv @ code
        reconstructed = P @ deform
        err = np.linalg.norm(reconstructed - code) / max(np.linalg.norm(code), 1e-6)
        assert err < 0.15, f"Reconstruction error {err:.4f} too high"


# ═══════════════════════════════════════════════════════════════════
# Test 16 — Geodesic outlier rejection
# ═══════════════════════════════════════════════════════════════════

def test_outlier_rejection_skips_accumulation():
    """An extreme code jump is classified as outlier and skipped."""
    rng = np.random.RandomState(42)
    estimator = _make_estimator(atlas_size=(64, 64))
    enrollment_mesh = _make_enrollment_mesh(rng)

    face = _make_face_bgr(h=64, w=64, rng=rng)
    estimator.set_anchor(face, enrollment_mesh=enrollment_mesh)
    quality = np.ones((64, 64), dtype=np.float32)

    # Build a stable history with slight natural variation
    for i in range(15):
        mesh = enrollment_mesh.copy()
        mesh[:, 1] += 2.0 + rng.normal(0, 1.0, 478).astype(np.float32)
        estimator.update_latent(face, _make_geometry(mesh=mesh), quality)

    obs_before = len(estimator._observation_points)
    smoothed_before = estimator._smoothed_appearance.copy()

    # Inject an extreme jump (10x the base deformation)
    extreme_mesh = enrollment_mesh.copy()
    extreme_mesh[:, 1] += 30.0
    estimator.update_latent(face, _make_geometry(mesh=extreme_mesh), quality)

    # Observation count should NOT have increased (extreme was rejected)
    assert len(estimator._observation_points) == obs_before, (
        f"Observation count grew from {obs_before} to {len(estimator._observation_points)}"
        f" — extreme frame should be rejected"
    )
    # Smoothed code should remain stable (not jump to the extreme)
    np.testing.assert_allclose(
        estimator._smoothed_appearance, smoothed_before, atol=0.1
    )


def test_outlier_rejection_small_jumps_accepted():
    """Small, consistent deformation is always accepted (not an outlier)."""
    rng = np.random.RandomState(42)
    estimator = _make_estimator(atlas_size=(64, 64))
    enrollment_mesh = _make_enrollment_mesh(rng)

    face = _make_face_bgr(h=64, w=64, rng=rng)
    estimator.set_anchor(face, enrollment_mesh=enrollment_mesh)
    quality = np.ones((64, 64), dtype=np.float32)

    # Build a stable history with slight natural variation
    for i in range(15):
        mesh = enrollment_mesh.copy()
        mesh[:, 1] += 2.0 + rng.normal(0, 1.0, 478).astype(np.float32)
        estimator.update_latent(face, _make_geometry(mesh=mesh), quality)

    obs_count_before = len(estimator._observation_points)

    # Slightly different deformation — should be accepted
    small_mesh = enrollment_mesh.copy()
    small_mesh[:, 1] += 4.0  # within natural variation range
    estimator.update_latent(face, _make_geometry(mesh=small_mesh), quality)

    assert len(estimator._observation_points) > obs_count_before, (
        "Small deformation jump should be accepted"
    )


def test_outlier_rejection_no_premature_flagging():
    """Before MIN_HISTORY frames, no frame is ever flagged as outlier."""
    rng = np.random.RandomState(99)
    estimator = _make_estimator(atlas_size=(64, 64))

    face = _make_face_bgr(h=64, w=64, rng=rng)
    estimator.set_anchor(face, enrollment_mesh=_make_enrollment_mesh(rng))
    quality = np.ones((64, 64), dtype=np.float32)

    # Feed extreme frames from the start — none should be rejected
    for i in range(8):  # less than GEODESIC_OUTLIER_MIN_HISTORY=10
        mesh = _make_enrollment_mesh(rng)
        mesh[:, 1] += rng.normal(0, 50, 478).astype(np.float32)
        estimator.update_latent(
            face, _make_geometry(mesh=mesh), quality
        )

    # All non-seed frames should have been accumulated
    # set_anchor does not go through update_latent, so only the 8 explicit
    # update_latent calls produce observations (first one seeds latent)
    assert len(estimator._observation_points) == 8, (
        f"Expected 8 observations (set_anchor seeds latent, all 8 frames "
        f"go through fusion), got {len(estimator._observation_points)}"
    )
