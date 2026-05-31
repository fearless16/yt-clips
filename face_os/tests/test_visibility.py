"""Tests for §16.6 Visibility / Occlusion Field (D-05 / C_recon prerequisite).

arch.md §16.6:
    V(u,v,t) ∈ [0,1]            (geometry-derived self-occlusion, NOT a 2D
                                 sharpness proxy)
    C_new(u,v) = C_old(u,v) + q_t · V(u,v,t)
    Invariant: when V(u,v,t)=0, C(u,v) and the stored appearance at (u,v) are
        unchanged by frame t.
    Required test: synthesize a profile observation; assert the occluded-side
        region memory is byte-identical before/after the update.

V is the geometry self-occlusion factor of the §16.8 composite
    C_recon = C_obs · Coverage_pose · Coverage_light · Visibility
derived from the canonical-UV per-pixel normal map already produced at runtime
(IntrinsicComponents.normal_map): the camera views down +Z, so a surface point
is visible iff N·view = N_z > 0; back-facing points (N_z ≤ 0) are occluded.

Geometry-FREE frames (normal_source != 'mesh', i.e. the face-prior dome) carry
NO self-occlusion evidence, so V≡1 there — the same "no evidence ⇒ no penalty"
stance as §16.7's coverage fallback. Only mesh normals gate memory.

Determinism: fixed synthetic inputs, no randomness (arch §3).
"""
from __future__ import annotations

import numpy as np
import pytest

from face_os.visibility import compute_visibility
from face_os.intrinsic_decomposition import IntrinsicComponents
from face_os.types import GeometryState


# ─── compute_visibility(): V = clip(N·view, 0, 1) ────────────────────────────

def _normal_field(hw, nx=0.0, ny=0.0, nz=1.0):
    h, w = hw
    n = np.zeros((h, w, 3), dtype=np.float32)
    n[..., 0] = nx
    n[..., 1] = ny
    n[..., 2] = nz
    return n


class TestComputeVisibility:
    def test_frontal_normals_fully_visible(self):
        """N = +Z everywhere ⇒ V = 1 everywhere (facing the camera)."""
        V = compute_visibility(_normal_field((16, 16), nz=1.0))
        assert V.shape == (16, 16)
        assert np.allclose(V, 1.0)

    def test_back_facing_normals_occluded(self):
        """N = -Z (pointing away) ⇒ N·view = -1 ⇒ clipped to V = 0."""
        V = compute_visibility(_normal_field((16, 16), nz=-1.0))
        assert np.allclose(V, 0.0)

    def test_grazing_normals_zero(self):
        """N ⟂ view (N_z = 0) ⇒ V = 0 (silhouette edge, not visible)."""
        V = compute_visibility(_normal_field((8, 8), nx=1.0, nz=0.0))
        assert np.allclose(V, 0.0)

    def test_oblique_normal_is_cosine(self):
        """V is the cosine N·view, not a hard 0/1 mask."""
        n = _normal_field((4, 4), ny=0.6, nz=0.8)  # unit, N_z = 0.8
        V = compute_visibility(n)
        assert np.allclose(V, 0.8, atol=1e-6)

    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(0)
        n = rng.standard_normal((32, 32, 3)).astype(np.float32)
        n /= (np.linalg.norm(n, axis=2, keepdims=True) + 1e-8)
        V = compute_visibility(n)
        assert V.min() >= 0.0 and V.max() <= 1.0
        assert V.dtype == np.float32

    def test_degenerate_zero_normals_are_occluded(self):
        """A zero normal (no geometry at that texel) ⇒ N·view = 0 ⇒ V = 0."""
        V = compute_visibility(np.zeros((8, 8, 3), dtype=np.float32))
        assert np.allclose(V, 0.0)

    def test_custom_view_direction(self):
        """View direction is configurable; V = clip(N·view_hat, 0, 1) with the
        view direction normalized."""
        n = _normal_field((4, 4), nx=1.0, nz=0.0)   # N = +X
        V = compute_visibility(n, view_direction=(1.0, 0.0, 0.0))
        assert np.allclose(V, 1.0)

    def test_left_right_split_mask(self):
        """A profile-like field: right half front-facing, left half back-facing."""
        n = np.zeros((10, 10, 3), dtype=np.float32)
        n[:, 5:, 2] = 1.0    # right half faces camera
        n[:, :5, 2] = -1.0   # left half faces away
        V = compute_visibility(n)
        assert np.allclose(V[:, 5:], 1.0)
        assert np.allclose(V[:, :5], 0.0)


# ─── §16.6 REQUIRED test: profile occlusion ⇒ occluded memory byte-identical ──

def _make_identity_estimator(atlas_size=(32, 32)):
    from face_os.subsystems.identity_estimator import IdentityEstimator

    class _MockState:
        pass

    return IdentityEstimator(_MockState(), atlas_size=atlas_size)


def _make_geometry(pose=(0.0, 0.0, 0.0)):
    return GeometryState(
        pose=pose,
        canonical_transform=np.eye(3, dtype=np.float32),
        inverse_transform=np.eye(3, dtype=np.float32),
    )


def _mesh_intrinsic(hw, albedo_value, normal_map):
    """An IntrinsicComponents tagged normal_source='mesh' so the visibility
    gate engages (face-prior frames carry no occlusion evidence)."""
    h, w = hw
    albedo = np.full((h, w, 3), albedo_value, dtype=np.float32)
    shading = np.full((h, w, 1), 0.5, dtype=np.float32)
    specular = np.zeros((h, w, 3), dtype=np.float32)
    confidence = np.full((h, w, 1), 1.0, dtype=np.float32)
    return IntrinsicComponents(
        albedo=albedo,
        shading=shading,
        specular=specular,
        normal_map=normal_map,
        confidence=confidence,
        reconstruction_error=0.0,
        albedo_uncertainty=np.full((h, w), 0.2, dtype=np.float32),
        normal_source="mesh",
    )


class TestVisibilityGatesMemory:
    def test_profile_occlusion_leaves_occluded_region_byte_identical(self):
        """arch §16.6 REQUIRED test. After enrolling frontally, a profile
        observation (left half back-facing) must leave the stored albedo +
        observation_count in the occluded (left) half BYTE-IDENTICAL, while the
        visible (right) half updates.
        """
        hw = (32, 32)
        est = _make_identity_estimator(atlas_size=hw)
        geom = _make_geometry()
        face = np.full((*hw, 3), 100, dtype=np.uint8)  # BGR observation

        # 1. Frontal enroll — everything visible, seeds the latent.
        frontal_normals = _normal_field(hw, nz=1.0)
        est.update_latent(
            face, geom, np.full(hw, 0.9, dtype=np.float32),
            intrinsic=_mesh_intrinsic(hw, albedo_value=0.4, normal_map=frontal_normals),
        )
        before_albedo = est.latent().albedo.copy()
        before_count = est.latent().observation_count.copy()

        # 2. Profile observation: left half back-facing (V=0), DIFFERENT albedo.
        profile_normals = np.zeros((*hw, 3), dtype=np.float32)
        profile_normals[:, 16:, 2] = 1.0    # right half visible
        profile_normals[:, :16, 2] = -1.0   # left half occluded
        est.update_latent(
            face, geom, np.full(hw, 0.9, dtype=np.float32),
            intrinsic=_mesh_intrinsic(hw, albedo_value=0.9, normal_map=profile_normals),
        )
        after_albedo = est.latent().albedo
        after_count = est.latent().observation_count

        # Occluded (left) half: byte-identical albedo AND count (the invariant).
        assert np.array_equal(after_albedo[:, :16], before_albedo[:, :16])
        assert np.array_equal(after_count[:, :16], before_count[:, :16])
        # Visible (right) half: actually moved toward the new observation.
        assert not np.array_equal(after_albedo[:, 16:], before_albedo[:, 16:])
        assert np.any(after_count[:, 16:] > before_count[:, 16:])

    def test_mean_visibility_recorded_for_mesh_update(self):
        """The estimator records last_mean_visibility for telemetry; a fully
        frontal mesh observation reads ~1.0."""
        hw = (16, 16)
        est = _make_identity_estimator(atlas_size=hw)
        geom = _make_geometry()
        face = np.full((*hw, 3), 100, dtype=np.uint8)
        est.update_latent(
            face, geom, np.full(hw, 0.9, dtype=np.float32),
            intrinsic=_mesh_intrinsic(hw, 0.4, _normal_field(hw, nz=1.0)),
        )
        assert est.last_mean_visibility == pytest.approx(1.0, abs=1e-6)

    def test_half_occluded_mesh_update_records_partial_visibility(self):
        hw = (16, 16)
        est = _make_identity_estimator(atlas_size=hw)
        geom = _make_geometry()
        face = np.full((*hw, 3), 100, dtype=np.uint8)
        normals = np.zeros((*hw, 3), dtype=np.float32)
        normals[:, 8:, 2] = 1.0
        normals[:, :8, 2] = -1.0
        est.update_latent(
            face, geom, np.full(hw, 0.9, dtype=np.float32),
            intrinsic=_mesh_intrinsic(hw, 0.4, normals),
        )
        assert est.last_mean_visibility == pytest.approx(0.5, abs=1e-6)

    def test_face_prior_frame_does_not_gate_visibility(self):
        """A geometry-free (face_prior) observation carries no self-occlusion
        evidence ⇒ V≡1, so last_mean_visibility stays 1.0 and memory updates
        exactly as before this feature (no regression on the legacy path)."""
        hw = (16, 16)
        est = _make_identity_estimator(atlas_size=hw)
        geom = _make_geometry()
        face = np.full((*hw, 3), 100, dtype=np.uint8)
        # normal_source defaults to 'face_prior' on a plain intrinsic; emulate
        # with an explicit tag and back-facing normals that must be IGNORED.
        ic = _mesh_intrinsic(hw, 0.4, _normal_field(hw, nz=-1.0))
        ic.normal_source = "face_prior"
        est.update_latent(face, geom, np.full(hw, 0.9, dtype=np.float32), intrinsic=ic)
        assert est.last_mean_visibility == 1.0
        # latent still initialized normally despite back-facing normals
        assert est.latent().initialized is True
