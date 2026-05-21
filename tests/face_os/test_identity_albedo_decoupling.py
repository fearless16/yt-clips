"""Tests for I-05: Identity Anchor Full Decoupling.

Verifies that identity is stored and corrected in albedo space (lighting-invariant),
not RGB space (lighting-entangled).

RULE 5: Albedo must be stored separately from appearance_latent.
Anchor correction must apply to albedo only, not RGB.
Query path must normalize albedo before blending.
"""

import numpy as np
import cv2
import pytest


class TestAlbedoStorage:
    """Albedo must be stored separately from appearance."""

    def test_identity_state_has_anchor_albedo(self):
        """After set_anchor, _anchor_albedo must exist."""
        from face_os.identity_state import IdentityState
        state = IdentityState()
        ref = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state.set_anchor(ref)
        assert hasattr(state, '_anchor_albedo')
        assert state._anchor_albedo is not None

    def test_anchor_albedo_is_float32(self):
        """Anchor albedo must be float32 for precision."""
        from face_os.identity_state import IdentityState
        state = IdentityState()
        ref = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state.set_anchor(ref)
        assert state._anchor_albedo.dtype == np.float32

    def test_anchor_albedo_range_01(self):
        """Anchor albedo must be in [0, 1] range."""
        from face_os.identity_state import IdentityState
        state = IdentityState()
        ref = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state.set_anchor(ref)
        assert state._anchor_albedo.min() >= 0.0
        assert state._anchor_albedo.max() <= 1.0

    def test_anchor_albedo_shape_matches_atlas(self):
        """Anchor albedo shape must match atlas size."""
        from face_os.identity_state import IdentityState
        state = IdentityState(atlas_size=(256, 256))
        ref = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state.set_anchor(ref)
        assert state._anchor_albedo.shape == (256, 256, 3)


class TestAlbedoAnchorCorrection:
    """Anchor correction must apply to albedo, not RGB."""

    def test_query_albedo_method_exists(self):
        """query_albedo() must exist and return (albedo, confidence)."""
        from face_os.identity_state import IdentityState
        state = IdentityState()
        ref = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state.set_anchor(ref)
        # Initialize belief
        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state.update(frame, quality)
        result = state.query_albedo(quality)
        assert isinstance(result, tuple)
        assert len(result) == 2
        albedo, confidence = result
        assert albedo.dtype == np.float32
        assert albedo.shape == (256, 256, 3)

    def test_query_albedo_range_01(self):
        """Albedo from query_albedo must be in [0, 1]."""
        from face_os.identity_state import IdentityState
        state = IdentityState()
        ref = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state.set_anchor(ref)
        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state.update(frame, quality)
        albedo, _ = state.query_albedo(quality)
        assert albedo.min() >= 0.0
        assert albedo.max() <= 1.0 + 1e-6

    def test_anchor_correction_in_albedo_space(self):
        """Anchor correction must pull albedo toward anchor_albedo, not anchor_rgb."""
        from face_os.identity_state import IdentityState
        state = IdentityState()
        # Create a reference with known color
        ref = np.full((256, 256, 3), [200, 100, 50], dtype=np.uint8)
        state.set_anchor(ref)
        # Update with different color
        frame = np.full((256, 256, 3), [50, 200, 100], dtype=np.uint8)
        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        state.update(frame, quality)
        # Query albedo — should be pulled toward anchor albedo
        albedo, _ = state.query_albedo(quality)
        # Albedo mean should be different from raw frame mean
        # (anchor correction is applied)
        frame_f = frame.astype(np.float32) / 255.0
        assert not np.allclose(albedo, frame_f, atol=0.01)


class TestQueryIntrinsicAlbedoDecoupling:
    """query_intrinsic must return albedo that is lighting-decoupled."""

    def test_query_intrinsic_returns_albedo(self):
        """query_intrinsic must return components with albedo."""
        from face_os.identity_state import IdentityState
        state = IdentityState()
        ref = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state.set_anchor(ref)
        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state.update(frame, quality)
        intrinsic, conf = state.query_intrinsic(quality)
        assert intrinsic is not None
        assert hasattr(intrinsic, 'albedo')
        assert intrinsic.albedo.dtype == np.float32

    def test_white_balance_applied_to_albedo(self):
        """query_intrinsic albedo must be white-balanced (RULE 5)."""
        from face_os.identity_state import IdentityState
        state = IdentityState()
        # Create reference with color cast
        ref = np.zeros((256, 256, 3), dtype=np.uint8)
        ref[:, :, 0] = 200  # Strong red channel
        ref[:, :, 1] = 100
        ref[:, :, 2] = 50
        state.set_anchor(ref)
        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        frame = ref.copy()
        state.update(frame, quality)
        intrinsic, _ = state.query_intrinsic(quality)
        # White-balanced albedo should have more equal channel means
        raw_means = np.mean(intrinsic.albedo, axis=(0, 1))
        # After white balance, channels should be closer to each other
        # than the raw 200:100:50 ratio
        ratio_max_min = raw_means.max() / (raw_means.min() + 1e-8)
        assert ratio_max_min < 4.0, f"White balance not applied: channel ratio {ratio_max_min}"


class TestAlbedoVsRGBDecoupling:
    """Albedo-based identity must be corrected in albedo space."""

    def test_anchor_correction_pulls_albedo_toward_anchor(self):
        """Anchor correction must move albedo closer to anchor_albedo."""
        from face_os.identity_state import IdentityState
        state = IdentityState()
        # Create a reference with texture
        ref = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            ref[i, :, 0] = int(100 + 80 * np.sin(i * np.pi / 128))
            ref[i, :, 1] = int(80 + 60 * np.cos(i * np.pi / 64))
            ref[i, :, 2] = int(60 + 40 * np.sin(i * np.pi / 32))
        state.set_anchor(ref)
        quality = np.ones((256, 256), dtype=np.float32) * 0.8

        # Update with different texture
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            frame[i, :, 0] = int(50 + 40 * np.cos(i * np.pi / 64))
            frame[i, :, 1] = int(150 + 70 * np.sin(i * np.pi / 32))
            frame[i, :, 2] = int(100 + 50 * np.cos(i * np.pi / 128))
        state.update(frame, quality)

        # Get raw intrinsic albedo (without anchor correction)
        raw_albedo = state._intrinsic_components.albedo.copy()
        raw_albedo = state._normalize_white_balance(raw_albedo)

        # Get anchor-corrected albedo
        corrected_albedo, _ = state.query_albedo(quality)

        # Anchor albedo
        anchor_albedo = state._anchor_albedo

        # Compute distances
        raw_dist = float(np.sqrt(np.mean((raw_albedo - anchor_albedo) ** 2)))
        corrected_dist = float(np.sqrt(np.mean((corrected_albedo - anchor_albedo) ** 2)))

        # Corrected albedo should be closer to anchor than raw
        assert corrected_dist < raw_dist, (
            f"Anchor correction didn't pull albedo closer: "
            f"raw_dist={raw_dist:.4f}, corrected_dist={corrected_dist:.4f}"
        )

    def test_albedo_anchor_correction_independent_of_rgb(self):
        """Albedo anchor correction uses anchor_albedo, not anchor_rgb."""
        from face_os.identity_state import IdentityState
        state = IdentityState()
        ref = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state.set_anchor(ref)
        # Verify anchor_albedo != anchor_rgb (white-balance changes it)
        anchor_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        anchor_albedo = state._anchor_albedo
        # They should be different because white balance was applied
        assert not np.allclose(anchor_rgb, anchor_albedo, atol=0.01), (
            "anchor_albedo should differ from anchor_rgb after white balance"
        )
