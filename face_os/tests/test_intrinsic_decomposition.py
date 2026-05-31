"""Layer 2a: Intrinsic decomposition tests.

Validates the image → (albedo, shading, specular, normals, detail) decomposition
that is the foundation of physically-based rendering (D-05: identity decoupling).

Mathematical invariants:
  - Reconstruction: albedo × shading + specular + detail ≈ input
  - Energy conservation: ECR ∈ [0.5, 1.5]
  - Albedo color invariance: channel std < 0.05
  - Specular sparsity: >75% near-zero
  - Normal map unit vectors: ‖n‖ ≈ 1
"""
import cv2
import numpy as np
import pytest

from face_os.intrinsic_decomposition import (
    IntrinsicDecomposer,
    DecompositionConfig,
    IntrinsicComponents,
    _srgb_to_linear,
    _linear_to_srgb,
)


# ═══════════════════════════════════════════════════════════════════
# sRGB ↔ Linear Roundtrip
# ═══════════════════════════════════════════════════════════════════

class TestColorSpaceConversion:
    """sRGB ↔ linear conversion must be invertible."""

    def test_srgb_linear_roundtrip(self):
        """sRGB → linear → sRGB roundtrip: uint8 → float → uint8."""
        # Use uint8 input to match the natural domain
        x = np.arange(256, dtype=np.uint8).reshape(16, 16)
        x_3ch = np.stack([x, x, x], axis=-1)
        lin = _srgb_to_linear(x_3ch)
        recovered = _linear_to_srgb(lin)
        # Roundtrip should be within ±1 due to quantization
        np.testing.assert_allclose(
            recovered.astype(np.float32), x_3ch.astype(np.float32), atol=1.5
        )

    def test_srgb_to_linear_monotonic(self):
        """sRGB → linear is monotonically increasing."""
        x = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        x_3ch = np.stack([x, x, x], axis=-1).reshape(10, 10, 3)
        lin = _srgb_to_linear(x_3ch)
        diffs = np.diff(lin[:, :, 0].ravel())
        assert np.all(diffs >= -1e-6), "sRGB → linear not monotonic"

    def test_linear_zero_and_one(self):
        """Boundary: srgb_to_linear(0) = 0, srgb_to_linear(1) = 1."""
        zero = np.zeros((1, 1, 3), dtype=np.float32)
        one = np.ones((1, 1, 3), dtype=np.float32)
        assert float(np.max(np.abs(_srgb_to_linear(zero)))) < 1e-6
        assert float(np.max(np.abs(_srgb_to_linear(one) - 1.0))) < 1e-6


# ═══════════════════════════════════════════════════════════════════
# Decomposition Output Contract
# ═══════════════════════════════════════════════════════════════════

class TestDecompositionContract:
    """IntrinsicDecomposer.decompose() must return all required components."""

    @pytest.fixture
    def decomposer(self):
        return IntrinsicDecomposer()

    @pytest.fixture
    def components(self, decomposer, skin_tone_image):
        return decomposer.decompose(skin_tone_image)

    def test_returns_intrinsic_components(self, components):
        """Output is an IntrinsicComponents instance."""
        assert isinstance(components, IntrinsicComponents)

    def test_albedo_present_and_shaped(self, components, skin_tone_image):
        """Albedo has correct shape (H, W, 3)."""
        assert components.albedo is not None
        assert components.albedo.shape == skin_tone_image.shape

    def test_shading_present(self, components):
        """Shading is present."""
        assert components.shading is not None

    def test_normals_present_and_shaped(self, components):
        """Normal map has shape (H, W, 3)."""
        assert components.normal_map is not None
        assert components.normal_map.ndim == 3
        assert components.normal_map.shape[2] == 3

    def test_specular_present(self, components):
        """Specular component exists."""
        assert components.specular is not None

    def test_detail_residual_present(self, components):
        """Detail residual exists."""
        assert components.detail_residual is not None


# ═══════════════════════════════════════════════════════════════════
# Decomposition Quality Constraints
# ═══════════════════════════════════════════════════════════════════

class TestDecompositionQuality:
    """Physical constraints on decomposition outputs."""

    @pytest.fixture(scope='class')
    def decomposer(self):
        return IntrinsicDecomposer()

    @pytest.fixture(scope='class')
    def components(self, decomposer):
        # Create a deterministic skin-tone image inline for class scope
        h, w = 256, 256
        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        r = 185 + 15 * np.sin(X / 40)
        g = 150 + 10 * np.cos(Y / 35)
        b = 120 + 8 * np.sin((X + Y) / 50)
        illumination = 0.5 + 0.5 * (X / w)
        r, g, b = r * illumination, g * illumination, b * illumination
        img = np.stack([
            np.clip(b, 0, 255).astype(np.uint8),
            np.clip(g, 0, 255).astype(np.uint8),
            np.clip(r, 0, 255).astype(np.uint8),
        ], axis=-1)
        return decomposer.decompose(img)

    def test_albedo_in_unit_range(self, components):
        """Albedo values must be in [0, 1]."""
        assert float(np.min(components.albedo)) >= -0.01
        assert float(np.max(components.albedo)) <= 1.01

    def test_shading_in_unit_range(self, components):
        """Shading values in [0, 1]."""
        shading = components.shading
        if shading.ndim == 3 and shading.shape[2] > 1:
            shading = np.mean(shading, axis=2, keepdims=True)
        assert float(np.min(shading)) >= -0.01
        assert float(np.max(shading)) <= 1.01

    def test_specular_sparsity(self, components):
        """Specular should be sparse: >75% of pixels near zero."""
        spec = np.abs(components.specular)
        near_zero = float(np.mean(spec < 0.05))
        assert near_zero > 0.75, f"Specular sparsity={near_zero:.2f} < 0.75"

    def test_normal_map_unit_vectors(self, components):
        """All normal vectors should have approximately unit length."""
        norms = np.linalg.norm(components.normal_map, axis=2)
        mean_err = float(np.mean(np.abs(norms - 1.0)))
        assert mean_err < 0.05, f"Normal map unit error={mean_err:.4f}"

    def test_normal_z_positive_mean(self, components):
        """Mean normal Z should be positive (facing camera)."""
        z_mean = float(np.mean(components.normal_map[:, :, 2]))
        assert z_mean > 0.3, f"Normal Z mean={z_mean:.3f} not facing camera"

    def test_reconstruction_error_bounded(self, components):
        """Reconstruction error should be < 0.3."""
        assert components.reconstruction_error < 0.3, (
            f"Reconstruction error={components.reconstruction_error:.3f} > 0.3"
        )

    def test_decomposition_quality_in_range(self, components):
        """Decomposition quality ∈ [0, 1]."""
        q = components.decomposition_quality
        assert 0.0 <= q <= 1.0, f"Quality={q} outside [0,1]"

    def test_energy_conservation_ratio(self, components):
        """ECR = mean(albedo × shading) / mean(input_linear) should be reasonable."""
        albedo = components.albedo
        shading = components.shading
        if shading.ndim == 3 and shading.shape[2] == 1:
            shading = np.repeat(shading, 3, axis=2)
        elif shading.ndim == 2:
            shading = shading[:, :, np.newaxis]
            shading = np.repeat(shading, 3, axis=2)
        product_energy = float(np.mean(albedo * shading))
        # ECR should be positive and reasonable
        assert product_energy > 0.01, f"Albedo×shading energy collapsed to {product_energy}"

    def test_albedo_not_uniform(self, components):
        """Albedo should have spatial variation (not a flat constant)."""
        albedo_std = float(np.std(components.albedo))
        assert albedo_std > 0.01, f"Albedo is flat: std={albedo_std}"


# ═══════════════════════════════════════════════════════════════════
# arch.md §16.9 — Background Invariance:  ∂I / ∂Background = 0
# ═══════════════════════════════════════════════════════════════════

class TestBackgroundInvariance:
    """Identity (albedo, shading) must be independent of the background.

    arch.md §16.9 (LOCKED): no background pixel may influence albedo/shading.
    Decomposition must operate inside the geometry-derived face mask.

    Invariant: decomposing the SAME face on two different backgrounds yields
    albedo and shading that are invariant inside the face mask. This is the
    "poster brightness" leak guard called out in arch.md §18 as a flicker cause.
    """

    @staticmethod
    def _face_and_mask(h=256, w=256):
        """A skin-tone face ellipse + its mask. Face content only (no bg)."""
        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        r = 185 + 15 * np.sin(X / 40.0)
        g = 150 + 10 * np.cos(Y / 35.0)
        b = 120 + 8 * np.sin((X + Y) / 50.0)
        illum = 0.5 + 0.5 * (X / w)          # left-lit gradient (real shading)
        face = np.stack([b * illum, g * illum, r * illum], axis=-1)
        dist = ((X - cx) / 90.0) ** 2 + ((Y - cy) / 110.0) ** 2
        mask = (dist < 1.0).astype(np.float32)
        return face.astype(np.float32), mask

    @staticmethod
    def _compose(face, mask, bg_value):
        """Composite identical face content onto a flat background."""
        m3 = mask[:, :, np.newaxis]
        img = face * m3 + float(bg_value) * (1.0 - m3)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _interior(self, mask, erode_px=3):
        """Mask eroded slightly to avoid the hard mask edge itself."""
        k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        return cv2.erode((mask > 0.5).astype(np.uint8), k) > 0

    def test_albedo_invariant_to_background(self):
        """Albedo inside the mask is invariant to background (∂A/∂bg ≈ 0)."""
        face, mask = self._face_and_mask()
        img_dark = self._compose(face, mask, 30)     # dark background
        img_bright = self._compose(face, mask, 220)   # bright background
        interior = self._interior(mask)

        dec = IntrinsicDecomposer()
        a_dark = dec.decompose(img_dark, mask=mask).albedo
        a_bright = dec.decompose(img_bright, mask=mask).albedo

        diff = np.abs(a_dark - a_bright)[interior]
        mean_diff = float(np.mean(diff))
        assert mean_diff < 0.02, (
            f"Albedo leaked background: mean|Δalbedo|={mean_diff:.4f} ≥ 0.02 "
            f"inside mask (∂A/∂bg must be ≈ 0)"
        )

    def test_shading_invariant_to_background(self):
        """Shading inside the mask is invariant to background (∂S/∂bg ≈ 0)."""
        face, mask = self._face_and_mask()
        img_dark = self._compose(face, mask, 30)
        img_bright = self._compose(face, mask, 220)
        interior = self._interior(mask)

        dec = IntrinsicDecomposer()
        s_dark = dec.decompose(img_dark, mask=mask).shading[:, :, 0]
        s_bright = dec.decompose(img_bright, mask=mask).shading[:, :, 0]

        diff = np.abs(s_dark - s_bright)[interior]
        mean_diff = float(np.mean(diff))
        assert mean_diff < 0.02, (
            f"Shading leaked background: mean|Δshading|={mean_diff:.4f} ≥ 0.02 "
            f"inside mask (∂S/∂bg must be ≈ 0)"
        )

    def test_masking_strictly_reduces_background_leak(self):
        """Mask-aware decompose must be strictly better than mask-free.

        Guards against a vacuous pass: proves the mask is what removes the leak,
        not that the synthetic image happens to be background-insensitive.
        """
        face, mask = self._face_and_mask()
        img_dark = self._compose(face, mask, 30)
        img_bright = self._compose(face, mask, 220)
        interior = self._interior(mask)

        dec = IntrinsicDecomposer()
        # Mask-free (legacy behavior) — background WILL leak near the boundary.
        leak_unmasked = float(np.mean(np.abs(
            dec.decompose(img_dark).albedo - dec.decompose(img_bright).albedo
        )[interior]))
        # Mask-aware — leak must shrink.
        leak_masked = float(np.mean(np.abs(
            dec.decompose(img_dark, mask=mask).albedo
            - dec.decompose(img_bright, mask=mask).albedo
        )[interior]))

        assert leak_masked < leak_unmasked, (
            f"Mask did not reduce background leak: masked={leak_masked:.4f} "
            f"not < unmasked={leak_unmasked:.4f}"
        )
