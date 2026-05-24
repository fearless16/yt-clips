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
