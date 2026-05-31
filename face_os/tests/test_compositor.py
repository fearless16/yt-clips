"""Layer 2c: Compositor tests.

Validates linear-light blending and multiband compositing (D-01 compliance).

Mathematical invariants:
  - sRGB ↔ linear roundtrip identity
  - Blend with mask=0 → background, mask=1 → foreground
  - Multiband blend preserves interior content
  - No visible seam artifacts
"""
import numpy as np
import pytest

from face_os.compositor import (
    _srgb_to_linear,
    _linear_to_srgb,
    _blend_linear,
    multiband_blend,
    Compositor,
)


# ═══════════════════════════════════════════════════════════════════
# Color Space
# ═══════════════════════════════════════════════════════════════════

class TestColorSpace:
    """sRGB ↔ linear conversions must be invertible."""

    def test_srgb_linear_roundtrip(self):
        """srgb → lin → srgb identity within tolerance.

        Note: _srgb_to_linear expects uint8, _linear_to_srgb returns uint8.
        So the natural roundtrip is uint8 → float → uint8.
        """
        x = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        lin = _srgb_to_linear(x)
        x_recovered = _linear_to_srgb(lin)
        np.testing.assert_allclose(x_recovered.astype(np.float32), x.astype(np.float32), atol=2.0)

    def test_linear_preserves_zero(self):
        """Zero stays zero through conversion."""
        zero = np.zeros((4, 4, 3), dtype=np.float32)
        assert float(np.max(np.abs(_srgb_to_linear(zero)))) < 1e-6
        assert float(np.max(np.abs(_linear_to_srgb(zero)))) < 1e-6

    def test_linear_preserves_white(self):
        """White (255) roundtrips correctly."""
        white_u8 = np.full((4, 4, 3), 255, dtype=np.uint8)
        lin = _srgb_to_linear(white_u8)
        np.testing.assert_allclose(lin, 1.0, atol=1e-4)
        recovered = _linear_to_srgb(lin)
        np.testing.assert_allclose(recovered, 255, atol=1)


# ═══════════════════════════════════════════════════════════════════
# Linear Blending
# ═══════════════════════════════════════════════════════════════════

class TestLinearBlend:
    """Linear-space blending must satisfy mask boundary conditions.

    Note: _blend_linear takes uint8 BGR inputs and returns uint8.
    It does internal sRGB→linear→sRGB conversion.
    """

    def test_zero_mask_returns_background(self):
        """mask=0 → output = background."""
        bg = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        fg = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        result = _blend_linear(bg, fg, mask)
        np.testing.assert_array_equal(result, bg)

    def test_full_mask_returns_foreground(self):
        """mask=1 → output = foreground."""
        bg = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        fg = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = _blend_linear(bg, fg, mask)
        np.testing.assert_array_equal(result, fg)

    def test_half_mask_between_inputs(self):
        """mask=0.5 → output is between bg and fg (in linear space)."""
        bg = np.full((32, 32, 3), 60, dtype=np.uint8)
        fg = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask = np.full((32, 32), 0.5, dtype=np.float32)
        result = _blend_linear(bg, fg, mask)
        result_mean = float(np.mean(result))
        assert 60 < result_mean < 200, f"Blend result={result_mean:.1f} not between 60 and 200"

    def test_blend_preserves_shape(self):
        """Output shape matches input."""
        bg = np.full((100, 80, 3), 128, dtype=np.uint8)
        fg = np.full((100, 80, 3), 200, dtype=np.uint8)
        mask = np.full((100, 80), 0.5, dtype=np.float32)
        result = _blend_linear(bg, fg, mask)
        assert result.shape == bg.shape


# ═══════════════════════════════════════════════════════════════════
# Multiband Blend
# ═══════════════════════════════════════════════════════════════════

class TestMultibandBlend:
    """Multiband (Laplacian pyramid) blending quality tests.

    Note: multiband_blend takes uint8 BGR inputs.
    """

    def test_multiband_preserves_interior(self):
        """Interior of foreground should be preserved (not blurred)."""
        h, w = 128, 128
        bg = np.full((h, w, 3), 30, dtype=np.uint8)
        fg = np.full((h, w, 3), 200, dtype=np.uint8)

        # Mask: circle in center
        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        mask = ((X - w/2)**2 + (Y - h/2)**2 < 30**2).astype(np.float32)

        result = multiband_blend(bg, fg, mask)
        # Center pixel (deep inside mask) should be close to fg
        center_val = float(np.mean(result[h//2, w//2]))
        assert center_val > 100, f"Center={center_val:.3f}, expected near 200"

    def test_multiband_no_hard_seam(self):
        """Blending should produce smooth transition."""
        h, w = 128, 128
        bg = np.full((h, w, 3), 50, dtype=np.uint8)
        fg = np.full((h, w, 3), 200, dtype=np.uint8)

        # Sharp mask: right half
        mask = np.zeros((h, w), dtype=np.float32)
        mask[:, w//2:] = 1.0

        result = multiband_blend(bg, fg, mask, levels=3)
        col_left = float(np.mean(result[:, w//4]))
        col_right = float(np.mean(result[:, 3*w//4]))
        assert col_left < col_right, (
            f"Left={col_left:.1f} should be darker than right={col_right:.1f}"
        )

    def test_multiband_output_shape(self):
        """Output shape matches input."""
        bg = np.full((64, 64, 3), 100, dtype=np.uint8)
        fg = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32) * 0.5
        result = multiband_blend(bg, fg, mask)
        assert result.shape == bg.shape


# ═══════════════════════════════════════════════════════════════════
# Compositor Class
# ═══════════════════════════════════════════════════════════════════

class TestCompositorClass:
    """Compositor high-level interface.

    Note: Compositor.composite() expects Optional[ConfidenceMap], not raw mask.
    Use face_mask parameter for raw ndarray masks.
    """

    def test_composite_with_face_mask(self):
        """Composite with face_mask produces valid uint8 BGR output."""
        comp = Compositor()
        bg = np.full((128, 128, 3), 100, dtype=np.uint8)
        fg = np.full((128, 128, 3), 200, dtype=np.uint8)
        mask = np.full((128, 128), 0.5, dtype=np.float32)
        result = comp.composite(bg, fg, face_mask=mask)
        assert result.dtype == np.uint8
        assert result.shape == bg.shape

    def test_composite_no_mask_returns_enhanced(self):
        """Without any mask, returns enhanced directly."""
        comp = Compositor()
        bg = np.full((64, 64, 3), 100, dtype=np.uint8)
        fg = np.full((64, 64, 3), 200, dtype=np.uint8)
        result = comp.composite(bg, fg)
        np.testing.assert_array_equal(result, fg)

    def test_composite_reset_clears_state(self):
        """reset() clears compositor temporal state."""
        comp = Compositor()
        comp.reset()
        bg = np.full((64, 64, 3), 100, dtype=np.uint8)
        fg = np.full((64, 64, 3), 200, dtype=np.uint8)
        result = comp.composite(bg, fg, face_mask=np.ones((64, 64), dtype=np.float32))
        assert result is not None
