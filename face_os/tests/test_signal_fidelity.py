"""Layer 3: Signal fidelity tests.

Validates the HF preservation chain that addresses D-01:
  - USM sharpening
  - Source HF re-injection
  - Detail residual injection
  - Frequency retention metric ≥ 0.6
  - Contrast preservation

These are unit tests on the signal-processing primitives, not the
full pipeline. Integration-level fidelity tests are in test_integration.py.
"""
import numpy as np
import pytest
import cv2


# ═══════════════════════════════════════════════════════════════════
# USM Sharpening
# ═══════════════════════════════════════════════════════════════════

class TestUSMSharpening:
    """Unsharp mask must increase high-frequency content."""

    @pytest.fixture
    def sharpen(self):
        """Import face_enhance._sharpen."""
        from face_os.face_enhance import _sharpen
        return _sharpen

    def test_usm_increases_sharpness(self, sharpen):
        """USM output has higher Laplacian variance than input."""
        # Create a slightly blurry image
        img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 1.5)

        sharpened = sharpen(img, amount=1.5, radius=1.0)

        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_out = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sharp_in = float(np.var(cv2.Laplacian(gray_in, cv2.CV_32F)))
        sharp_out = float(np.var(cv2.Laplacian(gray_out, cv2.CV_32F)))
        assert sharp_out > sharp_in, (
            f"USM failed: sharp_in={sharp_in:.1f}, sharp_out={sharp_out:.1f}"
        )

    def test_usm_output_valid_range(self, sharpen):
        """USM output stays in [0, 255] uint8."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = sharpen(img, amount=2.0, radius=0.8)
        assert result.dtype == np.uint8
        assert int(np.min(result)) >= 0
        assert int(np.max(result)) <= 255

    def test_usm_preserves_shape(self, sharpen):
        """Output shape matches input."""
        img = np.zeros((100, 80, 3), dtype=np.uint8)
        result = sharpen(img, amount=1.0, radius=1.0)
        assert result.shape == img.shape

    def test_usm_zero_amount_identity(self, sharpen):
        """amount=0 → output ≈ input."""
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        result = sharpen(img, amount=0.0, radius=1.0)
        # Should be very close to input
        diff = float(np.mean(np.abs(result.astype(np.float32) - img.astype(np.float32))))
        assert diff < 1.0, f"Zero-amount USM changed image by {diff:.1f}"


# ═══════════════════════════════════════════════════════════════════
# HF Re-injection
# ═══════════════════════════════════════════════════════════════════

class TestHFReinjection:
    """Source HF re-injection must boost high-frequency content."""

    @pytest.fixture
    def reinject(self):
        """Import pipeline._reinject_source_hf as standalone function."""
        from face_os.pipeline import FaceOSPipeline
        pipe = FaceOSPipeline.__new__(FaceOSPipeline)
        return pipe._reinject_source_hf

    def test_reinject_increases_hf(self, reinject):
        """HF re-injection increases Laplacian variance."""
        # Source with texture
        source = np.random.randint(80, 180, (128, 128, 3), dtype=np.uint8)
        # Rendered: smoothed version (simulating canonical warp loss)
        rendered = cv2.GaussianBlur(source, (7, 7), 2.0)

        result = reinject(rendered, source, face_mask=None, strength=0.5)

        gray_ren = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_out = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sharp_ren = float(np.var(cv2.Laplacian(gray_ren, cv2.CV_32F)))
        sharp_out = float(np.var(cv2.Laplacian(gray_out, cv2.CV_32F)))
        assert sharp_out > sharp_ren, (
            f"HF reinject failed: sharp_ren={sharp_ren:.1f}, sharp_out={sharp_out:.1f}"
        )

    def test_reinject_with_none_mask(self, reinject):
        """mask=None → full-frame re-injection (no crash)."""
        rendered = np.full((64, 64, 3), 128, dtype=np.uint8)
        source = np.random.randint(100, 200, (64, 64, 3), dtype=np.uint8)
        result = reinject(rendered, source, face_mask=None, strength=0.3)
        assert result.shape == rendered.shape
        assert result.dtype == np.uint8

    def test_reinject_preserves_dc(self, reinject):
        """Mean brightness should be preserved within ±15%."""
        source = np.random.randint(80, 180, (128, 128, 3), dtype=np.uint8)
        rendered = cv2.GaussianBlur(source, (5, 5), 1.5)
        result = reinject(rendered, source, face_mask=None, strength=0.5)

        mean_in = float(np.mean(rendered))
        mean_out = float(np.mean(result))
        ratio = mean_out / max(mean_in, 1.0)
        assert 0.85 < ratio < 1.15, (
            f"DC shift: mean_in={mean_in:.1f}, mean_out={mean_out:.1f}, ratio={ratio:.3f}"
        )

    def test_reinject_with_empty_mask_noop(self, reinject):
        """mask with max < 0.01 → no change."""
        rendered = np.full((64, 64, 3), 128, dtype=np.uint8)
        source = np.random.randint(100, 200, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        result = reinject(rendered, source, face_mask=mask, strength=0.5)
        np.testing.assert_array_equal(result, rendered)


# ═══════════════════════════════════════════════════════════════════
# Frequency Retention Metric
# ═══════════════════════════════════════════════════════════════════

class TestFrequencyRetention:
    """Frequency retention = output_hf / source_hf metric."""

    def _freq_ret(self, output, source):
        """Compute frequency retention as in audit.py."""
        gray_o = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_s = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hf_o = float(np.var(cv2.Laplacian(gray_o, cv2.CV_32F)))
        hf_s = float(np.var(cv2.Laplacian(gray_s, cv2.CV_32F)))
        if hf_s < 1e-6:
            return 1.0
        return hf_o / hf_s

    def test_identity_gives_one(self):
        """freq_ret(x, x) = 1.0."""
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        assert abs(self._freq_ret(img, img) - 1.0) < 1e-6

    def test_blurred_gives_less_than_one(self):
        """freq_ret(blur(x), x) < 1.0."""
        img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(img, (7, 7), 2.0)
        fr = self._freq_ret(blurred, img)
        assert fr < 1.0, f"Blurred freq_ret={fr:.3f} should be < 1.0"

    def test_sharpened_gives_more_than_one(self):
        """freq_ret(sharpen(x), x) > 1.0."""
        from face_os.face_enhance import _sharpen
        img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0.8)  # Slight blur first
        sharpened = _sharpen(img, amount=2.0, radius=1.0)
        fr = self._freq_ret(sharpened, img)
        assert fr > 1.0, f"Sharpened freq_ret={fr:.3f} should be > 1.0"


# ═══════════════════════════════════════════════════════════════════
# Postprocess Chain
# ═══════════════════════════════════════════════════════════════════

class TestPostprocessChain:
    """The full postprocess chain must produce valid output."""

    def test_postprocess_produces_valid_output(self):
        """_postprocess_rendered_crop returns valid uint8 BGR."""
        from face_os.pipeline import FaceOSPipeline
        pipe = FaceOSPipeline.__new__(FaceOSPipeline)
        # Need to initialize minimal state for postprocess
        from face_os.photometric import reset_photometric_lock
        reset_photometric_lock()

        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        result = pipe._postprocess_rendered_crop(img, face_mask=None)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_contrast_preserved(self):
        """Contrast (std of grayscale) should not collapse."""
        from face_os.pipeline import FaceOSPipeline
        from face_os.photometric import reset_photometric_lock
        reset_photometric_lock()

        pipe = FaceOSPipeline.__new__(FaceOSPipeline)
        img = np.random.randint(30, 220, (128, 128, 3), dtype=np.uint8)
        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast_in = float(np.std(gray_in))

        result = pipe._postprocess_rendered_crop(img, face_mask=None)
        gray_out = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        contrast_out = float(np.std(gray_out))

        assert contrast_out > contrast_in * 0.5, (
            f"Contrast collapsed: {contrast_in:.1f} → {contrast_out:.1f}"
        )
