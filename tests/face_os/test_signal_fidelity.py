"""Layer 3: Signal fidelity tests.

Validates the HF preservation chain that addresses D-01:
  - USM sharpening (including adaptive)
  - Source HF re-injection
  - Detail residual injection
  - Frequency retention metric ≥ 0.6
  - Contrast preservation (including adaptive enhancement)
  - Energy conservation on alpha path

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


# ═══════════════════════════════════════════════════════════════════
# A-1: Adaptive Sharpening
# ═══════════════════════════════════════════════════════════════════

class TestAdaptiveSharpening:
    """Adaptive USM scales amount based on content sharpness deficit."""

    def _sharpness(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.var(cv2.Laplacian(gray, cv2.CV_32F)))

    def test_blurry_gets_more_sharpening_than_sharp(self):
        """Blurred input → higher relative sharpness improvement than sharp input."""
        from face_os.face_enhance import adaptive_sharpen

        sharp = np.random.randint(40, 220, (128, 128, 3), dtype=np.uint8)
        blurry = cv2.GaussianBlur(sharp, (9, 9), 3.0)

        result_blurry = adaptive_sharpen(blurry, face_mask=None)
        result_sharp = adaptive_sharpen(sharp, face_mask=None)

        sb_in, sb_out = self._sharpness(blurry), self._sharpness(result_blurry)
        ss_in, ss_out = self._sharpness(sharp), self._sharpness(result_sharp)

        rel_blurry = (sb_out - sb_in) / max(sb_in, 1e-6)
        rel_sharp = (ss_out - ss_in) / max(ss_in, 1e-6)

        assert rel_blurry > rel_sharp, (
            f"Adaptive failed: rel_blurry={rel_blurry:.2f}, rel_sharp={rel_sharp:.2f}"
        )

    def test_already_sharp_receives_reduced_amount(self):
        """Sharp input gets ≤ base amount (avoids over-sharpening)."""
        from face_os.face_enhance import adaptive_sharpen, _sharpen

        img = np.random.randint(40, 220, (128, 128, 3), dtype=np.uint8)
        result = adaptive_sharpen(img, face_mask=None)

        sharp_in = self._sharpness(img)
        sharp_out = self._sharpness(result)

        max_expected = self._sharpness(_sharpen(img, amount=2.0, radius=1.0))
        assert sharp_out <= max_expected + 5.0, (
            f"Over-sharpened: out={sharp_out:.1f} > max_expected={max_expected:.1f}"
        )
        assert sharp_out >= sharp_in, "Sharpening must not reduce sharpness"

    def test_flat_frame_no_crash(self):
        """Constant flat frame → doesn't crash, returns valid output."""
        from face_os.face_enhance import adaptive_sharpen

        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = adaptive_sharpen(flat, face_mask=None)
        assert result.dtype == np.uint8
        assert result.shape == flat.shape
        assert int(np.min(result)) >= 0
        assert int(np.max(result)) <= 255

    def test_output_valid_range(self):
        """Output always uint8 [0, 255]."""
        from face_os.face_enhance import adaptive_sharpen

        img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        result = adaptive_sharpen(img, face_mask=None)
        assert result.dtype == np.uint8
        assert int(np.min(result)) >= 0
        assert int(np.max(result)) <= 255

    def test_sharpness_improves_on_blurred(self):
        """Any blurred input must see sharpness increase."""
        from face_os.face_enhance import adaptive_sharpen

        img = np.random.randint(40, 220, (128, 128, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(img, (7, 7), 2.0)
        result = adaptive_sharpen(blurred, face_mask=None)
        assert self._sharpness(result) > self._sharpness(blurred), (
            "Adaptive sharpening did not increase sharpness on blurred input"
        )

    def test_mask_aware_sharpening(self):
        """Masked region gets sharpened; outside region unchanged."""
        from face_os.face_enhance import adaptive_sharpen

        h, w = 128, 128
        img = np.random.randint(40, 220, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[32:96, 32:96] = 1.0

        result = adaptive_sharpen(img, face_mask=mask)
        outside_before = img[mask < 0.1]
        outside_after = result[mask < 0.1]
        np.testing.assert_array_equal(outside_before, outside_after,
            "Pixels outside mask modified")


# ═══════════════════════════════════════════════════════════════════
# A-2: Contrast Enhancement
# ═══════════════════════════════════════════════════════════════════

class TestContrastEnhancement:
    """Local contrast enhancement must improve low-contrast frames."""

    def _michelson_contrast(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        rng = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        mid = float(np.percentile(gray, 95) + np.percentile(gray, 5)) / 2.0
        mid = max(mid, 1.0)
        return rng / mid

    def test_low_contrast_increases(self):
        """Very low-contrast frame gets contrast boost."""
        from face_os.face_enhance import enhance_contrast

        low_contrast = np.full((128, 128, 3), 128, dtype=np.uint8)
        low_contrast = np.clip(
            low_contrast.astype(np.float32) + np.random.randn(128, 128, 3) * 10,
            0, 255
        ).astype(np.uint8)

        result = enhance_contrast(low_contrast, face_mask=None)
        contrast_in = self._michelson_contrast(low_contrast)
        contrast_out = self._michelson_contrast(result)
        assert contrast_out > contrast_in, (
            f"Contrast not improved: {contrast_in:.3f} → {contrast_out:.3f}"
        )

    def test_high_contrast_not_degraded(self):
        """Already-good contrast isn't substantially degraded."""
        from face_os.face_enhance import enhance_contrast

        high_contrast = np.random.randint(30, 220, (128, 128, 3), dtype=np.uint8)
        result = enhance_contrast(high_contrast, face_mask=None)
        contrast_in = self._michelson_contrast(high_contrast)
        contrast_out = self._michelson_contrast(result)
        assert contrast_out > contrast_in * 0.7, (
            f"High contrast degraded: {contrast_in:.3f} → {contrast_out:.3f}"
        )

    def test_output_valid_range(self):
        """Output stays uint8 [0, 255]."""
        from face_os.face_enhance import enhance_contrast

        img = np.random.randint(10, 240, (96, 96, 3), dtype=np.uint8)
        result = enhance_contrast(img, face_mask=None)
        assert result.dtype == np.uint8
        assert int(np.min(result)) >= 0
        assert int(np.max(result)) <= 255

    def test_preserves_shape(self):
        """Output shape matches input."""
        from face_os.face_enhance import enhance_contrast

        img = np.zeros((100, 80, 3), dtype=np.uint8)
        result = enhance_contrast(img, face_mask=None)
        assert result.shape == img.shape

    def test_mask_aware(self):
        """Outside-mask pixels unchanged (within LAB→BGR roundtrip tolerance)."""
        from face_os.face_enhance import enhance_contrast

        h, w = 128, 128
        img = np.random.randint(30, 200, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[32:96, 32:96] = 1.0

        result = enhance_contrast(img, face_mask=mask)
        outside = mask < 0.1
        diff = np.abs(img[outside].astype(np.float32) - result[outside].astype(np.float32))
        max_diff = float(np.max(diff))
        assert max_diff <= 10.0, (
            f"Contrast enhancement leaked outside mask: max_diff={max_diff:.0f}"
        )


# ═══════════════════════════════════════════════════════════════════
# A-3: Alpha Path Energy Conservation
# ═══════════════════════════════════════════════════════════════════

class TestAlphaPathECR:
    """Energy conservation must normalize the alpha composite path."""

    def test_ecr_on_too_bright_composite(self):
        """Bright composite gets scaled down (ECR ∈ [0.5, 1.5])."""
        from face_os.face_enhance import apply_energy_conservation

        source = np.full((64, 64, 3), 100, dtype=np.uint8)
        composite = np.full((64, 64, 3), 200, dtype=np.uint8)
        result = apply_energy_conservation(composite, source, energy_limit=0.95)

        ecr = float(np.mean(result)) / max(float(np.mean(source)), 1.0)
        assert ecr <= 0.95 + 0.05, f"ECR too high: {ecr:.3f}"
        mean_in = float(np.mean(composite))
        mean_out = float(np.mean(result))
        assert mean_out < mean_in, (
            f"Bright composite not scaled down: {mean_in:.1f} → {mean_out:.1f}"
        )

    def test_ecr_on_too_dark_composite(self):
        """Dark composite gets scaled up."""
        from face_os.face_enhance import apply_energy_conservation

        source = np.full((64, 64, 3), 150, dtype=np.uint8)
        composite = np.full((64, 64, 3), 30, dtype=np.uint8)
        result = apply_energy_conservation(composite, source, energy_limit=0.95)

        mean_out = float(np.mean(result))
        mean_in = float(np.mean(composite))
        assert mean_out > mean_in, (
            f"Dark composite not scaled up: {mean_in:.1f} → {mean_out:.1f}"
        )

    def test_balanced_composite_unchanged(self):
        """Energy-balanced composite stays nearly identical (ratio within [0.5, 0.95])."""
        from face_os.face_enhance import apply_energy_conservation

        source = np.random.randint(80, 180, (96, 96, 3), dtype=np.uint8)
        composite = (source.astype(np.float32) * 0.8).clip(0, 255).astype(np.uint8)
        result = apply_energy_conservation(composite, source, energy_limit=1.1)

        diff = float(np.mean(np.abs(result.astype(np.float32) - composite.astype(np.float32))))
        assert diff < 2.0, f"Balanced composite altered by ECR: mean_diff={diff:.1f}"

    def test_ecr_within_range(self):
        """ECR always in [0.5, 1.5] after normalization."""
        from face_os.face_enhance import apply_energy_conservation

        rng = np.random.default_rng(42)
        for _ in range(20):
            source = rng.integers(30, 200, (64, 64, 3), dtype=np.uint8)
            composite = rng.integers(10, 240, (64, 64, 3), dtype=np.uint8)
            result = apply_energy_conservation(composite, source, energy_limit=0.95)

            src_mean = float(np.mean(source))
            out_mean = float(np.mean(result))
            ecr = out_mean / max(src_mean, 1e-8)
            assert 0.4 < ecr < 1.6, (
                f"ECR out of range: {ecr:.3f} (src={src_mean:.1f}, out={out_mean:.1f})"
            )

    def test_output_valid_range(self):
        """Output always uint8 [0, 255]."""
        from face_os.face_enhance import apply_energy_conservation

        source = np.random.randint(50, 180, (64, 64, 3), dtype=np.uint8)
        composite = np.random.randint(5, 250, (64, 64, 3), dtype=np.uint8)
        result = apply_energy_conservation(composite, source, energy_limit=0.95)

        assert result.dtype == np.uint8
        assert int(np.min(result)) >= 0
        assert int(np.max(result)) <= 255

    def test_mask_aware_ecr(self):
        """ECR only scales pixels inside mask; outside unchanged."""
        from face_os.face_enhance import apply_energy_conservation

        h, w = 96, 96
        source = np.random.randint(50, 180, (h, w, 3), dtype=np.uint8)
        composite = source.astype(np.float32) * 2.0
        composite = np.clip(composite, 0, 255).astype(np.uint8)

        mask = np.zeros((h, w), dtype=np.float32)
        mask[24:72, 24:72] = 1.0

        result = apply_energy_conservation(composite, source, face_mask=mask, energy_limit=0.95)
        outside = mask < 0.05
        np.testing.assert_array_equal(composite[outside], result[outside],
            "ECR leaked outside mask")


# ═══════════════════════════════════════════════════════════════════
# A-4: Flicker Reduction — Adaptive Chroma + Luma Temporal Filtering
# ═══════════════════════════════════════════════════════════════════

class TestFlickerReduction:
    """Temporal chroma+luma smoothing must reduce frame-to-frame variance."""

    def test_chroma_smoothed_across_frames(self):
        """Frames with systematic brightness drift get smoothed toward EMA."""
        from face_os.photometric import photometric_lock, reset_photometric_lock
        reset_photometric_lock()

        mean_values = [120.0, 140.0, 160.0, 140.0, 120.0, 100.0, 120.0, 140.0]
        frames = []
        results = []
        for mv in mean_values:
            frame = np.full((64, 64, 3), mv, dtype=np.uint8)
            frames.append(frame)
            results.append(photometric_lock(frame))

        raw_means = [float(np.mean(f)) for f in frames]
        locked_means = [float(np.mean(r)) for r in results]

        raw_range = max(raw_means) - min(raw_means)
        locked_range = max(locked_means) - min(locked_means)
        assert locked_range < raw_range, (
            f"Photometric lock did not smooth drift: raw_range={raw_range:.1f}, locked_range={locked_range:.1f}"
        )

    def test_chroma_channels_stabilized(self):
        """U and V channel mean drift decreases after photometric lock sequence."""
        from face_os.photometric import photometric_lock, reset_photometric_lock
        reset_photometric_lock()

        frames_u_means = []
        results_u_means = []
        for t in range(10):
            b = 100 + int(40 * np.sin(t * 0.8))
            g = 150 + int(30 * np.cos(t * 1.2))
            r = 180 + int(35 * np.sin(t * 0.5))
            frame = np.full((64, 64, 3), [b, g, r], dtype=np.uint8)
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV).astype(np.float32)
            frames_u_means.append(float(yuv[:, :, 1].mean()))
            result = photometric_lock(frame)
            yuv_r = cv2.cvtColor(result, cv2.COLOR_BGR2YUV).astype(np.float32)
            results_u_means.append(float(yuv_r[:, :, 1].mean()))

        raw_u_range = max(frames_u_means) - min(frames_u_means)
        locked_u_range = max(results_u_means) - min(results_u_means)
        assert locked_u_range <= raw_u_range, (
            f"Chroma U not stabilized: raw_range={raw_u_range:.1f}, locked_range={locked_u_range:.1f}"
        )

    def test_output_preserves_dimensions(self):
        """Photometric lock preserves shape and dtype."""
        from face_os.photometric import photometric_lock, reset_photometric_lock
        reset_photometric_lock()

        img = np.random.randint(50, 200, (128, 96, 3), dtype=np.uint8)
        result = photometric_lock(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_reset_clears_state(self):
        """reset_photometric_lock() clears prev_luminance."""
        from face_os.photometric import photometric_lock, reset_photometric_lock
        reset_photometric_lock()

        img1 = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        img2 = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)

        r1 = photometric_lock(img1)
        reset_photometric_lock()
        r2 = photometric_lock(img2)

        diff_after_reset = float(np.mean(np.abs(r2.astype(np.float32) - img2.astype(np.float32))))
        diff_no_reset = float(np.mean(np.abs(r2.astype(np.float32) - img1.astype(np.float32))))
        assert diff_after_reset < max(diff_no_reset, 5.0), (
            "Reset did not clear state — second frame reflects first frame's luma"
        )
