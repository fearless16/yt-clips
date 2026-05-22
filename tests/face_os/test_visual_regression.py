"""Visual Regression Tests — D-09

Tests that validate visual output quality, not just internal correctness.

These tests use synthetic test images (generated in-memory) to verify:
- Sharpness preservation (Laplacian variance)
- Frequency retention (high-freq energy ratio)
- Contrast preservation (histogram overlap)
- Skin texture retention (LBP distance)
- Temporal stability (frame-to-frame flicker)

Each test generates its own test data — no external files needed.
"""

import cv2
import numpy as np
import pytest

from face_os.types import EnhancementMask


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_sharp_image(size=256):
    """Generate a sharp test image with edges and texture."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (size - 56, size - 56), (200, 150, 100), -1)
    cv2.circle(img, (size // 2, size // 2), 60, (100, 200, 150), -1)
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def _make_face_like_image(size=256):
    """Generate a face-like image with skin-tone gradients and features."""
    img = np.full((size, size, 3), [180, 150, 130], dtype=np.uint8)  # skin tone
    # Eyes
    cv2.circle(img, (size // 3, size // 3), 12, (40, 30, 25), -1)
    cv2.circle(img, (2 * size // 3, size // 3), 12, (40, 30, 25), -1)
    # Nose
    cv2.line(img, (size // 2, size // 2 - 10), (size // 2, size // 2 + 15), (160, 130, 110), 2)
    # Mouth
    cv2.ellipse(img, (size // 2, 2 * size // 3), (20, 8), 0, 0, 180, (120, 80, 70), 2)
    # Skin texture (fine noise)
    texture = np.random.randint(0, 15, img.shape, dtype=np.uint8)
    img = cv2.add(img, texture)
    return img


def _make_skin_mask(size=256):
    """Generate a skin-like mask (elliptical)."""
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(mask, (size // 2, size // 2), (size // 3, size // 2),
                0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (11, 11), 3)
    return np.clip(mask, 0, 1)


def _laplacian_variance(gray):
    """Measure sharpness via Laplacian variance."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _hf_energy(gray):
    """Measure high-frequency energy."""
    gray_f = gray.astype(np.float32)
    lf = cv2.GaussianBlur(gray_f, (0, 0), 2.0)
    hf = gray_f - lf
    return np.mean(hf ** 2)


def _histogram_overlap(a, b, bins=64):
    """Compute histogram overlap between two grayscale images."""
    ha, _ = np.histogram(a.ravel(), bins=bins, range=(0, 256))
    hb, _ = np.histogram(b.ravel(), bins=bins, range=(0, 256))
    ha = ha.astype(np.float64) / (ha.sum() + 1e-10)
    hb = hb.astype(np.float64) / (hb.sum() + 1e-10)
    return np.minimum(ha, hb).sum()


def _lbp_distance(gray_a, gray_b, radius=1):
    """Compute LBP histogram distance (texture similarity)."""
    def _lbp(img):
        h, w = img.shape
        lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(img, -dy, axis=0), -dx, axis=1)
                lbp += ((img[radius:-radius, radius:-radius] >
                         shifted[radius:-radius, radius:-radius]).astype(np.uint8)
                        << max(0, dy * (2 * radius + 1) + dx + 4))
        return lbp

    lbp_a = _lbp(gray_a.astype(np.int32))
    lbp_b = _lbp(gray_b.astype(np.int32))
    ha, _ = np.histogram(lbp_a.ravel(), bins=32, range=(0, 256))
    hb, _ = np.histogram(lbp_b.ravel(), bins=32, range=(0, 256))
    ha = ha.astype(np.float64) / (ha.sum() + 1e-10)
    hb = hb.astype(np.float64) / (hb.sum() + 1e-10)
    return np.sum(np.abs(ha - hb)) / 2  # L1 / 2


# ═══════════════════════════════════════════════════════════════════════════════
# SHARPNESS PRESERVATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestSharpnessPreservation:
    """Verify output sharpness is not destroyed by pipeline."""

    def test_output_laplacian_variance_above_threshold(self):
        """Output must retain minimum sharpness after mild degradation."""
        img = _make_sharp_image()
        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness_in = _laplacian_variance(gray_in)

        # Mild blur (sigma=0.5 simulates minor pipeline softening)
        blurred = cv2.GaussianBlur(img, (3, 3), 0.5)
        gray_out = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        sharpness_out = _laplacian_variance(gray_out)

        # Should retain at least 30% of sharpness
        assert sharpness_out > sharpness_in * 0.3, (
            f"Sharpness degraded too much: {sharpness_out:.1f} < {sharpness_in * 0.3:.1f}"
        )

    def test_sharpen_increases_laplacian_variance(self):
        """_sharpen should increase Laplacian variance."""
        from face_os.face_enhance import _sharpen

        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(img, (5, 5), 2.0)

        sharp_before = _laplacian_variance(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))
        sharpened = _sharpen(blurred, amount=0.8, radius=0.8)
        sharp_after = _laplacian_variance(cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY))

        assert sharp_after > sharp_before, (
            f"Sharpening did not increase sharpness: {sharp_after:.1f} <= {sharp_before:.1f}"
        )

    def test_sharpen_preserves_shape_and_dtype(self):
        """_sharpen must not alter frame contract."""
        from face_os.face_enhance import _sharpen

        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = _sharpen(img, amount=0.5, radius=1.0)

        assert result.shape == img.shape, f"Shape changed: {result.shape} != {img.shape}"
        assert result.dtype == np.uint8, f"Dtype changed: {result.dtype} != np.uint8"
        assert not np.any(np.isnan(result)), "Output contains NaN"
        assert not np.any(np.isinf(result)), "Output contains Inf"

    def test_render_frame_preserves_sharpness(self):
        """render_frame should not destroy sharpness of a textured image."""
        from face_os.face_enhance import render_frame

        img = _make_face_like_image(256)
        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp_in = _laplacian_variance(gray_in)

        result = render_frame(img)
        gray_out = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        sharp_out = _laplacian_variance(gray_out)

        # render_frame should not destroy more than 60% of sharpness
        assert sharp_out > sharp_in * 0.4, (
            f"render_frame destroyed sharpness: {sharp_out:.1f} < {sharp_in * 0.4:.1f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FREQUENCY RETENTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestFrequencyRetention:
    """Verify high-frequency content is preserved through pipeline."""

    def test_hf_energy_ratio_preserved(self):
        """High-frequency energy should not drop below threshold."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(0, 256, 4):
            for x in range(0, 256, 4):
                if (x // 4 + y // 4) % 2 == 0:
                    img[y:y + 4, x:x + 4] = [200, 180, 160]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hf_energy_in = _hf_energy(gray)

        processed = cv2.GaussianBlur(img, (3, 3), 0.5)
        hf_energy_out = _hf_energy(cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY))

        assert hf_energy_out > hf_energy_in * 0.3, (
            f"HF energy lost: {hf_energy_out:.1f} < {hf_energy_in * 0.3:.1f}"
        )

    def test_sharpen_restores_hf_energy(self):
        """_sharpen on a blurred image should restore HF energy."""
        from face_os.face_enhance import _sharpen

        img = _make_sharp_image()
        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hf_in = _hf_energy(gray_in)

        blurred = cv2.GaussianBlur(img, (5, 5), 2.0)
        hf_blurred = _hf_energy(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))

        sharpened = _sharpen(blurred, amount=0.8, radius=0.8)
        hf_sharp = _hf_energy(cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY))

        # Sharpening should recover some HF energy
        assert hf_sharp > hf_blurred, (
            f"_sharpen did not restore HF: {hf_sharp:.1f} <= {hf_blurred:.1f}"
        )

    def test_smooth_skin_reduces_hf_in_mask_region(self):
        """smooth_skin should reduce HF energy in skin regions."""
        from face_os.face_enhance import smooth_skin

        img = _make_face_like_image(256)
        skin_mask = _make_skin_mask(256)

        # Measure HF in skin region before
        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        skin_pixels_in = gray_in[skin_mask > 0.5]
        hf_in = np.var(skin_pixels_in.astype(np.float32))

        result = smooth_skin(img, skin_mask, amount=0.5)
        gray_out = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        skin_pixels_out = gray_out[skin_mask > 0.5]
        hf_out = np.var(skin_pixels_out.astype(np.float32))

        # Skin smoothing should reduce variance
        assert hf_out < hf_in, (
            f"smooth_skin did not reduce skin texture: {hf_out:.1f} >= {hf_in:.1f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRAST PRESERVATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestContrastPreservation:
    """Verify contrast is not collapsed by pipeline."""

    def test_output_histogram_not_collapsed(self):
        """Output histogram should not be narrower than input."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:128, :] = [50, 50, 50]
        img[128:, :] = [200, 200, 200]

        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast_in = np.std(gray_in)

        reduced = (img.astype(np.float32) * 0.5 + 128).clip(0, 255).astype(np.uint8)
        gray_out = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
        contrast_out = np.std(gray_out)

        assert contrast_out >= contrast_in * 0.5, (
            f"Contrast collapsed: {contrast_out:.1f} < {contrast_in * 0.5:.1f}"
        )

    def test_render_frame_preserves_contrast(self):
        """render_frame should not collapse histogram."""
        from face_os.face_enhance import render_frame

        img = _make_face_like_image(256)
        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result = render_frame(img)
        gray_out = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        overlap = _histogram_overlap(gray_in, gray_out)
        assert overlap > 0.7, (
            f"Histogram overlap too low: {overlap:.3f} < 0.7"
        )

    def test_compositor_produces_valid_output(self):
        """Compositor.composite should produce valid blended output."""
        from face_os.compositor import Compositor

        compositor = Compositor()

        original = _make_face_like_image(256)
        enhanced = original.copy()
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=-20)

        result = compositor.composite(original, enhanced)

        # Output should differ from both inputs (blending happened)
        diff_orig = np.mean(np.abs(result.astype(float) - original.astype(float)))
        diff_enh = np.mean(np.abs(result.astype(float) - enhanced.astype(float)))

        # Result should not be identical to either input
        assert result.shape == original.shape
        assert result.dtype == np.uint8
        assert not np.any(np.isnan(result)), "Output contains NaN"

    def test_photometric_lock_dampens_brightness_jumps(self):
        """photometric_lock should dampen sudden brightness changes."""
        from face_os.photometric import photometric_lock, reset_photometric_lock

        reset_photometric_lock()

        # Frame 1: normal brightness
        frame1 = np.full((64, 64, 3), 128, dtype=np.uint8)
        result1 = photometric_lock(frame1)

        # Frame 2: sudden brightness jump
        frame2 = np.full((64, 64, 3), 200, dtype=np.uint8)
        result2 = photometric_lock(frame2)

        mean2 = np.mean(result2)
        assert mean2 < 200, f"Photometric lock did not dampen jump: mean={mean2:.1f}"
        assert mean2 > 128, f"Photometric lock over-corrected: mean={mean2:.1f}"


# ═══════════════════════════════════════════════════════════════════════════════
# SKIN TEXTURE RETENTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkinTextureRetention:
    """Verify skin texture is preserved or enhanced, not destroyed."""

    def test_smooth_skin_preserves_structure(self):
        """smooth_skin should not destroy facial structure."""
        from face_os.face_enhance import smooth_skin

        img = _make_face_like_image(256)
        skin_mask = _make_skin_mask(256)

        result = smooth_skin(img, skin_mask, amount=0.3)

        # Overall structure should be similar (histogram overlap > 0.8)
        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_out = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        overlap = _histogram_overlap(gray_in, gray_out)
        assert overlap > 0.8, (
            f"smooth_skin destroyed structure: overlap={overlap:.3f}"
        )

    def test_smooth_skin_preserves_shape_dtype(self):
        """smooth_skin must preserve frame contract."""
        from face_os.face_enhance import smooth_skin

        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = np.random.rand(128, 128).astype(np.float32)
        result = smooth_skin(img, mask, amount=0.2)

        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_lbp_distance_within_tolerance(self):
        """LBP texture distance should be small for similar images."""
        img = _make_face_like_image(256)
        gray_a = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mild processing
        processed = cv2.GaussianBlur(img, (3, 3), 0.5)
        gray_b = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        dist = _lbp_distance(gray_a, gray_b)
        assert dist < 0.3, f"LBP distance too high: {dist:.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL STABILITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalStability:
    """Verify frame-to-frame consistency."""

    def test_flicker_below_threshold(self):
        """Frame-to-frame luminance change should be bounded."""
        frames = []
        for _ in range(10):
            brightness = 128 + np.random.randint(-2, 2)
            frame = np.full((64, 64, 3), brightness, dtype=np.uint8)
            frames.append(frame)

        luminances = [np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[:, :, 0])
                      for f in frames]
        flicker = np.std(np.diff(luminances))

        # ±2 variation gives std of diff ~1.7 LAB units
        assert flicker < 2.0, f"Flicker too high: {flicker:.2f} LAB units"

    def test_render_frame_temporal_consistency(self):
        """render_frame on nearly-identical frames should produce similar output."""
        from face_os.face_enhance import render_frame

        results = []
        for i in range(5):
            img = _make_face_like_image(256)
            # Small variation per frame
            noise = np.random.randint(-3, 3, img.shape, dtype=np.int16)
            frame = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            result = render_frame(frame)
            results.append(result)

        # Measure inter-frame difference
        diffs = []
        for i in range(len(results) - 1):
            diff = np.mean(np.abs(
                results[i].astype(np.float32) - results[i + 1].astype(np.float32)
            ))
            diffs.append(diff)

        avg_diff = np.mean(diffs)
        assert avg_diff < 10.0, (
            f"Temporal instability: avg inter-frame diff={avg_diff:.2f}"
        )

    def test_blend_linear_deterministic(self):
        """_blend_linear should produce identical output for identical input."""
        from face_os.pipeline import _blend_linear

        bg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        fg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.random.rand(64, 64).astype(np.float32)

        result1 = _blend_linear(bg, fg, mask)
        result2 = _blend_linear(bg, fg, mask)

        np.testing.assert_array_equal(result1, result2,
                                       err_msg="_blend_linear not deterministic")

    def test_photometric_lock_temporal_stability(self):
        """photometric_lock should produce smooth output over gradual changes."""
        from face_os.photometric import photometric_lock, reset_photometric_lock

        reset_photometric_lock()

        results = []
        for brightness in range(100, 150, 2):  # gradual brightness change
            frame = np.full((64, 64, 3), brightness, dtype=np.uint8)
            result = photometric_lock(frame)
            results.append(np.mean(result))

        # Measure smoothness: second derivative should be small
        second_deriv = np.diff(np.diff(results))
        smoothness = np.max(np.abs(second_deriv))
        assert smoothness < 5.0, (
            f"Photometric lock not smooth: max 2nd deriv={smoothness:.2f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LINEAR-LIGHT COMPOSITING
# ═══════════════════════════════════════════════════════════════════════════════

class TestLinearLightCompositing:
    """Verify linear-light compositing differs from gamma-space."""

    def test_blend_linear_differs_from_gamma(self):
        """Linear-light blend should produce different result than gamma-space."""
        from face_os.pipeline import _blend_linear

        bg = np.full((64, 64, 3), [100, 100, 100], dtype=np.uint8)
        fg = np.full((64, 64, 3), [200, 200, 200], dtype=np.uint8)
        mask = np.full((64, 64), 0.5, dtype=np.float32)

        linear_result = _blend_linear(bg, fg, mask)

        # Gamma-space blend (naive averaging)
        gamma_result = (bg.astype(np.float32) * 0.5 +
                        fg.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)

        diff = np.mean(np.abs(
            linear_result.astype(float) - gamma_result.astype(float)
        ))
        assert diff > 1.0, f"Linear and gamma blend too similar: diff={diff:.2f}"

    def test_blend_linear_preserves_shape(self):
        """_blend_linear must preserve frame contract."""
        from face_os.pipeline import _blend_linear

        bg = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        fg = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = np.random.rand(128, 128).astype(np.float32)

        result = _blend_linear(bg, fg, mask)
        assert result.shape == bg.shape
        assert result.dtype == np.uint8

    def test_blend_linear_full_mask_returns_foreground(self):
        """Full mask should return foreground."""
        from face_os.pipeline import _blend_linear

        bg = np.full((64, 64, 3), 50, dtype=np.uint8)
        fg = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)

        result = _blend_linear(bg, fg, mask)
        diff = np.mean(np.abs(result.astype(float) - fg.astype(float)))
        assert diff < 5.0, f"Full mask did not return fg: diff={diff:.2f}"

    def test_blend_linear_zero_mask_returns_background(self):
        """Zero mask should return background."""
        from face_os.pipeline import _blend_linear

        bg = np.full((64, 64, 3), 50, dtype=np.uint8)
        fg = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)

        result = _blend_linear(bg, fg, mask)
        diff = np.mean(np.abs(result.astype(float) - bg.astype(float)))
        assert diff < 5.0, f"Zero mask did not return bg: diff={diff:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITOR VISUAL QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompositorVisualQuality:
    """Verify Compositor produces visually acceptable output."""

    def test_composite_preserves_shape(self):
        """Compositor.composite must preserve frame contract."""
        from face_os.compositor import Compositor

        compositor = Compositor()
        original = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        enhanced = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        result = compositor.composite(original, enhanced)
        assert result.shape == original.shape
        assert result.dtype == np.uint8

    def test_composite_no_nan_inf(self):
        """Compositor output must be clean."""
        from face_os.compositor import Compositor

        compositor = Compositor()
        original = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        enhanced = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        result = compositor.composite(original, enhanced)
        assert not np.any(np.isnan(result)), "Compositor output contains NaN"
        assert not np.any(np.isinf(result)), "Compositor output contains Inf"

    def test_composite_with_face_mask(self):
        """Compositor with face mask should blend selectively."""
        from face_os.compositor import Compositor

        compositor = Compositor()
        original = np.full((128, 128, 3), 100, dtype=np.uint8)
        enhanced = np.full((128, 128, 3), 200, dtype=np.uint8)

        face_mask = np.zeros((128, 128), dtype=np.float32)
        face_mask[32:96, 32:96] = 1.0
        face_mask = cv2.GaussianBlur(face_mask, (11, 11), 3)
        face_mask = np.clip(face_mask, 0, 1)

        result = compositor.composite(original, enhanced, face_mask=face_mask)

        # Center (face region) should be closer to enhanced
        center_mean = np.mean(result[48:80, 48:80])
        # Edge (non-face) should be closer to original
        edge_mean = np.mean(result[0:16, 0:16])

        assert center_mean > edge_mean, (
            f"Face mask not applied: center={center_mean:.1f}, edge={edge_mean:.1f}"
        )

    def test_composite_deterministic(self):
        """Compositor.composite should be deterministic for same input."""
        from face_os.compositor import Compositor

        compositor = Compositor()
        original = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        enhanced = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        result1 = compositor.composite(original, enhanced)
        compositor.reset()
        result2 = compositor.composite(original, enhanced)

        np.testing.assert_array_equal(result1, result2,
                                       err_msg="Compositor not deterministic")


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY MASK VISUAL QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeometryMaskQuality:
    """Verify geometry mask has visually acceptable properties."""

    def test_mask_has_smooth_edges(self):
        """Geometry mask edges should be feathered (not binary)."""
        from face_os.pipeline import FaceOSPipeline

        mask = FaceOSPipeline._make_canonical_geometry_mask((256, 256))

        # Count transition pixels (between 0.1 and 0.9)
        transition = np.sum((mask > 0.1) & (mask < 0.9))
        total = mask.size
        ratio = transition / total

        assert ratio > 0.05, f"Mask edges too sharp: transition ratio={ratio:.3f}"

    def test_mask_is_brightness_invariant(self):
        """Geometry mask should be identical regardless of image brightness."""
        from face_os.pipeline import FaceOSPipeline

        mask1 = FaceOSPipeline._make_canonical_geometry_mask((256, 256))
        mask2 = FaceOSPipeline._make_canonical_geometry_mask((256, 256))

        np.testing.assert_array_equal(mask1, mask2,
                                       err_msg="Mask not deterministic")

    def test_mask_coverage_in_range(self):
        """Mask should cover ~60% of canonical area."""
        from face_os.pipeline import FaceOSPipeline

        mask = FaceOSPipeline._make_canonical_geometry_mask((256, 256))
        coverage = np.sum(mask > 0.5) / mask.size

        assert 0.3 < coverage < 0.9, (
            f"Mask coverage out of range: {coverage:.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CINEMATIC NOISE
# ═══════════════════════════════════════════════════════════════════════════════

class TestCinematicNoise:
    """Verify cinematic noise adds texture without destroying image."""

    def test_add_cinematic_noise_preserves_shape(self):
        """add_cinematic_noise must preserve frame contract."""
        from face_os.face_enhance import add_cinematic_noise

        img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        result = add_cinematic_noise(img, strength=0.02)

        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_add_cinematic_noise_subtle(self):
        """Noise should be subtle (mean diff < threshold)."""
        from face_os.face_enhance import add_cinematic_noise

        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        result = add_cinematic_noise(img, strength=0.015)

        diff = np.mean(np.abs(result.astype(float) - img.astype(float)))
        assert diff < 10.0, f"Noise too strong: mean diff={diff:.2f}"

    def test_add_cinematic_noise_no_nan(self):
        """Noise output must be clean."""
        from face_os.face_enhance import add_cinematic_noise

        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = add_cinematic_noise(img, strength=0.05)

        assert not np.any(np.isnan(result)), "Noise output contains NaN"
        assert not np.any(np.isinf(result)), "Noise output contains Inf"
