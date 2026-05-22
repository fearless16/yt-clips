"""Regression tests for identity belief, temporal solver, and compositor.

Tests the current architecture:
- Intrinsic snapshot selection (best over latest)
- Temporal solver (single reference, detail preservation)
- Compositor (reset, lighting decoupled from compositor)
- RendererMode (proper transitions, telemetry consistency)
- Pipeline orchestration (mode update in orchestration layer)
"""

import numpy as np
import pytest
import cv2

from face_os.identity_state import IdentityState
from face_os.intrinsic_decomposition import IntrinsicComponents
from face_os.temporal_solve import BidirectionalSolver, FrameQuality
from face_os.compositor import Compositor
from face_os.photometric import photometric_lock, reset_photometric_lock
from face_os.renderer_mode import RendererModeState, RendererMode


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_intrinsic(
    albedo_mean: float = 0.5,
    confidence_mean: float = 0.8,
    reconstruction_error: float = 0.05,
    h: int = 64,
    w: int = 64,
) -> IntrinsicComponents:
    """Create synthetic IntrinsicComponents."""
    return IntrinsicComponents(
        albedo=np.full((h, w, 3), albedo_mean, dtype=np.float32),
        shading=np.full((h, w, 1), 0.5, dtype=np.float32),
        specular=np.zeros((h, w, 3), dtype=np.float32),
        normal_map=np.stack([np.zeros((h, w)), np.zeros((h, w)), np.ones((h, w))], axis=2).astype(np.float32),
        confidence=np.full((h, w, 1), confidence_mean, dtype=np.float32),
        reconstruction_error=reconstruction_error,
        albedo_uncertainty=np.zeros((h, w, 1), dtype=np.float32),
        shading_uncertainty=np.zeros((h, w, 1), dtype=np.float32),
        specular_uncertainty=np.zeros((h, w, 3), dtype=np.float32),
        decomposition_quality=1.0 - reconstruction_error,
    )


def _quality_map(h: int = 64, w: int = 64, value: float = 0.8) -> np.ndarray:
    return np.full((h, w), value, dtype=np.float32)


def _make_sharp_face(h=64, w=64, seed=42, brightness=128.0):
    rng = np.random.RandomState(seed)
    base = np.full((h, w, 3), brightness, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            if (y + x) % 4 < 2:
                base[y, x] += 30
            else:
                base[y, x] -= 30
    return np.clip(base + rng.randn(h, w, 3) * 10, 0, 255).astype(np.uint8)


def _make_blurry_face(h=64, w=64, seed=42, brightness=128.0):
    return cv2.GaussianBlur(_make_sharp_face(h, w, seed, brightness), (15, 15), 5.0)


def _hf_energy(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ─── Identity State: Intrinsic Snapshot Selection ────────────────────────────

class TestIntrinsicSnapshotSelection:
    """Intrinsic snapshot history must prefer best over latest."""

    def test_high_quality_snapshot_survives_low_quality_flood(self):
        """High-confidence snapshot must remain selectable after many low-quality updates."""
        state = IdentityState(atlas_size=(64, 64))
        h, w = 64, 64

        init = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        for _ in range(5):
            state.update(init, _quality_map(h, w, 0.8), pose=(0, 0, 0))

        high = _make_intrinsic(confidence_mean=0.9, reconstruction_error=0.02)
        state._store_intrinsic_snapshot(high, _quality_map(h, w, 0.95))

        for _ in range(30):
            low = _make_intrinsic(confidence_mean=0.05, reconstruction_error=0.8)
            state._store_intrinsic_snapshot(low, _quality_map(h, w, 0.05))

        best = state._select_intrinsic_snapshot(_quality_map(h, w, 0.8))
        assert best is not None
        assert best["confidence_scalar"] > 0.5, (
            f"High-quality snapshot lost, best confidence={best['confidence_scalar']:.3f}"
        )

    def test_query_intrinsic_selects_best_snapshot(self):
        """query_intrinsic must return the best-scoring snapshot."""
        state = IdentityState(atlas_size=(64, 64))
        h, w = 64, 64

        init = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        for _ in range(5):
            state.update(init, _quality_map(h, w, 0.8), pose=(0, 0, 0))

        # Store snapshot with albedo_mean=0.7 (high quality)
        good = _make_intrinsic(albedo_mean=0.7, confidence_mean=0.85, reconstruction_error=0.03)
        state._store_intrinsic_snapshot(good, _quality_map(h, w, 0.9))

        # Overwrite with many bad snapshots
        for _ in range(20):
            bad = _make_intrinsic(albedo_mean=0.2, confidence_mean=0.1, reconstruction_error=0.6)
            state._store_intrinsic_snapshot(bad, _quality_map(h, w, 0.1))

        result, _ = state.query_intrinsic(_quality_map(h, w, 0.8))
        assert result is not None
        albedo_mean = float(np.mean(result.albedo))
        assert albedo_mean > 0.5, f"Expected best albedo (~0.7), got {albedo_mean:.3f}"

    def test_selection_is_deterministic(self):
        """Same query must return same result."""
        state = IdentityState(atlas_size=(64, 64))
        h, w = 64, 64

        init = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        for _ in range(5):
            state.update(init, _quality_map(h, w, 0.7), pose=(0, 0, 0))

        for i in range(10):
            snap = _make_intrinsic(confidence_mean=0.3 + (i % 4) * 0.2)
            state._store_intrinsic_snapshot(snap, _quality_map(h, w, 0.5))

        qm = _quality_map(h, w, 0.7)
        s1 = state._select_intrinsic_snapshot(qm)
        s2 = state._select_intrinsic_snapshot(qm)
        assert s1["update_index"] == s2["update_index"]

    def test_query_albedo_does_not_mutate_history(self):
        """query_albedo must not change intrinsic history."""
        state = IdentityState(atlas_size=(64, 64))
        h, w = 64, 64

        init = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        for _ in range(5):
            state.update(init, _quality_map(h, w, 0.8), pose=(0, 0, 0))

        snap = _make_intrinsic(confidence_mean=0.8)
        state._store_intrinsic_snapshot(snap, _quality_map(h, w, 0.8))

        len_before = len(state._intrinsic_history)
        for _ in range(10):
            state.query_albedo(_quality_map(h, w, 0.8))
        len_after = len(state._intrinsic_history)

        assert len_before == len_after

    def test_reset_clears_history_preserves_anchor(self):
        """reset() must clear intrinsic history but keep anchors."""
        state = IdentityState(atlas_size=(64, 64))
        h, w = 64, 64

        ref = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        state.set_anchor(ref)
        anchor = state._anchor_albedo.copy()

        for _ in range(5):
            snap = _make_intrinsic()
            state._store_intrinsic_snapshot(snap, _quality_map(h, w, 0.5))

        assert len(state._intrinsic_history) > 0
        state.reset()
        assert len(state._intrinsic_history) == 0
        assert state._anchor_albedo is not None
        np.testing.assert_array_equal(state._anchor_albedo, anchor)


# ─── Temporal Solver ─────────────────────────────────────────────────────────

class TestTemporalSolver:
    """Solver must select best reference and preserve detail."""

    def test_hq_frames_identified_correctly(self):
        """High-quality frames must be identified based on quality threshold."""
        solver = BidirectionalSolver(lookback_frames=5, lookahead_frames=5)
        h, w = 64, 64

        for i in range(5):
            face = _make_sharp_face(h, w, seed=i)
            fq = FrameQuality(frame_idx=i, sharpness=0.9, detection_confidence=0.95)
            solver.add_frame(i, face, _quality_map(h, w, 0.9), fq)

        hq = solver.identify_hq_frames()
        assert len(hq) > 0

    def test_solver_is_deterministic(self):
        """Same inputs must produce same outputs."""
        h, w = 64, 64

        def make_solver():
            s = BidirectionalSolver(lookback_frames=3, lookahead_frames=3)
            for i in range(5):
                face = _make_sharp_face(h, w, seed=i)
                fq = FrameQuality(frame_idx=i, sharpness=0.7, detection_confidence=0.9)
                s.add_frame(i, face, _quality_map(h, w, 0.7), fq)
            s.identify_hq_frames()
            return s

        s1, s2 = make_solver(), make_solver()
        r1, c1 = s1.solve_frame(2, (h, w))
        r2, c2 = s2.solve_frame(2, (h, w))
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(c1, c2)

    def test_solve_all_preserves_hq_frames(self):
        """High-quality frames must not lose significant HF energy."""
        solver = BidirectionalSolver(lookback_frames=3, lookahead_frames=3)
        h, w = 64, 64

        for i in range(5):
            face = _make_sharp_face(h, w, seed=i)
            fq = FrameQuality(frame_idx=i, sharpness=0.9, detection_confidence=0.95)
            solver.add_frame(i, face, _quality_map(h, w, 0.9), fq)

        solver.identify_hq_frames()
        results = solver.solve_all((h, w))

        for idx in range(5):
            original = solver._canonical_faces[idx]
            solved, _ = results[idx]
            orig_hf = _hf_energy(original)
            solved_hf = _hf_energy(solved)
            assert solved_hf > orig_hf * 0.5, (
                f"Frame {idx}: HF collapsed from {orig_hf:.1f} to {solved_hf:.1f}"
            )

    def test_confidence_nonzero_for_valid_frames(self):
        """Confidence must be > 0 for frames with valid data."""
        solver = BidirectionalSolver(lookback_frames=3, lookahead_frames=3)
        h, w = 64, 64

        face = _make_sharp_face(h, w, seed=0)
        fq = FrameQuality(frame_idx=0, sharpness=0.8, detection_confidence=0.9)
        solver.add_frame(0, face, _quality_map(h, w, 0.8), fq)
        solver.identify_hq_frames()

        _, conf = solver.solve_frame(0, (h, w))
        assert float(np.mean(conf)) > 0.0


# ─── Compositor ──────────────────────────────────────────────────────────────

class TestCompositor:
    """Compositor must reset state and produce valid output."""

    def test_reset_clears_luma_ema(self):
        """reset() must clear compositor-local state. Photometric locking is upstream."""
        comp = Compositor()
        comp._feather_kernel = np.ones((3, 3))
        comp.reset()
        assert comp._feather_kernel is None

    def test_composite_preserves_shape(self):
        """Output must have same shape as input."""
        comp = Compositor()
        h, w = 64, 64
        original = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        enhanced = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        mask = np.ones((h, w), dtype=np.float32) * 0.5

        result = comp.composite(original, enhanced, face_mask=mask)
        assert result.shape == original.shape
        assert result.dtype == np.uint8

    def test_composite_no_nan_inf(self):
        """Output must not contain NaN or Inf."""
        comp = Compositor()
        h, w = 64, 64
        original = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        enhanced = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        mask = np.ones((h, w), dtype=np.float32) * 0.5

        result = comp.composite(original, enhanced, face_mask=mask)
        assert not np.any(np.isnan(result.astype(float)))
        assert not np.any(np.isinf(result.astype(float)))


# ─── RendererMode ────────────────────────────────────────────────────────────

class TestRendererMode:
    """Mode transitions must be correct and consistent."""

    def test_transitions_to_physical_on_high_confidence(self):
        state = RendererModeState()
        for _ in range(10):
            mode = state.update(True, 0.9, 0.05)
        assert mode == RendererMode.PHYSICAL

    def test_transitions_to_hybrid_on_medium_confidence(self):
        state = RendererModeState()
        for _ in range(10):
            mode = state.update(True, 0.5, 0.1)
        assert mode == RendererMode.HYBRID

    def test_stays_alpha_on_low_confidence(self):
        state = RendererModeState()
        for _ in range(10):
            mode = state.update(True, 0.1, 0.5)
        assert mode == RendererMode.ALPHA_FALLBACK

    def test_no_intrinsic_goes_alpha(self):
        state = RendererModeState()
        mode = state.update(False, 0.0, 1.0)
        assert mode == RendererMode.ALPHA_FALLBACK

    def test_hysteresis_prevents_thrashing(self):
        state = RendererModeState()
        for i in range(20):
            conf = 0.9 if i % 2 == 0 else 0.1
            state.update(True, conf, 0.1)
        assert state.transition_count <= 4

    def test_blend_weight_physical(self):
        state = RendererModeState()
        state.current_mode = RendererMode.PHYSICAL
        assert state.get_blend_weight() == 1.0

    def test_blend_weight_alpha(self):
        state = RendererModeState()
        state.current_mode = RendererMode.ALPHA_FALLBACK
        assert state.get_blend_weight() == 0.0

    def test_blend_weight_hybrid_range(self):
        state = RendererModeState()
        state.current_mode = RendererMode.HYBRID
        state.mode_confidence = 0.45
        w = state.get_blend_weight()
        assert 0.0 <= w <= 1.0


# ─── Photometric Lock ────────────────────────────────────────────────────────

class TestPhotometricLock:
    """photometric_lock must stabilize temporal luminance."""

    def test_photometric_lock_returns_same_shape(self):
        """Output must have same shape and dtype as input."""
        reset_photometric_lock()
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = photometric_lock(frame)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_photometric_lock_first_frame_unchanged(self):
        """First frame must pass through unchanged (no reference yet)."""
        reset_photometric_lock()
        frame = np.ones((64, 64, 3), dtype=np.uint8) * 128
        result = photometric_lock(frame)
        np.testing.assert_array_equal(result, frame)

    def test_photometric_lock_no_nan(self):
        """Must not produce NaN or Inf."""
        reset_photometric_lock()
        for _ in range(5):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            result = photometric_lock(frame)
            assert not np.any(np.isnan(result.astype(float)))
            assert not np.any(np.isinf(result.astype(float)))

    def test_photometric_lock_with_mask(self):
        """Must work with face mask."""
        reset_photometric_lock()
        frame = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        result = photometric_lock(frame, mask=mask)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_photometric_lock_dampens_sudden_brightness_jump(self):
        """EMA must dampen sudden luminance jumps."""
        reset_photometric_lock()
        bright = np.ones((64, 64, 3), dtype=np.uint8) * 200
        dark = np.ones((64, 64, 3), dtype=np.uint8) * 50

        photometric_lock(bright)  # establish baseline
        result = photometric_lock(dark)  # sudden jump

        # Result should be brighter than input (EMA pulls toward bright baseline)
        assert float(np.mean(result)) > float(np.mean(dark))

    def test_reset_photometric_lock_clears_state(self):
        """reset_photometric_lock must clear temporal state."""
        bright = np.ones((64, 64, 3), dtype=np.uint8) * 200
        photometric_lock(bright)
        reset_photometric_lock()
        # After reset, next frame should pass through unchanged
        dark = np.ones((64, 64, 3), dtype=np.uint8) * 50
        result = photometric_lock(dark)
        np.testing.assert_array_equal(result, dark)


# ─── Linear-light blending ───────────────────────────────────────────────────

class TestBlendLinear:
    """_blend_linear must blend in linear-light space."""

    def test_blend_linear_returns_correct_shape(self):
        """Output must match input shape and dtype."""
        from face_os.pipeline import _blend_linear
        bg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        fg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32) * 0.5
        result = _blend_linear(bg, fg, mask)
        assert result.shape == bg.shape
        assert result.dtype == np.uint8

    def test_blend_linear_full_mask_returns_fg(self):
        """Full mask (1.0) must return foreground."""
        from face_os.pipeline import _blend_linear
        bg = np.zeros((64, 64, 3), dtype=np.uint8)
        fg = np.ones((64, 64, 3), dtype=np.uint8) * 200
        mask = np.ones((64, 64), dtype=np.float32)
        result = _blend_linear(bg, fg, mask)
        # Should be close to fg (linear roundtrip may cause minor quantization)
        assert np.abs(result.astype(float) - fg.astype(float)).mean() < 2.0

    def test_blend_linear_zero_mask_returns_bg(self):
        """Zero mask (0.0) must return background."""
        from face_os.pipeline import _blend_linear
        bg = np.ones((64, 64, 3), dtype=np.uint8) * 200
        fg = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        result = _blend_linear(bg, fg, mask)
        assert np.abs(result.astype(float) - bg.astype(float)).mean() < 2.0

    def test_blend_linear_no_nan(self):
        """Must not produce NaN or Inf."""
        from face_os.pipeline import _blend_linear
        bg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        fg = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.random.rand(64, 64).astype(np.float32)
        result = _blend_linear(bg, fg, mask)
        assert not np.any(np.isnan(result.astype(float)))
        assert not np.any(np.isinf(result.astype(float)))

    def test_blend_linear_differs_from_gamma_blend(self):
        """Linear-light blend must differ from naive gamma-space blend."""
        from face_os.pipeline import _blend_linear
        bg = np.ones((64, 64, 3), dtype=np.uint8) * 128
        fg = np.ones((64, 64, 3), dtype=np.uint8) * 200
        mask = np.ones((64, 64), dtype=np.float32) * 0.5

        linear_result = _blend_linear(bg, fg, mask)
        # Gamma-space blend (naive)
        gamma_result = (bg.astype(float) * 0.5 + fg.astype(float) * 0.5).astype(np.uint8)

        # They should differ because gamma-space is nonlinear
        diff = np.abs(linear_result.astype(float) - gamma_result.astype(float)).mean()
        assert diff > 1.0, (
            f"Linear and gamma blends differ by only {diff:.2f} — "
            f"_blend_linear may not be doing actual linear-light conversion"
        )
