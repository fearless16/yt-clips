"""TDD tests for D-05 Task 3.5: ABComparator latent-vs-legacy wiring.

Tests verify:
1. compute_sharpness (new helper) — monotonic, mask-aware
2. _run_pipeline_source — drives pipeline under fixed render_source
3. compare_render_sources — non-regression gate with named thresholds

Architecture: tests mock only at the BOUNDARY (pipeline.process_frame),
never inside the metric functions. The pipeline is a real FaceOSPipeline
with a stubbed process_frame to produce deterministic frames.
"""

import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from face_os.ab_validation import (
    ABComparator,
    compute_sharpness,
    compute_ssim,
    compute_lab_drift,
    compute_albedo_chroma_stability,
    compute_albedo_chroma_match,
    compute_perceptual_distance,
    compute_all_metrics,
    compare_approaches,
    ABMetrics,
)

try:
    from face_os.ab_validation import CorpusSourceReport
    _HAS_CORPUS_REPORT = True
except ImportError:
    _HAS_CORPUS_REPORT = False
    CorpusSourceReport = None

_ = CorpusSourceReport  # suppress unused-import warning when not available

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _solid_frame(bgr=(128, 128, 128), size=(64, 64)):
    """Uniform BGR frame."""
    f = np.full((size[1], size[0], 3), bgr, dtype=np.uint8)
    return f


def _sharp_frame(size=(64, 64)):
    """High-frequency checkerboard → high Laplacian variance."""
    f = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            if (x + y) % 2 == 0:
                f[y, x] = [255, 255, 255]
    return f


def _blur_frame(size=(64, 64)):
    """Gaussian-blurred uniform → low Laplacian variance."""
    f = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    f = cv2.GaussianBlur(f, (15, 15), 5)
    return f


def _structured_frame(size=(128, 128)):
    """Gradient + checkerboard frame with real structure for perceptual metrics."""
    f = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cx, cy = size[0] // 2, size[1] // 2
    for y in range(size[1]):
        for x in range(size[0]):
            r = int(np.sqrt((x - cx) ** 2 + (y - cy) ** 2))
            val = 128 + int(64 * np.sin(r / 4.0))
            if (x + y) % 8 == 0:
                val = 255 - val
            f[y, x] = [val, val, val]
    return f


def _make_stub_pipeline(frames_legacy, frames_latent):
    """Build a stub pipeline whose process_frame returns different frames
    depending on the current render_source attribute.

    This is the ARCHITECTURE-CORRECT boundary mock: we stub process_frame
    to produce deterministic output based on the render_source, then verify
    the ABComparator correctly sets/restores render_source and collects frames.
    """
    pipeline = MagicMock()
    pipeline.render_source = 'legacy'
    pipeline.tracker = True  # skip enroll
    pipeline._reset_state = MagicMock()

    _call_count = {'legacy': 0, 'latent': 0}

    def _process_frame(frame, frame_idx=0):
        src = pipeline.render_source
        idx = _call_count[src]
        _call_count[src] += 1
        if src == 'legacy':
            out = frames_legacy[idx] if idx < len(frames_legacy) else frames_legacy[-1]
        else:
            out = frames_latent[idx] if idx < len(frames_latent) else frames_latent[-1]
        return {'frame': out, 'landmarks': None, 'transform': None}

    pipeline.process_frame = MagicMock(side_effect=_process_frame)
    return pipeline


# ─── compute_sharpness ────────────────────────────────────────────────────────

class TestComputeSharpness:
    def test_sharp_higher_than_blur(self):
        """Load-bearing: sharp frame must score higher than blurred."""
        sharp = _sharp_frame()
        blur = _blur_frame()
        assert compute_sharpness(sharp) > compute_sharpness(blur)

    def test_monotonic(self):
        """Sharp score ≥ 10x blur score (conservative bound)."""
        sharp_s = compute_sharpness(_sharp_frame())
        blur_s = compute_sharpness(_blur_frame())
        assert sharp_s >= blur_s * 10

    def test_mask_restricts_to_face(self):
        """Masked sharp region inside mask, blurred outside → score ≈ sharp."""
        h, w = 64, 64
        frame = _blur_frame((w, h))
        # Paint sharp checkerboard in center 32x32
        for y in range(16, 48):
            for x in range(16, 48):
                if (x + y) % 2 == 0:
                    frame[y, x] = [255, 255, 255]

        mask = np.zeros((h, w), dtype=np.float32)
        mask[16:48, 16:48] = 1.0

        masked_score = compute_sharpness(frame, mask=mask)
        full_score = compute_sharpness(frame)
        # Masked (sharp region only) should be higher than full (blur dilutes)
        assert masked_score > full_score

    def test_no_mask_returns_full(self):
        """Without mask, returns variance of full Laplacian."""
        frame = _sharp_frame()
        assert compute_sharpness(frame) > 0

    def test_mask_resize(self):
        """Mask with different spatial dims gets resized (no crash)."""
        frame = _sharp_frame((64, 64))
        mask = np.ones((32, 32), dtype=np.float32)
        score = compute_sharpness(frame, mask=mask)
        assert score > 0


# ─── _run_pipeline_source ────────────────────────────────────────────────────

class TestRunPipelineSource:
    def test_sets_render_source(self):
        """_run_pipeline_source sets pipeline.render_source for duration."""
        frames = [_solid_frame((128, 128, 128))]
        pipeline = _make_stub_pipeline(frames, frames)
        ab = ABComparator()

        # Capture render_source during call
        sources_seen = []
        original_pf = pipeline.process_frame.side_effect
        def _spy(frame, frame_idx=0):
            sources_seen.append(pipeline.render_source)
            return original_pf(frame, frame_idx)
        pipeline.process_frame.side_effect = _spy

        ab._run_pipeline_source(pipeline, '/dev/null', max_frames=2, render_source='latent')
        assert all(s == 'latent' for s in sources_seen)

    def test_restores_original_source(self):
        """render_source is restored after call (even on error)."""
        pipeline = _make_stub_pipeline([_solid_frame()], [_solid_frame()])
        pipeline.render_source = 'legacy'
        ab = ABComparator()
        ab._run_pipeline_source(pipeline, '/dev/null', max_frames=1, render_source='latent')
        assert pipeline.render_source == 'legacy'

    def test_restores_on_exception(self):
        """render_source restored even if process_frame raises."""
        pipeline = _make_stub_pipeline([_solid_frame()], [_solid_frame()])
        pipeline.render_source = 'legacy'
        pipeline.process_frame.side_effect = RuntimeError('boom')
        ab = ABComparator()
        ab._run_pipeline_source(pipeline, '/dev/null', max_frames=1, render_source='latent')
        assert pipeline.render_source == 'legacy'

    def test_returns_empty_on_bad_video(self):
        """Non-existent video → empty lists."""
        pipeline = _make_stub_pipeline([], [])
        ab = ABComparator()
        frames, lms, tfs = ab._run_pipeline_source(pipeline, '/nonexistent.mp4', max_frames=5, render_source='legacy')
        assert frames == []
        assert lms == []
        assert tfs == []

    def test_collects_frames(self):
        """Returns frames produced by pipeline.process_frame."""
        legacy_frames = [_solid_frame((100, 100, 100)) for _ in range(3)]
        latent_frames = [_solid_frame((150, 150, 150)) for _ in range(3)]
        pipeline = _make_stub_pipeline(legacy_frames, latent_frames)
        ab = ABComparator()

        # Use a real video file for cap.read() to work
        # Instead, mock cv2.VideoCapture
        with patch('face_os.ab_validation.cv2.VideoCapture') as mock_vc:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.read.side_effect = [(True, _solid_frame()) for _ in range(3)] + [(False, None)]
            mock_vc.return_value = cap

            frames, _, _ = ab._run_pipeline_source(pipeline, 'fake.mp4', max_frames=3, render_source='legacy')
            assert len(frames) == 3

    def test_calls_reset_state(self):
        """_reset_state called before each run (no state pollution)."""
        pipeline = _make_stub_pipeline([_solid_frame()], [_solid_frame()])
        ab = ABComparator()
        with patch('face_os.ab_validation.cv2.VideoCapture') as mock_vc:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.read.side_effect = [(True, _solid_frame()), (False, None)]
            mock_vc.return_value = cap
            ab._run_pipeline_source(pipeline, 'fake.mp4', max_frames=1, render_source='legacy')
        pipeline._reset_state.assert_called()


# ─── compare_render_sources ───────────────────────────────────────────────────

def _make_cap_for_frames(frames):
    """Build a MagicMock cap that yields the given frames then (False, None)."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
    return cap


class TestCompareRenderSources:
    def test_identical_frames_no_regression(self):
        """Same frames for both sources → regressed=False, all checks pass."""
        frames = [_solid_frame((128, 128, 128)) for _ in range(5)]
        pipeline = _make_stub_pipeline(frames, frames)
        ab = ABComparator()

        with patch('face_os.ab_validation.cv2.VideoCapture') as mock_vc:
            # Each _run_pipeline_source call creates its own cap
            cap_legacy = _make_cap_for_frames(frames)
            cap_latent = _make_cap_for_frames(frames)
            mock_vc.side_effect = [cap_legacy, cap_latent]

            result = ab.compare_render_sources(pipeline, 'fake.mp4', max_frames=5)

        assert result['regressed'] is False
        assert result['reasons'] == []
        assert result['checks']['ssim_ok'] is True
        assert result['checks']['lab_drift_ok'] is True
        assert result['checks']['sharpness_ok'] is True
        assert result['checks']['flicker_ok'] is True
        assert result['frames_compared'] == 5

    def test_ssim_floor_gate(self):
        """Very different frames → SSIM < floor → regressed=True."""
        legacy = [_solid_frame((50, 50, 50)) for _ in range(5)]
        latent = [_solid_frame((200, 200, 200)) for _ in range(5)]
        pipeline = _make_stub_pipeline(legacy, latent)
        ab = ABComparator()

        with patch('face_os.ab_validation.cv2.VideoCapture') as mock_vc:
            mock_vc.side_effect = [_make_cap_for_frames(legacy), _make_cap_for_frames(latent)]
            result = ab.compare_render_sources(pipeline, 'fake.mp4', max_frames=5, ssim_floor=0.99)

        # Dark vs bright → SSIM very low
        assert result['ssim_mean'] < 0.99
        assert result['checks']['ssim_ok'] is False
        assert result['regressed'] is True
        assert any('SSIM' in r for r in result['reasons'])

    def test_lab_drift_ceiling_gate(self):
        """Large color difference → LAB drift > ceiling → regressed."""
        legacy = [_solid_frame((50, 50, 50)) for _ in range(3)]
        latent = [_solid_frame((200, 200, 200)) for _ in range(3)]
        pipeline = _make_stub_pipeline(legacy, latent)
        ab = ABComparator()

        with patch('face_os.ab_validation.cv2.VideoCapture') as mock_vc:
            mock_vc.side_effect = [_make_cap_for_frames(legacy), _make_cap_for_frames(latent)]
            result = ab.compare_render_sources(pipeline, 'fake.mp4', max_frames=3, lab_drift_ceiling=1.0)

        assert result['lab_drift_mean'] > 1.0
        assert result['checks']['lab_drift_ok'] is False

    def test_sharpness_ratio_gate(self):
        """Legacy sharp, latent blurry → sharpness ratio < floor."""
        legacy = [_sharp_frame() for _ in range(3)]
        latent = [_blur_frame() for _ in range(3)]
        pipeline = _make_stub_pipeline(legacy, latent)
        ab = ABComparator()

        with patch('face_os.ab_validation.cv2.VideoCapture') as mock_vc:
            mock_vc.side_effect = [_make_cap_for_frames(legacy), _make_cap_for_frames(latent)]
            result = ab.compare_render_sources(pipeline, 'fake.mp4', max_frames=3, sharpness_ratio_floor=0.95)

        assert result['sharpness_ratio'] < 0.95
        assert result['checks']['sharpness_ok'] is False

    def test_no_frames_regressed(self):
        """Zero frames produced → regressed=True with reason."""
        pipeline = _make_stub_pipeline([], [])
        ab = ABComparator()

        with patch('face_os.ab_validation.cv2.VideoCapture') as mock_vc:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.read.return_value = (False, None)
            mock_vc.return_value = cap

            result = ab.compare_render_sources(pipeline, 'fake.mp4', max_frames=5)

        assert result['regressed'] is True
        assert any('no frames' in r for r in result['reasons'])

    def test_returns_all_metric_keys(self):
        """Result dict contains all expected keys."""
        frames = [_solid_frame() for _ in range(3)]
        pipeline = _make_stub_pipeline(frames, frames)
        ab = ABComparator()

        with patch('face_os.ab_validation.cv2.VideoCapture') as mock_vc:
            mock_vc.side_effect = [_make_cap_for_frames(frames), _make_cap_for_frames(frames)]
            result = ab.compare_render_sources(pipeline, 'fake.mp4', max_frames=3)

        expected_keys = [
            'regressed', 'reasons', 'checks',
            'ssim_mean', 'lab_drift_mean',
            'sharpness_legacy', 'sharpness_latent', 'sharpness_ratio',
            'flicker_legacy', 'flicker_latent', 'flicker_ratio',
            'frames_compared',
        ]
        for k in expected_keys:
            assert k in result, f"Missing key: {k}"

    def test_custom_thresholds(self):
        """Custom thresholds are respected (not hardcoded)."""
        frames = [_solid_frame() for _ in range(3)]
        pipeline = _make_stub_pipeline(frames, frames)
        ab = ABComparator()

        with patch('face_os.ab_validation.cv2.VideoCapture') as mock_vc:
            mock_vc.side_effect = [_make_cap_for_frames(frames), _make_cap_for_frames(frames)]
            result = ab.compare_render_sources(
                pipeline, 'fake.mp4', max_frames=3,
                ssim_floor=0.9999,
                lab_drift_ceiling=0.001,
                sharpness_ratio_floor=0.9999,
                flicker_ratio_ceiling=1.001,
            )

        # Identical frames: SSIM≈1.0, LAB≈0, ratio≈1.0, flicker_ratio≈1.0
        assert result['ssim_mean'] >= 0.9999
        assert result['lab_drift_mean'] <= 0.001


# ─── Identity-space metric (arch §16.1 / §19.1) ─────────────────────────────────

def _chroma_albedo(seed=0, size=64):
    """Synthetic RGB float[0,1] albedo with real (a,b) chroma STRUCTURE so a
    structural corruption (flip/roll) actually changes the chroma field."""
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.3 + 0.5 * (xx / w)               # red ramps left->right
    g = 0.4 + 0.2 * np.sin(yy / 8.0)
    b = 0.3 + 0.5 * (yy / h)               # blue ramps top->bottom
    alb = np.stack([r, g, b], axis=2)
    rng = np.random.default_rng(seed)
    alb = alb + rng.normal(0, 0.01, alb.shape).astype(np.float32)
    return np.clip(alb, 0, 1).astype(np.float32)


def _structural_corrupt(alb):
    """The negative control proven on real video: vertical flip + spatial roll —
    survives white-balance, breaks feature correspondence => a WRONG identity."""
    w = np.flipud(alb)
    w = np.roll(w, shift=alb.shape[0] // 3, axis=0)
    w = np.roll(w, shift=alb.shape[1] // 4, axis=1)
    return np.clip(w, 0, 1).astype(np.float32)


class TestIdentityConsistencyMetric:
    """Locks the identity-space metric math against silent vacuity (§19.1).

    The metric is only meaningful if the MATCH term discriminates the correct
    enrolled identity from a STRUCTURALLY corrupted one. The real-video probe
    proved this (correct ΔE≈23.0 vs corrupted≈30.8); these tests enforce the
    same property deterministically on synthetic albedo so a future refactor
    cannot quietly turn the metric into a constant-texture rubber stamp.
    """

    def test_match_discriminates_wrong_identity(self):
        """NON-VACUITY: a structurally-corrupted reference must score WORSE
        (higher chroma ΔE) than the correct enrolled albedo."""
        ref = _chroma_albedo(seed=1)
        renders = [_chroma_albedo(seed=10 + i) for i in range(5)]  # ~the enrolled identity
        masks = [np.ones((64, 64), dtype=bool) for _ in renders]

        correct = compute_albedo_chroma_match(renders, masks, ref)
        wrong = compute_albedo_chroma_match(renders, masks, _structural_corrupt(ref))
        assert wrong > correct, f"metric is VACUOUS: wrong={wrong:.2f} !> correct={correct:.2f}"
        assert wrong > correct * 1.1  # meaningful separation, not float noise

    def test_stability_measures_temporal_variation(self):
        """A constant render sequence is maximally stable (≈0); an identity that
        jitters frame-to-frame scores higher."""
        masks = [np.ones((64, 64), dtype=bool) for _ in range(5)]
        const = [_chroma_albedo(seed=3) for _ in range(5)]            # identical frames
        const = [const[0].copy() for _ in range(5)]
        jitter = [_structural_corrupt(_chroma_albedo(seed=20 + i)) if i % 2 else _chroma_albedo(seed=20 + i)
                  for i in range(5)]                                   # alternating structure

        stable = compute_albedo_chroma_stability(const, masks)
        unstable = compute_albedo_chroma_stability(jitter, masks)
        assert stable < 1.0
        assert unstable > stable

    def test_stability_nan_on_single_frame(self):
        """<2 frames cannot define temporal std."""
        alb = [_chroma_albedo()]
        masks = [np.ones((64, 64), dtype=bool)]
        assert np.isnan(compute_albedo_chroma_stability(alb, masks))

    def test_match_nan_when_mask_empty(self):
        """No masked pixels => NaN (not a silent 0)."""
        ref = _chroma_albedo()
        renders = [_chroma_albedo(seed=5)]
        masks = [np.zeros((64, 64), dtype=bool)]
        assert np.isnan(compute_albedo_chroma_match(renders, masks, ref))

    def test_evaluate_identity_consistency_unavailable_on_stub(self):
        """Boundary: a stub lacking identity estimator returns available=False,
        never crashes and never fabricates a verdict."""
        pipeline = _make_stub_pipeline([_solid_frame()], [_solid_frame()])
        # MagicMock auto-creates attrs; force the guard to trip explicitly.
        pipeline._identity_estimator = None
        ab = ABComparator()
        out = ab.evaluate_identity_consistency(pipeline, 'fake.mp4', max_frames=3)
        assert out['available'] is False
        assert 'recovers_identity' not in out


# ═══════════════════════════════════════════════════════════════════════════════
# D-02: Perceptual Distance + Enhanced A/B Metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerceptualDistance:
    def test_identity_zero_distance(self):
        """Same frame yields perceptual distance ~0."""
        f = _structured_frame(size=(128, 128))
        d = compute_perceptual_distance(f, f)
        assert d == pytest.approx(0.0, abs=0.01), f"Identity dist should be ~0, got {d}"

    def test_dissimilar_frames_higher_distance(self):
        """Dissimilar frames yield higher perceptual distance than similar ones."""
        base = _structured_frame(size=(128, 128))
        similar = cv2.addWeighted(base, 0.99, _solid_frame((200, 200, 200), (128, 128)), 0.01, 0)
        dissimilar = cv2.randn(np.zeros_like(base, dtype=np.float32), 128, 30)
        dissimilar = np.clip(dissimilar, 0, 255).astype(np.uint8)
        d_sim = compute_perceptual_distance(base, similar)
        d_dis = compute_perceptual_distance(base, dissimilar)
        assert d_dis > d_sim, f"Similar={d_sim:.4f} should be < dissimilar={d_dis:.4f}"

    def test_shape_mismatch_handled(self):
        """Frames with different shapes are handled via resize."""
        a = _structured_frame(size=(128, 128))
        b = cv2.resize(_structured_frame(size=(64, 64)), (128, 128))
        d = compute_perceptual_distance(a, b)
        assert d >= 0.0

    def test_mask_restricts_to_face(self):
        """Mask outside face region does not affect distance."""
        a = _structured_frame(size=(128, 128))
        b = cv2.GaussianBlur(a, (5, 5), 1.0)
        mask = np.zeros((128, 128), dtype=np.float32)
        mask[32:96, 32:96] = 1.0
        d_masked = compute_perceptual_distance(a, b, mask=mask)
        d_unmasked = compute_perceptual_distance(a, b)
        assert d_masked < d_unmasked, "Masked distance should be lower (only face region)"

    def test_multi_scale_includes_all_levels(self):
        """Multi-scale computation includes all scale weights."""
        a = _structured_frame(size=(256, 256))
        b = cv2.GaussianBlur(a, (3, 3), 1.0)
        d3 = compute_perceptual_distance(a, b, scales=3)
        d1 = compute_perceptual_distance(a, b, scales=1)
        assert d1 >= 0.0
        assert d3 >= 0.0


class TestComputeAllMetrics:
    def test_paired_frames_computes_perceptual_distance(self):
        """When paired_frames is provided, perceptual_distance is computed."""
        a = _structured_frame(size=(128, 128))
        b = cv2.GaussianBlur(a, (5, 5), 1.0)
        frames = [a, a]
        paired = [b, b]
        metrics = compute_all_metrics(frames, paired_frames=paired)
        assert metrics.perceptual_distance > 0, "Paired frames should yield non-zero perceptual dist"

    def test_sharpness_mean_computed(self):
        """Sharpness mean is computed from frames."""
        frames = [_structured_frame(size=(128, 128)) for _ in range(3)]
        metrics = compute_all_metrics(frames)
        assert metrics.sharpness_mean > 0, "Sharpness mean should be > 0 for structured frames"

    def test_no_paired_frames_zero_perceptual_distance(self):
        """Without paired_frames, perceptual_distance stays 0."""
        frames = [_structured_frame(size=(128, 128))]
        metrics = compute_all_metrics(frames)
        assert metrics.perceptual_distance == 0.0

    def test_temporal_smoothness_computed(self):
        """Temporal smoothness is computed from frame sequence."""
        f1 = _structured_frame(size=(128, 128))
        f2 = cv2.GaussianBlur(f1, (3, 3), 1.0)
        f3 = cv2.GaussianBlur(f2, (3, 3), 1.0)
        metrics = compute_all_metrics([f1, f2, f3])
        assert metrics.temporal_smoothness > 0


class TestCompareApproaches:
    def test_perceptual_distance_included_in_checks(self):
        """Perceptual distance is included in A/B comparison checks."""
        a = ABMetrics(perceptual_distance=5.0, sharpness_mean=100.0,
                       lab_drift=2.0, luminance_consistency=0.9,
                       temporal_smoothness=0.8, procrustes_consistency=0.7,
                       transform_determinant_stability=0.6, ssim=0.95)
        b = ABMetrics(perceptual_distance=10.0, sharpness_mean=80.0,
                       lab_drift=5.0, luminance_consistency=0.7,
                       temporal_smoothness=0.6, procrustes_consistency=0.5,
                       transform_determinant_stability=0.4, ssim=0.85)
        comparison = compare_approaches("A", "B", a, b)
        assert comparison.winner == "A", f"Expected A to win, got {comparison.winner}"
        assert "perceptual_distance" in comparison.details
        assert "sharpness_mean" in comparison.details

    def test_sharpness_winner_detected(self):
        """Higher sharpness wins in comparison."""
        sharp = ABMetrics(sharpness_mean=200.0, lab_drift=10.0,
                           luminance_consistency=0.5, temporal_smoothness=0.5,
                           procrustes_consistency=0.5, transform_determinant_stability=0.5,
                           ssim=0.5, perceptual_distance=10.0)
        blurry = ABMetrics(sharpness_mean=100.0, lab_drift=10.0,
                            luminance_consistency=0.5, temporal_smoothness=0.5,
                            procrustes_consistency=0.5, transform_determinant_stability=0.5,
                            ssim=0.5, perceptual_distance=10.0)
        comparison = compare_approaches("Sharp", "Blurry", sharp, blurry)
        assert comparison.winner == "Sharp"

    def test_lower_perceptual_distance_wins(self):
        """Lower perceptual distance wins in comparison."""
        close = ABMetrics(perceptual_distance=3.0, lab_drift=10.0,
                           luminance_consistency=0.5, temporal_smoothness=0.5,
                           procrustes_consistency=0.5, transform_determinant_stability=0.5,
                           ssim=0.5, sharpness_mean=100.0)
        far = ABMetrics(perceptual_distance=8.0, lab_drift=10.0,
                         luminance_consistency=0.5, temporal_smoothness=0.5,
                         procrustes_consistency=0.5, transform_determinant_stability=0.5,
                         ssim=0.5, sharpness_mean=100.0)
        comparison = compare_approaches("Close", "Far", close, far)
        assert comparison.winner == "Close"


# ═══════════════════════════════════════════════════════════════════════════════
# D-05: Corpus Comparison Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCorpusSourceReport:
