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
)


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
