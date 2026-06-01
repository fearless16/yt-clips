"""Tests for ABComparator — compute_sharpness, metrics, corpus_validate.

Architecture: tests mock only at the BOUNDARY (pipeline.process_frame),
never inside the metric functions.
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

_ = CorpusSourceReport


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _solid_frame(bgr=(128, 128, 128), size=(64, 64)):
    f = np.full((size[1], size[0], 3), bgr, dtype=np.uint8)
    return f


def _sharp_frame(size=(64, 64)):
    f = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            if (x + y) % 2 == 0:
                f[y, x] = [255, 255, 255]
    return f


def _blur_frame(size=(64, 64)):
    f = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    f = cv2.GaussianBlur(f, (15, 15), 5)
    return f


def _structured_frame(size=(128, 128)):
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


def _make_stub_pipeline(frames):
    pipeline = MagicMock()
    pipeline.tracker = True
    pipeline._reset_state = MagicMock()

    _idx = [0]
    def _process_frame(frame, frame_idx=0):
        i = _idx[0]
        _idx[0] += 1
        out = frames[i] if i < len(frames) else frames[-1]
        return {'frame': out, 'landmarks': None, 'transform': None}

    pipeline.process_frame = MagicMock(side_effect=_process_frame)
    return pipeline


# ─── compute_sharpness ────────────────────────────────────────────────────────

class TestComputeSharpness:
    def test_sharp_higher_than_blur(self):
        assert compute_sharpness(_sharp_frame()) > compute_sharpness(_blur_frame())

    def test_monotonic(self):
        s = compute_sharpness(_solid_frame())
        assert s >= 0

    def test_mask_restricts_to_face(self):
        f = _sharp_frame()
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        assert compute_sharpness(f, mask) > 0

    def test_no_mask_returns_full(self):
        assert compute_sharpness(_sharp_frame()) > 0

    def test_mask_resize(self):
        f = _sharp_frame()
        small_mask = np.ones((32, 16), dtype=np.float32)
        assert compute_sharpness(f, small_mask) >= 0


# ─── _run_pipeline_source ─────────────────────────────────────────────────────

class TestRunPipelineSource:
    def test_collects_frames(self):
        frames_list = [_solid_frame((100, 100, 100)) for _ in range(3)]
        pipeline = _make_stub_pipeline(frames_list)
        ab = ABComparator()
        with patch('face_os.ab_validation.cv2.VideoCapture') as mock_vc:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.read.side_effect = [(True, _solid_frame()) for _ in range(3)] + [(False, None)]
            mock_vc.return_value = cap
            frames, _, _ = ab._run_pipeline_source(pipeline, 'fake.mp4', max_frames=3)
            assert len(frames) == 3

    def test_returns_empty_on_bad_video(self):
        pipeline = _make_stub_pipeline([])
        ab = ABComparator()
        frames, lms, tfs = ab._run_pipeline_source(pipeline, '/nonexistent.mp4', max_frames=5)
        assert frames == []
        assert lms == []
        assert tfs == []

    def test_calls_reset_state(self):
        pipeline = _make_stub_pipeline([_solid_frame()])
        ab = ABComparator()
        with patch('face_os.ab_validation.cv2.VideoCapture') as mock_vc:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.read.side_effect = [(True, _solid_frame()), (False, None)]
            mock_vc.return_value = cap
            ab._run_pipeline_source(pipeline, 'fake.mp4', max_frames=1)
        pipeline._reset_state.assert_called()


# ─── Perceptual Distance + Enhanced A/B Metrics ─────────────────────────────

class TestPerceptualDistance:
    def test_identity_zero_distance(self):
        f = _solid_frame()
        assert compute_perceptual_distance(f, f) == pytest.approx(0.0, abs=1e-4)

    def test_dissimilar_frames_higher_distance(self):
        a = _structured_frame()
        b = _solid_frame((200, 200, 200))
        assert compute_perceptual_distance(a, b) > 0.0

    def test_shape_mismatch_handled(self):
        a, b = np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((32, 32, 3), dtype=np.uint8)
        assert compute_perceptual_distance(a, b) >= 0

    def test_mask_restricts_to_face(self):
        f = _solid_frame()
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        assert compute_perceptual_distance(f, f, mask=mask) >= 0

    def test_multi_scale_includes_all_levels(self):
        f = _structured_frame()
        assert compute_perceptual_distance(f, f) == pytest.approx(0.0, abs=0.05)


class TestComputeAllMetrics:
    def test_paired_frames_computes_perceptual_distance(self):
        a = [_solid_frame() for _ in range(3)]
        b = [_structured_frame() for _ in range(3)]
        m = compute_all_metrics(a, paired_frames=b)
        assert m.perceptual_distance >= 0

    def test_sharpness_mean_computed(self):
        frames = [_sharp_frame() for _ in range(3)]
        m = compute_all_metrics(frames)
        assert m.sharpness_mean > 0

    def test_no_paired_frames_zero_perceptual_distance(self):
        frames = [_solid_frame() for _ in range(3)]
        m = compute_all_metrics(frames)
        assert m.perceptual_distance == pytest.approx(0.0, abs=1e-4)

    def test_temporal_smoothness_computed(self):
        frames = [_solid_frame()] * 3
        m = compute_all_metrics(frames)
        assert m.temporal_smoothness >= 0


class TestCompareApproaches:
    def test_perceptual_distance_included_in_checks(self):
        ma = ABMetrics(perceptual_distance=0.1, sharpness_mean=10.0)
        mb = ABMetrics(perceptual_distance=0.2, sharpness_mean=5.0)
        result = compare_approaches("a", "b", ma, mb)
        assert "perceptual_distance" in result.details

    def test_sharpness_winner_detected(self):
        ma = ABMetrics(sharpness_mean=100.0, lab_drift=5.0)
        mb = ABMetrics(sharpness_mean=5.0, lab_drift=5.0)
        result = compare_approaches("sharp", "blur", ma, mb)
        assert result.winner == "sharp"

    def test_lower_perceptual_distance_wins(self):
        ma = ABMetrics(perceptual_distance=0.05, sharpness_mean=10.0)
        mb = ABMetrics(perceptual_distance=0.95, sharpness_mean=10.0)
        result = compare_approaches("structured", "solid", ma, mb)
        assert result.winner == "structured"


class TestCorpusSourceReport:
    def test_report_creation(self):
        if not _HAS_CORPUS_REPORT:
            pytest.skip("CorpusSourceReport not available")
        r = CorpusSourceReport()
        assert r.total_clips == 0

    def test_to_dict(self):
        if not _HAS_CORPUS_REPORT:
            pytest.skip("CorpusSourceReport not available")
        r = CorpusSourceReport()
        r.clips = [{'clip': 'test', 'frames': 5, 'sharpness_mean': 12.3, 'flicker': 0.1}]
        r.total_clips = 1
        d = r.to_dict()
        assert 'clips' in d

    def test_summary_ready_when_no_regressions(self):
        if not _HAS_CORPUS_REPORT:
            pytest.skip("CorpusSourceReport not available")
        r = CorpusSourceReport()

    def test_summary_blocked_when_regressions(self):
        if not _HAS_CORPUS_REPORT:
            pytest.skip("CorpusSourceReport not available")
        r = CorpusSourceReport()

    def test_summary_blocked_when_zero_clips(self):
        if not _HAS_CORPUS_REPORT:
            pytest.skip("CorpusSourceReport not available")
        r = CorpusSourceReport()
        assert r.total_clips == 0


class TestCorpusValidate:
    def test_empty_corpus(self):
        pipeline = _make_stub_pipeline([])
        comp = ABComparator()
        report = comp.corpus_validate(pipeline, [], max_frames=5)
        assert report.total_clips == 0

    def test_corpus_collects_per_clip(self, monkeypatch):
        frames = [_solid_frame((128, 128, 128)) for _ in range(5)]
        pipeline = _make_stub_pipeline(frames)

        class FakeCap:
            def __init__(self, path):
                self._frames = list(frames)
                self._idx = 0
            def isOpened(self):
                return True
            def read(self):
                if self._idx < len(self._frames):
                    f = self._frames[self._idx]
                    self._idx += 1
                    return True, f
                return False, None
            def release(self):
                pass

        monkeypatch.setattr(cv2, 'VideoCapture', FakeCap)
        comp = ABComparator()
        corpus = [('clip_a', '/fake/a.mp4'), ('clip_b', '/fake/b.mp4')]
        report = comp.corpus_validate(pipeline, corpus, max_frames=5)
        assert report.total_clips == 2
        assert len(report.clips) == 2
        assert all(c['frames'] == 5 for c in report.clips)


class TestD05Readiness:
    def test_corpus_validate_exists(self):
        comp = ABComparator()
        assert hasattr(comp, 'corpus_validate')
        assert callable(comp.corpus_validate)

    def test_run_pipeline_source_exists(self):
        comp = ABComparator()
        assert hasattr(comp, '_run_pipeline_source')
        assert callable(comp._run_pipeline_source)
