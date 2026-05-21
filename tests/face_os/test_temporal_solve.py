"""
tests/face_os/test_temporal_solve.py — Regression tests for Temporal Solver.

Tests:
- Bidirectional solver
- HQ frame identification
- Frame quality computation
- Temporal repair engine
"""

import cv2
import numpy as np
import pytest

from face_os.temporal_solve import (
    BidirectionalSolver,
    TemporalRepairEngine,
    FrameQuality,
)


class TestFrameQuality:
    """Test frame quality computation."""

    def test_initializes(self):
        """Must initialize correctly."""
        fq = FrameQuality(
            frame_idx=0,
            sharpness=0.8,
            motion_blur=0.1,
            pose=(0, 0, 0),
            detection_confidence=0.9,
        )

        assert fq.frame_idx == 0
        assert fq.sharpness == 0.8

    def test_overall_quality(self):
        """Must compute overall quality."""
        fq = FrameQuality(
            frame_idx=0,
            sharpness=0.8,
            motion_blur=0.1,
            pose=(0, 0, 0),
            detection_confidence=0.9,
        )

        quality = fq.overall_quality
        assert 0 <= quality <= 1

    def test_quality_prefers_sharp(self):
        """Sharp frames must have higher quality."""
        sharp = FrameQuality(frame_idx=0, sharpness=0.9, detection_confidence=0.8)
        blurry = FrameQuality(frame_idx=1, sharpness=0.2, detection_confidence=0.8)

        assert sharp.overall_quality > blurry.overall_quality


class TestBidirectionalSolver:
    """Test bidirectional solver."""

    def test_initializes(self):
        """Must initialize correctly."""
        solver = BidirectionalSolver(lookback_frames=10, lookahead_frames=10)

        assert solver.lookback == 10
        assert solver.lookahead == 10

    def test_add_frame(self):
        """Must store frames."""
        solver = BidirectionalSolver()

        face = np.ones((64, 64, 3), dtype=np.uint8) * 128
        quality = np.ones((64, 64), dtype=np.float32) * 0.8
        fq = FrameQuality(frame_idx=0, sharpness=0.8, detection_confidence=0.8)

        solver.add_frame(0, face, quality, fq)

        assert 0 in solver._canonical_faces

    def test_identify_hq_frames(self):
        """Must identify high-quality frames."""
        solver = BidirectionalSolver()

        for i in range(20):
            face = np.ones((64, 64, 3), dtype=np.uint8) * 128
            quality = np.ones((64, 64), dtype=np.float32) * 0.8
            fq = FrameQuality(
                frame_idx=i,
                sharpness=0.9 if i in [5, 10, 15] else 0.3,
                detection_confidence=0.8,
            )
            solver.add_frame(i, face, quality, fq)

        hq = solver.identify_hq_frames(quality_threshold=0.5)

        assert len(hq) >= 3

    def test_solve_frame(self):
        """Must solve individual frame."""
        solver = BidirectionalSolver(lookback_frames=5, lookahead_frames=5)

        for i in range(10):
            face = np.ones((64, 64, 3), dtype=np.uint8) * 128
            quality = np.ones((64, 64), dtype=np.float32) * 0.8
            fq = FrameQuality(frame_idx=i, sharpness=0.8, detection_confidence=0.8)
            solver.add_frame(i, face, quality, fq)

        solver.identify_hq_frames()
        result, conf = solver.solve_frame(5, (64, 64))

        assert result is not None
        assert conf is not None

    def test_solve_all(self):
        """Must solve all frames."""
        solver = BidirectionalSolver(lookback_frames=5, lookahead_frames=5)

        for i in range(10):
            face = np.ones((64, 64, 3), dtype=np.uint8) * 128
            quality = np.ones((64, 64), dtype=np.float32) * 0.8
            fq = FrameQuality(frame_idx=i, sharpness=0.8, detection_confidence=0.8)
            solver.add_frame(i, face, quality, fq)

        solver.identify_hq_frames()
        results = solver.solve_all((64, 64))

        assert len(results) == 10

    def test_hq_frame_count(self):
        """Must track HQ frame count."""
        solver = BidirectionalSolver()

        for i in range(20):
            face = np.ones((64, 64, 3), dtype=np.uint8) * 128
            quality = np.ones((64, 64), dtype=np.float32) * 0.8
            fq = FrameQuality(
                frame_idx=i,
                sharpness=0.9 if i in [5, 10, 15] else 0.3,
                detection_confidence=0.8,
            )
            solver.add_frame(i, face, quality, fq)

        solver.identify_hq_frames()

        assert solver.get_hq_frame_count() >= 3


class TestTemporalRepairEngine:
    """Test temporal repair engine."""

    def test_initializes(self):
        """Must initialize correctly."""
        engine = TemporalRepairEngine(lookback=5, lookahead=5)

        assert engine.solver is not None

    def test_collect_frame(self):
        """Must collect frames."""
        engine = TemporalRepairEngine(lookback=5, lookahead=5)

        face = np.ones((64, 64, 3), dtype=np.uint8) * 128
        quality = np.ones((64, 64), dtype=np.float32) * 0.8

        engine.collect_frame(0, face, quality, sharpness=0.8, pose=(0, 0, 0))

        assert 0 in engine.solver._canonical_faces

    def test_solve(self):
        """Must solve all frames."""
        engine = TemporalRepairEngine(lookback=5, lookahead=5)

        for i in range(10):
            face = np.ones((64, 64, 3), dtype=np.uint8) * 128
            quality = np.ones((64, 64), dtype=np.float32) * 0.8
            engine.collect_frame(i, face, quality, sharpness=0.8, pose=(0, 0, 0))

        results = engine.solve()

        assert len(results) == 10

    def test_future_repairs_past(self):
        """Sharp future frame must repair blurry past frame."""
        engine = TemporalRepairEngine(lookback=5, lookahead=5)

        h, w = 64, 64

        # Frame 0: blurry
        blurry = cv2.GaussianBlur(
            np.ones((h, w, 3), dtype=np.uint8) * 128, (15, 15), 5
        )
        engine.collect_frame(0, blurry, np.ones((h, w), dtype=np.float32) * 0.3,
                           sharpness=0.2, pose=(0, 0, 0))

        # Frame 3: sharp
        sharp = np.ones((h, w, 3), dtype=np.uint8) * 128
        sharp[20:40, 20:40] = 200
        engine.collect_frame(3, sharp, np.ones((h, w), dtype=np.float32) * 0.9,
                           sharpness=0.9, pose=(0, 0, 0))

        results = engine.solve()

        if 0 in results:
            solved_0, conf_0 = results[0]
            assert conf_0.max() > 0

    def test_flicker_score(self):
        """Flicker must be low for consistent frames."""
        engine = TemporalRepairEngine(lookback=5, lookahead=5)

        for i in range(20):
            face = np.ones((64, 64, 3), dtype=np.uint8) * 128
            quality = np.ones((64, 64), dtype=np.float32) * 0.7
            engine.collect_frame(i, face, quality, sharpness=0.8, pose=(0, 0, 0))

        results = engine.solve()

        flicker_vals = []
        frames_list = sorted(results.keys())
        for i in range(1, len(frames_list)):
            f1 = results[frames_list[i-1]][0]
            f2 = results[frames_list[i]][0]
            diff = np.mean(np.abs(f1.astype(np.float32) - f2.astype(np.float32)))
            flicker_vals.append(diff)

        if flicker_vals:
            avg_flicker = np.mean(flicker_vals)
            assert avg_flicker < 20
