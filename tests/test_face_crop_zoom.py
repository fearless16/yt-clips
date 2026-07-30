"""TDD: Face crop must never produce extreme zoom on tiny/erroneous detections.

Bug: _apply_top_padding() computes scale = 270 / face_width. When face_width is
tiny (e.g. 20px), scale = 13.5 → crop_w = 80px → extreme zoom on teeth/mouth.

Fixes needed:
1. Clamp face_width min in _apply_top_padding
2. Reject faces below min area in scoring
3. Reject tiny crops in export._sanitize_strategy
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from frame_analyzer import detect_face_crop


class TestFaceCropZoomGuard:
    """These tests FAIL with current code and PASS after fix."""

    FRAME = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    MIN_CROP_W = 320
    MIN_CROP_H = 568
    # Use bottom-half positions to avoid center-position skip heuristic (norm_y >= 0.55)
    BOTTOM_TINY = (100, 600, 20, 25)       # tiny → should be rejected
    BOTTOM_NORMAL = (100, 600, 200, 280)    # normal → should pass
    BOTTOM_MEDIUM = (100, 600, 120, 160)    # medium → should pass
    BOTTOM_HOST_TINY = {"x": 100, "y": 600, "width": 25, "height": 30, "confidence": 0.9}

    def _run_crop(self, face_bboxes, host_return=None):
        with patch("frame_analyzer.find_host_in_frame") as mock_host:
            mock_host.return_value = host_return
            with patch("utils.face_detect.detect_faces_yunet") as mock_dnn:
                mock_dnn.return_value = face_bboxes
                return detect_face_crop(self.FRAME, 1920, 1080)

    def _assert_reasonable_crop(self, result, msg=""):
        if result is None:
            return
        assert result["width"] >= self.MIN_CROP_W, (
            f"{msg}: Extreme zoom! crop width {result['width']} < {self.MIN_CROP_W}"
        )
        assert result["height"] >= self.MIN_CROP_H, (
            f"{msg}: Extreme zoom! crop height {result['height']} < {self.MIN_CROP_H}"
        )
        assert result["width"] <= 1920
        assert result["height"] <= 1080

    # ── DNN path tests ─────────────────────────────────────────────────────

    def test_tiny_bbox_produces_reasonable_crop(self):
        """Tiny bbox should NOT cause extreme zoom. Either rejected or clamped."""
        result = self._run_crop([self.BOTTOM_TINY])
        self._assert_reasonable_crop(result, "tiny 20x25")

    def test_extremely_tiny_bbox_is_rejected(self):
        """A bbox < 40x40 should be rejected entirely."""
        result = self._run_crop([(100, 600, 5, 5)])
        assert result is None, "Extremely tiny bbox (5x5) should be rejected"

    def test_normal_face_bbox_produces_normal_crop(self):
        """Normal face should produce a valid crop."""
        result = self._run_crop([self.BOTTOM_NORMAL])
        assert result is not None
        self._assert_reasonable_crop(result, "normal face 200x280")
        assert result["face_w"] == 200

    def test_medium_face_bbox(self):
        result = self._run_crop([self.BOTTOM_MEDIUM])
        assert result is not None
        self._assert_reasonable_crop(result, "medium face 120x160")

    def test_mixed_tiny_and_normal_ignores_tiny(self):
        """When one tiny and one normal face, should pick the larger."""
        result = self._run_crop([self.BOTTOM_TINY, self.BOTTOM_NORMAL])
        assert result is not None
        assert result["face_w"] >= 80, "Should have picked the larger face"
        self._assert_reasonable_crop(result, "mixed detections")

    # ── Host match path tests ──────────────────────────────────────────────

    def test_host_match_tiny_bbox_still_reasonable(self):
        """Host match path should also guard against extreme zoom."""
        result = self._run_crop([], host_return=self.BOTTOM_HOST_TINY)
        assert result is not None
        self._assert_reasonable_crop(result, "host match tiny 25x30")

    def test_host_match_normal_bbox(self):
        result = self._run_crop([], host_return={
            "x": 100, "y": 600, "width": 200, "height": 280, "confidence": 0.9,
        })
        assert result is not None
        self._assert_reasonable_crop(result, "host match normal 200x280")


class TestSanitizeStrategyRejectsTinyCrop:
    """_sanitize_strategy must reject crops smaller than threshold."""

    def test_tiny_crop_width_is_rejected(self):
        from export import _sanitize_strategy
        strategy = _sanitize_strategy({
            "active_crop": {"x": 0, "y": 0, "width": 50, "height": 500},
            "use_solo_frame": True,
        })
        assert strategy["active_crop"] is None, "50px wide crop should be rejected"

    def test_tiny_crop_height_is_rejected(self):
        from export import _sanitize_strategy
        strategy = _sanitize_strategy({
            "active_crop": {"x": 0, "y": 0, "width": 500, "height": 50},
            "use_solo_frame": True,
        })
        assert strategy["active_crop"] is None, "50px tall crop should be rejected"

    def test_reasonable_crop_passes(self):
        from export import _sanitize_strategy
        strategy = _sanitize_strategy({
            "active_crop": {"x": 100, "y": 100, "width": 400, "height": 700},
            "use_solo_frame": True,
        })
        assert strategy["active_crop"] is not None
        assert strategy["active_crop"]["width"] == 400
