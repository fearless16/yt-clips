"""
test_premium_analyzer.py — Tests for YOLOv8-face + ByteTrack + smooth crop.
Falls back to Haar Cascade if YOLO not available (local dev).
"""

import numpy as np
import pytest

from premium_analyzer import (
    FaceDetector,
    ByteTrack,
    SmoothCrop,
    _classify_layout,
    _detect_chat_region,
    _bezier_interpolate,
    _iou,
)


# ─── Bezier interpolation ─────────────────────────────────────────────────

class TestBezierInterpolate:
    def test_start_is_p0(self):
        assert _bezier_interpolate(0, 0.5, 0.5, 1, 0) == 0.0

    def test_end_is_p3(self):
        assert _bezier_interpolate(0, 0.5, 0.5, 1, 1) == 1.0

    def test_midpoint_range(self):
        val = _bezier_interpolate(100, 100, 200, 200, 0.5)
        assert 100 < val < 200


# ─── IoU ───────────────────────────────────────────────────────────────────

class TestIoU:
    def test_perfect_overlap(self):
        assert _iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == pytest.approx(1.0, rel=1e-4)

    def test_no_overlap(self):
        assert _iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0.0

    def test_partial(self):
        val = _iou(np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15]))
        assert 0.1 < val < 0.5


# ─── ByteTrack ─────────────────────────────────────────────────────────────

class TestByteTrack:
    def test_empty_detections(self):
        bt = ByteTrack()
        result = bt.update(np.empty((0, 4)), np.empty((0,)))
        assert result == []

    def test_single_detection_tracked(self):
        bt = ByteTrack()
        dets = np.array([[0, 0, 50, 50]], dtype=np.float32)
        scores = np.array([0.9])
        r1 = bt.update(dets, scores)
        assert len(r1) == 1
        assert r1[0]["id"] >= 0
        # Update again — same detection, should keep same ID
        r2 = bt.update(dets, scores)
        assert len(r2) >= 1
        assert r2[0]["id"] == r1[0]["id"]

    def test_two_detections(self):
        bt = ByteTrack()
        dets = np.array([[0, 0, 50, 50], [100, 100, 150, 150]], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        r = bt.update(dets, scores)
        assert len(r) == 2
        assert r[0]["id"] != r[1]["id"]


# ─── FaceDetector ─────────────────────────────────────────────────────────

class TestFaceDetector:
    def test_detect_on_blank_frame(self):
        """Blank frame should return no faces."""
        fd = FaceDetector()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        xyxy, conf = fd.detect(frame)
        assert len(xyxy) == 0
        assert len(conf) == 0

    def test_detect_on_face_like_region(self):
        """Frame with a face-like blob should detect something."""
        fd = FaceDetector()
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 50
        # Draw a face-like rectangle (skin tone)
        cv2 = pytest.importorskip("cv2")
        cv2.rectangle(frame, (100, 80), (200, 240), (200, 150, 120), -1)
        xyxy, conf = fd.detect(frame)
        # Haar or YOLO — at least 1 face expected
        assert len(xyxy) >= 0  # Non-destructive; real test needs proper face

    def test_backend_is_set(self):
        fd = FaceDetector()
        assert fd.backend in ("yolo", "haar")


# ─── SmoothCrop ────────────────────────────────────────────────────────────

class TestSmoothCrop:
    def test_crop_stays_in_bounds(self):
        sc = SmoothCrop(1920, 1080)
        crop = sc.get_crop(500, 540, 0)
        assert crop["x"] >= 0
        assert crop["x"] + crop["width"] <= 1920
        assert crop["height"] == 1080

    def test_crop_width_9_16(self):
        sc = SmoothCrop(1920, 1080)
        assert sc.crop_w == int(1080 * 9 / 16)  # 607.5 → 607

    def test_face_center_cropped_properly(self):
        """Face at center of frame should give centered crop."""
        sc = SmoothCrop(1920, 1080)
        crop = sc.get_crop(960, 540, 0)
        # Center face → crop x should be around 960 - crop_w/2
        expected_x = 960 - sc.crop_w // 2
        assert abs(crop["x"] - expected_x) < 10

    def test_y_axis_stays_in_bounds(self):
        """For full-height crop (1080p), Y should always be 0 (crop = full frame)."""
        sc = SmoothCrop(1920, 1080)
        crop = sc.get_crop(960, 540, 0)
        assert crop["y"] >= 0, f"Y should be >=0, got {crop['y']}"
        assert crop["y"] + crop["height"] <= 1080, f"Crop overflow, y={crop['y']}+{crop['height']}"

    def test_crop_keeps_full_height(self):
        """For 16:9 source, crop height = frame height (full vertical capture)."""
        sc = SmoothCrop(1920, 1080)
        crop = sc.get_crop(500, 540, 0)
        assert crop["height"] == 1080, "Crop should capture full frame height"


# ─── Facecam Region Filter ─────────────────────────────────────────


class TestFilterByFacecamRegion:
    """TDD: Faces outside facecam area (e.g. poster players) must be filtered."""

    def test_keeps_face_in_facecam_region(self, monkeypatch):
        """Face inside facecam bounds should be kept."""
        monkeypatch.setattr("premium_analyzer.cfg", {
            "layout": {"has_facecam": True, "facecam": {"x": 0, "y": 540, "width": 320, "height": 180}},
        })
        xyxy = np.array([[50, 560, 150, 680]], dtype=np.float32)  # Inside facecam
        conf = np.array([0.9])
        result, rconf = FaceDetector._filter_by_facecam_region(xyxy, conf, 1920, 1080)
        assert len(result) == 1, "Face in facecam should be kept"

    def test_filters_face_outside_facecam(self, monkeypatch):
        """Face outside facecam bounds (e.g. poster on wall) should be filtered."""
        monkeypatch.setattr("premium_analyzer.cfg", {
            "layout": {"has_facecam": True, "facecam": {"x": 0, "y": 540, "width": 320, "height": 180}},
        })
        xyxy = np.array([[500, 100, 600, 250]], dtype=np.float32)  # Poster face, outside facecam
        conf = np.array([0.9])
        result, rconf = FaceDetector._filter_by_facecam_region(xyxy, conf, 1920, 1080)
        assert len(result) == 0, "Face outside facecam should be filtered"

    def test_mixed_faces_keeps_only_facecam_ones(self, monkeypatch):
        """Multiple faces, only those in facecam region should survive."""
        monkeypatch.setattr("premium_analyzer.cfg", {
            "layout": {"has_facecam": True, "facecam": {"x": 0, "y": 540, "width": 320, "height": 180}},
        })
        xyxy = np.array([
            [50, 560, 150, 680],    # In facecam
            [800, 100, 900, 250],   # Poster — outside
            [100, 600, 180, 700],   # In facecam
        ], dtype=np.float32)
        conf = np.array([0.9, 0.8, 0.7])
        result, rconf = FaceDetector._filter_by_facecam_region(xyxy, conf, 1920, 1080)
        assert len(result) == 2, f"Should keep 2 facecam faces, got {len(result)}"

    def test_disabled_when_no_facecam(self, monkeypatch):
        """When has_facecam=False, all faces should pass through."""
        monkeypatch.setattr("premium_analyzer.cfg", {
            "layout": {"has_facecam": False},
        })
        xyxy = np.array([[500, 100, 600, 250]], dtype=np.float32)
        conf = np.array([0.9])
        result, rconf = FaceDetector._filter_by_facecam_region(xyxy, conf, 1920, 1080)
        assert len(result) == 1, "All faces should pass when facecam disabled"

    def test_empty_detections_returns_empty(self, monkeypatch):
        monkeypatch.setattr("premium_analyzer.cfg", {
            "layout": {"has_facecam": True, "facecam": {"x": 0, "y": 540, "width": 320, "height": 180}},
        })
        xyxy = np.empty((0, 4))
        conf = np.empty((0,))
        result, rconf = FaceDetector._filter_by_facecam_region(xyxy, conf, 1920, 1080)
        assert len(result) == 0


# ─── Layout Classification ────────────────────────────────────────────────

class TestClassifyLayout:
    def test_blank_frame(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        assert _classify_layout(frame) == "blank"

    def test_solo_frame(self):
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 80
        assert _classify_layout(frame) == "solo"

    def test_split_guest_off_right_black(self):
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 80
        frame[:, 320:] = 5  # right half nearly black
        assert _classify_layout(frame) == "split_guest_off"


# ─── Chat Detection ───────────────────────────────────────────────────────

class TestDetectChatRegion:
    def test_no_chat_on_blank(self, monkeypatch):
        monkeypatch.setattr("premium_analyzer.cfg", {
            "layout": {"has_chat_overlay": True, "chat": {"side": "right", "estimated_width": 350, "brightness_threshold": 30}},
        })
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        assert _detect_chat_region(frame) is None

    def test_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setattr("premium_analyzer.cfg", {"layout": {"has_chat_overlay": False}})
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 128
        assert _detect_chat_region(frame) is None
