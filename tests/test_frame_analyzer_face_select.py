"""TDD: face crop selection must never crash on empty candidates."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from frame_analyzer import _apply_top_padding, select_face_crop, analyze_clip


def _face(x=10, y=20, w=80, h=90, conf=0.9, dynamic=False):
    return {
        "face_x": x,
        "face_y": y,
        "face_w": w,
        "face_h": h,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "confidence": conf,
        "is_dynamic_match": dynamic,
    }


def test_face_crop_is_vertical_and_places_face_near_upper_third():
    crop_y, crop_h, crop_w = _apply_top_padding(
        face_top_y=400,
        face_height=90,
        face_width=80,
        frame_height=1080,
        frame_width=1920,
    )

    assert (crop_w, crop_h) == (320, 568)
    assert crop_y == 230
    assert crop_w / crop_h == pytest.approx(9 / 16, abs=0.002)


class TestSelectFaceCropEmpty:
    def test_empty_candidates_has_facecam_no_crash(self):
        crop, no_face = select_face_crop(
            [],
            has_facecam=True,
            facecam={"x": 0, "y": 800, "width": 320, "height": 180},
        )
        assert crop is None
        assert no_face is True

    def test_empty_candidates_no_facecam_no_crash(self):
        crop, no_face = select_face_crop([], has_facecam=False, facecam={})
        assert crop is None
        assert no_face is True


class TestSelectFaceCropMatches:
    def test_prefers_dynamic_match(self):
        faces = [
            _face(x=0, y=0, w=40, h=40, conf=0.5, dynamic=False),
            _face(x=100, y=100, w=50, h=50, conf=0.8, dynamic=True),
        ]
        crop, no_face = select_face_crop(faces, has_facecam=True, facecam={})
        assert no_face is False
        assert crop["is_dynamic_match"] is True
        assert crop["confidence"] == 0.8

    def test_facecam_bounds_hit_best_in_bounds(self):
        # Centers: (40,855) and (60,860) both in facecam; larger should win
        faces = [
            _face(x=10, y=820, w=60, h=70, conf=0.5),
            _face(x=20, y=830, w=80, h=80, conf=0.7),
            _face(x=900, y=50, w=200, h=200, conf=0.99),
        ]
        fc = {"x": 0, "y": 800, "width": 320, "height": 180}
        crop, no_face = select_face_crop(faces, has_facecam=True, facecam=fc)
        assert no_face is False
        assert crop["face_w"] == 80
        assert crop["face_h"] == 80

    def test_facecam_miss_falls_back_to_largest(self):
        faces = [
            _face(x=900, y=50, w=40, h=40, conf=0.5),
            _face(x=500, y=100, w=100, h=120, conf=0.6),
        ]
        fc = {"x": 0, "y": 800, "width": 320, "height": 180}
        crop, no_face = select_face_crop(faces, has_facecam=True, facecam=fc)
        assert no_face is False
        assert crop["face_w"] == 100
        assert crop["face_h"] == 120

    def test_no_facecam_picks_best_area_conf(self):
        faces = [
            _face(w=30, h=30, conf=1.0),
            _face(w=80, h=90, conf=0.5),
        ]
        crop, no_face = select_face_crop(faces, has_facecam=False, facecam={})
        assert no_face is False
        assert crop["face_w"] == 80

    def test_multi_dynamic_highest_conf(self):
        faces = [
            _face(conf=0.4, dynamic=True),
            _face(conf=0.95, dynamic=True),
        ]
        crop, no_face = select_face_crop(faces, has_facecam=False, facecam={})
        assert no_face is False
        assert crop["confidence"] == 0.95


class TestAnalyzeClipEmptyFacesRegression:
    """Production crash: max([]) when has_facecam and zero detections."""

    def test_empty_faces_has_facecam_no_valueerror(self, tmp_path):
        import frame_analyzer as fa

        video = tmp_path / "blank.mp4"
        # Minimal valid-ish path; all heavy IO mocked
        video.write_bytes(b"")

        fake_layout = {
            "layout_type": "solo",
            "is_screen_share": False,
            "guest_cam_on": False,
            "guest_cam_off": False,
            "has_black_panel": False,
            "black_panel_side": None,
            "chat_overlay": None,
        }

        with patch.object(fa, "_sample_brightness_series", return_value=np.array([100.0] * 10)), \
             patch.object(fa, "detect_black_frames", return_value={"is_mostly_black": False}), \
             patch.object(fa, "analyze_lighting", return_value={"needs_correction": False, "lighting_filter": ""}), \
             patch.object(fa, "detect_layout", return_value=fake_layout), \
             patch.object(fa, "detect_dead_air", return_value={"has_dead_air": False}), \
             patch.object(fa, "_get_video_dimensions", return_value={"width": 1920, "height": 1080}), \
             patch.object(fa, "_detect_face_at_timestamp", return_value=None), \
             patch.dict(fa.cfg, {"layout": {"has_facecam": True, "facecam": {"x": 0, "y": 800, "width": 320, "height": 180}}}, clear=False):
            result = analyze_clip(str(video), 0.0, 10.0, clip_id="reg")

        strategy = result["export_strategy"]
        assert strategy["active_crop"] is None
        assert strategy["no_face"] is True
        assert strategy["should_drop"] is True


def test_obstructive_source_lower_third_is_detected():
    import frame_analyzer as fa

    # Detector unit test: force the guard on regardless of runtime config
    # (production config may intentionally disable it).
    original_guard = fa.cfg.get("layout", {}).get("source_overlay_guard", {})
    fa.cfg.setdefault("layout", {})["source_overlay_guard"] = {
        **original_guard, "enabled": True,
    }
    try:
        frame = np.full((180, 320), 100, dtype=np.uint8)
        region = frame[136:174, 96:166]
        region[:] = 10
        region.flat[: max(1, int(region.size * 0.04))] = 255

        assert fa._is_obstructive_lower_third(frame) is True
        assert fa._is_obstructive_lower_third(np.full((180, 320), 100, dtype=np.uint8)) is False
    finally:
        fa.cfg["layout"]["source_overlay_guard"] = original_guard
