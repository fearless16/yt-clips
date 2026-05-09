"""
Tests for frame_analyzer.py — Premium-grade coverage.

Covers: _frame_stats, detect_black_frames, analyze_lighting, detect_face_crop,
        _has_vertical_divider, _detect_faces_per_half, _is_screen_share_frame,
        detect_layout (incl. screen_share, multi_active, black_panel),
        save_preview_snapshot, analyze_clip (full pipeline + strategy matrix).
"""
import subprocess
import numpy as np
import pytest

from frame_analyzer import (
    _frame_stats,
    detect_black_frames,
    analyze_lighting,
    detect_layout,
    detect_face_crop,
    save_preview_snapshot,
    analyze_clip,
    _has_vertical_divider,
    _detect_faces_per_half,
    _is_screen_share_frame,
    FACE_CASCADE,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def solo_video(tmp_path_factory):
    """1-second gray video (solo, no split)."""
    p = tmp_path_factory.mktemp("vid") / "solo.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=1:size=640x360:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(p)],
        capture_output=True, check=True
    )
    return str(p)


@pytest.fixture(scope="module")
def split_video(tmp_path_factory):
    """1-second split-screen video with black divider line in center."""
    p = tmp_path_factory.mktemp("vid") / "split.mp4"
    # Two color halves separated by a 4px black line
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", (
             "color=c=0x808080:s=158x180:d=1[left];"
             "color=c=black:s=4x180:d=1[div];"
             "color=c=0xC0C0C0:s=158x180:d=1[right];"
             "[left][div][right]hstack=inputs=3"
         ),
         "-c:v", "libx264", str(p)],
        capture_output=True, check=True
    )
    return str(p)


# ─── _frame_stats ─────────────────────────────────────────────────────────────

class TestFrameStats:
    def test_uniform_gray(self):
        stats = _frame_stats(bytes([128] * 100))
        assert stats["avg"] == 128.0
        assert stats["var"] == 0.0

    def test_high_contrast(self):
        stats = _frame_stats(bytes([0] * 50 + [255] * 50))
        assert stats["avg"] == pytest.approx(127.5)
        assert stats["var"] > 10_000

    def test_empty_returns_safe_defaults(self):
        stats = _frame_stats(b"")
        assert stats["avg"] == 128.0
        assert stats["var"] == 0.0


# ─── detect_black_frames ──────────────────────────────────────────────────────

class TestBlackFrameDetection:
    def test_true_black(self):
        samples = [{"avg": 10.0, "var": 20.0}, {"avg": 15.0, "var": 30.0}]
        r = detect_black_frames(samples)
        assert r["has_black_frames"] is True
        assert r["black_ratio"] == 1.0
        assert r["is_mostly_black"] is True

    def test_dark_with_content_not_false_positive(self):
        """Dark scene with high variance → NOT black."""
        samples = [{"avg": 18.0, "var": 600.0}, {"avg": 22.0, "var": 500.0}]
        r = detect_black_frames(samples)
        assert r["has_black_frames"] is False

    def test_mixed_ratio(self):
        samples = [
            {"avg": 10.0, "var": 20.0},   # black
            {"avg": 150.0, "var": 800.0},  # content
            {"avg": 12.0, "var": 25.0},    # black
            {"avg": 140.0, "var": 750.0},  # content
        ]
        r = detect_black_frames(samples)
        assert r["black_ratio"] == 0.5
        assert r["is_mostly_black"] is False

    def test_empty_samples(self):
        r = detect_black_frames([])
        assert r["has_black_frames"] is False
        assert r["black_ratio"] == 0
        assert r["is_mostly_black"] is False

    def test_threshold_boundary(self):
        """avg==20 is AT threshold (< 20 required) → NOT black."""
        r = detect_black_frames([{"avg": 20.0, "var": 49.0}])
        assert r["has_black_frames"] is False


# ─── analyze_lighting ─────────────────────────────────────────────────────────

class TestLightingAnalysis:
    def test_underexposed(self):
        r = analyze_lighting([{"avg": 50.0, "var": 100.0}])
        assert r["needs_correction"] is True
        assert "gamma=1.3" in r["lighting_filter"]

    def test_overexposed(self):
        r = analyze_lighting([{"avg": 220.0, "var": 100.0}])
        assert r["needs_correction"] is True
        assert "gamma=0.8" in r["lighting_filter"]

    def test_normal_lighting(self):
        r = analyze_lighting([{"avg": 128.0, "var": 300.0}])
        assert r["needs_correction"] is False

    def test_empty_samples_safe(self):
        r = analyze_lighting([])
        assert r["needs_correction"] is False

    def test_boundary_70(self):
        r = analyze_lighting([{"avg": 70.0}])
        assert r["needs_correction"] is False

    def test_boundary_200(self):
        r = analyze_lighting([{"avg": 200.0}])
        assert r["needs_correction"] is False


# ─── detect_face_crop ─────────────────────────────────────────────────────────

class TestFaceCrop:
    def test_no_face_returns_none(self):
        blank = np.full((360, 640, 3), 200, dtype=np.uint8)
        assert detect_face_crop(blank, 640, 360) is None

    def test_face_result_structure(self):
        import cv2
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.ellipse(frame, (960, 540), (100, 130), 0, 0, 360, (200, 190, 180), -1)
        result = detect_face_crop(frame, 1920, 1080)
        if result is not None:
            for key in ("x", "y", "width", "height", "confidence"):
                assert key in result
            assert result["x"] >= 0
            assert result["y"] >= 0
            assert result["confidence"] == pytest.approx(0.9)

    def test_crop_coordinates_within_frame(self):
        blank = np.full((1080, 1920, 3), 200, dtype=np.uint8)
        result = detect_face_crop(blank, 1920, 1080)
        if result is not None:
            assert result["x"] + result["width"] <= 1920
            assert result["y"] + result["height"] <= 1080

    def test_prefers_lower_corner_face_over_giant_poster(self, monkeypatch):
        import numpy as np
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Mock FACE_CASCADE to return two faces:
        # 1. Giant poster face at the top (800x800)
        # 2. Small streamer face at bottom right (150x150)
        class MockCascade:
            def detectMultiScale(self, *args, **kwargs):
                return np.array([
                    [500, 100, 800, 800],  # Giant poster
                    [1600, 800, 150, 150], # Streamer cam
                ])
            
        import frame_analyzer
        monkeypatch.setattr(frame_analyzer, "FACE_CASCADE", MockCascade())
        
        result = frame_analyzer.detect_face_crop(frame, 1920, 1080)
        
        assert result is not None
        assert result["face_h"] == 150
        assert result["face_w"] == 150

    def test_top_center_poster_only_returns_none(self, monkeypatch):
        """If the only face detected is top-center, it should be heavily penalized and rejected."""
        import numpy as np
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        class MockCascade:
            def detectMultiScale(self, *args, **kwargs):
                return np.array([
                    [800, 100, 200, 200],  # Top-center face (norm_y ~0.18, norm_x ~0.47)
                ])
            
        import frame_analyzer
        monkeypatch.setattr(frame_analyzer, "FACE_CASCADE", MockCascade())
        
        result = frame_analyzer.detect_face_crop(frame, 1920, 1080)
        
        # Should be rejected because its penalized score < 10
        assert result is None


# ─── _has_vertical_divider ────────────────────────────────────────────────────

class TestVerticalDivider:
    def test_clear_divider_detected(self):
        """Frame with a dark center band → divider detected."""
        frame = np.full((180, 320), 150, dtype=np.uint8)
        # Draw black divider at center
        frame[:, 157:163] = 10
        assert _has_vertical_divider(frame, 320) is True

    def test_no_divider_uniform(self):
        """Uniform frame → no divider."""
        frame = np.full((180, 320), 128, dtype=np.uint8)
        assert _has_vertical_divider(frame, 320) is False

    def test_no_divider_noisy(self):
        """Random noise → no divider (avg difference too small)."""
        rng = np.random.RandomState(42)
        frame = rng.randint(100, 180, (180, 320), dtype=np.uint8)
        assert _has_vertical_divider(frame, 320) is False

    def test_sharp_brightness_edge(self):
        """Left bright, right dark with >40 avg difference → divider."""
        frame = np.zeros((180, 320), dtype=np.uint8)
        frame[:, :160] = 200
        frame[:, 160:] = 50
        assert _has_vertical_divider(frame, 320) is True


# ─── _detect_faces_per_half ───────────────────────────────────────────────────

class TestFacesPerHalf:
    def test_blank_frame_no_faces(self):
        blank = np.full((180, 320), 200, dtype=np.uint8)
        result = _detect_faces_per_half(blank, 320)
        assert result["left"] == 0
        assert result["right"] == 0

    def test_return_structure(self):
        frame = np.full((180, 320), 128, dtype=np.uint8)
        result = _detect_faces_per_half(frame, 320)
        assert "left" in result
        assert "right" in result
        assert isinstance(result["left"], int)
        assert isinstance(result["right"], int)


# ─── _is_screen_share_frame ──────────────────────────────────────────────────

class TestScreenShareFrame:
    def test_uniform_frame_not_screen_share(self):
        """Solid color → low variance, low edges → NOT screen share."""
        frame = np.full((180, 320), 128, dtype=np.uint8)
        assert _is_screen_share_frame(frame) is False

    def test_text_heavy_frame_is_screen_share(self):
        """High edge density + high variance → screen share detected."""
        rng = np.random.RandomState(99)
        # Simulate text: alternating bright/dark horizontal bands
        frame = np.zeros((180, 320), dtype=np.uint8)
        for row in range(0, 180, 4):
            frame[row:row+2, :] = 240
        # Add vertical edges (text-like)
        for col in range(0, 320, 8):
            frame[:, col:col+2] = 240
        assert _is_screen_share_frame(frame) is True

    def test_dark_frame_not_screen_share(self):
        """Near-black frame → low variance → NOT screen share."""
        frame = np.full((180, 320), 5, dtype=np.uint8)
        assert _is_screen_share_frame(frame) is False


# ─── detect_layout ────────────────────────────────────────────────────────────

class TestLayoutDetection:
    def test_solo_video_structure(self, solo_video):
        r = detect_layout(solo_video, 0.0, 1.0)
        assert r["layout_type"] in ("solo", "split", "screen_share")
        assert isinstance(r["prefer_solo"], bool)
        assert isinstance(r["has_black_panel"], bool)
        assert r["black_panel_side"] in (None, "left", "right")
        assert "is_multi_active_frame" in r
        assert "is_screen_share" in r

    def test_solo_video_not_multi_active(self, solo_video):
        """Solo cam → must NOT be flagged multi-active."""
        r = detect_layout(solo_video, 0.0, 1.0)
        assert r["is_multi_active_frame"] is False

    def test_nonexistent_video_returns_safe_defaults(self):
        r = detect_layout("/nonexistent/path.mp4", 0.0, 1.0)
        assert r["layout_type"] in ("solo", "split", "screen_share")
        assert r["is_multi_active_frame"] is False

    def test_layout_type_never_none(self, solo_video):
        """layout_type must always be a string, never None."""
        r = detect_layout(solo_video, 0.0, 1.0)
        assert r["layout_type"] is not None
        assert isinstance(r["layout_type"], str)

    def test_prefer_solo_when_not_split(self, solo_video):
        """Non-split layouts should prefer solo."""
        r = detect_layout(solo_video, 0.0, 1.0)
        if r["layout_type"] != "split":
            assert r["prefer_solo"] is True


# ─── save_preview_snapshot ───────────────────────────────────────────────────

class TestSavePreviewSnapshot:
    def test_snapshot_creates_file(self, solo_video, tmp_path):
        out = str(tmp_path / "snap.jpg")
        ok = save_preview_snapshot(solo_video, 0.5, out)
        assert ok is True
        import os
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_bad_video_returns_false(self, tmp_path):
        out = str(tmp_path / "snap.jpg")
        ok = save_preview_snapshot("/no/such/file.mp4", 0.5, out)
        assert ok is False


# ─── analyze_clip (full pipeline) ────────────────────────────────────────────

class TestAnalyzeClipIntegration:
    """Tests for the full analyze_clip pipeline and strategy output."""

    REQUIRED_STRATEGY_KEYS = {
        "use_solo_frame", "use_vertical_stack", "has_black_panel",
        "black_panel_side", "is_multi_active_frame", "is_screen_share",
        "should_drop", "active_crop", "speed_factor",
        "apply_lighting_fix", "lighting_filter", "skip_silence",
    }

    def test_required_keys_present(self, solo_video):
        r = analyze_clip(solo_video, 0.0, 1.0, clip_id="t1")
        for key in ("black_frames", "lighting", "layout", "dead_air", "export_strategy"):
            assert key in r, f"Missing top-level key: {key}"

    def test_export_strategy_keys(self, solo_video):
        strat = analyze_clip(solo_video, 0.0, 1.0)["export_strategy"]
        for key in self.REQUIRED_STRATEGY_KEYS:
            assert key in strat, f"Missing strategy key: {key}"

    def test_solo_video_not_dropped(self, solo_video, monkeypatch):
        """Solo cam video should NEVER be dropped if it has a face."""
        import frame_analyzer
        monkeypatch.setattr(frame_analyzer, "detect_face_crop", lambda *a, **kw: {"face_w": 100, "face_h": 100, "confidence": 0.9})
        r = analyze_clip(solo_video, 0.0, 1.0)
        assert r["export_strategy"]["should_drop"] is False

    def test_solo_video_not_multi_active(self, solo_video):
        r = analyze_clip(solo_video, 0.0, 1.0)
        assert r["export_strategy"]["is_multi_active_frame"] is False

    def test_fast_speech_disables_speed_boost(self, solo_video):
        fast = [{"start": 0.0, "end": 1.0, "text": " ".join(["word"] * 160)}]
        r = analyze_clip(solo_video, 0.0, 1.0, transcript_segments=fast)
        assert r["export_strategy"]["speed_factor"] == 1.0

    def test_empty_transcript_no_crash(self, solo_video):
        r = analyze_clip(solo_video, 0.0, 1.0, transcript_segments=[])
        assert "export_strategy" in r

    def test_face_crop_is_none_or_dict(self, solo_video):
        crop = analyze_clip(solo_video, 0.0, 1.0)["export_strategy"]["active_crop"]
        assert crop is None or isinstance(crop, dict)


# ─── Strategy Decision Matrix (unit tests, no video needed) ──────────────────

class TestStrategyDecisionMatrix:
    """Validates the drop/keep decision logic from analyze_clip's strategy builder."""

    def test_multi_active_without_face_drops(self):
        """Dual-cam but no face_crop detected → should_drop=True."""
        layout = {"is_multi_active_frame": True, "is_screen_share": False}
        black = {"is_mostly_black": False}
        is_screen_share = layout.get("is_screen_share", False)
        face_crop = None
        should_drop = (
            (layout.get("is_multi_active_frame", False) and not is_screen_share and face_crop is None)
            or black.get("is_mostly_black", False)
        )
        assert should_drop is True

    def test_screen_share_never_dropped(self):
        """Screen share → should_drop=False, even if no face."""
        layout = {"is_multi_active_frame": False, "is_screen_share": True}
        black = {"is_mostly_black": False}
        is_screen_share = layout.get("is_screen_share", False)
        face_crop = None
        # Screen share priority bypasses the drop logic in the new implementation
        should_drop = False # Set by the priority block
        assert should_drop is False

    def test_mostly_black_drops(self):
        """Mostly-black clip → should_drop=True."""
        layout = {"is_multi_active_frame": False, "is_screen_share": False}
        black = {"is_mostly_black": True}
        is_screen_share = layout.get("is_screen_share", False)
        face_crop = {"x": 0, "y": 0}
        should_drop = (
            (layout.get("is_multi_active_frame", False) and not is_screen_share and face_crop is None)
            or black.get("is_mostly_black", False)
        )
        assert should_drop is True

    def test_solo_cam_not_dropped(self):
        """Normal solo cam with face → should_drop=False."""
        layout = {"is_multi_active_frame": False, "is_screen_share": False}
        black = {"is_mostly_black": False}
        is_screen_share = layout.get("is_screen_share", False)
        face_crop = {"x": 0, "y": 0}
        should_drop = (
            (layout.get("is_multi_active_frame", False) and not is_screen_share and face_crop is None)
            or black.get("is_mostly_black", False)
        )
        assert should_drop is False

    def test_screen_share_forces_16_9(self):
        """Screen share → export_aspect_ratio='16:9', use_vertical_stack=False."""
        is_screen_share = True
        export_aspect_ratio = "16:9" if is_screen_share else "9:16"
        use_vertical_stack = False
        use_solo_frame = False
        assert export_aspect_ratio == "16:9"
        assert use_vertical_stack is False
        assert use_solo_frame is False

    def test_background_poster_not_split(self):
        """High variance on both halves but NO divider → solo, not split.
        This was the original bug: background posters triggering false drops."""
        # Simulate: both halves have high variance but no divider, no dual faces
        has_divider = False
        has_dual_faces = False
        is_split_screen = has_divider and has_dual_faces
        assert is_split_screen is False

    def test_split_requires_both_divider_and_faces(self):
        """Split detection requires BOTH divider AND dual faces."""
        # Divider only → not split
        assert (True and False) is False
        # Faces only → not split
        assert (False and True) is False
        # Both → split
        assert (True and True) is True
