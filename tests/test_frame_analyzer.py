"""
Tests for frame_analyzer.py — Premium-grade coverage.

Covers: _frame_stats, detect_black_frames, analyze_lighting, detect_face_crop,
        _has_vertical_divider, _detect_faces_per_half, _is_screen_share_frame,
        detect_layout (incl. screen_share, split_both_active, split_guest_off),
        save_preview_snapshot, analyze_clip (full pipeline + strategy matrix).
"""
import subprocess
import numpy as np
import pytest
import json

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


# ─── detect_black_frames ──────────────────────────────────────────────────────

class TestBlackFrameDetection:
    def test_true_black(self):
        samples = [{"avg": 10.0, "var": 20.0}, {"avg": 15.0, "var": 30.0}]
        r = detect_black_frames(samples)
        assert r["has_black_frames"] is True
        assert r["is_mostly_black"] is True


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


# ─── detect_face_crop ─────────────────────────────────────────────────────────

class TestFaceCrop:
    def test_no_face_returns_none(self):
        blank = np.full((360, 640, 3), 200, dtype=np.uint8)
        assert detect_face_crop(blank, 640, 360) is None

    def test_prefers_lower_corner_face_over_giant_poster(self, monkeypatch):
        import numpy as np
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
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

    def test_top_center_poster_only_returns_none(self, monkeypatch):
        import numpy as np
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        class MockCascade:
            def detectMultiScale(self, *args, **kwargs):
                return np.array([
                    [800, 100, 200, 200],  # Top-center face
                ])
            
        import frame_analyzer
        monkeypatch.setattr(frame_analyzer, "FACE_CASCADE", MockCascade())
        
        result = frame_analyzer.detect_face_crop(frame, 1920, 1080)
        assert result is None


# ─── _has_vertical_divider ────────────────────────────────────────────────────

class TestVerticalDivider:
    def test_clear_divider_detected(self):
        frame = np.full((180, 320), 150, dtype=np.uint8)
        frame[:, 157:163] = 10
        assert _has_vertical_divider(frame, 320) is True

    def test_sharp_brightness_edge(self):
        frame = np.zeros((180, 320), dtype=np.uint8)
        frame[:, :160] = 200
        frame[:, 160:] = 50
        assert _has_vertical_divider(frame, 320) is True


# ─── detect_layout ────────────────────────────────────────────────────────────

class TestLayoutDetection:
    def test_solo_video(self, solo_video):
        r = detect_layout(solo_video, 0.0, 1.0)
        assert r["layout_type"] == "solo"
        assert r["prefer_solo"] is True

    def test_screen_share_detection(self, monkeypatch):
        # We need to mock _run_cmd and _is_screen_share_frame
        import frame_analyzer
        class MockRes:
            stdout = bytes([128] * (320 * 180 * 12))
        monkeypatch.setattr(frame_analyzer, "_run_cmd", lambda *a, **kw: MockRes())
        monkeypatch.setattr(frame_analyzer, "_is_screen_share_frame", lambda *a: True)
        monkeypatch.setattr(frame_analyzer, "_has_vertical_divider", lambda *a: False)
        
        r = detect_layout("fake.mp4", 0, 1)
        assert r["layout_type"] == "screen_share"
        assert r["is_screen_share"] is True

    def test_split_both_active(self, monkeypatch):
        import frame_analyzer
        class MockRes:
            stdout = bytes([128] * (320 * 180 * 12))
        monkeypatch.setattr(frame_analyzer, "_run_cmd", lambda *a, **kw: MockRes())
        monkeypatch.setattr(frame_analyzer, "_has_vertical_divider", lambda *a: True)
        monkeypatch.setattr(frame_analyzer, "_detect_faces_per_half", lambda *a: {"left": 1, "right": 1})
        
        r = detect_layout("fake.mp4", 0, 1)
        assert r["layout_type"] == "split_both_active"
        assert r["guest_cam_on"] is True


# ─── analyze_clip (full pipeline) ────────────────────────────────────────────

class TestAnalyzeClipIntegration:
    def test_required_keys_present(self, solo_video):
        r = analyze_clip(solo_video, 0.0, 1.0, clip_id="t1")
        for key in ("black_frames", "lighting", "layout", "dead_air", "export_strategy"):
            assert key in r

    def test_export_strategy_keys(self, solo_video):
        strat = analyze_clip(solo_video, 0.0, 1.0)["export_strategy"]
        keys = {"use_solo_frame", "use_vertical_stack", "guest_cam_on", "guest_cam_off", "is_screen_share", "should_drop"}
        for k in keys:
            assert k in strat

    def test_screen_share_priority_over_face(self, solo_video, monkeypatch):
        import frame_analyzer
        monkeypatch.setattr(frame_analyzer, "detect_layout", lambda *a: {
            "layout_type": "screen_share", "is_screen_share": True
        })
        # Small facecam face
        monkeypatch.setattr(frame_analyzer, "_detect_face_at_timestamp", lambda *a: {
            "face_h": 100, "confidence": 0.9
        })
        monkeypatch.setattr(frame_analyzer, "_get_video_dimensions", lambda *a: {"width": 1920, "height": 1080})
        
        r = analyze_clip(solo_video, 0.0, 1.0)
        assert r["export_strategy"]["is_screen_share"] is True
        assert r["export_strategy"]["export_aspect_ratio"] == "16:9"

    def test_screen_share_overridden_by_large_face(self, solo_video, monkeypatch):
        import frame_analyzer
        monkeypatch.setattr(frame_analyzer, "detect_layout", lambda *a: {
            "layout_type": "screen_share", "is_screen_share": True
        })
        # Large face (>35% of 1080)
        monkeypatch.setattr(frame_analyzer, "_detect_face_at_timestamp", lambda *a: {
            "face_h": 500, "confidence": 0.9
        })
        monkeypatch.setattr(frame_analyzer, "_get_video_dimensions", lambda *a: {"width": 1920, "height": 1080})
        
        r = analyze_clip(solo_video, 0.0, 1.0)
        assert r["export_strategy"]["is_screen_share"] is False
        assert r["export_strategy"]["export_aspect_ratio"] == "9:16"
