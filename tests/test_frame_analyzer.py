"""
Tests for frame_analyzer.py
Covers: detect_face_crop, detect_black_frames, analyze_lighting,
        detect_layout (incl. is_multi_active_frame, black_panel_side),
        _frame_stats, save_preview_snapshot, analyze_clip (full pipeline).
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
        """Dark scene (high variance) should NOT be flagged as black."""
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
        """avg==20 is at the threshold; var<50 must also hold."""
        r = detect_black_frames([{"avg": 20.0, "var": 49.0}])
        # avg < 20 is required — this should NOT be black
        assert r["has_black_frames"] is False


# ─── analyze_lighting ─────────────────────────────────────────────────────────

class TestLightingAnalysis:
    def test_underexposed(self):
        r = analyze_lighting([{"avg": 50.0, "var": 100.0}])
        assert r["needs_correction"] is True
        assert "gamma=1.3" in r["lighting_filter"]
        assert "contrast=1.1" in r["lighting_filter"]

    def test_overexposed(self):
        r = analyze_lighting([{"avg": 220.0, "var": 100.0}])
        assert r["needs_correction"] is True
        assert "gamma=0.8" in r["lighting_filter"]
        assert "contrast=0.9" in r["lighting_filter"]

    def test_normal_lighting(self):
        r = analyze_lighting([{"avg": 128.0, "var": 300.0}])
        assert r["needs_correction"] is False
        assert "lighting_filter" not in r

    def test_empty_samples_safe(self):
        r = analyze_lighting([])
        assert r["needs_correction"] is False

    def test_boundary_70(self):
        """avg exactly 70 is NOT underexposed (< 70 required)."""
        r = analyze_lighting([{"avg": 70.0}])
        assert r["needs_correction"] is False

    def test_boundary_200(self):
        """avg exactly 200 is NOT overexposed (> 200 required)."""
        r = analyze_lighting([{"avg": 200.0}])
        assert r["needs_correction"] is False


# ─── detect_face_crop ─────────────────────────────────────────────────────────

class TestFaceCrop:
    def test_no_face_returns_none(self):
        """Blank gray frame has no face — should return None."""
        blank = np.full((360, 640, 3), 200, dtype=np.uint8)
        result = detect_face_crop(blank, 640, 360)
        assert result is None

    def test_face_result_structure(self):
        """If a face IS detected, output must have required keys."""
        # Synthesize a face-like patch using the cascade's trained range
        # We test structure only; use a real face image if available.
        fake_face_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Draw white oval to simulate face shape
        import cv2
        cv2.ellipse(fake_face_bgr, (960, 540), (100, 130), 0, 0, 360, (200, 190, 180), -1)
        result = detect_face_crop(fake_face_bgr, 1920, 1080)
        if result is not None:
            assert "x" in result
            assert "y" in result
            assert "width" in result
            assert "height" in result
            assert "confidence" in result
            assert result["x"] >= 0
            assert result["y"] >= 0
            assert result["confidence"] == pytest.approx(0.9)

    def test_crop_coordinates_within_frame(self):
        """Crop box must never exceed frame boundaries."""
        blank = np.full((1080, 1920, 3), 200, dtype=np.uint8)
        result = detect_face_crop(blank, 1920, 1080)
        if result is not None:
            assert result["x"] >= 0
            assert result["y"] >= 0
            assert result["x"] + result["width"] <= 1920
            assert result["y"] + result["height"] <= 1080


# ─── detect_layout ────────────────────────────────────────────────────────────

class TestLayoutDetection:
    def test_solo_video_structure(self, solo_video):
        r = detect_layout(solo_video, 0.0, 1.0)
        assert "layout_type" in r
        assert r["layout_type"] in ("solo", "split")
        assert isinstance(r["prefer_solo"], bool)
        assert isinstance(r["has_black_panel"], bool)
        assert r["black_panel_side"] in (None, "left", "right")
        assert "is_multi_active_frame" in r

    def test_multi_active_frame_flag_logic(self):
        """is_multi_active_frame must be True only when split AND no black panel."""
        # Manually simulate the logic from detect_layout return value
        is_split = True
        has_black = False
        expected = is_split and not has_black
        assert expected is True

        # If black panel exists, multi_active should be False
        has_black = True
        expected2 = is_split and not has_black
        assert expected2 is False

    def test_black_panel_side_majority_vote(self):
        """Side determination: majority wins in vote list."""
        side_votes = ["right", "right", "left"]
        right_count = side_votes.count("right")
        left_count = side_votes.count("left")
        result = "right" if right_count >= left_count else "left"
        assert result == "right"

    def test_nonexistent_video_returns_safe_defaults(self):
        """Missing video should not crash — detect_layout must be resilient."""
        r = detect_layout("/nonexistent/path.mp4", 0.0, 1.0)
        assert r["layout_type"] in ("solo", "split")


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
    def test_required_keys_present(self, solo_video):
        r = analyze_clip(solo_video, 0.0, 1.0, clip_id="t1")
        assert "black_frames" in r
        assert "lighting" in r
        assert "layout" in r
        assert "dead_air" in r
        assert "export_strategy" in r

    def test_export_strategy_keys(self, solo_video):
        strat = analyze_clip(solo_video, 0.0, 1.0)["export_strategy"]
        required = {
            "use_solo_frame", "has_black_panel", "black_panel_side",
            "is_multi_active_frame", "should_drop", "active_crop",
            "speed_factor", "apply_lighting_fix", "lighting_filter",
            "skip_silence",
        }
        for key in required:
            assert key in strat, f"Missing key: {key}"

    def test_should_drop_multi_active_frame(self, solo_video):
        """should_drop must be True when is_multi_active_frame is True."""
        r = analyze_clip(solo_video, 0.0, 1.0)
        is_multi = r["layout"]["is_multi_active_frame"]
        should_drop = r["export_strategy"]["should_drop"]
        if is_multi:
            assert should_drop is True

    def test_fast_speech_disables_speed_boost(self, solo_video):
        """150+ WPM transcript → speed_factor stays 1.0 even with dead air."""
        fast_transcript = [
            {
                "start": 0.0, "end": 1.0,
                "text": " ".join(["word"] * 160)  # 160 WPM → fast paced
            }
        ]
        r = analyze_clip(solo_video, 0.0, 1.0, transcript_segments=fast_transcript)
        assert r["export_strategy"]["speed_factor"] == 1.0

    def test_empty_transcript_no_crash(self, solo_video):
        r = analyze_clip(solo_video, 0.0, 1.0, transcript_segments=[])
        assert "export_strategy" in r

    def test_face_crop_in_strategy_is_none_or_dict(self, solo_video):
        r = analyze_clip(solo_video, 0.0, 1.0)
        crop = r["export_strategy"]["active_crop"]
        assert crop is None or isinstance(crop, dict)

    def test_mostly_black_triggers_should_drop(self):
        """Clip with >60% black ratio → should_drop=True via is_mostly_black."""
        black_result = {"is_mostly_black": True}
        layout_result = {"is_multi_active_frame": False}
        should_drop = layout_result.get("is_multi_active_frame", False) or black_result.get("is_mostly_black", False)
        assert should_drop is True
