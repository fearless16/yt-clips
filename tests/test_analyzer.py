"""
test_analyzer.py — Lightweight smoke tests for frame_analyzer public API.
Non-duplicate cross-checks for save_preview_snapshot and analyze_clip
using the session-scoped test_video fixture from conftest.py.
"""
from frame_analyzer import analyze_clip, detect_black_frames, analyze_lighting, save_preview_snapshot


def test_save_preview_snapshot(test_video, tmp_path):
    out = tmp_path / "preview.jpg"
    ok = save_preview_snapshot(str(test_video), 1.0, str(out))
    assert ok is True
    assert out.exists()
    assert out.stat().st_size > 0


def test_detect_black_frames_unit():
    """True black: low avg AND low variance."""
    assert detect_black_frames([{"avg": 10, "var": 5}])["has_black_frames"] is True
    # Dark with content (high var) — must NOT be flagged
    assert detect_black_frames([{"avg": 18, "var": 800}])["has_black_frames"] is False
    # Bright frames
    assert detect_black_frames([{"avg": 200, "var": 100}])["has_black_frames"] is False


def test_analyze_lighting_unit():
    assert analyze_lighting([{"avg": 50}])["needs_correction"] is True
    assert analyze_lighting([{"avg": 128}])["needs_correction"] is False
    assert analyze_lighting([{"avg": 230}])["needs_correction"] is True


def test_analyze_clip_with_fast_transcript(test_video):
    """Fast speech (>150 WPM) gets slight speedup (1.15x) for pacing."""
    transcript = [
        {"start": 0.0, "end": 1.0, "text": " ".join(["word"] * 200)},  # ~200 WPM
    ]
    res = analyze_clip(str(test_video), 0.0, 1.0, transcript_segments=transcript)
    assert "export_strategy" in res
    assert 1.0 <= res["export_strategy"]["speed_factor"] <= 1.4


def test_analyze_clip_export_strategy_all_keys(test_video):
    """All downstream export keys must be present."""
    strat = analyze_clip(str(test_video), 0.0, 1.0)["export_strategy"]
    for key in ("use_solo_frame", "should_drop", "guest_cam_on", "guest_cam_off",
                "active_crop", "speed_factor", "apply_lighting_fix",
                "lighting_filter", "skip_silence", "is_screen_share"):
        assert key in strat, f"Missing: {key}"
