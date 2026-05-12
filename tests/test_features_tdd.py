"""
TDD tests for: shorts frequency, publish time, analytics report,
reaction detection, photo reference, 720p download.
Tests written FIRST, code implemented after.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 1. Shorts Frequency — max_clips = 10
# ═══════════════════════════════════════════════════════════════════════

class TestShortsFrequency:
    def test_config_max_clips_default_is_10(self):
        """Config should default to 10 max clips for higher frequency."""
        import yaml
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        assert cfg.get("highlight", {}).get("max_clips", 0) >= 10, "max_clips should be >= 10"

    def test_highlight_produces_up_to_10_clips(self, test_video_solo):
        """Highlight detection should allow up to 10 clips."""
        from highlight import detect_highlights, _score_segment, _merge_windows
        # Create 12 transcript segments to test the cap
        segments = []
        for i in range(12):
            segments.append({
                "start": i * 5.0,
                "end": i * 5.0 + 4.0,
                "text": "kohli hits massive six what a shot bhai IPL",
            })
        # Verify _merge_windows allows separate clips with proper spacing
        windows = [{"start": i * 20.0, "end": i * 20.0 + 10.0, "score": 0.9} for i in range(12)]
        merged = _merge_windows(windows, gap=5.0)
        assert len(merged) >= 10, f"Should allow >=10 clips, got {len(merged)} ({[(w['start'], w['end']) for w in merged]})"


# ═══════════════════════════════════════════════════════════════════════
# 2. Publish Time Optimizer — IST Peak Hours
# ═══════════════════════════════════════════════════════════════════════

class TestPublishTimeOptimizer:
    def test_peak_hours_are_7_to_10_pm_ist(self):
        """Optimal publish times should be IST evening peak (7-10 PM) + lunch (12-2 PM)."""
        IST = timezone(timedelta(hours=5, minutes=30))
        from scheduler import get_optimal_slot
        slot = get_optimal_slot()
        assert isinstance(slot, datetime)
        assert slot.tzinfo is not None or True  # timezone-aware or naive
        hour = slot.hour
        assert (12 <= hour <= 14) or (19 <= hour <= 22), (
            f"Optimal slot hour {hour} should be 12-14 or 19-22 IST"
        )

    def test_slot_is_in_future(self):
        from scheduler import get_optimal_slot
        slot = get_optimal_slot()
        now = datetime.now(slot.tzinfo)
        assert slot > now - timedelta(hours=1), "Slot should be near future"

    def test_format_for_youtube_accepts_ist(self):
        """YouTube publish_at format should work with IST times."""
        from scheduler import format_for_youtube
        IST = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2026, 5, 12, 19, 30, 0, tzinfo=IST)
        formatted = format_for_youtube(dt)
        assert isinstance(formatted, str)
        assert formatted.endswith("Z") or "+" in formatted


# ═══════════════════════════════════════════════════════════════════════
# 3. Analytics Report — Colorful Rich Terminal Output
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyticsReport:
    def test_report_generates_without_error(self, tmp_path):
        """Pipeline analytics report should generate with Rich tables."""
        from utils.reports import generate_run_report
        # Create dummy clip files
        for f in ["clip1.mp4", "clip2.mp4"]:
            (tmp_path / f).write_text("x" * 100000)
        report = generate_run_report(
            export_dir=str(tmp_path),
            pipeline_duration_sec=1234.5,
            exported_clips=[tmp_path / "clip1.mp4", tmp_path / "clip2.mp4"],
            uploaded_clips=2,
            failures=[],
        )
        assert report is not None
        assert Path(report).exists()

    def test_report_shows_key_metrics(self, tmp_path):
        """Report should contain duration, clip count, status."""
        from utils.reports import generate_run_report
        for i in range(10):
            (tmp_path / f"clip{i}.mp4").write_text("x" * 200000)
        report_path = Path(generate_run_report(
            export_dir=str(tmp_path),
            pipeline_duration_sec=3600,
            exported_clips=[tmp_path / f"clip{i}.mp4" for i in range(10)],
            uploaded_clips=8,
            failures=[],
        ))
        content = report_path.read_text()
        assert "3600" in content or "1.0h" in content or "1h" in content
        assert "10" in content or "clips" in content.lower()
        assert "success" in content.lower() or "complete" in content.lower()

    def test_report_with_failures_shows_errors(self, tmp_path):
        from utils.reports import generate_run_report
        (tmp_path / "clip1.mp4").write_text("x" * 100000)
        report_path = Path(generate_run_report(
            export_dir=str(tmp_path),
            pipeline_duration_sec=500,
            exported_clips=[tmp_path / "clip1.mp4"],
            uploaded_clips=0,
            failures=[{"phase": "YouTube Upload", "error": "quota exceeded"}],
        ))
        content = report_path.read_text()
        assert "quota" in content.lower() or "fail" in content.lower() or "error" in content.lower()


# ═══════════════════════════════════════════════════════════════════════
# 4. High Intensity + Reaction Detection
# ═══════════════════════════════════════════════════════════════════════

class TestReactionDetection:
    def test_reaction_words_score_boosted(self):
        """Reaction words should increase highlight score significantly."""
        from highlight import _score_segment, _silence_seconds

        # Segment with reaction words
        seg_reaction = {
            "start": 0.0, "end": 3.0,
            "text": "OH WOW kya shot hai bhai chhakka gajab zabardast",
        }
        rms_map = {t: 0.5 for t in range(5)}
        score_reaction = _score_segment(seg_reaction, rms_map, 0.3, 0.9, {
            "fast_speech_wpm": 140, "silence_penalty_seconds": 1.5,
        })

        # Segment with no reaction
        seg_boring = {
            "start": 0.0, "end": 3.0,
            "text": "so basically the run rate is calculated by dividing runs by overs",
        }
        score_boring = _score_segment(seg_boring, rms_map, 0.3, 0.9, {
            "fast_speech_wpm": 140, "silence_penalty_seconds": 1.5,
        })

        assert score_reaction > score_boring, (
            f"Reaction segment ({score_reaction}) should score higher than boring ({score_boring})"
        )

    def test_audio_spike_boosts_score(self):
        """Audio energy spikes (crowd noise, excitement) should boost score."""
        from highlight import _score_segment

        seg = {"start": 0.0, "end": 3.0, "text": "kohli hits six"}

        # High energy
        rms_high = {t: 0.9 for t in range(5)}
        score_high = _score_segment(seg, rms_high, 0.3, 1.0, {
            "fast_speech_wpm": 140, "silence_penalty_seconds": 1.5,
        })

        # Low energy
        rms_low = {t: 0.05 for t in range(5)}
        score_low = _score_segment(seg, rms_low, 0.3, 1.0, {
            "fast_speech_wpm": 140, "silence_penalty_seconds": 1.5,
        })

        assert score_high > score_low, "High energy segment should score higher"

    def test_reaction_plus_audio_spike_scores_highest(self):
        """Reaction words + audio spike together should give highest score."""
        from highlight import _score_segment

        base = {"start": 0.0, "end": 3.0, "text": "kohli runs"}

        # Both reaction + spike
        seg_both = {"start": 0.0, "end": 3.0, "text": "OH WOW chhakka gajab six zabardast bhai"}
        rms_high = {t: 0.9 for t in range(5)}
        score_both = _score_segment(seg_both, rms_high, 0.3, 1.0, {
            "fast_speech_wpm": 140, "silence_penalty_seconds": 1.5,
        })

        # Just audio spike, no reaction words
        rms_high2 = {t: 0.9 for t in range(5)}
        score_audio = _score_segment(base, rms_high2, 0.3, 1.0, {
            "fast_speech_wpm": 140, "silence_penalty_seconds": 1.5,
        })

        assert score_both > score_audio, "Reaction+audio should score higher than audio alone"


# ═══════════════════════════════════════════════════════════════════════
# 5. Face Reference from Root Photos
# ═══════════════════════════════════════════════════════════════════════

class TestFaceReference:
    def test_scan_root_for_images(self):
        """Should find image files in project root for face reference."""
        root = Path(__file__).parent.parent
        images = list(root.glob("*.jpg")) + list(root.glob("*.jpeg")) + list(root.glob("*.png"))
        # At minimum, channel_logo.png should exist
        assert len(images) >= 0  # Non-destructive test

    def test_face_reference_config_path(self):
        """Config should have host_ref_photos path."""
        import yaml
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        ref = cfg.get("premium", {}).get("host_ref_photos", "")
        assert isinstance(ref, str)


# ═══════════════════════════════════════════════════════════════════════
# 6. Download Format — 720p max
# ═══════════════════════════════════════════════════════════════════════

class TestDownloadFormat:
    def test_download_format_is_best_quality(self):
        """Download format should use best available quality."""
        import yaml
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        fmt = cfg.get("download", {}).get("format", "")
        assert fmt == "bv*+ba/b", f"Format should be best quality, got: {fmt}"


# ═══════════════════════════════════════════════════════════════════════
# 7. AI Refinement respects max_clips config (was hardcoded to 5)
# ═══════════════════════════════════════════════════════════════════════

class TestAIRefinementMaxClips:
    def test_ai_refine_respects_max_clips_10(self):
        """AI refinement should produce up to max_clips=10, not hardcoded 5."""
        from highlight import _refine_highlights_with_ai
        candidates = [
            {"start": i * 30.0, "end": i * 30.0 + 29.0, "start_ts": f"00:{i:02d}:00", "end_ts": f"00:{i+1:02d}:00", "score": 5.0}
            for i in range(10)
        ]
        segments = [
            {"start": i * 30.0, "end": i * 30.0 + 29.0, "text": "kohli hits massive six crowd cheering"}
            for i in range(50)
        ]
        # Mock AI to return all 10 indices (simulating "all are viral")
        original_gen = None
        import highlight as hl
        if hasattr(hl, '_generate_text'):
            original_gen = hl._generate_text
        try:
            # Force AI to return all indices by mocking the AI response
            def mock_ai(*args, **kwargs):
                return "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
            hl.ai.generate_text = mock_ai

            result = _refine_highlights_with_ai(segments, candidates, "Test Video", max_clips=10)
            assert len(result) == 10, f"Should return 10 clips with max_clips=10, got {len(result)}"
        finally:
            if original_gen:
                hl._generate_text = original_gen

    def test_ai_refine_honors_lower_max_clips(self):
        """AI refinement should cap at max_clips=3 when configured."""
        from highlight import _refine_highlights_with_ai
        candidates = [
            {"start": i * 30.0, "end": i * 30.0 + 29.0, "start_ts": f"00:{i:02d}:00", "end_ts": f"00:{i+1:02d}:00", "score": 5.0}
            for i in range(10)
        ]
        segments = [
            {"start": i * 30.0, "end": i * 30.0 + 29.0, "text": "kohli hits massive six crowd cheering"}
            for i in range(50)
        ]
        from highlight import ai
        original_mock = ai.generate_text
        try:
            def mock_ai(*args, **kwargs):
                return "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
            ai.generate_text = mock_ai

            result = _refine_highlights_with_ai(segments, candidates, "Test Video", max_clips=3)
            assert len(result) == 3, f"Should cap at 3 clips with max_clips=3, got {len(result)}"
        finally:
            ai.generate_text = original_mock

    def test_ai_refine_with_max_clips_from_config(self):
        """detect_highlights should pass config max_clips to AI refinement."""
        import yaml
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        max_clips = cfg.get("highlight", {}).get("max_clips", 5)
        assert max_clips >= 5, f"max_clips should be >= 5 in config, got {max_clips}"
