"""
tests/test_video_analyzer.py — Comprehensive tests for video_analyzer.py

Covers: lighting analysis, face detection, reference matching, quality scoring,
        frame aggregation, segment selection, and full pipeline integration.
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest

# Ensure project root is on sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_analyzer import (
    _analyze_lighting,
    _score_frame,
    _aggregate_to_seconds,
    _find_best_segments,
    _detect_faces,
    _probe_video,
    analyze_video,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def bright_face_frame():
    """Synthetic 640x480 frame with a bright face-like rectangle."""
    frame = np.full((480, 640, 3), 80, dtype=np.uint8)  # Dark background
    # Draw a bright face-like region (skin-tone-ish)
    cv2.rectangle(frame, (220, 120), (420, 360), (180, 170, 160), -1)
    # Add eyes (dark spots)
    cv2.circle(frame, (280, 220), 12, (40, 40, 40), -1)
    cv2.circle(frame, (360, 220), 12, (40, 40, 40), -1)
    return frame


@pytest.fixture
def dark_frame():
    """Very dark frame — poor lighting."""
    frame = np.full((480, 640, 3), 10, dtype=np.uint8)
    return frame


@pytest.fixture
def overexposed_frame():
    """Blown-out / overexposed frame."""
    frame = np.full((480, 640, 3), 250, dtype=np.uint8)
    return frame


@pytest.fixture
def balanced_frame():
    """Well-balanced mid-tone frame."""
    frame = np.full((480, 640, 3), 140, dtype=np.uint8)
    return frame


@pytest.fixture
def tmp_dir():
    """Temporary directory for test outputs."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_video(tmp_dir):
    """Create a small synthetic 3-second test video."""
    video_path = os.path.join(tmp_dir, "test_video.mp4")
    fps = 10
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(30):  # 3 seconds at 10fps
        frame = np.full((height, width, 3), 80, dtype=np.uint8)
        # Bright region that moves — simulates a face
        cx = int(80 + (i / 30) * 160)  # Moves left to right
        cv2.rectangle(frame, (cx - 30, 60), (cx + 30, 180), (180, 170, 160), -1)
        writer.write(frame)
    writer.release()
    return video_path


@pytest.fixture
def synthetic_video_with_audio(tmp_dir):
    """Create a 3-second video with audio using ffmpeg."""
    video_path = os.path.join(tmp_dir, "test_video_audio.mp4")
    # Generate with ffmpeg — adds silent audio track
    os.system(
        f'ffmpeg -y -f lavfi -i testsrc=duration=3:size=320x240:rate=10 '
        f'-f lavfi -i sine=frequency=440:duration=3 '
        f'-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest '
        f'"{video_path}" 2>/dev/null'
    )
    return video_path


@pytest.fixture
def photos_dir(tmp_dir):
    """Create a photos directory with a reference face image."""
    photos = Path(tmp_dir) / "photos"
    photos.mkdir()

    # Create a synthetic face-like image (skin tone rectangle)
    img = np.full((200, 200, 3), 140, dtype=np.uint8)
    cv2.rectangle(img, (40, 20), (160, 180), (180, 170, 160), -1)
    cv2.circle(img, (80, 80), 10, (40, 40, 40), -1)
    cv2.circle(img, (140, 80), 10, (40, 40, 40), -1)
    cv2.imwrite(str(photos / "ref1.jpg"), img)
    cv2.imwrite(str(photos / "ref2.png"), img)
    return photos


# ─── Lighting Analysis Tests ───────────────────────────────────────────────

class TestAnalyzeLighting:
    def test_bright_frame(self, bright_face_frame):
        result = _analyze_lighting(bright_face_frame)
        assert "brightness" in result
        assert "contrast" in result
        assert "entropy" in result
        assert "face_brightness" in result
        assert "face_count" in result
        assert result["face_count"] >= 0
        assert 0 <= result["overexposed_pct"] <= 100
        assert 0 <= result["underexposed_pct"] <= 100

    def test_dark_frame(self, dark_frame):
        result = _analyze_lighting(dark_frame)
        assert result["brightness"] < 30, "Dark frame should have low brightness"
        assert result["underexposed_pct"] > 50, "Most pixels should be underexposed"

    def test_overexposed_frame(self, overexposed_frame):
        result = _analyze_lighting(overexposed_frame)
        assert result["brightness"] > 240, "Overexposed frame should be very bright"
        assert result["overexposed_pct"] > 80, "Most pixels should be overexposed"

    def test_balanced_frame(self, balanced_frame):
        result = _analyze_lighting(balanced_frame)
        assert 100 <= result["brightness"] <= 180
        assert result["overexposed_pct"] < 5
        assert result["underexposed_pct"] < 5

    def test_face_brightness_populated(self, bright_face_frame):
        result = _analyze_lighting(bright_face_frame)
        # face_brightness should be non-zero if face detected
        if result["face_count"] > 0:
            assert result["face_brightness"] > 0
            assert result["face_area_ratio"] > 0

    def test_entropy_range(self, bright_face_frame):
        result = _analyze_lighting(bright_face_frame)
        assert result["entropy"] >= 0, "Entropy should be non-negative"
        assert result["entropy"] <= 8, "Entropy for 8-bit image max ~8"


# ─── Face Detection Tests ──────────────────────────────────────────────────

class TestDetectFaces:
    def test_returns_list(self, bright_face_frame):
        faces = _detect_faces(bright_face_frame)
        assert isinstance(faces, list)

    def test_returns_tuples(self, bright_face_frame):
        faces = _detect_faces(bright_face_frame)
        for f in faces:
            assert len(f) == 4, "Each face should be (top, right, bottom, left)"
            top, right, bottom, left = f
            assert bottom > top, "bottom must be > top"
            assert right > left, "right must be > left"

    def test_empty_on_blank(self, dark_frame):
        faces = _detect_faces(dark_frame)
        # Dark frame may or may not detect faces depending on cascade
        assert isinstance(faces, list)

    def test_multiple_faces(self):
        """Frame with two face-like regions."""
        frame = np.full((480, 640, 3), 80, dtype=np.uint8)
        cv2.rectangle(frame, (50, 100), (150, 250), (180, 170, 160), -1)
        cv2.rectangle(frame, (400, 100), (500, 250), (180, 170, 160), -1)
        faces = _detect_faces(frame)
        assert isinstance(faces, list)


# ─── Quality Scoring Tests ─────────────────────────────────────────────────

class TestScoreFrame:
    def test_no_face_low_score(self):
        lighting = {
            "brightness": 140, "contrast": 50, "entropy": 6.0,
            "face_brightness": 0, "face_contrast": 0,
            "face_area_ratio": 0, "overexposed_pct": 0,
            "underexposed_pct": 0, "face_count": 0,
        }
        score = _score_frame(lighting, ref_match=0.0, face_detected=False)
        assert score < 0.1, "No face should give very low score"

    def test_face_present_adds_score(self):
        lighting = {
            "brightness": 140, "contrast": 50, "entropy": 6.0,
            "face_brightness": 150, "face_contrast": 40,
            "face_area_ratio": 0.15, "overexposed_pct": 0,
            "underexposed_pct": 0, "face_count": 1,
        }
        score_no_ref = _score_frame(lighting, ref_match=0.0, face_detected=True)
        score_high_ref = _score_frame(lighting, ref_match=0.9, face_detected=True)
        assert score_no_ref > 0.15, "Face present should boost score"
        assert score_high_ref > score_no_ref, "Higher ref match should increase score"

    def test_ideal_lighting_highest(self):
        ideal = {
            "brightness": 140, "contrast": 60, "entropy": 6.0,
            "face_brightness": 150, "face_contrast": 50,
            "face_area_ratio": 0.2, "overexposed_pct": 0,
            "underexposed_pct": 0, "face_count": 1,
        }
        dark = {
            "brightness": 20, "contrast": 10, "entropy": 3.0,
            "face_brightness": 25, "face_contrast": 8,
            "face_area_ratio": 0.1, "overexposed_pct": 0,
            "underexposed_pct": 70, "face_count": 1,
        }
        score_ideal = _score_frame(ideal, ref_match=0.5, face_detected=True)
        score_dark = _score_frame(dark, ref_match=0.5, face_detected=True)
        assert score_ideal > score_dark, "Ideal lighting should score higher than dark"

    def test_overexposure_penalty(self):
        good = {
            "brightness": 140, "contrast": 60, "entropy": 6.0,
            "face_brightness": 150, "face_contrast": 50,
            "face_area_ratio": 0.2, "overexposed_pct": 0,
            "underexposed_pct": 0, "face_count": 1,
        }
        bad_exposure = {
            "brightness": 140, "contrast": 60, "entropy": 6.0,
            "face_brightness": 150, "face_contrast": 50,
            "face_area_ratio": 0.2, "overexposed_pct": 30,
            "underexposed_pct": 20, "face_count": 1,
        }
        s_good = _score_frame(good, ref_match=0.5, face_detected=True)
        s_bad = _score_frame(bad_exposure, ref_match=0.5, face_detected=True)
        assert s_good > s_bad, "Clipped exposure should penalize score"

    def test_score_bounds(self):
        """Score must always be between 0 and 1."""
        for brightness in [0, 50, 128, 200, 255]:
            for face_count in [0, 1]:
                lighting = {
                    "brightness": brightness, "contrast": 40, "entropy": 5.0,
                    "face_brightness": brightness, "face_contrast": 30,
                    "face_area_ratio": 0.15, "overexposed_pct": 5,
                    "underexposed_pct": 5, "face_count": face_count,
                }
                score = _score_frame(lighting, ref_match=0.5, face_detected=face_count > 0)
                assert 0.0 <= score <= 1.0, f"Score {score} out of range for brightness={brightness}"

    def test_high_ref_match_beats_low(self):
        lighting = {
            "brightness": 140, "contrast": 60, "entropy": 6.0,
            "face_brightness": 150, "face_contrast": 50,
            "face_area_ratio": 0.2, "overexposed_pct": 0,
            "underexposed_pct": 0, "face_count": 1,
        }
        scores = [_score_frame(lighting, ref_match=r, face_detected=True) for r in [0.0, 0.3, 0.6, 0.9]]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], f"Score should increase with ref_match: {scores}"


# ─── Aggregation Tests ─────────────────────────────────────────────────────

class TestAggregateToSeconds:
    def test_groups_by_second(self):
        per_frame = [
            {"timestamp": 0.0, "quality": 0.5, "lighting": {"brightness": 100, "face_count": 1}, "ref_match": 0.3},
            {"timestamp": 0.5, "quality": 0.7, "lighting": {"brightness": 120, "face_count": 1}, "ref_match": 0.4},
            {"timestamp": 1.0, "quality": 0.6, "lighting": {"brightness": 110, "face_count": 0}, "ref_match": 0.0},
            {"timestamp": 1.5, "quality": 0.8, "lighting": {"brightness": 130, "face_count": 1}, "ref_match": 0.5},
        ]
        result = _aggregate_to_seconds(per_frame)
        assert len(result) == 2, "Should have 2 seconds"
        assert result[0]["second"] == 0
        assert result[1]["second"] == 1

    def test_averages_quality(self):
        per_frame = [
            {"timestamp": 0.0, "quality": 0.4, "lighting": {"brightness": 100, "face_count": 1}, "ref_match": 0.2},
            {"timestamp": 0.5, "quality": 0.8, "lighting": {"brightness": 100, "face_count": 1}, "ref_match": 0.6},
        ]
        result = _aggregate_to_seconds(per_frame)
        assert abs(result[0]["quality"] - 0.6) < 0.01

    def test_face_present_detection(self):
        per_frame = [
            {"timestamp": 0.0, "quality": 0.5, "lighting": {"brightness": 100, "face_count": 0}, "ref_match": 0.0},
            {"timestamp": 0.3, "quality": 0.6, "lighting": {"brightness": 100, "face_count": 1}, "ref_match": 0.3},
        ]
        result = _aggregate_to_seconds(per_frame)
        assert result[0]["face_present"] is True, "At least one frame had a face"

    def test_empty_input(self):
        result = _aggregate_to_seconds([])
        assert result == []


# ─── Segment Detection Tests ───────────────────────────────────────────────

class TestFindBestSegments:
    def test_basic_segment(self):
        per_second = [
            {"second": i, "quality": 0.5 + (0.3 if 10 <= i <= 15 else 0),
             "brightness": 140, "ref_match": 0.4 if 10 <= i <= 15 else 0.1,
             "face_present": True}
            for i in range(30)
        ]
        segments = _find_best_segments(per_second, total_duration=30.0, window=10, top_n=3)
        assert len(segments) > 0
        assert segments[0]["composite_score"] > 0
        # Best segment should be around second 10-15
        best_start = segments[0]["start_sec"]
        assert 5 <= best_start <= 20, f"Best segment starts at {best_start}, expected ~10"

    def test_no_overlap(self):
        per_second = [
            {"second": i, "quality": 0.8 if i < 10 else 0.3,
             "brightness": 140, "ref_match": 0.5, "face_present": True}
            for i in range(30)
        ]
        segments = _find_best_segments(per_second, total_duration=30.0, window=8, top_n=5)
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                a, b = segments[i], segments[j]
                overlap = a["start_sec"] < b["end_sec"] and a["end_sec"] > b["start_sec"]
                assert not overlap, f"Segments {i} and {j} overlap"

    def test_top_n_limit(self):
        per_second = [
            {"second": i, "quality": 0.9, "brightness": 140,
             "ref_match": 0.8, "face_present": True}
            for i in range(100)
        ]
        segments = _find_best_segments(per_second, total_duration=100.0, window=10, top_n=5)
        assert len(segments) <= 5

    def test_empty_input(self):
        segments = _find_best_segments([], total_duration=0)
        assert segments == []

    def test_single_second(self):
        per_second = [
            {"second": 0, "quality": 0.7, "brightness": 140,
             "ref_match": 0.5, "face_present": True}
        ]
        segments = _find_best_segments(per_second, total_duration=1.0, window=5, top_n=3)
        assert len(segments) >= 1
        assert segments[0]["start_sec"] == 0

    def test_composite_score_weights(self):
        """Best segment should have highest composite score."""
        per_second = [
            {"second": i, "quality": 0.9 if 5 <= i <= 15 else 0.2,
             "brightness": 140,
             "ref_match": 0.8 if 5 <= i <= 15 else 0.1,
             "face_present": 5 <= i <= 15}
            for i in range(30)
        ]
        segments = _find_best_segments(per_second, total_duration=30.0, window=10, top_n=5)
        scores = [s["composite_score"] for s in segments]
        assert scores == sorted(scores, reverse=True), "Segments should be sorted by score desc"


# ─── Probe Video Tests ─────────────────────────────────────────────────────

class TestProbeVideo:
    def test_synthetic_video(self, synthetic_video):
        result = _probe_video(synthetic_video)
        assert result["width"] == 320
        assert result["height"] == 240
        assert result["fps"] > 0
        assert result["duration"] > 0

    def test_nonexistent_video(self):
        result = _probe_video("/nonexistent/video.mp4")
        assert result["width"] == 0
        assert result["duration"] == 0


# ─── Full Integration Test ─────────────────────────────────────────────────

class TestAnalyzeVideo:
    def test_end_to_end(self, synthetic_video, photos_dir, tmp_dir):
        output_path = os.path.join(tmp_dir, "analysis_result.json")
        result = analyze_video(
            video_path=synthetic_video,
            photos_dir=str(photos_dir),
            reference_image="expectation.png",
            sample_interval=1.0,
            output_path=output_path,
        )
        # Should return a valid result
        assert "summary" in result
        assert "per_second" in result
        assert "best_segments" in result

        # Summary checks
        s = result["summary"]
        assert s["duration_sec"] > 0
        assert s["frames_sampled"] > 0
        assert 0 <= s["face_detection_rate"] <= 100
        assert 0 <= s["avg_quality"] <= 1
        assert 0 <= s["max_quality"] <= 1

        # Per-second should have entries
        assert len(result["per_second"]) > 0
        for ps in result["per_second"]:
            assert "second" in ps
            assert "quality" in ps
            assert "face_present" in ps
            assert 0 <= ps["quality"] <= 1

        # Output file should exist
        assert os.path.exists(output_path)

        # Output should be valid JSON
        with open(output_path) as f:
            data = json.load(f)
        assert "summary" in data

    def test_with_audio_video(self, synthetic_video_with_audio, photos_dir, tmp_dir):
        """Test with a video that has an audio track."""
        if not os.path.exists(synthetic_video_with_audio):
            pytest.skip("ffmpeg not available for audio test")
        output_path = os.path.join(tmp_dir, "analysis_audio.json")
        result = analyze_video(
            video_path=synthetic_video_with_audio,
            photos_dir=str(photos_dir),
            reference_image="expectation.png",
            sample_interval=1.5,
            output_path=output_path,
        )
        assert "summary" in result

    def test_no_photos_dir(self, synthetic_video, tmp_dir):
        """Should work without photos directory — just no ref matching."""
        fake_photos = os.path.join(tmp_dir, "nonexistent_photos")
        output_path = os.path.join(tmp_dir, "analysis_nophoto.json")
        result = analyze_video(
            video_path=synthetic_video,
            photos_dir=fake_photos,
            reference_image="nonexistent.png",
            sample_interval=1.0,
            output_path=output_path,
        )
        assert "summary" in result
        assert result["summary"]["reference_photos_used"] == 0

    def test_large_interval(self, synthetic_video, photos_dir, tmp_dir):
        """Large sampling interval should still work."""
        output_path = os.path.join(tmp_dir, "analysis_large.json")
        result = analyze_video(
            video_path=synthetic_video,
            photos_dir=str(photos_dir),
            reference_image="expectation.png",
            sample_interval=5.0,  # Sample every 5s from 3s video = ~1 frame
            output_path=output_path,
        )
        assert result["summary"]["frames_sampled"] >= 1

    def test_nonexistent_video(self, photos_dir, tmp_dir):
        """Should handle missing video gracefully — returns empty/error result."""
        result = analyze_video(
            video_path="/nonexistent/video.mp4",
            photos_dir=str(photos_dir),
            reference_image="expectation.png",
            output_path=os.path.join(tmp_dir, "fail.json"),
        )
        # Should return a result with zero frames sampled, not crash
        assert result["summary"]["frames_sampled"] == 0


# ─── Edge Case Tests ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_black_frames(self, tmp_dir):
        """Video with all-black frames."""
        video_path = os.path.join(tmp_dir, "black.mp4")
        fps = 10
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
        for _ in range(10):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        result = analyze_video(video_path, output_path=os.path.join(tmp_dir, "black_analysis.json"))
        assert result["summary"]["avg_brightness"] < 5

    def test_all_white_frames(self, tmp_dir):
        """Video with all-white frames."""
        video_path = os.path.join(tmp_dir, "white.mp4")
        fps = 10
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
        for _ in range(10):
            frame = np.full((120, 160, 3), 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        result = analyze_video(video_path, output_path=os.path.join(tmp_dir, "white_analysis.json"))
        assert result["summary"]["avg_brightness"] > 250

    def test_single_frame_video(self, tmp_dir):
        """Video with only 1 frame."""
        video_path = os.path.join(tmp_dir, "single.mp4")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
        frame = np.full((120, 160, 3), 128, dtype=np.uint8)
        writer.write(frame)
        writer.release()

        result = analyze_video(video_path, output_path=os.path.join(tmp_dir, "single_analysis.json"))
        assert result["summary"]["frames_sampled"] >= 1

    def test_noisy_frame(self):
        """Random noise frame — should not crash."""
        noisy = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = _analyze_lighting(noisy)
        assert "brightness" in result
        assert result["contrast"] > 10  # Noisy frames have high contrast
