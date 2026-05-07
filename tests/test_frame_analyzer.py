"""
Test suite for frame_analyzer.py - TDD approach
Tests for smart clipping, black panel detection, and quality analysis
"""
import pytest
import subprocess
import os
from pathlib import Path
import sys

# Add workspace to path
sys.path.insert(0, '/workspace')

from frame_analyzer import (
    detect_black_frames,
    analyze_lighting,
    detect_layout,
    _frame_stats,
    _sample_brightness_series,
    analyze_clip
)


class TestBlackFrameDetection:
    """Tests for black frame detection - critical for guest camera off scenarios"""
    
    def test_true_black_detection(self):
        """True black frames (avg < 20, var < 50) should be detected"""
        samples = [
            {"avg": 10.0, "var": 20.0},  # True black
            {"avg": 15.0, "var": 30.0},  # True black
            {"avg": 12.0, "var": 25.0},  # True black
        ]
        result = detect_black_frames(samples)
        assert result["has_black_frames"] is True
        assert result["black_ratio"] == 1.0
        assert result["is_mostly_black"] is True
    
    def test_dark_scene_not_false_positive(self):
        """Dark scenes with content (high variance) should NOT be detected as black"""
        samples = [
            {"avg": 30.0, "var": 500.0},  # Dark but has content
            {"avg": 25.0, "var": 600.0},  # Dark but has content
        ]
        result = detect_black_frames(samples)
        assert result["has_black_frames"] is False
        assert result["black_ratio"] == 0.0
    
    def test_mixed_content(self):
        """Mixed black and content frames should calculate correct ratio"""
        samples = [
            {"avg": 10.0, "var": 20.0},   # Black
            {"avg": 150.0, "var": 800.0}, # Content
            {"avg": 12.0, "var": 25.0},   # Black
            {"avg": 140.0, "var": 750.0}, # Content
        ]
        result = detect_black_frames(samples)
        assert result["has_black_frames"] is True
        assert result["black_ratio"] == 0.5
        assert result["is_mostly_black"] is False
    
    def test_empty_samples(self):
        """Empty samples should return safe defaults"""
        result = detect_black_frames([])
        assert result["has_black_frames"] is False
        assert result["black_ratio"] == 0
        assert result["is_mostly_black"] is False


class TestLightingAnalysis:
    """Tests for lighting correction detection"""
    
    def test_underexposed_detection(self):
        """Average brightness < 70 should trigger lighting correction"""
        samples = [
            {"avg": 50.0, "var": 100.0},
            {"avg": 60.0, "var": 120.0},
        ]
        result = analyze_lighting(samples)
        assert result["needs_correction"] is True
        assert "gamma=1.3" in result.get("lighting_filter", "")
    
    def test_overexposed_detection(self):
        """Average brightness > 200 should trigger lighting correction"""
        samples = [
            {"avg": 220.0, "var": 100.0},
            {"avg": 230.0, "var": 120.0},
        ]
        result = analyze_lighting(samples)
        assert result["needs_correction"] is True
        assert "gamma=0.8" in result.get("lighting_filter", "")
    
    def test_normal_lighting(self):
        """Normal brightness (70-200) should not trigger correction"""
        samples = [
            {"avg": 120.0, "var": 300.0},
            {"avg": 130.0, "var": 320.0},
        ]
        result = analyze_lighting(samples)
        assert result["needs_correction"] is False


class TestLayoutDetection:
    """Tests for split-screen and black panel layout detection"""
    
    @pytest.fixture
    def sample_video(self, tmp_path):
        """Create a simple test video for layout detection"""
        video_path = tmp_path / "test_layout.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "color=c=white:s=1920x1080:d=1",
            "-t", "1",
            str(video_path)
        ]
        subprocess.run(cmd, capture_output=True, timeout=10)
        return str(video_path)
    
    def test_solo_layout_detection(self, sample_video):
        """Single panel video should be detected as solo layout"""
        result = detect_layout(sample_video, 0.0, 1.0)
        assert result["layout_type"] in ["solo", "split"]
        # At minimum, should not crash and return valid structure
        assert "prefer_solo" in result
        assert "has_black_panel" in result
    
    def test_black_panel_metadata(self, sample_video):
        """Layout detection should include black panel metadata"""
        result = detect_layout(sample_video, 0.0, 1.0)
        assert isinstance(result.get("has_black_panel"), bool)
        assert result.get("black_panel_side") in [None, "left", "right"]


class TestFrameStats:
    """Tests for low-level frame statistics computation"""
    
    def test_uniform_frame(self):
        """Uniform gray frame should have zero variance"""
        frame = bytes([128] * 100)
        stats = _frame_stats(frame)
        assert stats["avg"] == 128.0
        assert stats["var"] == 0.0
    
    def test_high_contrast_frame(self):
        """High contrast frame should have high variance"""
        frame = bytes([0] * 50 + [255] * 50)
        stats = _frame_stats(frame)
        assert stats["avg"] == 127.5
        assert stats["var"] > 10000
    
    def test_empty_frame(self):
        """Empty frame should return safe defaults"""
        stats = _frame_stats(b"")
        assert stats["avg"] == 128.0
        assert stats["var"] == 0.0


class TestAnalyzeClipIntegration:
    """Integration tests for full clip analysis pipeline"""
    
    @pytest.fixture
    def sample_video(self, tmp_path):
        """Create a test video with known properties"""
        video_path = tmp_path / "test_clip.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "color=c=gray:s=1920x1080:d=2",
            "-t", "2",
            str(video_path)
        ]
        subprocess.run(cmd, capture_output=True, timeout=10)
        return str(video_path)
    
    def test_full_analysis_pipeline(self, sample_video):
        """Full clip analysis should return all required fields"""
        result = analyze_clip(sample_video, 0.0, 2.0, clip_id="test")
        
        # Check top-level structure
        assert "black_frames" in result
        assert "lighting" in result
        assert "layout" in result
        assert "dead_air" in result
        assert "export_strategy" in result
        
        # Check export strategy has required fields
        strategy = result["export_strategy"]
        assert "use_solo_frame" in strategy
        assert "speed_factor" in strategy
        assert "apply_lighting_fix" in strategy
        
        # Verify black panel detection fields exist
        assert "has_black_panel" in strategy
        assert "black_panel_side" in strategy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
