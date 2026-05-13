"""
Tests for 16:9 to 9:16 static frame conversion.
Verifies that static 16:9 frames are correctly converted to vertical 9:16 Shorts format.
"""
import subprocess
import pytest
from pathlib import Path
import numpy as np
import cv2


@pytest.fixture(scope="module")
def static_16x9_gray():
    """Create a 1-second static gray 16:9 video (no motion)."""
    p = Path("temp/test_16x9_static.mp4")
    p.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=c=gray:s=1920x1080:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(p)
    ], capture_output=True, check=True)
    return str(p)


@pytest.fixture(scope="module")
def static_16x9_pattern():
    """Create a 16:9 video with a visible pattern (for visual verification)."""
    p = Path("temp/test_16x9_pattern.mp4")
    p.parent.mkdir(parents=True, exist_ok=True)
    # Create a gradient pattern with text-like elements
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc2=size=1920x1080:rate=30:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(p)
    ], capture_output=True, check=True)
    return str(p)


class Test16x9To9x16StaticConversion:
    """Test 16:9 static frame conversion to 9:16."""

    def test_static_16x9_center_crop_to_9x16(self, static_16x9_gray):
        """Static 16:9 should center-crop to 9:16."""
        from export import _build_enhance_stack
        from frame_analyzer import analyze_clip

        # Analyze the static video
        analysis = analyze_clip(static_16x9_gray, 0.0, 1.0, clip_id="test_center_crop")

        # Build enhance stack - should result in center 9:16 crop
        enhance = _build_enhance_stack(analysis, source_fps=30.0, use_logo=False)

        # Verify the filter contains 9:16 crop
        assert "crop=" in enhance
        # 1920x1080 -> 1080x1920 means crop width should be 1080 (9/16 of height)
        assert "1080" in enhance or "ih*9/16" in enhance.lower()

    def test_static_16x9_pattern_conversion(self, static_16x9_pattern):
        """Pattern video converts to 9:16 preserving key visual elements."""
        from export import _build_enhance_stack
        from frame_analyzer import analyze_clip

        analysis = analyze_clip(static_16x9_pattern, 0.0, 1.0, clip_id="test_pattern")
        enhance = _build_enhance_stack(analysis, source_fps=30.0, use_logo=False)

        # Should contain scaling and cropping
        assert "scale=" in enhance
        assert "crop=" in enhance

    def test_16x9_to_9x16_aspect_ratio_preserved(self):
        """Output must be exactly 9:16 (1080x1920)."""
        # This test verifies the target dimensions are correct
        from utils.config import load_config

        cfg = load_config()
        width = cfg["export"]["width"]
        height = cfg["export"]["height"]

        assert width == 1080
        assert height == 1920
        assert height / width == 16 / 9

    def test_no_face_solo_center_crop_fallback(self):
        """When no face detected, use center 9:16 crop."""
        from export import _build_enhance_stack
        import numpy as np

        # Create mock analysis with no face detected
        mock_analysis = {
            "export_strategy": {
                "use_solo_frame": False,  # No face detected
                "use_vertical_stack": False,
                "guest_cam_on": False,
                "guest_cam_off": False,
                "is_screen_share": False,
                "should_drop": False,
                "active_crop": None,
                "speed_factor": 1.0,
            }
        }

        enhance = _build_enhance_stack(mock_analysis, source_fps=30.0, use_logo=False)

        # Should fall back to center crop
        assert "crop=" in enhance
        # Default: crop to 9:16 center
        assert "ih" in enhance  # Uses input height


class Test16x9LayoutDetection:
    """Test that layout detection works for various 16:9 inputs."""

    def test_detect_layout_returns_valid_structure(self, static_16x9_gray):
        """detect_layout should return required keys."""
        from frame_analyzer import detect_layout

        result = detect_layout(static_16x9_gray, 0.0, 1.0)

        required_keys = ["layout_type", "prefer_solo", "has_black_panel", "guest_cam_off"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_solo_layout_for_static_16x9(self, static_16x9_gray):
        """Static 16:9 without faces should be classified as solo."""
        from frame_analyzer import detect_layout

        result = detect_layout(static_16x9_gray, 0.0, 1.0)

        # Gray video = solo layout (no split, no screen share)
        assert result["layout_type"] in ["solo", "screen_share"]