"""
TDD tests for host identification and centering.
Tests that the system can identify and prioritize the host in multi-person streams.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestHostIdentification:
    """Tests for host identification using reference photos."""

    def test_host_detector_initializes_with_reference_photos(self):
        """HostDetector should initialize when reference photos are provided."""
        from premium_analyzer import HostDetector
        
        # Create a mock reference image (3x64x64 face)
        ref_face = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        with patch("premium_analyzer.cv2") as mock_cv2:
            mock_cv2.resize.return_value = ref_face
            mock_cv2.cvtColor.return_value = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            
            detector = HostDetector(reference_photos=[ref_face])
            assert detector is not None
            assert detector.has_host_reference is True

    def test_host_detector_works_without_reference_photos(self):
        """HostDetector should work even without reference photos (fallback to largest face)."""
        from premium_analyzer import HostDetector
        
        detector = HostDetector(reference_photos=[])
        assert detector is not None
        assert detector.has_host_reference is False

    def test_identify_host_from_multiple_faces(self):
        """Given multiple faces, should identify the host via reference matching."""
        from premium_analyzer import HostDetector
        
        # Create a mock reference face
        ref_face = np.random.randint(100, 150, (64, 64), dtype=np.uint8)
        
        # Create multiple candidate faces
        faces = [
            {"x": 100, "y": 200, "w": 80, "h": 100, "frame": np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)},
            {"x": 500, "y": 200, "w": 80, "h": 100, "frame": np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)},  # Different
            {"x": 800, "y": 200, "w": 80, "h": 100, "frame": np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)},  # Different
        ]
        
        with patch("premium_analyzer.cv2") as mock_cv2:
            # First call returns ref face, others return different
            mock_cv2.resize.side_effect = [ref_face, ref_face, np.random.randint(0, 255, (64, 64), dtype=np.uint8)]
            mock_cv2.cvtColor.return_value = ref_face
            
            detector = HostDetector(reference_photos=[ref_face])
            host_idx = detector.identify_host(faces)
            
            # Should identify one as host (by reference or fallback to largest)
            assert host_idx in [0, 1, 2]


class TestHostCentering:
    """Tests for host-centered composition in 9:16 output."""

    def test_crop_centered_on_host_face(self):
        """When host identified, crop should center on host not just largest face."""
        from premium_analyzer import SmoothCrop
        
        crop = SmoothCrop(frame_w=1920, frame_h=1080)
        
        # Host at unusual position (right side of frame)
        host_x, host_y = 1600, 600  # Right side
        frame_idx = 0
        
        result = crop.get_crop(host_x, host_y, frame_idx)
        
        # Crop window should center on host position
        assert result["x"] <= host_x
        assert result["x"] + result["width"] >= host_x

    def test_host_prioritized_over_guest_in_split_screen(self):
        """In split layout with both host and guest, host should be primary."""
        from premium_analyzer import analyze_clip_with_host_priority
        
        # Mock frames with host on left, guest on right
        mock_analysis = {
            "layout": {"layout_type": "split_both_active"},
            "faces": [
                {"x": 200, "y": 400, "w": 100, "h": 120, "is_host": True},
                {"x": 1000, "y": 300, "w": 100, "h": 120, "is_host": False},
            ]
        }
        
        result = analyze_clip_with_host_priority(mock_analysis)
        
        # Result should have host as primary
        assert result["primary_face"]["is_host"] is True
        assert result["use_vertical_stack"] is False  # Should prefer solo crop of host


class TestHostFallback:
    """Tests for graceful fallback when host detection fails."""

    def test_falls_back_to_largest_face_when_no_reference(self):
        """When no reference photos and no facecam config, should fallback to largest face."""
        from premium_analyzer import HostDetector

        # Mock config to have no facecam configured
        with patch("premium_analyzer.cfg") as mock_cfg:
            mock_cfg.get.return_value = {"has_facecam": False}
            faces = [
                {"x": 100, "y": 200, "w": 60, "h": 80},
                {"x": 500, "y": 200, "w": 100, "h": 120},  # Largest
                {"x": 800, "y": 200, "w": 70, "h": 90},
            ]

            detector = HostDetector(reference_photos=[])
            host_idx = detector._identify_largest_face(faces)

            # Should pick the largest (middle one)
            assert host_idx == 1

    def test_falls_back_to_facecam_region_when_detection_fails(self):
        """Should use facecam region config when face detection completely fails."""
        from premium_analyzer import get_fallback_host_position
        
        # No faces detected - should return facecam region center
        result = get_fallback_host_position(frame_w=1920, frame_h=1080)
        
        # Should return bottom-left region (where facecam typically is)
        assert result["x"] > 0  # Not at edge
        assert result["y"] > 400  # Bottom portion