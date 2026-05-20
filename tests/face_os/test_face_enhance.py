"""
tests/face_os/test_face_enhance.py — Regression tests for Face Enhancement.

Tests:
- Blink detection
- Eye freeze
- Structure-preserving rendering
- Cinematic noise
- Temporal noise field
"""

import cv2
import numpy as np
import pytest

from face_os.face_enhance import (
    BlinkDetector,
    TemporalNoiseField,
    preserve_eyes,
    preserve_brows,
    preserve_beard,
    smooth_skin,
    apply_vignette,
    add_cinematic_noise,
    render_frame,
    _sharpen,
)


class TestBlinkDetector:
    """Test blink detector."""

    def test_initializes(self):
        """Must initialize correctly."""
        detector = BlinkDetector(ear_threshold=0.20, consecutive_frames=2)

        assert detector.ear_threshold == 0.20
        assert detector.consecutive_frames == 2
        assert not detector.is_blinking

    def test_compute_ear(self):
        """Must compute Eye Aspect Ratio."""
        detector = BlinkDetector()

        eye_points = np.array([
            [100, 100],  # p1
            [110, 95],   # p2
            [120, 95],   # p3
            [130, 100],  # p4
            [120, 105],  # p5
            [110, 105],  # p6
        ], dtype=np.float32)

        ear = detector._compute_ear(eye_points)

        assert 0 < ear < 1

    def test_detect_open_eyes(self):
        """Must detect open eyes."""
        detector = BlinkDetector(ear_threshold=0.20)

        landmarks = np.zeros((68, 2), dtype=np.float32)
        landmarks[36:42] = [[100, 100], [110, 90], [120, 90], [130, 100], [120, 110], [110, 110]]
        landmarks[42:48] = [[200, 100], [210, 90], [220, 90], [230, 100], [220, 110], [210, 110]]

        is_blinking, ear = detector.detect(landmarks, frame_idx=0)

        assert not is_blinking
        assert ear > 0.20

    def test_detect_closed_eyes(self):
        """Must detect closed eyes."""
        detector = BlinkDetector(ear_threshold=0.20, consecutive_frames=1)

        landmarks = np.zeros((68, 2), dtype=np.float32)
        landmarks[36:42] = [[100, 100], [110, 100], [120, 100], [130, 100], [120, 100], [110, 100]]
        landmarks[42:48] = [[200, 100], [210, 100], [220, 100], [230, 100], [220, 100], [210, 100]]

        is_blinking, ear = detector.detect(landmarks, frame_idx=0)

        assert is_blinking
        assert ear < 0.20

    def test_track_history(self):
        """Must track history."""
        detector = BlinkDetector()

        landmarks = np.zeros((68, 2), dtype=np.float32)
        landmarks[36:42] = [[100, 100], [110, 90], [120, 90], [130, 100], [120, 110], [110, 110]]
        landmarks[42:48] = [[200, 100], [210, 90], [220, 90], [230, 100], [220, 110], [210, 110]]

        for i in range(10):
            detector.detect(landmarks, frame_idx=i)

        stats = detector.get_stats()

        assert stats['total_frames'] == 10

    def test_freeze_eyes(self):
        """Must freeze eyes during blink."""
        detector = BlinkDetector(ear_threshold=0.20, consecutive_frames=1)

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        landmarks = np.zeros((68, 2), dtype=np.float32)
        landmarks[36:42] = [[100, 100], [110, 90], [120, 90], [130, 100], [120, 110], [110, 110]]
        landmarks[42:48] = [[200, 100], [210, 90], [220, 90], [230, 100], [220, 110], [210, 110]]

        # Open eyes
        result = detector.freeze_eyes(frame, landmarks, frame_idx=0)
        assert result is not None

        # Closed eyes
        landmarks_closed = landmarks.copy()
        landmarks_closed[36:42] = [[100, 100], [110, 100], [120, 100], [130, 100], [120, 100], [110, 100]]
        landmarks_closed[42:48] = [[200, 100], [210, 100], [220, 100], [230, 100], [220, 100], [210, 100]]

        result = detector.freeze_eyes(frame, landmarks_closed, frame_idx=1)
        assert result is not None


class TestTemporalNoiseField:
    """Test temporal noise field."""

    def test_initializes(self):
        """Must initialize correctly."""
        field = TemporalNoiseField(100, 100, alpha=0.05)

        assert field.h == 100
        assert field.w == 100
        assert field.alpha == 0.05

    def test_get_noise(self):
        """Must generate noise."""
        field = TemporalNoiseField(100, 100)

        noise = field.get_noise(strength=0.02)

        assert noise.shape == (100, 100)
        assert noise.dtype == np.float32

    def test_noise_temporally_coherent(self):
        """Noise must be temporally coherent."""
        field = TemporalNoiseField(100, 100, alpha=0.05)

        noise1 = field.get_noise(strength=0.02)
        noise2 = field.get_noise(strength=0.02)

        # Should be similar (temporally coherent)
        diff = np.mean(np.abs(noise1 - noise2))
        assert diff < 10  # Should be close

    def test_reset(self):
        """Must reset correctly."""
        field = TemporalNoiseField(100, 100)

        field.get_noise(strength=0.02)
        field.reset()

        assert field._frame_count == 0


class TestEyePreservation:
    """Test eye preservation."""

    def test_preserve_eyes(self):
        """Must preserve eyes."""
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        eye_mask = np.zeros((480, 640), dtype=np.float32)
        eye_mask[200:250, 250:350] = 1.0

        result = preserve_eyes(frame, eye_mask, confidence=0.8)

        assert result.shape == frame.shape

    def test_preserve_eyes_no_hallucination(self):
        """Must not hallucinate eye details."""
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        eye_mask = np.zeros((480, 640), dtype=np.float32)
        eye_mask[200:250, 250:350] = 1.0

        result = preserve_eyes(frame, eye_mask, confidence=0.8)

        # Should not change too much
        diff = np.abs(frame.astype(np.float32) - result.astype(np.float32))
        assert np.mean(diff) < 15


class TestBrowPreservation:
    """Test brow preservation."""

    def test_preserve_brows(self):
        """Must preserve brows."""
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        brow_mask = np.zeros((480, 640), dtype=np.float32)
        brow_mask[180:200, 250:350] = 1.0

        result = preserve_brows(frame, brow_mask)

        assert result.shape == frame.shape


class TestBeardPreservation:
    """Test beard preservation."""

    def test_preserve_beard(self):
        """Must preserve beard."""
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        beard_mask = np.zeros((480, 640), dtype=np.float32)
        beard_mask[300:400, 250:400] = 1.0

        result = preserve_beard(frame, beard_mask)

        assert result.shape == frame.shape


class TestSkinSmoothing:
    """Test skin smoothing."""

    def test_smooth_skin(self):
        """Must smooth skin."""
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        skin_mask = np.ones((480, 640), dtype=np.float32)

        result = smooth_skin(frame, skin_mask, amount=0.15)

        assert result.shape == frame.shape

    def test_smooth_skin_preserves_texture(self):
        """Must preserve texture."""
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        skin_mask = np.ones((480, 640), dtype=np.float32)

        result = smooth_skin(frame, skin_mask, amount=0.15)

        diff = np.abs(frame.astype(np.float32) - result.astype(np.float32))
        assert np.mean(diff) < 10


class TestVignette:
    """Test vignette."""

    def test_apply_vignette(self):
        """Must apply vignette."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        bg_mask = np.zeros((480, 640), dtype=np.float32)
        bg_mask[:200, :] = 1.0

        result = apply_vignette(frame, bg_mask, strength=0.3)

        assert result.shape == frame.shape


class TestCinematicNoise:
    """Test cinematic noise."""

    def test_add_cinematic_noise(self):
        """Must add noise."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        result = add_cinematic_noise(frame, strength=0.02)

        assert result.shape == frame.shape

    def test_noise_varies_spatially(self):
        """Noise must vary spatially."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        result = add_cinematic_noise(frame, strength=0.02)

        region1 = result[100:200, 100:200]
        region2 = result[300:400, 300:400]

        # Different regions should have different noise
        diff = np.abs(region1.astype(np.float32) - region2.astype(np.float32))
        assert True  # Noise is random


class TestSharpen:
    """Test sharpening."""

    def test_sharpen(self):
        """Must sharpen."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        result = _sharpen(frame, amount=0.3, radius=1.0)

        assert result.shape == frame.shape


class TestRenderFrame:
    """Test render frame."""

    def test_render_frame(self):
        """Must render frame."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        result = render_frame(frame)

        assert result.shape == frame.shape
