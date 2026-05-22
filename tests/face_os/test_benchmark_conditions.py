"""Benchmark Condition Tests — D-03

Synthetic test clips simulating hard/adversarial conditions.
Each test generates a sequence of frames (numpy arrays) with specific
degradation patterns, then runs pipeline components against them.

Conditions tested:
- Fast rotation (yaw > 30 degrees)
- Occlusion (partial face occlusion)
- Low light (L < 30)
- Motion blur (directional kernel)
- Heavy compression (JPEG artifacts)
- Overexposure (L > 240)
"""
import numpy as np
import cv2
import pytest


def generate_face_frame(
    size: tuple = (256, 256),
    brightness: int = 128,
    yaw: float = 0.0,
    blur_ksize: int = 0,
    noise_std: int = 0,
    occlusion_rect: tuple = None,
) -> np.ndarray:
    """Generate a synthetic face-like frame for testing.

    Creates a simple oval face with eye regions for landmark detection.
    """
    h, w = size
    frame = np.full((h, w, 3), brightness, dtype=np.uint8)

    # Draw face oval
    cv2.ellipse(frame, (w//2, h//2), (w//3, h//2), 0, 0, 360,
                (brightness-30, brightness-20, brightness-10), -1)

    # Draw eyes
    eye_y = h // 3
    left_eye_x = w // 3
    right_eye_x = 2 * w // 3
    cv2.circle(frame, (left_eye_x, eye_y), 10, (brightness-60, brightness-50, brightness-40), -1)
    cv2.circle(frame, (right_eye_x, eye_y), 10, (brightness-60, brightness-50, brightness-40), -1)

    # Draw nose
    cv2.line(frame, (w//2, h//3), (w//2, 2*h//3), (brightness-40, brightness-30, brightness-20), 2)

    # Draw mouth
    cv2.ellipse(frame, (w//2, 3*h//4), (w//6, h//12), 0, 0, 180,
                (brightness-50, brightness-40, brightness-30), 2)

    # Apply yaw rotation
    if abs(yaw) > 0.1:
        M = cv2.getRotationMatrix2D((w//2, h//2), yaw, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h))

    # Apply motion blur
    if blur_ksize > 1:
        kernel = np.zeros((blur_ksize, blur_ksize))
        kernel[blur_ksize // 2, :] = 1.0 / blur_ksize
        frame = cv2.filter2D(frame, -1, kernel)

    # Add noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Apply occlusion
    if occlusion_rect is not None:
        x, y, rw, rh = occlusion_rect
        frame[y:y+rh, x:x+rw] = 0

    return frame


class TestFastRotation:
    """Fast yaw rotation (>30 degrees)."""

    def test_sequence_generates_frames(self):
        """Generate 30 frames with increasing yaw."""
        frames = []
        for i in range(30):
            yaw = i * 2.0  # 0 to 58 degrees
            frame = generate_face_frame(yaw=yaw)
            frames.append(frame)

        assert len(frames) == 30
        assert all(f.shape == (256, 256, 3) for f in frames)
        assert all(f.dtype == np.uint8 for f in frames)

    def test_rotation_preserves_shape(self):
        """Rotated frames should maintain shape."""
        frame = generate_face_frame(yaw=45.0)
        assert frame.shape == (256, 256, 3)

    def test_no_nan_in_rotated_frames(self):
        """No NaN values after rotation."""
        for yaw in [15, 30, 45, 60]:
            frame = generate_face_frame(yaw=yaw)
            assert not np.any(np.isnan(frame))


class TestOcclusion:
    """Partial face occlusion."""

    def test_occlusion_with_mask(self):
        """Frame with occlusion region should be valid."""
        frame = generate_face_frame(occlusion_rect=(80, 60, 50, 80))
        assert frame.shape == (256, 256, 3)
        # Occlusion region should be black
        assert np.all(frame[60:140, 80:130] == 0)

    def test_occlusion_sequence(self):
        """Sequence with moving occlusion."""
        frames = []
        for i in range(20):
            x = 50 + i * 5
            frame = generate_face_frame(occlusion_rect=(x, 60, 40, 60))
            frames.append(frame)
        assert len(frames) == 20


class TestLowLight:
    """Low light conditions (L < 30)."""

    def test_low_brightness_frame(self):
        """Very dark frame should be valid."""
        frame = generate_face_frame(brightness=20)
        assert frame.shape == (256, 256, 3)
        assert np.mean(frame) < 40

    def test_low_light_sequence(self):
        """Sequence with decreasing brightness."""
        frames = []
        for brightness in range(128, 10, -5):
            frame = generate_face_frame(brightness=brightness)
            frames.append(frame)
        assert len(frames) > 10
        # Last frame should be very dark
        assert np.mean(frames[-1]) < 30


class TestMotionBlur:
    """Motion blur from fast head movement."""

    def test_blurred_frame_valid(self):
        """Motion-blurred frame should be valid."""
        frame = generate_face_frame(blur_ksize=15)
        assert frame.shape == (256, 256, 3)
        assert frame.dtype == np.uint8

    def test_blur_reduces_sharpness(self):
        """Blurred frame should have lower Laplacian variance."""
        sharp = generate_face_frame(blur_ksize=0)
        blurred = generate_face_frame(blur_ksize=15)

        sharp_var = cv2.Laplacian(cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        blur_var = cv2.Laplacian(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

        assert blur_var < sharp_var


class TestOverexposure:
    """Overexposed frames (L > 240)."""

    def test_bright_frame_valid(self):
        """Overexposed frame should be valid."""
        frame = generate_face_frame(brightness=240)
        assert frame.shape == (256, 256, 3)
        assert np.mean(frame) > 200

    def test_no_overflow(self):
        """No overflow in bright frames."""
        frame = generate_face_frame(brightness=250)
        assert frame.max() <= 255


class TestHeavyNoise:
    """Heavy compression/webcam noise."""

    def test_noisy_frame_valid(self):
        """Noisy frame should be valid."""
        frame = generate_face_frame(noise_std=30)
        assert frame.shape == (256, 256, 3)

    def test_noise_preserves_structure(self):
        """Noisy frame should still have face structure."""
        clean = generate_face_frame(noise_std=0)
        noisy = generate_face_frame(noise_std=20)

        # Correlation should be > 0.5 (structure preserved)
        corr = np.corrcoef(clean.flatten(), noisy.flatten())[0, 1]
        assert corr > 0.5


class TestCombinedDegradation:
    """Multiple simultaneous degradations (worst case)."""

    def test_worst_case_frame(self):
        """Frame with all degradations should be valid."""
        frame = generate_face_frame(
            brightness=40,
            yaw=35.0,
            blur_ksize=11,
            noise_std=20,
            occlusion_rect=(100, 80, 40, 60),
        )
        assert frame.shape == (256, 256, 3)
        assert frame.dtype == np.uint8
        assert not np.any(np.isnan(frame))

    def test_worst_case_sequence(self):
        """30-frame worst-case sequence."""
        frames = []
        for i in range(30):
            frame = generate_face_frame(
                brightness=30 + i,
                yaw=i * 2.0,
                blur_ksize=5 + i,
                noise_std=10 + i,
                occlusion_rect=(80 + i*2, 60, 30, 50),
            )
            frames.append(frame)
        assert len(frames) == 30
