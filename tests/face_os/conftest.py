"""Shared fixtures for face_os integration tests."""
import os
import pytest
import cv2
import numpy as np


@pytest.fixture(scope='session')
def real_video_path():
    """Path to the real test video clip."""
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'input', 'video.mp4')
    path = os.path.abspath(path)
    if not os.path.exists(path):
        pytest.skip('Real video not available at input/video.mp4')
    return path


@pytest.fixture(scope='session')
def video_frames(real_video_path):
    """Load first 30 frames from the real video."""
    cap = cv2.VideoCapture(real_video_path)
    frames = []
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) > 0, "Failed to read any frames from video"
    return frames


@pytest.fixture(scope='session')
def video_metadata(real_video_path):
    """Video metadata: width, height, fps, total_frames."""
    cap = cv2.VideoCapture(real_video_path)
    meta = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


@pytest.fixture
def mock_face():
    """Simple synthetic face for unit-level tests."""
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    cv2.ellipse(img, (128, 128), (80, 100), 0, 0, 360, (180, 160, 140), -1)
    cv2.circle(img, (100, 100), 12, (60, 50, 40), -1)
    cv2.circle(img, (156, 100), 12, (60, 50, 40), -1)
    cv2.ellipse(img, (128, 160), (30, 10), 0, 0, 360, (100, 80, 80), -1)
    return img


@pytest.fixture
def canonical_face():
    """256x256 canonical-space face with skin-like gradient."""
    h, w = 256, 256
    Y, X = np.mgrid[0:h, 0:w]
    b = np.clip(140 + (X - 128) * 0.2 + (Y - 128) * 0.1, 0, 255).astype(np.uint8)
    g = np.clip(160 + (X - 128) * 0.15, 0, 255).astype(np.uint8)
    r = np.clip(180 - (Y - 128) * 0.1, 0, 255).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)
