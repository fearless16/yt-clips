"""Shared fixtures for face_os tests.

Provides synthetic data fixtures (fast, deterministic) and real-video
fixtures (slower, requires input/video.mp4).

Self-contained: bootstraps the project root onto sys.path so `import face_os`
works when this suite is run in isolation (`pytest face_os/tests/`), without
depending on a root-level conftest.py.
"""
import os
import sys
from pathlib import Path

# Project root is three levels up: face_os/tests/conftest.py -> <root>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Real Video Fixtures (session-scoped, shared across all test modules)
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# Synthetic Face Fixtures (instant, deterministic)
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# Renderer / Intrinsic Synthetic Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_albedo():
    """128x128 float32 skin-tone albedo in [0, 1]."""
    h, w = 128, 128
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    # Skin-like: warm tones with subtle spatial variation
    r = np.clip(0.65 + 0.05 * np.sin(X / 20), 0, 1)
    g = np.clip(0.50 + 0.03 * np.cos(Y / 25), 0, 1)
    b = np.clip(0.40 + 0.02 * np.sin((X + Y) / 30), 0, 1)
    return np.stack([b, g, r], axis=-1).astype(np.float32)


@pytest.fixture
def synthetic_shading():
    """128x128 float32 single-channel shading map in [0, 1]."""
    h, w = 128, 128
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    # Smooth left-to-right illumination gradient
    shading = 0.2 + 0.6 * (X / w)
    return shading[:, :, np.newaxis].astype(np.float32)


@pytest.fixture
def synthetic_normals():
    """128x128 float32 unit normal map (frontal face, Z-dominant)."""
    h, w = 128, 128
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    # Slight curvature from center
    nx = (X - cx) / (w * 2)
    ny = (Y - cy) / (h * 2)
    nz = np.sqrt(np.maximum(1.0 - nx**2 - ny**2, 0.01))
    normals = np.stack([nx, ny, nz], axis=-1).astype(np.float32)
    # Normalize to unit length
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)
    return normals


@pytest.fixture
def skin_tone_image():
    """256x256 uint8 BGR image with realistic skin-tone face patch.

    Useful for intrinsic decomposition tests — has spatial variation
    mimicking illumination gradient across a face.
    """
    h, w = 256, 256
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2

    # Base skin tone
    r = 185 + 15 * np.sin(X / 40)
    g = 150 + 10 * np.cos(Y / 35)
    b = 120 + 8 * np.sin((X + Y) / 50)

    # Illumination gradient (left lit, right shadow)
    illumination = 0.5 + 0.5 * (X / w)
    r = r * illumination
    g = g * illumination
    b = b * illumination

    # Face ellipse mask
    dist = ((X - cx) / 90) ** 2 + ((Y - cy) / 110) ** 2
    mask = (dist < 1.0).astype(np.float32)
    bg_val = 60
    r = r * mask + bg_val * (1 - mask)
    g = g * mask + bg_val * (1 - mask)
    b = b * mask + bg_val * (1 - mask)

    img = np.stack([
        np.clip(b, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        np.clip(r, 0, 255).astype(np.uint8),
    ], axis=-1)
    return img


# ═══════════════════════════════════════════════════════════════════
# Reference Image Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope='session')
def expectation_image():
    """Load expectation.png if available."""
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'expectation.png')
    path = os.path.abspath(path)
    if not os.path.exists(path):
        pytest.skip('expectation.png not available')
    img = cv2.imread(path)
    assert img is not None, "Failed to load expectation.png"
    return img


@pytest.fixture(scope='session')
def reference_photos_dir():
    """Path to photos/ directory if available."""
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'photos')
    path = os.path.abspath(path)
    if not os.path.exists(path):
        pytest.skip('photos/ directory not available')
    return path
