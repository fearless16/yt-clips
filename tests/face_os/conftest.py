"""
tests/face_os/conftest.py — Shared fixtures for Face OS tests.

Provides:
- Reference image loading
- Mock face generation
- Common test utilities
"""

import sys
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import cv2
import numpy as np
import pytest


@pytest.fixture
def reference_image():
    """Load reference image."""
    img = cv2.imread("expectation.png")
    if img is None:
        pytest.skip("expectation.png not found")
    return img


@pytest.fixture
def canonical_face(reference_image):
    """Resize reference to canonical size."""
    return cv2.resize(reference_image, (256, 256))


@pytest.fixture
def cascade():
    """Get Haar Cascade classifier."""
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


@pytest.fixture
def face_detection(cascade, reference_image):
    """Detect face in reference image."""
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(faces) == 0:
        pytest.skip("No face detected in reference")
    return max(faces, key=lambda f: f[2] * f[3])


@pytest.fixture
def landmarks(reference_image, face_detection):
    """Extract landmarks from reference image."""
    from face_os import landmarks as lm_module
    x, y, w, h = face_detection
    return lm_module.extract_landmarks(reference_image, (x, y, w, h))


@pytest.fixture
def mock_face():
    """Generate a mock face image."""
    face = np.ones((256, 256, 3), dtype=np.uint8) * 128
    face[80:180, 80:180] = 200  # Face region
    face[100:120, 110:130] = 180  # Left eye
    face[100:120, 140:160] = 180  # Right eye
    face[140:160, 120:150] = 160  # Mouth
    return face


@pytest.fixture
def quality_map():
    """Generate a quality map."""
    return np.ones((256, 256), dtype=np.float32) * 0.8


@pytest.fixture
def dark_face(canonical_face):
    """Generate a dark version of canonical face."""
    return (canonical_face * 0.6).astype(np.uint8)


@pytest.fixture
def bright_face(canonical_face):
    """Generate a bright version of canonical face."""
    return np.clip(canonical_face.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)


@pytest.fixture
def cold_face(canonical_face):
    """Generate a cold (low b-channel) version of canonical face."""
    lab = cv2.cvtColor(canonical_face, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 2] -= 20
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


@pytest.fixture
def warm_face(canonical_face):
    """Generate a warm (high b-channel) version of canonical face."""
    lab = cv2.cvtColor(canonical_face, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 2] += 20
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
