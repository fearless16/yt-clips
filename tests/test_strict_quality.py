"""
tests/test_strict_quality.py — STRICT quality tests for Face OS.

These tests enforce absolute quality standards.
LAB distance must be < 5.0. No Haar cascade allowed.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import subprocess
import cv2
import numpy as np
import pytest


def _get_reference_face_lab():
    """Get reference face LAB mean."""
    from face_os.detect_track import detect_faces
    
    exp = cv2.imread("expectation.png")
    if exp is None:
        pytest.skip("expectation.png not found")
    
    detections = detect_faces(exp)
    if not detections:
        pytest.skip("No face in reference")
    
    track = detections[0]
    x, y, w, h = track.smooth_bbox
    face = cv2.resize(exp[y:y+h, x:x+w], (200, 200))
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.mean(lab, axis=(0, 1))


def _get_output_face_lab(frame_idx):
    """Get output face LAB mean at given frame."""
    from face_os.detect_track import detect_faces
    
    cap = cv2.VideoCapture("output/face_os/output_v4.mp4")
    if not cap.isOpened():
        pytest.skip("Output video not found")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    detections = detect_faces(frame)
    if not detections:
        return None
    
    track = detections[0]
    x, y, w, h = track.smooth_bbox
    face = cv2.resize(frame[y:y+h, x:x+w], (200, 200))
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.mean(lab, axis=(0, 1))


class TestLABDistance:
    """Test LAB distance between output and reference."""

    def test_lab_distance_under_5(self):
        """LAB distance must be < 5.0. Currently 20.3 — WILL FAIL until identity blending fixed."""
        ref_lab = _get_reference_face_lab()

        # Sample multiple frames (25-125)
        distances = []
        for frame_idx in [25, 50, 75, 100, 125]:
            out_lab = _get_output_face_lab(frame_idx)
            if out_lab is not None:
                delta = out_lab - ref_lab
                dist = np.sqrt(np.sum(delta ** 2))
                distances.append(dist)

        if not distances:
            pytest.fail("Could not extract faces from output")

        mean_dist = np.mean(distances)
        print(f"\n  LAB distance: {mean_dist:.1f} (target <5.0)")
        print(f"  Reference: L={ref_lab[0]:.1f}, a={ref_lab[1]:.1f}, b={ref_lab[2]:.1f}")

        # STRICT: Must be under 5.0
        assert mean_dist < 5.0, f"LAB distance {mean_dist:.1f} >= 5.0 (target <5.0)"


class TestNoHaar:
    """Test that Haar cascade is completely removed."""

    def test_no_haar_in_codebase(self):
        """grep must return empty for Haar cascade usage."""
        result = subprocess.run(
            ["grep", "-rn", "CascadeClassifier", "face_os/", "--include=*.py"],
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "", f"Found Haar cascade usage:\n{result.stdout}"

    def test_no_haarcascade_in_codebase(self):
        """grep must return empty for haarcascade usage."""
        result = subprocess.run(
            ["grep", "-rn", "haarcascade", "face_os/", "--include=*.py"],
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "", f"Found haarcascade usage:\n{result.stdout}"


class TestVerificationGate:
    """Test VerificationGate rejects bad observations."""

    def test_rejects_tiny_face(self):
        """Face with < 4000 pixels must be rejected."""
        from face_os.identity_state import VerificationGate

        gate = VerificationGate(min_face_pixels=4000)

        # 50x50 = 2500 pixels < 4000
        canonical_face = np.ones((256, 256, 3), dtype=np.uint8) * 128
        face_bbox = (0, 0, 50, 50)

        ok, reason = gate.verify(canonical_face, face_bbox, None, None)
        assert not ok, f"Tiny face should be rejected: {reason}"
        assert "face_too_small" in reason

    def test_accepts_large_face(self):
        """Face with >= 4000 pixels must be accepted."""
        from face_os.identity_state import VerificationGate

        gate = VerificationGate(min_face_pixels=4000)

        # 100x100 = 10000 pixels >= 4000
        canonical_face = np.ones((256, 256, 3), dtype=np.uint8) * 128
        face_bbox = (0, 0, 100, 100)

        ok, reason = gate.verify(canonical_face, face_bbox, None, None)
        assert ok, f"Large face should be accepted: {reason}"
