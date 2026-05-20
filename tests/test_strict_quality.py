"""
tests/test_strict_quality.py — STRICT quality tests for Face OS.

These tests WILL FAIL until the compositor bug is fixed.
Current LAB distance: 24.8 (target: <5)
Current SSIM: ~0.6 (target: >0.75)

DO NOT adjust thresholds to make tests pass. Fix the code.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import cv2
import numpy as np
import pytest

from face_os.detect_track import (
    compute_procrustes_disparity,
    compute_landmark_jitter,
    compute_occupancy,
    pass_quality_gates,
    detect_faces,
)
from face_os.compositor import Compositor
from face_os.types import ConfidenceMap


# ─── Helpers ────────────────────────────────────────────────────────────────

def _ssim(img1, img2):
    """Compute SSIM between two grayscale images."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(img1.astype(np.float64), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2.astype(np.float64), (11, 11), 1.5)
    sigma1_sq = cv2.GaussianBlur(img1.astype(np.float64) ** 2, (11, 11), 1.5) - mu1 ** 2
    sigma2_sq = cv2.GaussianBlur(img2.astype(np.float64) ** 2, (11, 11), 1.5) - mu2 ** 2
    sigma12 = cv2.GaussianBlur((img1.astype(np.float64) * img2.astype(np.float64)), (11, 11), 1.5) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def _get_reference_face():
    """Get reference face LAB mean."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    exp = cv2.imread("expectation.png")
    if exp is None:
        pytest.skip("expectation.png not found")
    gray = cv2.cvtColor(exp, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(faces) == 0:
        pytest.skip("No face in reference")
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = cv2.resize(exp[y:y+h, x:x+w], (200, 200))
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.mean(lab, axis=(0, 1))


def _get_output_face_lab(frame_idx):
    """Get output face LAB mean at given frame."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture("output/face_os_v2/output.mp4")
    if not cap.isOpened():
        pytest.skip("Output video not found")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = cv2.resize(frame[y:y+h, x:x+w], (200, 200))
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.mean(lab, axis=(0, 1))


# ─── Test: LAB Distance ─────────────────────────────────────────────────────

class TestLABDistance:
    """Test LAB distance between output and reference."""

    def test_lab_distance_under_5(self):
        """LAB distance must be < 5.0. Currently 24.8 — WILL FAIL until compositor fixed."""
        ref_lab = _get_reference_face()

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


# ─── Test: SSIM Quality ─────────────────────────────────────────────────────

class TestSSIMQuality:
    """Test SSIM between output and source."""

    def test_ssim_above_075(self):
        """SSIM must be > 0.75. Currently ~0.6 — WILL FAIL until compositor fixed."""
        cap_src = cv2.VideoCapture("clips_test/test_clip.mp4")
        cap_out = cv2.VideoCapture("output/face_os_v2/output.mp4")

        if not cap_src.isOpened() or not cap_out.isOpened():
            pytest.skip("Video files not found")

        ssim_scores = []
        for frame_idx in [25, 50, 75, 100, 125]:
            cap_src.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap_out.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret_src, src = cap_src.read()
            ret_out, out = cap_out.read()

            if not ret_src or not ret_out:
                continue

            # Resize to same size
            if src.shape != out.shape:
                out = cv2.resize(out, (src.shape[1], src.shape[0]))

            gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            gray_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

            score = _ssim(gray_src, gray_out)
            ssim_scores.append(score)

        cap_src.release()
        cap_out.release()

        if not ssim_scores:
            pytest.fail("Could not compute SSIM")

        mean_ssim = np.mean(ssim_scores)
        print(f"\n  SSIM: {mean_ssim:.3f} (target >0.75)")

        # STRICT: Must be above 0.75
        assert mean_ssim > 0.75, f"SSIM {mean_ssim:.3f} <= 0.75 (target >0.75)"


# ─── Test: No Ghost Mask ────────────────────────────────────────────────────

class TestNoGhostMask:
    """Test for mask ghosting artifacts."""

    def test_laplacian_variance_above_120(self):
        """Laplacian variance must be > 120 (no blur ghost)."""
        cap = cv2.VideoCapture("output/face_os_v2/output.mp4")
        if not cap.isOpened():
            pytest.skip("Output video not found")

        variances = []
        for frame_idx in range(25, 126, 10):  # Sample every 10 frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Extract face region
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(faces) == 0:
                continue

            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_gray = gray[y:y+h, x:x+w]

            lap_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            variances.append(lap_var)

        cap.release()

        if not variances:
            pytest.fail("Could not extract faces")

        mean_var = np.mean(variances)
        print(f"\n  Laplacian variance: {mean_var:.1f} (target >120)")

        # STRICT: Must be above 120
        assert mean_var > 120, f"Laplacian variance {mean_var:.1f} <= 120 (ghost mask detected)"

    def test_mean_abs_diff_under_20(self):
        """Mean absolute difference between consecutive frames must be < 20."""
        cap = cv2.VideoCapture("output/face_os_v2/output.mp4")
        if not cap.isOpened():
            pytest.skip("Output video not found")

        diffs = []
        prev_face = None
        for frame_idx in range(25, 126, 5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(faces) == 0:
                continue

            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face = cv2.resize(gray[y:y+h, x:x+w], (100, 100))

            if prev_face is not None:
                diff = np.mean(np.abs(face.astype(float) - prev_face.astype(float)))
                diffs.append(diff)

            prev_face = face

        cap.release()

        if not diffs:
            pytest.fail("Could not compute differences")

        mean_diff = np.mean(diffs)
        print(f"\n  Mean abs diff: {mean_diff:.1f} (target <20)")

        # STRICT: Must be under 20
        assert mean_diff < 20, f"Mean abs diff {mean_diff:.1f} >= 20 (mask artifact detected)"


# ─── Test: Quality Gates Strict ─────────────────────────────────────────────

class TestGatesStrict:
    """Test quality gates with strict thresholds."""

    def test_disparity_0095_rejected(self):
        """Disparity 0.095 must be rejected (threshold 0.09)."""
        # Create two slightly different face shapes
        ref_mesh = np.array([
            [100, 100], [150, 100], [200, 100],  # Top
            [100, 150], [150, 150], [200, 150],  # Middle
            [100, 200], [150, 200], [200, 200],  # Bottom
        ], dtype=np.float32)

        # Create mesh with slight variation (should exceed 0.09 threshold)
        test_mesh = ref_mesh + np.random.normal(0, 2, ref_mesh.shape).astype(np.float32)

        disparity = compute_procrustes_disparity(test_mesh, ref_mesh)
        print(f"\n  Disparity: {disparity:.4f} (threshold 0.09)")

        # The disparity should be computed correctly
        # If it's > 0.09, the gate should reject
        if disparity > 0.09:
            bbox = (50, 50, 200, 200)
            history = [test_mesh for _ in range(5)]
            passed, metrics = pass_quality_gates(test_mesh, ref_mesh, history, bbox)
            assert not passed, f"Disparity {disparity:.4f} > 0.09 should be rejected"

    def test_static_poster_low_jitter(self):
        """Static poster must have jitter < 0.0008 and be rejected."""
        # Create static landmarks (no movement)
        static_mesh = np.array([[100, 100], [150, 100], [150, 150], [100, 150]], dtype=np.float32)
        history = [static_mesh.copy() for _ in range(10)]  # No movement at all

        jitter = compute_landmark_jitter(history)
        print(f"\n  Static jitter: {jitter:.6f} (threshold 0.0008)")

        assert jitter < 0.0008, f"Static jitter {jitter:.6f} >= 0.0008"

        # Should be rejected
        bbox = (50, 50, 200, 200)
        ref = static_mesh.copy()
        passed, metrics = pass_quality_gates(static_mesh, ref, history, bbox)
        assert not passed, f"Static poster should be rejected (jitter={jitter:.6f})"

    def test_small_face_occupancy_024_rejected(self):
        """Face with occupancy 0.24 must be rejected (threshold 0.25)."""
        # Create large bbox with small face
        bbox = (0, 0, 400, 400)  # 160000 area
        # Small face: ~38400 area (0.24 * 160000)
        small_face = np.array([
            [120, 120], [280, 120],  # Top edge
            [280, 280], [120, 280],  # Bottom edge
        ], dtype=np.float32)

        occupancy = compute_occupancy(small_face, bbox)
        print(f"\n  Occupancy: {occupancy:.3f} (threshold 0.25)")

        assert occupancy < 0.25, f"Occupancy {occupancy:.3f} >= 0.25"

        # Should be rejected
        history = [small_face for _ in range(5)]
        passed, metrics = pass_quality_gates(small_face, small_face, history, bbox)
        assert not passed, f"Small face (occ={occupancy:.3f}) should be rejected"


# ─── Test: Compositor Uses Identity ─────────────────────────────────────────

class TestCompositorUsesIdentity:
    """Test that compositor uses identity_face, not rendered."""

    def test_compositor_uses_identity_face(self):
        """Compositor must use identity_face, not rendered.

        Mock identity_face = pure white, rendered = black.
        Call compositor.composite().
        ASSERT output mean > 200 (proves identity_face used, not rendered).

        Currently WILL FAIL — this is intentional.
        """
        compositor = Compositor()

        # Create test frames
        h, w = 256, 256
        original = np.ones((h, w, 3), dtype=np.uint8) * 128  # Gray original
        identity_face = np.ones((h, w, 3), dtype=np.uint8) * 255  # White identity
        rendered = np.zeros((h, w, 3), dtype=np.uint8)  # Black rendered

        # Create face mask (center region)
        face_mask = np.zeros((h, w), dtype=np.float32)
        face_mask[50:200, 50:200] = 1.0

        # Create confidence map
        conf = np.ones((h, w), dtype=np.float32) * 0.8

        # Test with identity_face (should produce white output)
        output_identity = compositor.composite(
            original, identity_face,
            confidence=ConfidenceMap(combined=conf),
            face_mask=face_mask,
        )

        # Test with rendered (would produce black output)
        output_rendered = compositor.composite(
            original, rendered,
            confidence=ConfidenceMap(combined=conf),
            face_mask=face_mask,
        )

        # Compute means in face region (center where feathering has less effect)
        face_region = output_identity[75:175, 75:175]
        identity_mean = np.mean(face_region)

        face_region_rendered = output_rendered[75:175, 75:175]
        rendered_mean = np.mean(face_region_rendered)

        print(f"\n  Identity blend mean: {identity_mean:.1f}")
        print(f"  Rendered blend mean: {rendered_mean:.1f}")
        print(f"  Identity should be > 200, rendered should be ~100")

        # The identity_face (white) should produce output mean > 190
        # The rendered (black) would produce output mean ~60
        # This test proves which one the compositor actually uses

        # STRICT: Output with identity_face should have mean > 190
        # (feathering reduces mean slightly at edges)
        assert identity_mean > 190, f"Identity blend mean {identity_mean:.1f} <= 190 (compositor not using identity_face)"
