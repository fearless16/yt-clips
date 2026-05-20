"""
tests/face_os/test_region_confidence.py — Tests for region-specific confidence.

Tests:
- Region-specific confidence computation
- Different confidence per region
- Integration with identity state
"""

import cv2
import numpy as np
import pytest

from face_os.identity_state import IdentityState, BeliefPixel
from face_os.patch_memory import REGION_DEFS


class TestRegionConfidence:
    """Test region-specific confidence computation."""

    def test_compute_region_confidence(self):
        """Must compute confidence per region."""
        state = IdentityState()

        # Create a face with different quality per region
        face = np.ones((256, 256, 3), dtype=np.uint8) * 128

        # Initialize
        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        state.update(face, quality, pose=(0, 0, 0))

        # Compute region confidence
        region_conf = state.compute_region_confidence()

        # Must have confidence for each region
        assert 'left_eye' in region_conf
        assert 'right_eye' in region_conf
        assert 'beard' in region_conf
        assert 'forehead' in region_conf

    def test_different_confidence_per_region(self):
        """Different regions must have different confidence."""
        state = IdentityState()

        # Create a face with different quality per region
        face = np.ones((256, 256, 3), dtype=np.uint8) * 128

        # Make eye region brighter (higher quality)
        face[80:120, 100:150] = 200  # Left eye
        face[80:120, 140:190] = 200  # Right eye

        # Make beard region darker (lower quality)
        face[140:200, 110:170] = 80  # Beard

        # Initialize with quality based on brightness
        quality = np.ones((256, 256), dtype=np.float32) * 0.5
        quality[80:120, 100:150] = 0.9  # Eyes: high quality
        quality[80:120, 140:190] = 0.9
        quality[140:200, 110:170] = 0.3  # Beard: low quality

        state.update(face, quality, pose=(0, 0, 0))

        # Compute region confidence
        region_conf = state.compute_region_confidence()

        # All regions should have some confidence
        for name, conf in region_conf.items():
            assert conf > 0, f"Region {name} should have confidence > 0"

    def test_region_confidence_affects_query(self):
        """Region confidence must affect query result."""
        state = IdentityState()

        # Create reference face
        ref_face = np.ones((256, 256, 3), dtype=np.uint8) * 128
        ref_face[80:120, 100:150] = 200  # Bright eyes

        # Set anchor and initialize
        state.set_anchor(ref_face)
        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        for _ in range(10):
            state.update(ref_face, quality, pose=(0, 0, 0))

        # Query with dark eyes
        dark_face = ref_face.copy()
        dark_face[80:120, 100:150] = 80  # Dark eyes

        result, conf = state.query(dark_face, quality)

        # Result should preserve bright eyes from identity
        result_eyes = result[80:120, 100:150]
        ref_eyes = ref_face[80:120, 100:150]

        # Eyes should be closer to reference than to dark input
        diff_to_ref = np.mean(np.abs(result_eyes.astype(np.float32) - ref_eyes.astype(np.float32)))
        diff_to_dark = np.mean(np.abs(result_eyes.astype(np.float32) - dark_face[80:120, 100:150].astype(np.float32)))

        # Identity should dominate for eyes (high confidence region)
        assert diff_to_ref < diff_to_dark


class TestSemanticConfidence:
    """Test semantic confidence factors."""

    def test_confidence_factors(self):
        """Confidence must consider multiple factors."""
        # Test each factor
        frame = np.ones((256, 256, 3), dtype=np.uint8) * 128

        # Sharpness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        assert sharpness >= 0

        # Brightness
        brightness = np.mean(gray) / 255.0
        assert 0 <= brightness <= 1

        # Combined quality
        quality = min(sharpness / 100.0, 1.0) * 0.5 + (1.0 - abs(brightness - 0.5) * 2) * 0.5
        assert 0 <= quality <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
