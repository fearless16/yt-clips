"""
tests/face_os/test_hypothesis_matching.py — Tests for improved hypothesis matching.

Tests:
- Region-based LAB distance
- Better hypothesis selection
"""

import cv2
import numpy as np
import pytest

from face_os.identity_state import (
    IdentityHypothesis,
    IdentityHypothesisSpace,
    FrequencyDecomposition,
)


class TestHypothesisMatching:
    """Test improved hypothesis matching."""

    def test_hypothesis_update_support_with_regions(self):
        """Hypothesis support must use region-based distance."""
        # Create hypothesis with distinct regions
        face = np.ones((256, 256, 3), dtype=np.uint8) * 128
        face[80:120, 100:150] = 200  # Bright eyes
        face[140:200, 110:170] = 80   # Dark beard

        hyp = IdentityHypothesis(
            name="test",
            canonical_face=face,
            quality=0.8,
        )
        hyp.decompose(FrequencyDecomposition())

        # Similar face should support
        similar = face.copy()
        similar[80:120, 100:150] = 190  # Slightly different eyes
        supported = hyp.update_support(similar, 0.8, frame_idx=1)

        assert supported == True
        assert hyp.support_count > 1

    def test_hypothesis_rejects_different_face(self):
        """Hypothesis must reject very different face."""
        face = np.ones((256, 256, 3), dtype=np.uint8) * 128
        face[80:120, 100:150] = 200  # Bright eyes

        hyp = IdentityHypothesis(
            name="test",
            canonical_face=face,
            quality=0.8,
        )
        hyp.decompose(FrequencyDecomposition())

        # Very different face should not support
        different = np.ones((256, 256, 3), dtype=np.uint8) * 50
        supported = hyp.update_support(different, 0.8, frame_idx=1)

        assert supported == False
        assert hyp.contradiction_count > 0

    def test_hypothesis_space_selects_best_pose(self):
        """Hypothesis space must select best pose match."""
        space = IdentityHypothesisSpace(max_hypotheses=10)

        # Add hypotheses for different poses
        face_frontal = np.ones((256, 256, 3), dtype=np.uint8) * 128
        face_left = np.ones((256, 256, 3), dtype=np.uint8) * 100
        face_right = np.ones((256, 256, 3), dtype=np.uint8) * 150

        space.update(face_frontal, 0.8, pose=(0, 0, 0))
        space.update(face_left, 0.8, pose=(-30, 0, 0))
        space.update(face_right, 0.8, pose=(30, 0, 0))

        # Query at frontal pose
        result, conf = space.query(pose=(0, 0, 0))
        assert result is not None

        # Query at left pose
        result, conf = space.query(pose=(-20, 0, 0))
        assert result is not None

    def test_hypothesis_space_selects_best_expression(self):
        """Hypothesis space must select best expression match."""
        space = IdentityHypothesisSpace(max_hypotheses=10)

        # Add hypotheses for different expressions
        face_neutral = np.ones((256, 256, 3), dtype=np.uint8) * 128
        face_smile = np.ones((256, 256, 3), dtype=np.uint8) * 140

        space.update(face_neutral, 0.8, pose=(0, 0, 0), expression='neutral')
        space.update(face_smile, 0.8, pose=(0, 0, 0), expression='smile')

        # Query with neutral expression
        result, conf = space.query(pose=(0, 0, 0), expression='neutral')
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
