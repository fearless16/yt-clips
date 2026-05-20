"""
tests/face_os/test_identity_state.py — Regression tests for Identity State.

Tests:
- Identity gravity equation
- Anchor correction
- Frequency decomposition
- Belief distributions
- Brightness correction
- Warmth correction
"""

import cv2
import numpy as np
import pytest

from face_os.identity_state import (
    IdentityState,
    IdentityHypothesis,
    IdentityHypothesisSpace,
    FrequencyDecomposition,
    BeliefPixel,
)


class TestFrequencyDecomposition:
    """Test frequency decomposition."""

    def test_decompose_separates_low_high(self):
        """Must separate into low and high frequency."""
        freq = FrequencyDecomposition(low_pass_sigma=2.0)

        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        low, high = freq.decompose(img)

        assert low.shape == img.shape
        assert high.shape == img.shape

    def test_low_freq_smoother(self):
        """Low freq must be smoother than source."""
        freq = FrequencyDecomposition(low_pass_sigma=2.0)

        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        low, _ = freq.decompose(img)

        assert np.var(low) < np.var(img.astype(np.float32))

    def test_reconstruction_lossless(self):
        """Reconstruction must be lossless."""
        freq = FrequencyDecomposition(low_pass_sigma=2.0)

        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        low, high = freq.decompose(img)
        reconstructed = freq.reconstruct(low, high)

        diff = np.abs(img.astype(np.float32) - reconstructed.astype(np.float32))
        assert np.max(diff) < 1.0


class TestBeliefPixel:
    """Test per-pixel belief distribution."""

    def test_initializes_correctly(self):
        """Must initialize with correct dimensions."""
        belief = BeliefPixel(100, 100, 3)

        assert belief.best_low.shape == (100, 100, 3)
        assert belief.best_high.shape == (100, 100, 3)
        assert belief.quality_max.shape == (100, 100)

    def test_update_accumulates(self):
        """Update must accumulate observations."""
        belief = BeliefPixel(100, 100, 3)

        low = np.ones((100, 100, 3), dtype=np.float32) * 100
        high = np.ones((100, 100, 3), dtype=np.float32) * 10
        quality = np.ones((100, 100), dtype=np.float32) * 0.8

        belief.update(low, high, quality)

        assert np.mean(belief.observation_count) > 0

    def test_best_observation_not_averaged(self):
        """High freq must use BEST observation, not average."""
        belief = BeliefPixel(100, 100, 3)

        # First: sharp (high quality)
        high_sharp = np.ones((100, 100, 3), dtype=np.float32) * 50
        quality_sharp = np.ones((100, 100), dtype=np.float32) * 0.9
        belief.update(np.zeros((100, 100, 3), dtype=np.float32), high_sharp, quality_sharp)

        # Second: blurry (low quality)
        high_blurry = np.ones((100, 100, 3), dtype=np.float32) * 20
        quality_blurry = np.ones((100, 100), dtype=np.float32) * 0.3
        belief.update(np.zeros((100, 100, 3), dtype=np.float32), high_blurry, quality_blurry)

        # Should keep sharp (best)
        assert np.mean(belief.best_high) == 50.0


class TestIdentityState:
    """Test identity state."""

    def test_initializes(self, canonical_face):
        """Must initialize correctly."""
        state = IdentityState()

        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        state.update(canonical_face, quality, pose=(0, 0, 0))

        assert state.is_initialized()

    def test_set_anchor(self, canonical_face):
        """Must set anchor correctly."""
        state = IdentityState()
        state.set_anchor(canonical_face)

        assert state._anchor_low is not None
        assert state._anchor_high is not None
        assert state._anchor_lab is not None

    def test_anchor_preserved_across_reset(self, canonical_face):
        """Anchor must survive reset."""
        state = IdentityState()
        state.set_anchor(canonical_face)

        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        state.update(canonical_face, quality, pose=(0, 0, 0))

        state.reset()

        assert state._anchor_low is not None

    def test_get_anchor_distance(self, canonical_face):
        """Must compute anchor distance."""
        state = IdentityState()
        state.set_anchor(canonical_face)

        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        state.update(canonical_face, quality, pose=(0, 0, 0))

        dist = state.get_anchor_distance()
        assert dist < 5.0

    def test_query_returns_result(self, canonical_face):
        """Must return result from query."""
        state = IdentityState()

        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        state.update(canonical_face, quality, pose=(0, 0, 0))

        result, conf = state.query(canonical_face, quality)

        assert result is not None
        assert conf is not None

    def test_brightness_correction(self, canonical_face, dark_face):
        """Must pull brightness toward reference."""
        state = IdentityState()
        state.set_anchor(canonical_face)

        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        for _ in range(50):
            state.update(dark_face, quality, pose=(0, 0, 0))

        result, _ = state.query(dark_face, quality)

        result_L = np.mean(cv2.cvtColor(result, cv2.COLOR_BGR2LAB)[:, :, 0])
        dark_L = np.mean(cv2.cvtColor(dark_face, cv2.COLOR_BGR2LAB)[:, :, 0])
        ref_L = np.mean(cv2.cvtColor(canonical_face, cv2.COLOR_BGR2LAB)[:, :, 0])

        assert abs(result_L - ref_L) < abs(dark_L - ref_L)

    def test_warmth_correction(self, canonical_face, cold_face):
        """Must pull warmth toward reference."""
        state = IdentityState()
        state.set_anchor(canonical_face)

        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        for _ in range(50):
            state.update(cold_face, quality, pose=(0, 0, 0))

        result, _ = state.query(cold_face, quality)

        result_b = np.mean(cv2.cvtColor(result, cv2.COLOR_BGR2LAB)[:, :, 2])
        cold_b = np.mean(cv2.cvtColor(cold_face, cv2.COLOR_BGR2LAB)[:, :, 2])
        ref_b = np.mean(cv2.cvtColor(canonical_face, cv2.COLOR_BGR2LAB)[:, :, 2])

        assert abs(result_b - ref_b) < abs(cold_b - ref_b)

    def test_identity_slower_than_source(self, canonical_face):
        """Identity must change slower than source."""
        state = IdentityState()

        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        for _ in range(10):
            state.update(canonical_face, quality, pose=(0, 0, 0))

        source_frames = []
        identity_frames = []

        for i in range(20):
            source = np.ones((256, 256, 3), dtype=np.uint8) * (100 + i * 5)
            source_frames.append(source.astype(np.float32))

            state.update(source, quality, pose=(0, 0, 0))
            result, _ = state.query(source, quality)
            identity_frames.append(result.astype(np.float32))

        source_deltas = []
        identity_deltas = []
        for i in range(1, len(source_frames)):
            source_deltas.append(np.mean(np.abs(source_frames[i] - source_frames[i-1])))
            identity_deltas.append(np.mean(np.abs(identity_frames[i] - identity_frames[i-1])))

        assert np.mean(identity_deltas) < np.mean(source_deltas)


class TestIdentityHypothesis:
    """Test identity hypothesis."""

    def test_creates_hypothesis(self, canonical_face):
        """Must create hypothesis correctly."""
        hyp = IdentityHypothesis(
            name="test",
            canonical_face=canonical_face,
            quality=0.8,
            pose=(0, 0, 0),
        )

        assert hyp.name == "test"
        assert hyp.quality == 0.8

    def test_decompose(self, canonical_face):
        """Must decompose into frequencies."""
        hyp = IdentityHypothesis(
            name="test",
            canonical_face=canonical_face,
            quality=0.8,
        )
        hyp.decompose(FrequencyDecomposition())

        assert hyp.low_freq is not None
        assert hyp.high_freq is not None

    def test_update_support(self, canonical_face):
        """Must update support correctly."""
        hyp = IdentityHypothesis(
            name="test",
            canonical_face=canonical_face,
            quality=0.8,
        )
        hyp.decompose(FrequencyDecomposition())

        # Similar observation should support
        similar = canonical_face.copy()
        supported = hyp.update_support(similar, 0.9, frame_idx=1)

        assert supported == True
        assert hyp.support_count > 1


class TestIdentityHypothesisSpace:
    """Test identity hypothesis space."""

    def test_creates_space(self):
        """Must create space correctly."""
        space = IdentityHypothesisSpace(max_hypotheses=10)

        assert space.max_hypotheses == 10
        assert len(space.hypotheses) == 0

    def test_update_creates_hypothesis(self, canonical_face):
        """Must create hypothesis on update."""
        space = IdentityHypothesisSpace(max_hypotheses=10)

        space.update(canonical_face, 0.8, pose=(0, 0, 0))

        assert len(space.hypotheses) > 0

    def test_query_returns_result(self, canonical_face):
        """Must return result from query."""
        space = IdentityHypothesisSpace(max_hypotheses=10)

        space.update(canonical_face, 0.8, pose=(0, 0, 0))

        result, conf = space.query(pose=(0, 0, 0))

        assert result is not None

    def test_prune_when_full(self, canonical_face):
        """Must prune when full."""
        space = IdentityHypothesisSpace(max_hypotheses=5)

        for i in range(10):
            face = np.ones((256, 256, 3), dtype=np.uint8) * (i * 25)
            space.update(face, 0.8, pose=(i * 10, 0, 0))

        assert len(space.hypotheses) <= 5
