"""
tests/face_os/test_neural_codec.py — Regression tests for Neural Codec.

Tests:
- Personalized space
- Neural codec
- Identity operating system
- Identity preservation
"""

import cv2
import numpy as np
import pytest

from face_os.neural_codec import (
    PersonalizedSpace,
    NeuralCodec,
    IdentityOperatingSystem,
)


class TestPersonalizedSpace:
    """Test personalized space."""

    def test_initializes(self):
        """Must initialize correctly."""
        space = PersonalizedSpace(dimensions=16)

        assert space.dimensions == 16
        assert not space.is_trained()

    def test_trains(self):
        """Must train from reference faces."""
        space = PersonalizedSpace(dimensions=16)

        faces = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(20)]
        space.train(faces)

        assert space.is_trained()
        assert space.mean_face is not None
        assert space.basis is not None

    def test_encode(self):
        """Must encode face."""
        space = PersonalizedSpace(dimensions=16)

        faces = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(20)]
        space.train(faces)

        face = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        encoded = space.encode(face)

        assert encoded.shape[0] > 0

    def test_decode(self):
        """Must decode vector."""
        space = PersonalizedSpace(dimensions=16)

        faces = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(20)]
        space.train(faces)

        face = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        encoded = space.encode(face)
        decoded = space.decode(encoded)

        assert decoded.shape == face.shape

    def test_identity_distance(self):
        """Must compute identity distance."""
        space = PersonalizedSpace(dimensions=16)

        base = np.ones((64, 64, 3), dtype=np.uint8) * 128
        faces = [base.copy() for _ in range(20)]
        space.train(faces)

        dist = space.get_identity_distance(base)

        assert dist >= 0


class TestNeuralCodec:
    """Test neural codec."""

    def test_initializes(self):
        """Must initialize correctly."""
        codec = NeuralCodec(dimensions=16)

        assert codec.dimensions == 16
        assert not codec.is_trained()

    def test_trains(self):
        """Must train from reference faces."""
        codec = NeuralCodec(dimensions=16)

        faces = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(20)]
        codec.train(faces)

        assert codec.is_trained()

    def test_encode_and_correct(self):
        """Must encode and correct."""
        codec = NeuralCodec(dimensions=16)

        faces = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(20)]
        codec.train(faces)

        face = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        corrected, encoded = codec.encode_and_correct(face)

        assert corrected.shape == face.shape
        assert encoded.shape[0] > 0

    def test_identity_score(self):
        """Must compute identity score."""
        codec = NeuralCodec(dimensions=16)

        base = np.ones((64, 64, 3), dtype=np.uint8) * 128
        faces = [base.copy() for _ in range(20)]
        codec.train(faces)

        score = codec.get_identity_score(base)

        assert 0 <= score <= 1

    def test_identity_score_for_different(self):
        """Different face must have lower score."""
        codec = NeuralCodec(dimensions=16)

        base = np.ones((64, 64, 3), dtype=np.uint8) * 128
        faces = [base.copy() for _ in range(20)]
        codec.train(faces)

        different = np.ones((64, 64, 3), dtype=np.uint8) * 200
        score_different = codec.get_identity_score(different)
        score_base = codec.get_identity_score(base)

        assert score_different <= score_base

    def test_stats(self):
        """Must track statistics."""
        codec = NeuralCodec(dimensions=16)

        faces = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(20)]
        codec.train(faces)

        stats = codec.get_stats()

        assert stats['is_trained'] == True
        assert stats['dimensions'] == 16


class TestIdentityOperatingSystem:
    """Test identity operating system."""

    def test_initializes(self):
        """Must initialize correctly."""
        ios = IdentityOperatingSystem()

        assert not ios.is_initialized()

    def test_initialize(self):
        """Must initialize with reference faces."""
        ios = IdentityOperatingSystem()

        faces = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        ios.initialize(faces)

        assert ios.is_initialized()

    def test_process_frame(self):
        """Must process frames."""
        ios = IdentityOperatingSystem()

        faces = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        ios.initialize(faces)

        face = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        processed, score = ios.process_frame(face)

        assert processed.shape == face.shape
        assert 0 <= score <= 1

    def test_identity_stability(self):
        """Must track identity stability."""
        ios = IdentityOperatingSystem()

        faces = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        ios.initialize(faces)

        for i in range(20):
            face = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
            ios.process_frame(face)

        stability = ios.get_identity_stability()

        assert 0 <= stability <= 1

    def test_stats(self):
        """Must track statistics."""
        ios = IdentityOperatingSystem()

        faces = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        ios.initialize(faces)

        for i in range(10):
            face = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
            ios.process_frame(face)

        stats = ios.get_stats()

        assert stats['is_initialized'] == True
        assert stats['frame_count'] == 10
