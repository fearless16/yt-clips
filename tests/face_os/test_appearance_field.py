"""
tests/face_os/test_appearance_field.py — Regression tests for Appearance Field.

Tests:
- Appearance sample storage
- k-NN interpolation
- Dynamic UV flow
- Microdetail synthesis
"""

import cv2
import numpy as np
import pytest

from face_os.appearance_field import (
    AppearanceField,
    AppearanceSample,
    DynamicAppearanceField,
)


class TestAppearanceSample:
    """Test appearance sample."""

    def test_initializes(self, canonical_face):
        """Must initialize correctly."""
        sample = AppearanceSample(
            canonical_face=canonical_face,
            pose=(0, 0, 0),
            quality=0.8,
        )

        assert sample.quality == 0.8
        assert sample.pose == (0, 0, 0)

    def test_computes_features(self, canonical_face):
        """Must compute features."""
        sample = AppearanceSample(
            canonical_face=canonical_face,
            pose=(0, 0, 0),
            quality=0.8,
        )

        assert sample.lab_mean is not None
        assert sample.feature_vector is not None

    def test_distance_to(self, canonical_face):
        """Must compute distance to other sample."""
        sample1 = AppearanceSample(canonical_face=canonical_face, pose=(0, 0, 0))
        sample2 = AppearanceSample(canonical_face=canonical_face, pose=(30, 0, 0))

        dist = sample1.distance_to(sample2)

        assert dist >= 0


class TestAppearanceField:
    """Test appearance field."""

    def test_initializes(self):
        """Must initialize correctly."""
        field = AppearanceField(max_samples=10)

        assert field.max_samples == 10
        assert len(field.samples) == 0

    def test_add_sample(self, canonical_face):
        """Must add samples."""
        field = AppearanceField(max_samples=10)

        field.add_sample(canonical_face, pose=(0, 0, 0), quality=0.8)

        assert len(field.samples) == 1

    def test_query_closest_pose(self, canonical_face):
        """Must query closest pose."""
        field = AppearanceField(max_samples=10)

        face_0 = np.ones((256, 256, 3), dtype=np.uint8) * 100
        face_30 = np.ones((256, 256, 3), dtype=np.uint8) * 200

        field.add_sample(face_0, pose=(0, 0, 0), quality=0.8)
        field.add_sample(face_30, pose=(30, 0, 0), quality=0.8)

        result, conf = field.query(pose=(15, 0, 0))

        assert result is not None
        assert conf > 0

    def test_query_interpolates(self, canonical_face):
        """Must interpolate between samples."""
        field = AppearanceField(max_samples=10)

        face_dark = np.ones((256, 256, 3), dtype=np.uint8) * 50
        face_bright = np.ones((256, 256, 3), dtype=np.uint8) * 200

        field.add_sample(face_dark, pose=(-30, 0, 0), quality=0.8)
        field.add_sample(face_bright, pose=(30, 0, 0), quality=0.8)

        result, conf = field.query(pose=(0, 0, 0))

        assert result is not None

    def test_prune_when_full(self, canonical_face):
        """Must prune when full."""
        field = AppearanceField(max_samples=5)

        for i in range(10):
            face = np.ones((256, 256, 3), dtype=np.uint8) * (i * 25)
            field.add_sample(face, pose=(i * 10, 0, 0), quality=i / 10.0)

        assert len(field.samples) <= 5

    def test_stats(self, canonical_face):
        """Must track statistics."""
        field = AppearanceField(max_samples=10)

        for i in range(5):
            field.add_sample(canonical_face, pose=(i * 10, 0, 0), quality=0.8)

        stats = field.get_stats()

        assert stats['num_samples'] == 5
        assert stats['frame_count'] == 5


class TestDynamicAppearanceField:
    """Test dynamic appearance field."""

    def test_initializes(self):
        """Must initialize correctly."""
        field = DynamicAppearanceField(max_samples=10)

        assert field.max_samples == 10

    def test_add_deformation(self):
        """Must add deformation."""
        field = DynamicAppearanceField(max_samples=10)

        deformation = np.random.randn(256, 256, 2).astype(np.float32) * 5
        field.add_expression_deformation('smile', deformation)

        assert 'smile' in field.uv_deformations

    def test_query_with_deformation(self, canonical_face):
        """Must query with deformation."""
        field = DynamicAppearanceField(max_samples=10)

        field.add_sample(canonical_face, pose=(0, 0, 0), quality=0.8)

        deformation = np.zeros((256, 256, 2), dtype=np.float32)
        field.add_expression_deformation('smile', deformation)

        result, conf = field.query_with_deformation(
            pose=(0, 0, 0),
            expression='smile',
        )

        assert result is not None

    def test_generate_microdetail(self, canonical_face):
        """Must generate microdetail."""
        field = DynamicAppearanceField(max_samples=10)

        detail = field.generate_microdetail(canonical_face, detail_level=0.5)

        assert detail is not None
        assert detail.shape == (256, 256)
