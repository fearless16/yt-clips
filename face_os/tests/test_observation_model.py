"""Tests for §16.1 Observation Model — forward prediction residual and noise."""
import numpy as np
import pytest

from face_os.observation_model import compute_observation_residual


class TestObservationResidualComputation:
    def test_identical_images_zero_residual(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        res = compute_observation_residual(img, img)
        assert res.residual_mean == pytest.approx(0.0, abs=1e-3)
        assert res.residual_max == pytest.approx(0.0, abs=1e-3)
        assert res.observation_confidence == pytest.approx(1.0, abs=1e-3)

    def test_different_images_positive_residual(self):
        pred = np.full((64, 64, 3), 128, dtype=np.uint8)
        obs = np.full((64, 64, 3), 64, dtype=np.uint8)
        res = compute_observation_residual(pred, obs)
        assert res.residual_mean > 0.0
        assert res.observation_confidence < 1.0

    def test_mask_restricts_computation(self):
        pred = np.full((64, 64, 3), 128, dtype=np.uint8)
        obs = np.full((64, 64, 3), 64, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:40, 20:40] = 1.0
        res_masked = compute_observation_residual(pred, obs, mask=mask)
        res_full = compute_observation_residual(pred, obs)
        assert res_masked.residual_mean > 0.0
        assert res_full.residual_mean > 0.0

    def test_empty_mask_returns_zero(self):
        pred = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        res = compute_observation_residual(pred, obs, mask=mask)
        assert res.residual_mean == 0.0
        assert res.observation_confidence == 0.0

    def test_residual_finite_and_bounded(self):
        pred = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        obs = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        res = compute_observation_residual(pred, obs)
        assert np.isfinite(res.residual_mean)
        assert np.isfinite(res.residual_max)
        assert np.isfinite(res.noise_mean)
        assert 0.0 <= res.observation_confidence <= 1.0

    def test_float32_input(self):
        pred = np.random.rand(64, 64, 3).astype(np.float32)
        obs = np.random.rand(64, 64, 3).astype(np.float32)
        res = compute_observation_residual(pred, obs)
        assert np.isfinite(res.residual_mean)
        assert 0.0 <= res.observation_confidence <= 1.0

    def test_shape_mismatch_resizes(self):
        pred = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        res = compute_observation_residual(pred, obs)
        assert res.residual_map.shape == (64, 64)

    def test_confidence_decreases_with_larger_residual(self):
        pred = np.full((64, 64, 3), 128, dtype=np.uint8)
        obs_small = np.full((64, 64, 3), 120, dtype=np.uint8)
        obs_large = np.full((64, 64, 3), 64, dtype=np.uint8)
        res_small = compute_observation_residual(pred, obs_small)
        res_large = compute_observation_residual(pred, obs_large)
        assert res_small.observation_confidence > res_large.observation_confidence

    def test_noise_map_shape(self):
        pred = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        res = compute_observation_residual(pred, obs)
        assert res.noise_map.shape == (64, 64, 3)
        assert res.residual_map.shape == (64, 64)
