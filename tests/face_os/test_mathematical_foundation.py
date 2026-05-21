"""Tests for State Evolution, Energy Scaling, and Optimizer Architecture.

Tests for:
- StateEvolution: predict, update, Kalman gain
- EnergyScaler: normalization, weighting
- OptimizationEngine: convergence, divergence, stall

Target: 30 tests
"""

import numpy as np
import pytest

from face_os.state_evolution import StateEvolution, StateEvolutionConfig
from face_os.energy_scaling import EnergyScaler, EnergyScalingConfig
from face_os.optimizer_architecture import (
    OptimizationEngine,
    OptimizationConfig,
    OptimizationStatus,
)


class TestStateEvolution:
    """Test StateEvolution."""

    def test_initialization(self):
        """Must initialize with config."""
        model = StateEvolution()
        assert model.state_dim == 11
        assert model.transition_matrix.shape == (11, 11)

    def test_predict_shape(self):
        """Predict must return correct shape."""
        model = StateEvolution()
        state = np.zeros(11)
        predicted = model.predict(state)
        assert predicted.shape == (11,)

    def test_predict_damping(self):
        """Predict must apply damping."""
        model = StateEvolution()
        state = np.ones(11)
        predicted = model.predict(state)
        # Damping < 1 means predicted < state
        assert np.all(np.abs(predicted) <= np.abs(state) + 1e-10)

    def test_predict_covariance_shape(self):
        """Predict covariance must return correct shape."""
        model = StateEvolution()
        cov = np.eye(11)
        predicted_cov = model.predict_covariance(cov)
        assert predicted_cov.shape == (11, 11)

    def test_predict_covariance_increase(self):
        """Predict covariance must increase (add process noise)."""
        model = StateEvolution()
        cov = np.eye(11) * 0.1
        predicted_cov = model.predict_covariance(cov)
        # Trace should increase
        assert np.trace(predicted_cov) > np.trace(cov)

    def test_innovation_shape(self):
        """Innovation must have correct shape."""
        model = StateEvolution()
        state = np.zeros(11)
        obs = np.zeros(5)
        H = np.zeros((5, 11))
        for i in range(5):
            H[i, i] = 1.0
        innovation = model.compute_innovation(state, obs, H)
        assert innovation.shape == (5,)

    def test_kalman_gain_shape(self):
        """Kalman gain must have correct shape."""
        model = StateEvolution()
        P = np.eye(11)
        H = np.zeros((5, 11))
        for i in range(5):
            H[i, i] = 1.0
        S = np.eye(5)
        K = model.compute_kalman_gain(P, H, S)
        assert K.shape == (11, 5)

    def test_update_state_shape(self):
        """Updated state must have correct shape."""
        model = StateEvolution()
        state = np.zeros(11)
        K = np.zeros((11, 5))
        innovation = np.zeros(5)
        updated = model.update_state(state, K, innovation)
        assert updated.shape == (11,)

    def test_update_covariance_shape(self):
        """Updated covariance must have correct shape."""
        model = StateEvolution()
        P = np.eye(11)
        K = np.zeros((11, 5))
        H = np.zeros((5, 11))
        updated = model.update_covariance(P, K, H)
        assert updated.shape == (11, 11)

    def test_full_kalman_cycle(self):
        """Full Kalman cycle must work."""
        model = StateEvolution()

        # Initial state
        state = np.zeros(11)
        cov = np.eye(11)

        # Predict
        predicted_state = model.predict(state)
        predicted_cov = model.predict_covariance(cov)

        # Observation
        obs = np.ones(5)
        H = np.zeros((5, 11))
        for i in range(5):
            H[i, i] = 1.0
        R = np.eye(5) * 0.1

        # Innovation
        innovation = model.compute_innovation(predicted_state, obs, H)
        S = model.compute_innovation_covariance(predicted_cov, H, R)

        # Update
        K = model.compute_kalman_gain(predicted_cov, H, S)
        updated_state = model.update_state(predicted_state, K, innovation)
        updated_cov = model.update_covariance(predicted_cov, K, H)

        # Check shapes
        assert updated_state.shape == (11,)
        assert updated_cov.shape == (11, 11)

        # Check no NaN
        assert not np.any(np.isnan(updated_state))
        assert not np.any(np.isnan(updated_cov))


class TestEnergyScaler:
    """Test EnergyScaler."""

    def test_initialization(self):
        """Must initialize with config."""
        scaler = EnergyScaler()
        assert scaler.config is not None

    def test_normalize_zscore(self):
        """Z-score normalization must work."""
        config = EnergyScalingConfig(normalization_method='zscore')
        scaler = EnergyScaler(config)

        # Add many values to stabilize statistics
        for _ in range(10):
            for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
                scaler.normalize("test", v)

        # Normalize a value near the mean
        normalized = scaler.normalize("test", 3.0)
        # Should be near 0 (within 2 standard deviations)
        assert abs(normalized) < 3.0

    def test_normalize_minmax(self):
        """Min-max normalization must work."""
        config = EnergyScalingConfig(normalization_method='minmax')
        scaler = EnergyScaler(config)

        # Add some values
        for v in [0.0, 1.0, 2.0, 3.0, 4.0]:
            scaler.normalize("test", v)

        # Normalize
        normalized = scaler.normalize("test", 2.0)
        assert 0.0 <= normalized <= 1.0

    def test_compute_weight(self):
        """Weight computation must work."""
        scaler = EnergyScaler()

        weight = scaler.compute_weight("test", uncertainty=1.0)
        assert weight > 0

    def test_weight_inverse_uncertainty(self):
        """Weight must be inverse of uncertainty."""
        scaler = EnergyScaler()

        w1 = scaler.compute_weight("test", uncertainty=0.5)
        w2 = scaler.compute_weight("test", uncertainty=2.0)

        # Lower uncertainty = higher weight
        assert w1 > w2

    def test_compute_scaled_energy(self):
        """Scaled energy must work."""
        scaler = EnergyScaler()

        terms = {"geom": 0.5, "identity": 0.3, "temporal": 0.2}
        energy = scaler.compute_scaled_energy(terms)
        assert isinstance(energy, float)

    def test_get_stats(self):
        """Stats must be available."""
        scaler = EnergyScaler()

        scaler.normalize("test", 1.0)
        scaler.normalize("test", 2.0)

        stats = scaler.get_stats()
        assert "test" in stats
        assert stats["test"]["count"] == 2

    def test_reset(self):
        """Reset must clear stats."""
        scaler = EnergyScaler()

        scaler.normalize("test", 1.0)
        scaler.reset()

        stats = scaler.get_stats()
        assert len(stats) == 0


class TestOptimizationEngine:
    """Test OptimizationEngine."""

    def test_initialization(self):
        """Must initialize with config."""
        engine = OptimizationEngine()
        assert engine.config is not None

    def test_optimize_convex(self):
        """Must converge on convex function."""
        config = OptimizationConfig(max_iterations=500, convergence_threshold=1e-4)
        engine = OptimizationEngine(config)

        # Simple quadratic: f(x) = x^2
        def energy_fn(x):
            return float(np.sum(x**2))

        def gradient_fn(x):
            return 2 * x

        initial = np.array([10.0, 10.0])
        result = engine.optimize(initial, energy_fn, gradient_fn)

        # Should converge or reach max iterations with low energy
        assert result.final_energy < 1.0

    def test_optimize_max_iterations(self):
        """Must stop at max iterations."""
        config = OptimizationConfig(max_iterations=5)
        engine = OptimizationEngine(config)

        def energy_fn(x):
            return float(np.sum(x**2))

        def gradient_fn(x):
            return 2 * x

        initial = np.array([10.0, 10.0])
        result = engine.optimize(initial, energy_fn, gradient_fn)

        assert result.iterations <= 5

    def test_optimize_history(self):
        """Must record history."""
        engine = OptimizationEngine()

        def energy_fn(x):
            return float(np.sum(x**2))

        def gradient_fn(x):
            return 2 * x

        initial = np.array([5.0, 5.0])
        result = engine.optimize(initial, energy_fn, gradient_fn)

        assert len(result.history) > 0
        assert result.history[0].iteration == 0

    def test_convergence_rate(self):
        """Must compute convergence rate."""
        engine = OptimizationEngine()

        def energy_fn(x):
            return float(np.sum(x**2))

        def gradient_fn(x):
            return 2 * x

        initial = np.array([5.0, 5.0])
        result = engine.optimize(initial, energy_fn, gradient_fn)

        rate = result._compute_convergence_rate()
        assert rate >= 0

    def test_check_convergence(self):
        """Must check convergence correctly."""
        engine = OptimizationEngine()

        # Create history showing convergence
        from face_os.optimizer_architecture import IterationState
        history = [
            IterationState(0, 100.0, -10.0, 0.1, 10.0, 0.1, 1.0, "running"),
            IterationState(1, 90.0, -0.0001, 1e-7, 5.0, 0.1, 0.5, "running"),
        ]

        is_converged, status = engine.check_convergence(history)
        assert is_converged
        assert status == OptimizationStatus.CONVERGED

    def test_to_dict(self):
        """Result must convert to dict."""
        engine = OptimizationEngine()

        def energy_fn(x):
            return float(np.sum(x**2))

        def gradient_fn(x):
            return 2 * x

        initial = np.array([1.0, 1.0])
        result = engine.optimize(initial, energy_fn, gradient_fn)

        d = result.to_dict()
        assert "final_energy" in d
        assert "status" in d
        assert "iterations" in d
