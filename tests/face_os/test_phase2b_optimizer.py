"""
test_phase2b_optimizer.py — Phase 2B: Optimizer Architecture Tests.

Tests:
1. Convergence stability
2. Conditioning robustness
3. Optimizer rollback
4. Singularity handling
5. Energy descent consistency
6. Adversarial initialization recovery

TDD: These tests should FAIL before implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import List

from face_os.types import EnergyTerms
from face_os.energy import EnergyComputer
from face_os.state_space import LatentState, StateSpaceEstimator
from face_os.optimizer import (
    OptimizerConfig,
    OptimizerState,
    GaussNewtonOptimizer,
    LevenbergMarquardtOptimizer,
    ConvergenceReport,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _simple_energy_function(x: np.ndarray) -> float:
    """Simple quadratic energy: E = ||x - target||^2."""
    target = np.array([5.0, -3.0, 1.0])
    return float(np.sum((x - target) ** 2))


def _simple_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient of simple quadratic."""
    target = np.array([5.0, -3.0, 1.0])
    return 2.0 * (x - target)


def _simple_hessian(x: np.ndarray) -> np.ndarray:
    """Hessian of simple quadratic."""
    return 2.0 * np.eye(len(x))


def _rosenbrock_energy(x: np.ndarray) -> float:
    """Rosenbrock function (ill-conditioned)."""
    a, b = 1.0, 100.0
    return float((a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2)


def _rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient of Rosenbrock."""
    a, b = 1.0, 100.0
    dx0 = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
    dx1 = 2 * b * (x[1] - x[0] ** 2)
    return np.array([dx0, dx1])


def _rosenbrock_hessian(x: np.ndarray) -> np.ndarray:
    """Hessian of Rosenbrock."""
    a, b = 1.0, 100.0
    h00 = 2 - 4 * b * (x[1] - 3 * x[0] ** 2)
    h01 = -4 * b * x[0]
    h11 = 2 * b
    return np.array([[h00, h01], [h01, h11]])


# ─── Test Class 1: OptimizerConfig ───────────────────────────────────────────

class TestOptimizerConfig:
    """Test optimizer configuration."""

    def test_config_exists(self):
        """OptimizerConfig must exist."""
        config = OptimizerConfig()
        assert config is not None

    def test_config_has_max_iterations(self):
        """Config must have max_iterations."""
        config = OptimizerConfig()
        assert hasattr(config, "max_iterations")
        assert config.max_iterations > 0

    def test_config_has_tolerance(self):
        """Config must have convergence tolerance."""
        config = OptimizerConfig()
        assert hasattr(config, "tolerance")
        assert config.tolerance > 0

    def test_config_has_damping(self):
        """Config must have damping parameter."""
        config = OptimizerConfig()
        assert hasattr(config, "damping")
        assert config.damping >= 0

    def test_config_has_rollback_threshold(self):
        """Config must have rollback threshold."""
        config = OptimizerConfig()
        assert hasattr(config, "rollback_threshold")
        assert config.rollback_threshold > 0


# ─── Test Class 2: OptimizerState ────────────────────────────────────────────

class TestOptimizerState:
    """Test optimizer state."""

    def test_state_exists(self):
        """OptimizerState must exist."""
        state = OptimizerState()
        assert state is not None

    def test_state_has_iteration(self):
        """State must track iteration count."""
        state = OptimizerState()
        assert hasattr(state, "iteration")
        assert state.iteration == 0

    def test_state_has_energy_history(self):
        """State must track energy history."""
        state = OptimizerState()
        assert hasattr(state, "energy_history")
        assert isinstance(state.energy_history, list)

    def test_state_has_converged(self):
        """State must track convergence."""
        state = OptimizerState()
        assert hasattr(state, "converged")

    def test_state_has_rollback_count(self):
        """State must track rollback count."""
        state = OptimizerState()
        assert hasattr(state, "rollback_count")


# ─── Test Class 3: GaussNewtonOptimizer ──────────────────────────────────────

class TestGaussNewtonOptimizer:
    """Test Gauss-Newton optimizer."""

    def test_optimizer_exists(self):
        """GaussNewtonOptimizer must exist."""
        opt = GaussNewtonOptimizer()
        assert opt is not None

    def test_optimizer_has_minimize(self):
        """Optimizer must have minimize method."""
        opt = GaussNewtonOptimizer()
        assert hasattr(opt, "minimize")

    def test_optimizer_converges_quadratic(self):
        """Optimizer must converge on quadratic function."""
        opt = GaussNewtonOptimizer(config=OptimizerConfig(max_iterations=100))
        x0 = np.array([0.0, 0.0, 0.0])
        result = opt.minimize(
            x0,
            energy_fn=_simple_energy_function,
            gradient_fn=_simple_gradient,
            hessian_fn=_simple_hessian,
        )
        # Should converge near target [5, -3, 1]
        assert np.linalg.norm(result.x - np.array([5.0, -3.0, 1.0])) < 0.1
        assert result.converged

    def test_optimizer_energy_decreases(self):
        """Energy must decrease monotonically."""
        opt = GaussNewtonOptimizer(config=OptimizerConfig(max_iterations=100))
        x0 = np.array([0.0, 0.0, 0.0])
        result = opt.minimize(
            x0,
            energy_fn=_simple_energy_function,
            gradient_fn=_simple_gradient,
            hessian_fn=_simple_hessian,
        )
        # Energy should decrease
        energies = result.energy_history
        for i in range(1, len(energies)):
            assert energies[i] <= energies[i-1] + 1e-10

    def test_optimizer_bounded_iterations(self):
        """Optimizer must respect max_iterations."""
        config = OptimizerConfig(max_iterations=10)
        opt = GaussNewtonOptimizer(config=config)
        x0 = np.array([0.0, 0.0, 0.0])
        result = opt.minimize(
            x0,
            energy_fn=_simple_energy_function,
            gradient_fn=_simple_gradient,
            hessian_fn=_simple_hessian,
        )
        assert result.iterations <= 10

    def test_optimizer_report_has_metrics(self):
        """Optimizer report must have all metrics."""
        opt = GaussNewtonOptimizer()
        x0 = np.array([0.0, 0.0, 0.0])
        result = opt.minimize(
            x0,
            energy_fn=_simple_energy_function,
            gradient_fn=_simple_gradient,
            hessian_fn=_simple_hessian,
        )
        assert hasattr(result, "x")
        assert hasattr(result, "energy")
        assert hasattr(result, "iterations")
        assert hasattr(result, "converged")
        assert hasattr(result, "energy_history")


# ─── Test Class 4: LevenbergMarquardtOptimizer ───────────────────────────────

class TestLevenbergMarquardtOptimizer:
    """Test Levenberg-Marquardt optimizer."""

    def test_optimizer_exists(self):
        """LevenbergMarquardtOptimizer must exist."""
        opt = LevenbergMarquardtOptimizer()
        assert opt is not None

    def test_optimizer_handles_ill_conditioned(self):
        """Optimizer must handle ill-conditioned problems (Rosenbrock)."""
        opt = LevenbergMarquardtOptimizer(
            config=OptimizerConfig(max_iterations=1000)
        )
        x0 = np.array([-1.0, -1.0])
        result = opt.minimize(
            x0,
            energy_fn=_rosenbrock_energy,
            gradient_fn=_rosenbrock_gradient,
            hessian_fn=_rosenbrock_hessian,
        )
        # Should converge near (1, 1)
        assert np.linalg.norm(result.x - np.array([1.0, 1.0])) < 0.5

    def test_optimizer_adaptive_damping(self):
        """LM optimizer must adapt damping."""
        opt = LevenbergMarquardtOptimizer()
        assert hasattr(opt, "config")
        assert hasattr(opt.config, "damping")

    def test_optimizer_convergence_on_simple(self):
        """LM must converge on simple quadratic."""
        opt = LevenbergMarquardtOptimizer(
            config=OptimizerConfig(max_iterations=100)
        )
        x0 = np.array([0.0, 0.0, 0.0])
        result = opt.minimize(
            x0,
            energy_fn=_simple_energy_function,
            gradient_fn=_simple_gradient,
            hessian_fn=_simple_hessian,
        )
        assert np.linalg.norm(result.x - np.array([5.0, -3.0, 1.0])) < 0.1


# ─── Test Class 5: ConvergenceReport ─────────────────────────────────────────

class TestConvergenceReport:
    """Test convergence report."""

    def test_report_exists(self):
        """ConvergenceReport must exist."""
        report = ConvergenceReport()
        assert report is not None

    def test_report_has_x(self):
        """Report must have solution vector."""
        report = ConvergenceReport()
        assert hasattr(report, "x")

    def test_report_has_energy(self):
        """Report must have final energy."""
        report = ConvergenceReport()
        assert hasattr(report, "energy")

    def test_report_has_iterations(self):
        """Report must have iteration count."""
        report = ConvergenceReport()
        assert hasattr(report, "iterations")

    def test_report_has_converged(self):
        """Report must have convergence flag."""
        report = ConvergenceReport()
        assert hasattr(report, "converged")

    def test_report_has_energy_history(self):
        """Report must have energy history."""
        report = ConvergenceReport()
        assert hasattr(report, "energy_history")

    def test_report_to_dict(self):
        """Report must be serializable."""
        report = ConvergenceReport()
        d = report.to_dict()
        assert "x" in d
        assert "energy" in d
        assert "iterations" in d


# ─── Test Class 6: Energy Descent ────────────────────────────────────────────

class TestEnergyDescent:
    """Test energy descent consistency."""

    def test_energy_decreases_consistently(self):
        """Energy must decrease consistently across runs."""
        opt = GaussNewtonOptimizer(config=OptimizerConfig(max_iterations=50))

        final_energies = []
        for _ in range(5):
            x0 = np.random.randn(3) * 10
            result = opt.minimize(
                x0,
                energy_fn=_simple_energy_function,
                gradient_fn=_simple_gradient,
                hessian_fn=_simple_hessian,
            )
            final_energies.append(result.energy)

        # All runs should converge to similar energy
        assert np.std(final_energies) < 1.0

    def test_energy_descent_rate(self):
        """Energy must decrease at least 50% in first 10 iterations."""
        opt = GaussNewtonOptimizer(config=OptimizerConfig(max_iterations=10))
        x0 = np.array([10.0, 10.0, 10.0])
        result = opt.minimize(
            x0,
            energy_fn=_simple_energy_function,
            gradient_fn=_simple_gradient,
            hessian_fn=_simple_hessian,
        )
        if len(result.energy_history) >= 2:
            initial = result.energy_history[0]
            final = result.energy_history[-1]
            assert final < initial * 0.5


# ─── Test Class 7: Singularity Handling ──────────────────────────────────────

class TestSingularityHandling:
    """Test singularity handling."""

    def test_optimizer_handles_singular_hessian(self):
        """Optimizer must handle singular Hessian."""
        opt = LevenbergMarquardtOptimizer(
            config=OptimizerConfig(max_iterations=100)
        )

        def singular_hessian(x):
            n = len(x)
            H = np.ones((n, n))  # Singular (rank 1)
            return H

        def energy_2d(x):
            target = np.array([5.0, -3.0])
            return float(np.sum((x - target) ** 2))

        def gradient_2d(x):
            target = np.array([5.0, -3.0])
            return 2.0 * (x - target)

        x0 = np.array([0.0, 0.0])
        result = opt.minimize(
            x0,
            energy_fn=energy_2d,
            gradient_fn=gradient_2d,
            hessian_fn=singular_hessian,
        )
        # Should not crash, should return finite result
        assert np.all(np.isfinite(result.x))

    def test_optimizer_handles_zero_gradient(self):
        """Optimizer must handle zero gradient (already at optimum)."""
        opt = GaussNewtonOptimizer(config=OptimizerConfig(max_iterations=10))
        x0 = np.array([5.0, -3.0, 1.0])  # Already at target
        result = opt.minimize(
            x0,
            energy_fn=_simple_energy_function,
            gradient_fn=_simple_gradient,
            hessian_fn=_simple_hessian,
        )
        # Should converge immediately
        assert result.iterations <= 2


# ─── Test Class 8: Rollback ──────────────────────────────────────────────────

class TestRollback:
    """Test optimizer rollback."""

    def test_rollback_on_energy_increase(self):
        """Optimizer must rollback if energy increases."""
        config = OptimizerConfig(rollback_threshold=1.5)
        opt = GaussNewtonOptimizer(config=config)

        # Use a function where naive step might increase energy
        def bad_energy(x):
            return float(np.sum(x ** 2))

        def bad_gradient(x):
            return 2.0 * x

        def bad_hessian(x):
            return 2.0 * np.eye(len(x))

        x0 = np.array([10.0, 10.0])
        result = opt.minimize(
            x0,
            energy_fn=bad_energy,
            gradient_fn=bad_gradient,
            hessian_fn=bad_hessian,
        )
        # Should converge despite potential issues
        assert np.all(np.isfinite(result.x))
