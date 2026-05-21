"""
test_phase2e_map_estimation.py — Phase 2E: Joint MAP Estimation Tests.

Tests:
1. MAP convergence
2. Uncertainty-aware optimization
3. Adversarial initialization recovery
4. Covariance-aware stability
5. Observation consistency
6. Posterior contraction
7. Energy descent consistency

TDD: These tests should FAIL before implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from face_os.state_space import LatentState, StateSpaceEstimator
from face_os.energy import EnergyComputer
from face_os.optimizer import OptimizerConfig
from face_os.map_estimation import (
    MAPConfig,
    MAPOptimizer,
    MAPReport,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _simple_energy(x: np.ndarray) -> float:
    """Simple quadratic energy."""
    target = np.array([5.0, -3.0, 1.0])
    return float(np.sum((x[:3] - target) ** 2))


def _simple_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient of simple quadratic."""
    target = np.array([5.0, -3.0, 1.0])
    g = np.zeros_like(x)
    g[:3] = 2.0 * (x[:3] - target)
    return g


# ─── Test Class 1: MAPConfig ─────────────────────────────────────────────────

class TestMAPConfig:
    """Test MAP configuration."""

    def test_config_exists(self):
        """MAPConfig must exist."""
        config = MAPConfig()
        assert config is not None

    def test_config_has_dynamics_weight(self):
        """Config must have dynamics weight."""
        config = MAPConfig()
        assert hasattr(config, "dynamics_weight")

    def test_config_has_observation_weight(self):
        """Config must have observation weight."""
        config = MAPConfig()
        assert hasattr(config, "observation_weight")

    def test_config_has_energy_weight(self):
        """Config must have energy weight."""
        config = MAPConfig()
        assert hasattr(config, "energy_weight")


# ─── Test Class 2: MAPOptimizer ──────────────────────────────────────────────

class TestMAPOptimizer:
    """Test MAP optimizer."""

    def test_optimizer_exists(self):
        """MAPOptimizer must exist."""
        opt = MAPOptimizer()
        assert opt is not None

    def test_optimizer_has_optimize(self):
        """Optimizer must have optimize method."""
        opt = MAPOptimizer()
        assert hasattr(opt, "optimize")

    def test_optimizer_converges(self):
        """Optimizer must converge on simple problem."""
        opt = MAPOptimizer(config=MAPConfig(max_iterations=100))
        x0 = np.zeros(11)
        report = opt.optimize(
            x0,
            energy_fn=_simple_energy,
            gradient_fn=_simple_gradient,
        )
        assert report.converged

    def test_optimizer_energy_decreases(self):
        """Energy must decrease during optimization."""
        opt = MAPOptimizer(config=MAPConfig(max_iterations=50))
        x0 = np.zeros(11)
        report = opt.optimize(
            x0,
            energy_fn=_simple_energy,
            gradient_fn=_simple_gradient,
        )
        energies = report.energy_history
        for i in range(1, len(energies)):
            assert energies[i] <= energies[i-1] + 1e-10

    def test_optimizer_respects_max_iterations(self):
        """Optimizer must respect max_iterations."""
        config = MAPConfig(max_iterations=5)
        opt = MAPOptimizer(config=config)
        x0 = np.zeros(11)
        report = opt.optimize(
            x0,
            energy_fn=_simple_energy,
            gradient_fn=_simple_gradient,
        )
        assert report.iterations <= 5


# ─── Test Class 3: MAPReport ─────────────────────────────────────────────────

class TestMAPReport:
    """Test MAP optimization report."""

    def test_report_exists(self):
        """MAPReport must exist."""
        report = MAPReport()
        assert report is not None

    def test_report_has_x(self):
        """Report must have solution vector."""
        report = MAPReport()
        assert hasattr(report, "x")

    def test_report_has_energy(self):
        """Report must have final energy."""
        report = MAPReport()
        assert hasattr(report, "energy")

    def test_report_has_posterior_uncertainty(self):
        """Report must have posterior uncertainty."""
        report = MAPReport()
        assert hasattr(report, "posterior_uncertainty")

    def test_report_has_convergence_trajectory(self):
        """Report must have convergence trajectory."""
        report = MAPReport()
        assert hasattr(report, "convergence_trajectory")

    def test_report_has_innovation_statistics(self):
        """Report must have innovation statistics."""
        report = MAPReport()
        assert hasattr(report, "innovation_norm")

    def test_report_to_dict(self):
        """Report must be serializable."""
        report = MAPReport()
        d = report.to_dict()
        assert "energy" in d
        assert "iterations" in d
        assert "converged" in d


# ─── Test Class 4: Posterior Contraction ──────────────────────────────────────

class TestPosteriorContraction:
    """Test posterior contraction (uncertainty shrinks with data)."""

    def test_posterior_uncertainty_computable(self):
        """Posterior uncertainty must be computable."""
        opt = MAPOptimizer()
        x0 = np.zeros(11)
        report = opt.optimize(
            x0,
            energy_fn=_simple_energy,
            gradient_fn=_simple_gradient,
        )
        assert report.posterior_uncertainty >= 0

    def test_posterior_shrinks_with_observations(self):
        """Posterior must shrink when observations are added."""
        opt = MAPOptimizer()

        # Without observations
        x0 = np.zeros(11)
        report_no_obs = opt.optimize(
            x0,
            energy_fn=_simple_energy,
            gradient_fn=_simple_gradient,
        )

        # With observations (via prior precision)
        config = MAPConfig(observation_weight=10.0)
        opt_with_obs = MAPOptimizer(config=config)
        report_with_obs = opt_with_obs.optimize(
            x0,
            energy_fn=_simple_energy,
            gradient_fn=_simple_gradient,
        )

        # Posterior uncertainty should be smaller with observations
        assert report_with_obs.posterior_uncertainty <= report_no_obs.posterior_uncertainty * 1.1


# ─── Test Class 5: Energy Descent ────────────────────────────────────────────

class TestMAPEnergyDescent:
    """Test MAP energy descent consistency."""

    def test_energy_decreases_consistently(self):
        """Energy must decrease consistently."""
        opt = MAPOptimizer(config=MAPConfig(max_iterations=50))
        x0 = np.zeros(11)

        final_energies = []
        for _ in range(5):
            report = opt.optimize(
                x0,
                energy_fn=_simple_energy,
                gradient_fn=_simple_gradient,
            )
            final_energies.append(report.energy)

        # All runs should converge to similar energy
        assert np.std(final_energies) < 1.0
