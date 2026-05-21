"""
map_estimation.py — Joint MAP Estimation for Face OS.

Phase 2E: Joint MAP Estimation.

The MAP objective:
    x_t* = argmin_x ( E(x_t) + ||x_t - f(x_{t-1})||²_{Σ^{-1}} + ||z_t - h(x_t)||²_{R^{-1}} )

Unifies:
- Energy terms
- Dynamics (state transition)
- Observations
- Uncertainty (covariance-aware)
- Optimization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from face_os.state_space import LatentState, StateTransitionModel, ObservationModel
from face_os.optimizer import OptimizerConfig, GaussNewtonOptimizer, ConvergenceReport


# ─── MAP Config ──────────────────────────────────────────────────────────────

@dataclass
class MAPConfig:
    """MAP optimization configuration."""
    max_iterations: int = 100
    tolerance: float = 1e-6
    dynamics_weight: float = 1.0
    observation_weight: float = 1.0
    energy_weight: float = 1.0
    damping: float = 1.0
    rollback_threshold: float = 2.0


# ─── MAP Report ──────────────────────────────────────────────────────────────

@dataclass
class MAPReport:
    """MAP optimization report."""
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    energy: float = 0.0
    iterations: int = 0
    converged: bool = False
    energy_history: List[float] = field(default_factory=list)
    posterior_uncertainty: float = 0.0
    convergence_trajectory: List[float] = field(default_factory=list)
    innovation_norm: float = 0.0
    dynamics_residual: float = 0.0
    observation_residual: float = 0.0

    def to_dict(self) -> dict:
        return {
            "x": self.x.tolist() if len(self.x) > 0 else [],
            "energy": self.energy,
            "iterations": self.iterations,
            "converged": self.converged,
            "energy_history": self.energy_history,
            "posterior_uncertainty": self.posterior_uncertainty,
            "convergence_trajectory": self.convergence_trajectory,
            "innovation_norm": self.innovation_norm,
            "dynamics_residual": self.dynamics_residual,
            "observation_residual": self.observation_residual,
        }


# ─── MAP Optimizer ───────────────────────────────────────────────────────────

class MAPOptimizer:
    """Joint MAP optimizer.

    Minimizes:
        E_total(x) = w_e * E(x) + w_d * ||x - f(x_prev)||²_{Σ^{-1}} + w_o * ||z - h(x)||²_{R^{-1}}

    Where:
        E(x) = energy function
        f(x_prev) = state transition prediction
        h(x) = observation model
        Σ = process covariance
        R = observation covariance
    """

    def __init__(self, config: Optional[MAPConfig] = None):
        self.config = config or MAPConfig()
        self.transition = StateTransitionModel()
        self.observation = ObservationModel()

    def optimize(
        self,
        x0: np.ndarray,
        energy_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        x_prev: Optional[np.ndarray] = None,
        observation: Optional[np.ndarray] = None,
        prior_covariance: Optional[np.ndarray] = None,
        observation_covariance: Optional[np.ndarray] = None,
    ) -> MAPReport:
        """Optimize MAP objective.

        Args:
            x0: Initial point
            energy_fn: Energy function E(x)
            gradient_fn: Gradient of energy
            x_prev: Previous state (for dynamics term)
            observation: Observation vector (for observation term)
            prior_covariance: Process covariance Σ
            observation_covariance: Observation covariance R

        Returns:
            MAPReport with solution and metrics
        """
        x = x0.copy().astype(np.float64)
        n = len(x)

        # Default covariances
        if prior_covariance is None:
            prior_covariance = np.eye(n) * 10.0
        if observation_covariance is None:
            observation_covariance = np.eye(9) * 1.0  # 9 observation dimensions

        # Previous state prediction
        if x_prev is not None:
            x_pred = self.transition.predict(
                LatentState.from_vector(x_prev)
            ).to_vector()
        else:
            x_pred = x0.copy()

        # Observation vector (9 dimensions matching expanded observation model)
        if observation is None:
            observation = np.zeros(9)  # Default: zero observation

        # Inverse covariances
        Sigma_inv = np.linalg.inv(prior_covariance + np.eye(n) * 1e-10)
        R_inv = np.linalg.inv(observation_covariance + np.eye(9) * 1e-10)

        # Optimization loop
        energy_history = []
        convergence_trajectory = []

        for i in range(self.config.max_iterations):
            # Compute energy term
            E = energy_fn(x)
            g_E = gradient_fn(x)

            # Compute dynamics term: ||x - x_pred||²_{Σ^{-1}}
            dynamics_diff = x - x_pred
            E_dynamics = float(dynamics_diff.T @ Sigma_inv @ dynamics_diff)
            g_dynamics = 2.0 * Sigma_inv @ dynamics_diff

            # Compute observation term: ||z - h(x)||²_{R^{-1}}
            h_x = self._observation_vector(x)
            obs_diff = observation - h_x
            E_observation = float(obs_diff.T @ R_inv @ obs_diff)
            g_observation = -2.0 * R_inv @ obs_diff @ self._observation_jacobian(x)

            # Total MAP objective
            E_total = (
                self.config.energy_weight * E
                + self.config.dynamics_weight * E_dynamics
                + self.config.observation_weight * E_observation
            )
            energy_history.append(E_total)

            # Total gradient
            g_total = (
                self.config.energy_weight * g_E
                + self.config.dynamics_weight * g_dynamics
                + self.config.observation_weight * g_observation
            )

            # Check convergence
            grad_norm = float(np.linalg.norm(g_total))
            convergence_trajectory.append(grad_norm)

            if grad_norm < 1e-8:
                # Converged
                posterior_unc = self._compute_posterior_uncertainty(
                    x, Sigma_inv, R_inv
                )
                return MAPReport(
                    x=x,
                    energy=E_total,
                    iterations=i + 1,
                    converged=True,
                    energy_history=energy_history,
                    posterior_uncertainty=posterior_unc,
                    convergence_trajectory=convergence_trajectory,
                    innovation_norm=float(np.linalg.norm(obs_diff)),
                    dynamics_residual=float(np.linalg.norm(dynamics_diff)),
                    observation_residual=float(np.linalg.norm(obs_diff)),
                )

            # Gauss-Newton step with damping
            # Approximate Hessian: H ≈ 2 * (w_e * I + w_d * Σ^{-1} + w_o * J^T R^{-1} J)
            J = self._observation_jacobian(x)
            H_approx = (
                self.config.energy_weight * np.eye(n)
                + self.config.dynamics_weight * Sigma_inv
                + self.config.observation_weight * J.T @ R_inv @ J
                + self.config.damping * np.eye(n)
            )

            # Step
            try:
                delta = np.linalg.solve(H_approx, -g_total)
            except np.linalg.LinAlgError:
                delta = -g_total * 0.01

            # Update
            x = x + delta

        # Did not converge
        posterior_unc = self._compute_posterior_uncertainty(x, Sigma_inv, R_inv)
        return MAPReport(
            x=x,
            energy=energy_history[-1] if energy_history else 0.0,
            iterations=self.config.max_iterations,
            converged=False,
            energy_history=energy_history,
            posterior_uncertainty=posterior_unc,
            convergence_trajectory=convergence_trajectory,
            innovation_norm=float(np.linalg.norm(obs_diff)) if 'obs_diff' in dir() else 0.0,
            dynamics_residual=float(np.linalg.norm(dynamics_diff)) if 'dynamics_diff' in dir() else 0.0,
            observation_residual=float(np.linalg.norm(obs_diff)) if 'obs_diff' in dir() else 0.0,
        )

    def _observation_vector(self, x: np.ndarray) -> np.ndarray:
        """Map state to observation space (9 dimensions)."""
        state = LatentState.from_vector(x)
        obs = self.observation.observe(state)
        return np.array([
            obs["yaw"],
            obs["pitch"],
            obs["roll"],
            obs["identity_uncertainty"],
            obs["confidence"],
            obs["brightness_mean"],
            obs["contrast_mean"],
            obs["drift_score"],
            obs["continuity_score"],
        ])

    def _observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute observation Jacobian."""
        state = LatentState.from_vector(x)
        return self.observation.jacobian(state)

    def _compute_posterior_uncertainty(
        self,
        x: np.ndarray,
        Sigma_inv: np.ndarray,
        R_inv: np.ndarray,
    ) -> float:
        """Compute posterior uncertainty (trace of posterior covariance)."""
        J = self._observation_jacobian(x)
        n = len(x)
        # Posterior precision: H_post = Σ^{-1} + J^T R^{-1} J
        H_post = Sigma_inv + J.T @ R_inv @ J
        try:
            # Posterior covariance: P_post = H_post^{-1}
            P_post = np.linalg.inv(H_post + np.eye(n) * 1e-10)
            return float(np.trace(P_post))
        except np.linalg.LinAlgError:
            return float(np.trace(np.linalg.pinv(H_post)))
