"""
optimizer.py — Optimizer Architecture for Face OS.

Phase 2B: Optimizer Architecture.

Implements:
- OptimizerConfig — configuration for optimization
- OptimizerState — current optimization state
- GaussNewtonOptimizer — Gauss-Newton method
- LevenbergMarquardtOptimizer — LM method with adaptive damping
- ConvergenceReport — per-optimization convergence metrics

The optimizer minimizes energy functions of the form:
    E(x) where x is the latent state vector

Update rule (Gauss-Newton):
    x_{k+1} = x_k - H^{-1} * g

Where:
    g = gradient of E
    H = Hessian of E (or approximation)

Update rule (Levenberg-Marquardt):
    x_{k+1} = x_k - (H + lambda * I)^{-1} * g

Where:
    lambda = adaptive damping parameter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


# ─── Optimizer Config ────────────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    max_iterations: int = 100
    tolerance: float = 1e-6
    damping: float = 1.0
    damping_increase: float = 10.0
    damping_decrease: float = 0.1
    rollback_threshold: float = 2.0
    min_gradient_norm: float = 1e-8
    conditioning_threshold: float = 1e-12


# ─── Optimizer State ─────────────────────────────────────────────────────────

@dataclass
class OptimizerState:
    """Current optimization state."""
    iteration: int = 0
    energy_history: List[float] = field(default_factory=list)
    converged: bool = False
    rollback_count: int = 0
    damping: float = 1.0
    gradient_norm: float = 0.0
    step_norm: float = 0.0


# ─── Convergence Report ─────────────────────────────────────────────────────

@dataclass
class ConvergenceReport:
    """Convergence report after optimization."""
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    energy: float = 0.0
    iterations: int = 0
    converged: bool = False
    energy_history: List[float] = field(default_factory=list)
    gradient_norm: float = 0.0
    step_norm: float = 0.0
    rollback_count: int = 0

    def to_dict(self) -> dict:
        return {
            "x": self.x.tolist(),
            "energy": self.energy,
            "iterations": self.iterations,
            "converged": self.converged,
            "energy_history": self.energy_history,
            "gradient_norm": self.gradient_norm,
            "step_norm": self.step_norm,
            "rollback_count": self.rollback_count,
        }


# ─── Gauss-Newton Optimizer ─────────────────────────────────────────────────

class GaussNewtonOptimizer:
    """Gauss-Newton optimizer.

    Update rule:
        x_{k+1} = x_k - H^{-1} * g

    Where:
        g = gradient of E
        H = Hessian of E
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()

    def minimize(
        self,
        x0: np.ndarray,
        energy_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        hessian_fn: Callable[[np.ndarray], np.ndarray],
    ) -> ConvergenceReport:
        """Minimize energy function.

        Args:
            x0: Initial point
            energy_fn: Energy function E(x)
            gradient_fn: Gradient function g(x)
            hessian_fn: Hessian function H(x)

        Returns:
            ConvergenceReport with solution and metrics
        """
        x = x0.copy().astype(np.float64)
        state = OptimizerState()
        state.damping = self.config.damping

        for i in range(self.config.max_iterations):
            state.iteration = i

            # Compute energy
            energy = energy_fn(x)
            state.energy_history.append(energy)

            # Compute gradient
            g = gradient_fn(x)
            state.gradient_norm = float(np.linalg.norm(g))

            # Check convergence: gradient norm
            if state.gradient_norm < self.config.min_gradient_norm:
                state.converged = True
                break

            # Compute Hessian
            H = hessian_fn(x)

            # Add damping for numerical stability
            H_damped = H + state.damping * np.eye(len(x))

            # Check conditioning
            cond = np.linalg.cond(H_damped)
            if cond > 1.0 / self.config.conditioning_threshold:
                # Ill-conditioned, increase damping
                state.damping *= self.config.damping_increase
                H_damped = H + state.damping * np.eye(len(x))

            # Compute step: delta = -H^{-1} * g
            try:
                delta = np.linalg.solve(H_damped, -g)
            except np.linalg.LinAlgError:
                # Singular matrix, use gradient descent
                delta = -g * 0.01

            # Line search with rollback
            x_new = x + delta
            energy_new = energy_fn(x_new)

            # Check if energy decreased
            if energy_new > energy * self.config.rollback_threshold:
                # Rollback: increase damping and retry
                state.rollback_count += 1
                state.damping *= self.config.damping_increase
                continue

            # Accept step
            state.step_norm = float(np.linalg.norm(delta))
            x = x_new

            # Decrease damping on success
            state.damping = max(
                self.config.damping * self.config.damping_decrease,
                1e-8,
            )

            # Check convergence: step norm
            if state.step_norm < self.config.tolerance:
                state.converged = True
                break

        return ConvergenceReport(
            x=x,
            energy=state.energy_history[-1] if state.energy_history else 0.0,
            iterations=state.iteration + 1,
            converged=state.converged,
            energy_history=state.energy_history,
            gradient_norm=state.gradient_norm,
            step_norm=state.step_norm,
            rollback_count=state.rollback_count,
        )


# ─── Levenberg-Marquardt Optimizer ───────────────────────────────────────────

class LevenbergMarquardtOptimizer:
    """Levenberg-Marquardt optimizer with adaptive damping.

    Update rule:
        x_{k+1} = x_k - (H + lambda * I)^{-1} * g

    Where:
        lambda = adaptive damping parameter
        g = gradient of E
        H = Hessian of E

    Damping adaptation:
        If energy decreased: lambda *= decrease_factor
        If energy increased: lambda *= increase_factor
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()

    def minimize(
        self,
        x0: np.ndarray,
        energy_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        hessian_fn: Callable[[np.ndarray], np.ndarray],
    ) -> ConvergenceReport:
        """Minimize energy function.

        Args:
            x0: Initial point
            energy_fn: Energy function E(x)
            gradient_fn: Gradient function g(x)
            hessian_fn: Hessian function H(x)

        Returns:
            ConvergenceReport with solution and metrics
        """
        x = x0.copy().astype(np.float64)
        state = OptimizerState()
        state.damping = self.config.damping

        for i in range(self.config.max_iterations):
            state.iteration = i

            # Compute energy
            energy = energy_fn(x)
            state.energy_history.append(energy)

            # Compute gradient
            g = gradient_fn(x)
            state.gradient_norm = float(np.linalg.norm(g))

            # Check convergence: gradient norm
            if state.gradient_norm < self.config.min_gradient_norm:
                state.converged = True
                break

            # Compute Hessian
            H = hessian_fn(x)

            # LM step: (H + lambda * I)^{-1} * g
            H_damped = H + state.damping * np.eye(len(x))

            # Check conditioning
            try:
                cond = np.linalg.cond(H_damped)
                if cond > 1.0 / self.config.conditioning_threshold:
                    state.damping *= self.config.damping_increase
                    H_damped = H + state.damping * np.eye(len(x))
            except np.linalg.LinAlgError:
                state.damping *= self.config.damping_increase
                H_damped = H + state.damping * np.eye(len(x))

            # Compute step
            try:
                delta = np.linalg.solve(H_damped, -g)
            except np.linalg.LinAlgError:
                delta = -g * 0.01

            # Try step
            x_new = x + delta
            energy_new = energy_fn(x_new)

            # Adaptive damping
            if energy_new < energy:
                # Energy decreased: accept step, decrease damping
                x = x_new
                state.damping *= self.config.damping_decrease
                state.damping = max(state.damping, 1e-8)
                state.step_norm = float(np.linalg.norm(delta))
            else:
                # Energy increased: reject step, increase damping
                state.damping *= self.config.damping_increase
                state.rollback_count += 1

            # Check convergence: step norm
            if state.step_norm < self.config.tolerance:
                state.converged = True
                break

        return ConvergenceReport(
            x=x,
            energy=state.energy_history[-1] if state.energy_history else 0.0,
            iterations=state.iteration + 1,
            converged=state.converged,
            energy_history=state.energy_history,
            gradient_norm=state.gradient_norm,
            step_norm=state.step_norm,
            rollback_count=state.rollback_count,
        )
