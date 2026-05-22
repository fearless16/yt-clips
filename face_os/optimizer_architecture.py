"""Optimizer Architecture Module.

Defines the optimization strategy:
- Convergence policy
- Update ordering
- Damping policy
- Stopping conditions
- Divergence handling

This module provides:
- OptimizationEngine with iterative solve
- Convergence thresholds
- Numerical stability policy
- Divergence detection and recovery
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STRANDED MODULE — Status: SCHEDULED (D-10 / I-10)
#
# This module is fully implemented and tested but has ZERO runtime integration.
# It is NOT called by pipeline.py or any runtime path.
#
# Decision: SCHEDULED for integration in Phase C (Probabilistic Runtime)
# Action: Keep code + tests. Do not modify until integration phase.
# Tests: test_optimizer_architecture.py (32 tests), test_observability.py (28 tests)
# ═══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class OptimizationStatus(Enum):
    """Optimization status."""
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    DIVERGED = "diverged"
    STALLED = "stalled"
    NUMERICAL_ERROR = "numerical_error"


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""

    # Maximum iterations
    max_iterations: int = 100

    # Convergence threshold (relative change)
    convergence_threshold: float = 1e-6

    # Divergence threshold (energy increase)
    divergence_threshold: float = 10.0

    # Stall detection window
    stall_window: int = 10

    # Stall threshold (minimum improvement)
    stall_threshold: float = 1e-8

    # Damping factor
    damping: float = 0.1

    # Adaptive damping
    adaptive_damping: bool = True

    # Minimum damping
    min_damping: float = 0.01

    # Maximum damping
    max_damping: float = 1.0

    # Damping increase factor
    damping_increase: float = 2.0

    # Damping decrease factor
    damping_decrease: float = 0.5


@dataclass
class IterationState:
    """State of a single optimization iteration."""

    # Iteration number
    iteration: int

    # Energy value
    energy: float

    # Energy change from previous iteration
    energy_change: float

    # Relative energy change
    relative_change: float

    # Gradient norm
    gradient_norm: float

    # Damping used
    damping: float

    # Step size
    step_size: float

    # Status
    status: str = "running"


@dataclass
class OptimizationResult:
    """Result of optimization."""

    # Final state
    final_state: np.ndarray

    # Final energy
    final_energy: float

    # Status
    status: OptimizationStatus

    # Iterations taken
    iterations: int

    # Convergence history
    history: List[IterationState]

    # Total time
    time_ms: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "final_energy": self.final_energy,
            "status": self.status.value,
            "iterations": self.iterations,
            "time_ms": self.time_ms,
            "convergence_rate": self._compute_convergence_rate(),
        }

    def _compute_convergence_rate(self) -> float:
        """Compute convergence rate from history."""
        if len(self.history) < 2:
            return 0.0

        # Compute average relative change
        changes = [s.relative_change for s in self.history[1:]]
        return float(np.mean(np.abs(changes)))


class OptimizationEngine:
    """Optimization engine with convergence policy.

    Provides:
    - Iterative optimization with damping
    - Convergence detection
    - Divergence detection
    - Stall detection
    - Adaptive damping
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize optimization engine.

        Args:
            config: Configuration
        """
        self.config = config or OptimizationConfig()

    def optimize(
        self,
        initial_state: np.ndarray,
        energy_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        update_fn: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None,
    ) -> OptimizationResult:
        """Run optimization.

        Args:
            initial_state: Initial state
            energy_fn: Energy function
            gradient_fn: Gradient function
            update_fn: Update function (state, gradient, damping) -> new_state

        Returns:
            OptimizationResult
        """
        import time
        start_time = time.time()

        # Initialize
        state = initial_state.copy()
        energy = energy_fn(state)
        history = []
        damping = self.config.damping
        stall_count = 0

        # Default update function
        if update_fn is None:
            update_fn = lambda s, g, d: s - d * g

        # Optimization loop
        for iteration in range(self.config.max_iterations):
            # Compute gradient
            gradient = gradient_fn(state)
            gradient_norm = float(np.linalg.norm(gradient))

            # Compute step
            new_state = update_fn(state, gradient, damping)
            new_energy = energy_fn(new_state)

            # Compute changes
            energy_change = new_energy - energy
            relative_change = abs(energy_change) / (abs(energy) + 1e-8)
            step_size = float(np.linalg.norm(new_state - state))

            # Check for divergence
            if energy_change > self.config.divergence_threshold * abs(energy):
                # Diverged: increase damping and retry
                damping = min(damping * self.config.damping_increase, self.config.max_damping)
                history.append(IterationState(
                    iteration=iteration,
                    energy=energy,
                    energy_change=energy_change,
                    relative_change=relative_change,
                    gradient_norm=gradient_norm,
                    damping=damping,
                    step_size=step_size,
                    status="diverged",
                ))
                continue

            # Check for stall
            if relative_change < self.config.stall_threshold:
                stall_count += 1
                if stall_count >= self.config.stall_window:
                    # Stalled
                    history.append(IterationState(
                        iteration=iteration,
                        energy=new_energy,
                        energy_change=energy_change,
                        relative_change=relative_change,
                        gradient_norm=gradient_norm,
                        damping=damping,
                        step_size=step_size,
                        status="stalled",
                    ))
                    return OptimizationResult(
                        final_state=new_state,
                        final_energy=new_energy,
                        status=OptimizationStatus.STALLED,
                        iterations=iteration + 1,
                        history=history,
                        time_ms=(time.time() - start_time) * 1000,
                    )
            else:
                stall_count = 0

            # Update state
            state = new_state
            energy = new_energy

            # Adaptive damping
            if self.config.adaptive_damping:
                if energy_change < 0:
                    # Good step: decrease damping
                    damping = max(damping * self.config.damping_decrease, self.config.min_damping)
                else:
                    # Bad step: increase damping
                    damping = min(damping * self.config.damping_increase, self.config.max_damping)

            # Record iteration
            history.append(IterationState(
                iteration=iteration,
                energy=energy,
                energy_change=energy_change,
                relative_change=relative_change,
                gradient_norm=gradient_norm,
                damping=damping,
                step_size=step_size,
                status="running",
            ))

            # Check convergence
            if relative_change < self.config.convergence_threshold:
                return OptimizationResult(
                    final_state=state,
                    final_energy=energy,
                    status=OptimizationStatus.CONVERGED,
                    iterations=iteration + 1,
                    history=history,
                    time_ms=(time.time() - start_time) * 1000,
                )

        # Max iterations reached
        return OptimizationResult(
            final_state=state,
            final_energy=energy,
            status=OptimizationStatus.MAX_ITERATIONS,
            iterations=self.config.max_iterations,
            history=history,
            time_ms=(time.time() - start_time) * 1000,
        )

    def check_convergence(
        self,
        history: List[IterationState],
    ) -> Tuple[bool, OptimizationStatus]:
        """Check if optimization has converged.

        Args:
            history: Optimization history

        Returns:
            (is_converged, status)
        """
        if len(history) < 2:
            return False, OptimizationStatus.STALLED

        # Check last iteration
        last = history[-1]

        # Converged if relative change < threshold
        if last.relative_change < self.config.convergence_threshold:
            return True, OptimizationStatus.CONVERGED

        # Diverged if energy increased significantly
        if last.energy_change > self.config.divergence_threshold * abs(last.energy):
            return True, OptimizationStatus.DIVERGED

        return False, OptimizationStatus.STALLED

    def compute_convergence_rate(
        self,
        history: List[IterationState],
    ) -> float:
        """Compute convergence rate from history.

        Args:
            history: Optimization history

        Returns:
            Convergence rate (average relative change per iteration)
        """
        if len(history) < 2:
            return 0.0

        changes = [s.relative_change for s in history[1:]]
        return float(np.mean(np.abs(changes)))
