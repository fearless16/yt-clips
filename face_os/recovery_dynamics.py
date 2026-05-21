"""
recovery_dynamics.py — Probabilistic Recovery Dynamics for Face OS.

Phase 2G: Probabilistic Recovery Dynamics.

Replaces discrete recovery state machine with probabilistic transitions:
    P(x_meta,t | x_meta,t-1)

Components:
- RecoveryTransitionMatrix: State transition probabilities
- ProbabilisticRecoveryState: Soft state representation with probabilities
- RecoveryDynamics: Main dynamics engine with entropy tracking
- RecoveryReport: Per-frame recovery dynamics metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─── Recovery States ─────────────────────────────────────────────────────────

class RecoveryState(Enum):
    """Recovery states for the estimator."""
    STABLE = auto()
    UNCERTAIN = auto()
    DEGRADED = auto()
    RECOVERING = auto()
    RESET_REQUIRED = auto()


# ─── Recovery Transition Matrix ──────────────────────────────────────────────

@dataclass
class RecoveryTransitionMatrix:
    """State transition probabilities: P(x_meta,t | x_meta,t-1).

    Transition matrix T[i,j] = P(state_j at t | state_i at t-1).

    States: [STABLE, UNCERTAIN, DEGRADED, RECOVERING, RESET_REQUIRED]

    Default transitions (realistic):
    - STABLE → STABLE: 0.95, → UNCERTAIN: 0.05
    - UNCERTAIN → STABLE: 0.3, → UNCERTAIN: 0.5, → DEGRADED: 0.2
    - DEGRADED → UNCERTAIN: 0.2, → DEGRADED: 0.5, → RECOVERING: 0.3
    - RECOVERING → DEGRADED: 0.2, → RECOVERING: 0.5, → STABLE: 0.3
    - RESET_REQUIRED → RECOVERING: 0.5, → RESET_REQUIRED: 0.5
    """
    matrix: np.ndarray = field(default_factory=lambda: np.array([
        # STABLE    UNCERTAIN  DEGRADED   RECOVERING RESET_REQUIRED
        [0.95,      0.05,      0.00,      0.00,      0.00],  # STABLE
        [0.30,      0.50,      0.20,      0.00,      0.00],  # UNCERTAIN
        [0.00,      0.20,      0.50,      0.30,      0.00],  # DEGRADED
        [0.30,      0.00,      0.20,      0.50,      0.00],  # RECOVERING
        [0.00,      0.00,      0.00,      0.50,      0.50],  # RESET_REQUIRED
    ], dtype=np.float64))

    def __post_init__(self):
        # Validate: each row must sum to 1.0
        row_sums = np.sum(self.matrix, axis=1)
        assert np.allclose(row_sums, 1.0), f"Transition matrix rows must sum to 1.0, got {row_sums}"

    def step(self, state_probs: np.ndarray) -> np.ndarray:
        """Apply one transition step: p_{t+1} = T^T @ p_t.

        Args:
            state_probs: Current state probability vector (5,)

        Returns:
            Next state probability vector (5,)
        """
        return self.matrix.T @ state_probs

    def steady_state(self) -> np.ndarray:
        """Compute steady-state distribution: π = T^T @ π.

        Returns:
            Steady-state probability vector (5,)
        """
        eigvals, eigvecs = np.linalg.eig(self.matrix.T)
        # Find eigenvector with eigenvalue 1
        idx = np.argmin(np.abs(eigvals - 1.0))
        steady = np.real(eigvecs[:, idx])
        return steady / np.sum(steady)


# ─── Probabilistic Recovery State ────────────────────────────────────────────

@dataclass
class ProbabilisticRecoveryState:
    """Soft state representation with probabilities.

    Instead of a single discrete state, maintains a probability
    distribution over all recovery states.
    """
    # State probabilities: [STABLE, UNCERTAIN, DEGRADED, RECOVERING, RESET_REQUIRED]
    probabilities: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    # Entropy of the distribution
    entropy: float = 0.0
    # Most likely state
    dominant_state: RecoveryState = RecoveryState.STABLE
    # Confidence in dominant state (max probability)
    confidence: float = 1.0

    def __post_init__(self):
        self._update_metrics()

    def _update_metrics(self):
        """Update entropy, dominant state, confidence."""
        # Entropy: H = -Σ p_i * log(p_i)
        eps = 1e-10
        self.entropy = -np.sum(self.probabilities * np.log(self.probabilities + eps))

        # Dominant state
        idx = np.argmax(self.probabilities)
        self.dominant_state = list(RecoveryState)[idx]

        # Confidence
        self.confidence = float(self.probabilities[idx])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "probabilities": {
                state.name: float(self.probabilities[i])
                for i, state in enumerate(RecoveryState)
            },
            "entropy": float(self.entropy),
            "dominant_state": self.dominant_state.name,
            "confidence": float(self.confidence),
        }


# ─── Recovery Dynamics ───────────────────────────────────────────────────────

@dataclass
class RecoveryReport:
    """Per-frame recovery dynamics metrics."""
    state: ProbabilisticRecoveryState = field(
        default_factory=ProbabilisticRecoveryState
    )
    transition_applied: bool = False
    observation_update_applied: bool = False
    frame_idx: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "frame_idx": self.frame_idx,
            "state": self.state.to_dict(),
            "transition_applied": self.transition_applied,
            "observation_update_applied": self.observation_update_applied,
        }


class RecoveryDynamics:
    """Main dynamics engine for probabilistic recovery.

    Combines:
    1. Transition dynamics: P(x_meta,t | x_meta,t-1)
    2. Observation updates: P(x_meta,t | obs_t)
    3. Entropy tracking: uncertainty over recovery states
    """

    def __init__(
        self,
        transition_matrix: Optional[RecoveryTransitionMatrix] = None,
        observation_weight: float = 0.3,
    ):
        self.transition = transition_matrix or RecoveryTransitionMatrix()
        self.observation_weight = observation_weight
        self.state = ProbabilisticRecoveryState()
        self._frame_count = 0

    def predict(self) -> ProbabilisticRecoveryState:
        """Predict next state using transition matrix.

        Returns:
            Predicted state probabilities
        """
        self.state.probabilities = self.transition.step(self.state.probabilities)
        self.state._update_metrics()
        self._frame_count += 1
        return self.state

    def update(
        self,
        trace: float,
        occlusion_count: int,
        uncertainty: float = 0.0,
    ) -> ProbabilisticRecoveryState:
        """Update state based on observations.

        Uses soft evidence instead of hard thresholds:
        - trace: covariance trace (higher → more uncertain)
        - occlusion_count: frames since last observation
        - uncertainty: identity uncertainty

        Args:
            trace: Covariance trace
            occlusion_count: Frames since last observation
            uncertainty: Identity uncertainty (0-1)

        Returns:
            Updated state probabilities
        """
        # Compute observation likelihood for each state
        likelihood = self._compute_likelihood(trace, occlusion_count, uncertainty)

        # Bayesian update with uniform prior (not current state)
        # This allows the observation to shift the state significantly
        uniform_prior = np.ones(5) / 5.0
        posterior = likelihood * uniform_prior
        posterior = posterior / (np.sum(posterior) + 1e-10)

        # Blend with current state
        self.state.probabilities = (
            (1 - self.observation_weight) * self.state.probabilities
            + self.observation_weight * posterior
        )
        self.state.probabilities = self.state.probabilities / (np.sum(self.state.probabilities) + 1e-10)
        self.state._update_metrics()

        return self.state

        # Blend with prior (soft update)
        self.state.probabilities = (
            (1 - self.observation_weight) * self.state.probabilities
            + self.observation_weight * posterior
        )
        self.state.probabilities = self.state.probabilities / (np.sum(self.state.probabilities) + 1e-10)
        self.state._update_metrics()

        return self.state

    def _compute_likelihood(
        self,
        trace: float,
        occlusion_count: int,
        uncertainty: float,
    ) -> np.ndarray:
        """Compute observation likelihood for each state.

        Uses soft sigmoid-based likelihoods instead of hard thresholds.

        Args:
            trace: Covariance trace
            occlusion_count: Frames since last observation
            uncertainty: Identity uncertainty

        Returns:
            Likelihood vector (5,)
        """
        # Sigmoid function for soft thresholds
        def sigmoid(x, center, scale):
            return 1.0 / (1.0 + np.exp(-scale * (x - center)))

        # Likelihood based on trace
        # STABLE: low trace (< 20)
        p_stable_trace = 1.0 - sigmoid(trace, 20.0, 0.1)
        # UNCERTAIN: medium trace (20-50)
        p_uncertain_trace = sigmoid(trace, 20.0, 0.1) * (1.0 - sigmoid(trace, 50.0, 0.1))
        # DEGRADED: high trace (50-100)
        p_degraded_trace = sigmoid(trace, 50.0, 0.1) * (1.0 - sigmoid(trace, 100.0, 0.1))
        # RECOVERING: very high trace (100-200)
        p_recovering_trace = sigmoid(trace, 100.0, 0.1) * (1.0 - sigmoid(trace, 200.0, 0.1))
        # RESET_REQUIRED: extreme trace (> 200)
        p_reset_trace = sigmoid(trace, 200.0, 0.1)

        trace_likelihood = np.array([
            p_stable_trace,
            p_uncertain_trace,
            p_degraded_trace,
            p_recovering_trace,
            p_reset_trace,
        ])

        # Likelihood based on occlusion count
        p_stable_occ = 1.0 - sigmoid(occlusion_count, 5.0, 0.5)
        p_uncertain_occ = sigmoid(occlusion_count, 5.0, 0.5) * (1.0 - sigmoid(occlusion_count, 10.0, 0.5))
        p_degraded_occ = sigmoid(occlusion_count, 10.0, 0.5) * (1.0 - sigmoid(occlusion_count, 20.0, 0.5))
        p_recovering_occ = sigmoid(occlusion_count, 20.0, 0.5) * (1.0 - sigmoid(occlusion_count, 30.0, 0.5))
        p_reset_occ = sigmoid(occlusion_count, 30.0, 0.5)

        occ_likelihood = np.array([
            p_stable_occ,
            p_uncertain_occ,
            p_degraded_occ,
            p_recovering_occ,
            p_reset_occ,
        ])

        # Combine likelihoods (product of independent observations)
        combined = trace_likelihood * occ_likelihood

        # Normalize
        return combined / (np.sum(combined) + 1e-10)

    def step(
        self,
        trace: float,
        occlusion_count: int,
        uncertainty: float = 0.0,
    ) -> RecoveryReport:
        """Full predict-update cycle.

        Args:
            trace: Covariance trace
            occlusion_count: Frames since last observation
            uncertainty: Identity uncertainty

        Returns:
            Recovery report
        """
        # Predict
        self.predict()

        # Update
        self.update(trace, occlusion_count, uncertainty)

        return RecoveryReport(
            state=ProbabilisticRecoveryState(
                probabilities=self.state.probabilities.copy(),
                entropy=self.state.entropy,
                dominant_state=self.state.dominant_state,
                confidence=self.state.confidence,
            ),
            transition_applied=True,
            observation_update_applied=True,
            frame_idx=self._frame_count,
        )

    def get_state_vector(self) -> np.ndarray:
        """Get current state as vector for integration with LatentState.

        Returns:
            State vector (5,) with probabilities
        """
        return self.state.probabilities.copy()

    def set_state_vector(self, vec: np.ndarray):
        """Set state from vector.

        Args:
            vec: State vector (5,) with probabilities
        """
        self.state.probabilities = vec / (np.sum(vec) + 1e-10)
        self.state._update_metrics()
