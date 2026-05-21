"""
observability.py — Observability Analysis for Face OS.

Phase 2C: Observability Analysis.

The observability matrix:
    O = [H; HF; HF²; ...]

Where:
    F = state transition Jacobian
    H = observation Jacobian

Determines:
    - Observable dimensions
    - Weakly observable dimensions
    - Degenerate dimensions
    - Unstable latent coupling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from face_os.state_space import (
    LatentState,
    StateTransitionModel,
    ObservationModel,
)


# ─── Degeneracy Report ───────────────────────────────────────────────────────

@dataclass
class DegeneracyReport:
    """Report on state degeneracies."""
    lighting_ambiguity: float = 0.0      # [0, 1] — higher = more ambiguous
    pose_degeneracy: float = 0.0         # [0, 1] — higher = more degenerate
    identity_ambiguity: float = 0.0      # [0, 1] — higher = more ambiguous
    temporal_degeneracy: float = 0.0     # [0, 1] — higher = more degenerate
    worst_dimension: int = -1            # Most degenerate dimension
    worst_condition: float = 1.0         # Condition number of worst dimension

    def to_dict(self) -> dict:
        return {
            "lighting_ambiguity": self.lighting_ambiguity,
            "pose_degeneracy": self.pose_degeneracy,
            "identity_ambiguity": self.identity_ambiguity,
            "temporal_degeneracy": self.temporal_degeneracy,
            "worst_dimension": self.worst_dimension,
            "worst_condition": self.worst_condition,
        }


# ─── Observability Report ────────────────────────────────────────────────────

@dataclass
class ObservabilityReport:
    """Observability analysis report."""
    observability_matrix: Optional[np.ndarray] = None
    observability_rank: int = 0
    observable_dimensions: List[int] = field(default_factory=list)
    degenerate_dimensions: List[int] = field(default_factory=list)
    condition_number: float = 1.0
    latent_coupling_matrix: Optional[np.ndarray] = None
    degeneracy_report: Optional[DegeneracyReport] = None
    state_dimension: int = 0
    observation_dimension: int = 0

    def __post_init__(self):
        if self.latent_coupling_matrix is None:
            self.latent_coupling_matrix = np.zeros((0, 0))

    def to_dict(self) -> dict:
        return {
            "observability_rank": self.observability_rank,
            "observable_dimensions": self.observable_dimensions,
            "degenerate_dimensions": self.degenerate_dimensions,
            "condition_number": self.condition_number,
            "state_dimension": self.state_dimension,
            "observation_dimension": self.observation_dimension,
            "degeneracy_report": self.degeneracy_report.to_dict() if self.degeneracy_report else None,
            "latent_coupling_matrix_shape": list(self.latent_coupling_matrix.shape) if self.latent_coupling_matrix is not None else [],
        }


# ─── Observability Analyzer ──────────────────────────────────────────────────

class ObservabilityAnalyzer:
    """Analyzes observability of the state-space system.

    Computes:
    - Observability matrix O = [H; HF; HF²; ...]
    - Observable dimensions (rank of O)
    - Degenerate dimensions (near-zero singular values)
    - Latent coupling matrix (how state dimensions interact)
    """

    def __init__(self, horizon: int = 5):
        """Initialize analyzer.

        Args:
            horizon: Number of steps to include in observability matrix
        """
        self.horizon = horizon
        self.transition = StateTransitionModel()
        self.observation = ObservationModel()

    def analyze(self, state: LatentState) -> ObservabilityReport:
        """Analyze observability at given state.

        Args:
            state: Current latent state

        Returns:
            ObservabilityReport with all observability metrics
        """
        n = LatentState.STATE_DIM
        H = self.observation.jacobian(state)
        F = self._compute_transition_jacobian(state)
        m = H.shape[0]  # Observation dimension

        # Build observability matrix O = [H; HF; HF²; ...]
        O = np.zeros((m * self.horizon, n))
        F_power = np.eye(n)
        for i in range(self.horizon):
            O[i * m:(i + 1) * m, :] = H @ F_power
            F_power = F_power @ F

        # Compute rank via SVD
        U, S, Vt = np.linalg.svd(O, full_matrices=False)
        rank = int(np.sum(S > 1e-10))

        # Identify observable dimensions (columns of Vt with significant singular values)
        observable_dims = []
        degenerate_dims = []
        for i in range(len(S)):
            if S[i] > 1e-6:
                # This singular value corresponds to a combination of state dimensions
                pass

        # Identify dimensions by checking which state components contribute
        # to the observability matrix
        col_norms = np.linalg.norm(O, axis=0)
        threshold = np.max(col_norms) * 1e-6
        for i in range(n):
            if col_norms[i] > threshold:
                observable_dims.append(i)
            else:
                degenerate_dims.append(i)

        # Compute condition number
        if S[-1] > 1e-15:
            condition = S[0] / S[-1]
        else:
            condition = 1e15

        # Compute latent coupling matrix: O^T @ O
        coupling = O.T @ O

        # Compute degeneracy report
        degeneracy = self._compute_degeneracy(state, O, S)

        return ObservabilityReport(
            observability_matrix=O,
            observability_rank=rank,
            observable_dimensions=observable_dims,
            degenerate_dimensions=degenerate_dims,
            condition_number=float(condition),
            latent_coupling_matrix=coupling,
            degeneracy_report=degeneracy,
            state_dimension=n,
            observation_dimension=m,
        )

    def _compute_transition_jacobian(self, state: LatentState) -> np.ndarray:
        """Compute state transition Jacobian F = df/dx.

        For the linear damping model:
            x_t = A * x_{t-1}
            F = A (diagonal damping matrix)
        """
        return np.diag(self.transition.damping)

    def _compute_degeneracy(
        self,
        state: LatentState,
        O: np.ndarray,
        S: np.ndarray,
    ) -> DegeneracyReport:
        """Compute degeneracy metrics.

        Args:
            state: Current state
            O: Observability matrix
            S: Singular values

        Returns:
            DegeneracyReport with degeneracy metrics
        """
        n = LatentState.STATE_DIM

        # Lighting ambiguity: brightness/albedo cannot be separated
        # High brightness_mean → more ambiguous
        lighting_amb = min(1.0, state.brightness_mean / 255.0)

        # Pose degeneracy: extreme angles → less observable
        pose_mag = abs(state.yaw) + abs(state.pitch) + abs(state.roll)
        pose_deg = min(1.0, pose_mag / 90.0)

        # Identity ambiguity: high uncertainty → more ambiguous
        id_amb = state.identity_uncertainty

        # Temporal degeneracy: low confidence → less observable
        temp_deg = 1.0 - state.temporal_confidence

        # Find worst dimension
        col_norms = np.linalg.norm(O, axis=0)
        worst_dim = int(np.argmin(col_norms))
        worst_cond = float(np.max(col_norms) / max(np.min(col_norms), 1e-15))

        return DegeneracyReport(
            lighting_ambiguity=lighting_amb,
            pose_degeneracy=pose_deg,
            identity_ambiguity=id_amb,
            temporal_degeneracy=temp_deg,
            worst_dimension=worst_dim,
            worst_condition=worst_cond,
        )
