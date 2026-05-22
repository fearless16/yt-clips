"""Subsystem C — Temporal Estimation.

Maintains temporal consistency via Bayesian belief propagation.

Output: TemporalState (from face_os.types)
Delegates to: state_evolution.py, temporal_solve.py, lie_group.py

BOUNDARY CONTRACT:
- MUST NOT inject backward textures
- MUST NOT perform frame averaging
- MUST NOT handle identity or geometry logic
"""

import numpy as np

from face_os.types import TemporalState


class TemporalEstimator:
    """Subsystem C: Temporal consistency.

    Thin wrapper that delegates to state_evolution.py and lie_group.py.
    Enforces forward-only prediction (no backward texture injection).

    FORBIDDEN: backward texture injection, frame averaging, identity logic
    """

    def __init__(self, state_evolution):
        """Args:
        state_evolution: StateEvolution instance from state_evolution.py
        """
        self._evolution = state_evolution
        self._prev_sim2 = None

    def predict(self, current_sim2, observation, H, R) -> TemporalState:
        """Predict next temporal state via SIM(2) velocity extrapolation.

        Args:
            current_sim2: Current SIM(2) transform
            observation: Observation vector
            H: Observation matrix
            R: Observation noise

        Returns:
            TemporalState with prediction
        """
        state = TemporalState()

        if self._prev_sim2 is not None and current_sim2 is not None:
            try:
                predicted = self._evolution.predict_with_velocity(
                    self._prev_sim2, current_sim2
                )
                state.motion_field = predicted
            except Exception:
                pass

        self._prev_sim2 = current_sim2

        return state

    def update(self, state: TemporalState, latent_state, covariance) -> TemporalState:
        """Update temporal state with new observations."""
        state.smoothing_constraints["latent_state"] = latent_state
        state.smoothing_constraints["covariance"] = covariance
        return state

    def reset(self):
        """Reset temporal state between clips."""
        self._prev_sim2 = None
