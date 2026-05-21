"""State Evolution Module.

Defines the latent state evolution model:
    x_t = f(x_{t-1}, u_t, epsilon_t)

where:
    x_t = latent state at time t
    f = state transition function
    u_t = control input (optional)
    epsilon_t = process noise

This module provides:
- State transition model
- Process noise model
- Uncertainty propagation
- Motion priors
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


from face_os.lie_group import SIM2Transform


@dataclass
class StateEvolutionConfig:
    """Configuration for state evolution."""

    # State dimension
    state_dim: int = 11

    # Process noise base variance
    process_noise_base: float = 0.01

    # Damping factor for state persistence
    damping_factor: float = 0.95

    # Motion prior strength
    motion_prior_strength: float = 0.1

    # Maximum state change per step
    max_state_change: float = 1.0


@dataclass
class StateEvolution:
    """State evolution model.

    x_t = f(x_{t-1}, u_t, epsilon_t)

    where:
        f(x, u, e) = A * x + B * u + e
        A = state transition matrix (damping)
        B = control input matrix
        e = process noise
    """

    # State dimension
    state_dim: int

    # State transition matrix A (state_dim x state_dim)
    transition_matrix: np.ndarray

    # Process noise covariance Q (state_dim x state_dim)
    process_noise_cov: np.ndarray

    # Damping per dimension (state_dim,)
    damping: np.ndarray

    def __init__(self, config: Optional[StateEvolutionConfig] = None):
        """Initialize state evolution model.

        Args:
            config: Configuration
        """
        config = config or StateEvolutionConfig()
        self.state_dim = config.state_dim

        # State transition: diagonal damping
        self.damping = np.array([
            0.95,  # yaw
            0.95,  # pitch
            0.95,  # roll
            0.99,  # identity_uncertainty
            0.99,  # appearance_latent_norm
            0.98,  # temporal_confidence
            0.90,  # drift_score
            0.95,  # continuity_score
            1.00,  # recovery_state
            0.99,  # brightness_mean
            0.99,  # contrast_mean
        ])[:self.state_dim]

        self.transition_matrix = np.diag(self.damping)

        # Process noise: per-dimension variance
        self.process_noise_cov = np.diag([
            0.5,   # yaw
            0.5,   # pitch
            0.3,   # roll
            0.01,  # identity_uncertainty
            1.0,   # appearance_latent_norm
            0.02,  # temporal_confidence
            0.1,   # drift_score
            0.01,  # continuity_score
            0.001, # recovery_state
            5.0,   # brightness_mean
            2.0,   # contrast_mean
        ])[:self.state_dim, :self.state_dim]

    def predict(
        self,
        state: np.ndarray,
        control: Optional[np.ndarray] = None,
    ) -> tuple:
        """Predict next state.

        x_{t|t-1} = A * x_{t-1} + B * u_t
        P_{t|t-1} = A * P_{t-1} * A^T + Q

        Args:
            state: Current state (state_dim,)
            control: Control input (optional)

        Returns:
            (predicted_state, predicted_covariance)
        """
        # State prediction: x = A * x
        predicted_state = self.transition_matrix @ state

        # Add control input if provided
        if control is not None:
            predicted_state += control

        # Clamp state change
        state_change = predicted_state - state
        state_change = np.clip(state_change, -1.0, 1.0)
        predicted_state = state + state_change

        return predicted_state

    def predict_covariance(
        self,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """Predict covariance.

        P_{t|t-1} = A * P_{t-1} * A^T + Q

        Args:
            covariance: Current covariance (state_dim, state_dim)

        Returns:
            Predicted covariance
        """
        A = self.transition_matrix
        Q = self.process_noise_cov
        predicted_cov = A @ covariance @ A.T + Q
        return predicted_cov

    def compute_innovation(
        self,
        predicted_state: np.ndarray,
        observation: np.ndarray,
        observation_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute innovation (observation residual).

        innovation = y_t - H * x_{t|t-1}

        Args:
            predicted_state: Predicted state
            observation: Observation
            observation_matrix: Observation matrix H

        Returns:
            Innovation vector
        """
        predicted_obs = observation_matrix @ predicted_state
        innovation = observation - predicted_obs
        return innovation

    def compute_innovation_covariance(
        self,
        predicted_covariance: np.ndarray,
        observation_matrix: np.ndarray,
        observation_noise_cov: np.ndarray,
    ) -> np.ndarray:
        """Compute innovation covariance.

        S = H * P_{t|t-1} * H^T + R

        Args:
            predicted_covariance: Predicted covariance
            observation_matrix: Observation matrix H
            observation_noise_cov: Observation noise R

        Returns:
            Innovation covariance
        """
        H = observation_matrix
        P = predicted_covariance
        R = observation_noise_cov
        S = H @ P @ H.T + R
        return S

    def compute_kalman_gain(
        self,
        predicted_covariance: np.ndarray,
        observation_matrix: np.ndarray,
        innovation_covariance: np.ndarray,
    ) -> np.ndarray:
        """Compute Kalman gain.

        K = P_{t|t-1} * H^T * S^{-1}

        Args:
            predicted_covariance: Predicted covariance
            observation_matrix: Observation matrix H
            innovation_covariance: Innovation covariance S

        Returns:
            Kalman gain matrix
        """
        H = observation_matrix
        P = predicted_covariance
        S = innovation_covariance

        # Compute gain with numerical stability
        try:
            K = P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix: use pseudoinverse
            K = P @ H.T @ np.linalg.pinv(S)

        return K

    def update_state(
        self,
        predicted_state: np.ndarray,
        kalman_gain: np.ndarray,
        innovation: np.ndarray,
    ) -> np.ndarray:
        """Update state with observation.

        x_t = x_{t|t-1} + K * innovation

        Args:
            predicted_state: Predicted state
            kalman_gain: Kalman gain
            innovation: Innovation

        Returns:
            Updated state
        """
        updated_state = predicted_state + kalman_gain @ innovation
        return updated_state

    def update_covariance(
        self,
        predicted_covariance: np.ndarray,
        kalman_gain: np.ndarray,
        observation_matrix: np.ndarray,
    ) -> np.ndarray:
        """Update covariance.

        P_t = (I - K * H) * P_{t|t-1}

        Args:
            predicted_covariance: Predicted covariance
            kalman_gain: Kalman gain
            observation_matrix: Observation matrix H

        Returns:
            Updated covariance
        """
        I = np.eye(self.state_dim)
        K = kalman_gain
        H = observation_matrix

        # Joseph form for numerical stability
        IKH = I - K @ H
        updated_cov = IKH @ predicted_covariance @ IKH.T + K @ K.T

        return updated_cov

    def predict_with_velocity(
        self,
        T_prev: 'SIM2Transform',
        T_curr: 'SIM2Transform',
    ) -> 'SIM2Transform':
        """Constant-velocity prediction on SIM(2) using Lie algebra.

        RULE 6: Implements T_hat(t+1) = T(t) * exp(v_t)
        where v_t = log(T_t) - log(T_t-1)

        This predicts the next transform by extrapolating the velocity
        in Lie algebra space, then mapping back to the group.

        Args:
            T_prev: Previous transform T(t-1)
            T_curr: Current transform T(t)

        Returns:
            Predicted transform T_hat(t+1)
        """
        # Velocity in Lie algebra: v_t = log(T_t) - log(T_{t-1})
        v_prev = T_prev.log()
        v_curr = T_curr.log()
        velocity = v_curr - v_prev

        # Predict: T_hat(t+1) = T(t) * exp(v_t)
        T_velocity = SIM2Transform.exp(velocity)
        T_predicted = T_curr.compose(T_velocity)

        return T_predicted

    def predict_update_full(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        observation: np.ndarray,
        observation_matrix: np.ndarray,
        observation_noise_cov: np.ndarray,
    ) -> tuple:
        """Full predict-update cycle.

        RULE 6: StateEvolution must predict AND update (not just predict).

        Args:
            state: Current state
            covariance: Current covariance
            observation: Observation vector
            observation_matrix: Observation matrix H
            observation_noise_cov: Observation noise R

        Returns:
            (updated_state, updated_covariance)
        """
        # Predict
        predicted_state = self.predict(state)
        predicted_cov = self.predict_covariance(covariance)

        # Update
        innovation = self.compute_innovation(predicted_state, observation, observation_matrix)
        S = self.compute_innovation_covariance(predicted_cov, observation_matrix, observation_noise_cov)
        K = self.compute_kalman_gain(predicted_cov, observation_matrix, S)
        updated_state = self.update_state(predicted_state, K, innovation)
        updated_cov = self.update_covariance(predicted_cov, K, observation_matrix)

        return updated_state, updated_cov


@dataclass
class StateEvolutionReport:
    """Report for state evolution metrics."""

    predicted_state: np.ndarray
    predicted_covariance: np.ndarray
    innovation: Optional[np.ndarray]
    innovation_covariance: Optional[np.ndarray]
    kalman_gain: Optional[np.ndarray]
    updated_state: Optional[np.ndarray]
    updated_covariance: Optional[np.ndarray]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "predicted_state_norm": float(np.linalg.norm(self.predicted_state)),
            "predicted_cov_trace": float(np.trace(self.predicted_covariance)),
            "innovation_norm": float(np.linalg.norm(self.innovation)) if self.innovation is not None else None,
            "innovation_cov_trace": float(np.trace(self.innovation_covariance)) if self.innovation_covariance is not None else None,
            "kalman_gain_norm": float(np.linalg.norm(self.kalman_gain)) if self.kalman_gain is not None else None,
            "updated_state_norm": float(np.linalg.norm(self.updated_state)) if self.updated_state is not None else None,
            "updated_cov_trace": float(np.trace(self.updated_covariance)) if self.updated_covariance is not None else None,
        }
