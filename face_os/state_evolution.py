"""State Evolution Module.

BEAST MODE FIXES:
- Fixed the Joseph Form Covariance Nuke (added missing R matrix).
- Fixed the 360-Degree Spin Trap in SIM(2) velocity prediction.
- Replaced np.linalg.inv with np.linalg.solve for numerical stability.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from face_os.lie_group import SIM2Transform


@dataclass
class StateEvolutionConfig:
    """Configuration for state evolution."""
    state_dim: int = 11
    process_noise_base: float = 0.01
    damping_factor: float = 0.95
    motion_prior_strength: float = 0.1
    max_state_change: float = 1.0


@dataclass
class StateEvolution:
    """State evolution model."""
    state_dim: int
    transition_matrix: np.ndarray
    process_noise_cov: np.ndarray
    damping: np.ndarray

    def __init__(self, config: Optional[StateEvolutionConfig] = None):
        config = config or StateEvolutionConfig()
        self.state_dim = config.state_dim

        self.damping = np.array([
            0.95, 0.95, 0.95, 0.99, 0.99, 0.98, 0.90, 0.95, 1.00, 0.99, 0.99
        ])[:self.state_dim]

        self.transition_matrix = np.diag(self.damping)

        self.process_noise_cov = np.diag([
            0.5, 0.5, 0.3, 0.01, 1.0, 0.02, 0.1, 0.01, 0.001, 5.0, 2.0
        ])[:self.state_dim, :self.state_dim]

    def predict(self, state: np.ndarray, control: Optional[np.ndarray] = None) -> np.ndarray:
        predicted_state = self.transition_matrix @ state
        if control is not None:
            predicted_state += control

        state_change = predicted_state - state
        state_change = np.clip(state_change, -1.0, 1.0)
        predicted_state = state + state_change
        return predicted_state

    def predict_covariance(self, covariance: np.ndarray) -> np.ndarray:
        A = self.transition_matrix
        Q = self.process_noise_cov
        return A @ covariance @ A.T + Q

    def compute_innovation(self, predicted_state: np.ndarray, observation: np.ndarray, observation_matrix: np.ndarray) -> np.ndarray:
        return observation - (observation_matrix @ predicted_state)

    def compute_innovation_covariance(self, predicted_covariance: np.ndarray, observation_matrix: np.ndarray, observation_noise_cov: np.ndarray) -> np.ndarray:
        H = observation_matrix
        P = predicted_covariance
        R = observation_noise_cov
        return H @ P @ H.T + R

    def compute_kalman_gain(self, predicted_covariance: np.ndarray, observation_matrix: np.ndarray, innovation_covariance: np.ndarray) -> np.ndarray:
        H = observation_matrix
        P = predicted_covariance
        S = innovation_covariance

        # BEAST MODE: Use solve instead of inv for numerical stability
        # K = P @ H.T @ S^-1  =>  S @ K.T = H @ P.T  =>  K.T = solve(S, H @ P.T)
        try:
            K_T = np.linalg.solve(S, H @ P.T)
            K = K_T.T
        except np.linalg.LinAlgError:
            K = P @ H.T @ np.linalg.pinv(S)

        return K

    def update_state(self, predicted_state: np.ndarray, kalman_gain: np.ndarray, innovation: np.ndarray) -> np.ndarray:
        return predicted_state + kalman_gain @ innovation

    def update_covariance(
        self,
        predicted_covariance: np.ndarray,
        kalman_gain: np.ndarray,
        observation_matrix: np.ndarray,
        observation_noise_cov: np.ndarray,
    ) -> np.ndarray:
        I = np.eye(self.state_dim)
        K = kalman_gain
        H = observation_matrix
        R = observation_noise_cov

        # BEAST MODE FIX: Joseph form requires R! P = (I - KH)P(I - KH)^T + K R K^T
        IKH = I - K @ H
        updated_cov = IKH @ predicted_covariance @ IKH.T + K @ R @ K.T

        return updated_cov

    def predict_with_velocity(self, T_prev: 'SIM2Transform', T_curr: 'SIM2Transform') -> 'SIM2Transform':
        # BEAST MODE FIX: Unwrap angle difference to prevent 360-degree spin
        dtheta = T_curr.theta - T_prev.theta
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        
        dtx = T_curr.tx - T_prev.tx
        dty = T_curr.ty - T_prev.ty
        dlog_scale = np.log(T_curr.scale) - np.log(T_prev.scale)
        
        velocity = np.array([dtheta, dtx, dty, dlog_scale])
        
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
        predicted_state = self.predict(state)
        predicted_cov = self.predict_covariance(covariance)

        innovation = self.compute_innovation(predicted_state, observation, observation_matrix)
        S = self.compute_innovation_covariance(predicted_cov, observation_matrix, observation_noise_cov)
        K = self.compute_kalman_gain(predicted_cov, observation_matrix, S)
        updated_state = self.update_state(predicted_state, K, innovation)
        
        # BEAST MODE: Pass R to update_covariance
        updated_cov = self.update_covariance(predicted_cov, K, observation_matrix, observation_noise_cov)

        return updated_state, updated_cov


@dataclass
class StateEvolutionReport:
    predicted_state: np.ndarray
    predicted_covariance: np.ndarray
    innovation: Optional[np.ndarray]
    innovation_covariance: Optional[np.ndarray]
    kalman_gain: Optional[np.ndarray]
    updated_state: Optional[np.ndarray]
    updated_covariance: Optional[np.ndarray]

    def to_dict(self) -> dict:
        return {
            "predicted_state_norm": float(np.linalg.norm(self.predicted_state)),
            "predicted_cov_trace": float(np.trace(self.predicted_covariance)),
            "innovation_norm": float(np.linalg.norm(self.innovation)) if self.innovation is not None else None,
            "innovation_cov_trace": float(np.trace(self.innovation_covariance)) if self.innovation_covariance is not None else None,
            "kalman_gain_norm": float(np.linalg.norm(self.kalman_gain)) if self.kalman_gain is not None else None,
            "updated_state_norm": float(np.linalg.norm(self.updated_state)) if self.updated_state is not None else None,
            "updated_cov_trace": float(np.trace(self.updated_covariance)) if self.updated_covariance is not None else None,
        }