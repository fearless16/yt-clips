"""
state_space.py — State-Space Formulation for Face OS.

Phase 2A: State-Space Formulation.

The core model:
    x_t = f(x_{t-1}, u_t, ε_t)

Where:
    x_t = latent state (geometry, identity, lighting, temporal uncertainty, recovery)
    u_t = observations (yaw, pitch, roll, confidence, identity_uncertainty)
    ε_t = process noise

State-space components:
    - LatentState: the hidden state vector
    - StateTransitionModel: x_t = f(x_{t-1}) + ε_t
    - ProcessNoiseModel: ε_t ~ N(0, Q)
    - ObservationModel: z_t = h(x_t) + δ_t
    - StateSpaceEstimator: Kalman-like predict-update cycle
    - StateEvolutionReport: per-frame state evolution metrics
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


# ─── Latent State ────────────────────────────────────────────────────────────

@dataclass
class LatentState:
    """The hidden state vector.

    Components:
    - Geometry: yaw, pitch, roll (3)
    - Identity: identity_uncertainty, appearance_latent_norm (2)
    - Temporal: temporal_confidence, drift_score, continuity_score (3)
    - Recovery: recovery_state (encoded as float: 0=stable, 1=reset_required)
    - Lighting: brightness_mean, contrast_mean (2)

    Total dimension: 11
    """
    # Geometry
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    # Identity
    identity_uncertainty: float = 0.5
    appearance_latent_norm: float = 0.0
    # Temporal
    temporal_confidence: float = 1.0
    drift_score: float = 0.0
    continuity_score: float = 1.0
    # Recovery
    recovery_state: str = "stable"
    # Lighting
    brightness_mean: float = 128.0
    contrast_mean: float = 50.0
    # Covariance (stored separately)
    covariance: Optional[np.ndarray] = None

    # State dimension
    STATE_DIM = 11

    def __post_init__(self):
        if self.covariance is None:
            # Start with moderate uncertainty
            self.covariance = np.eye(self.STATE_DIM) * 1.0

    def to_vector(self) -> np.ndarray:
        """Convert state to vector representation."""
        recovery_float = {
            "stable": 0.0,
            "uncertain": 0.25,
            "degraded": 0.5,
            "recovering": 0.75,
            "reset_required": 1.0,
        }.get(self.recovery_state, 0.0)

        return np.array([
            self.yaw,
            self.pitch,
            self.roll,
            self.identity_uncertainty,
            self.appearance_latent_norm,
            self.temporal_confidence,
            self.drift_score,
            self.continuity_score,
            recovery_float,
            self.brightness_mean,
            self.contrast_mean,
        ], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> LatentState:
        """Construct state from vector representation."""
        recovery_map = {
            0.0: "stable",
            0.25: "uncertain",
            0.5: "degraded",
            0.75: "recovering",
            1.0: "reset_required",
        }
        recovery_float = float(np.clip(vec[8], 0, 1))
        # Find closest recovery state
        closest = min(recovery_map.keys(), key=lambda x: abs(x - recovery_float))
        recovery_state = recovery_map[closest]

        return cls(
            yaw=float(vec[0]),
            pitch=float(vec[1]),
            roll=float(vec[2]),
            identity_uncertainty=float(np.clip(vec[3], 0, 1)),
            appearance_latent_norm=float(max(0, vec[4])),
            temporal_confidence=float(np.clip(vec[5], 0, 1)),
            drift_score=float(max(0, vec[6])),
            continuity_score=float(np.clip(vec[7], 0, 1)),
            recovery_state=recovery_state,
            brightness_mean=float(np.clip(vec[9], 0, 255)),
            contrast_mean=float(max(0, vec[10])),
        )


# ─── Process Noise Model ─────────────────────────────────────────────────────

@dataclass
class ProcessNoiseModel:
    """Process noise model: ε_t ~ N(0, Q).

    Covariance Q defines how much each state component is expected
    to change between frames due to unmodeled dynamics.
    """
    covariance: np.ndarray = field(default_factory=lambda: np.diag([
        0.5,    # yaw (degrees^2)
        0.5,    # pitch
        0.3,    # roll
        0.01,   # identity_uncertainty
        1.0,    # appearance_latent_norm (reduced from 100 — identity should be stable)
        0.02,   # temporal_confidence
        0.1,    # drift_score
        0.01,   # continuity_score
        0.001,  # recovery_state (small noise for numerical stability)
        5.0,    # brightness_mean
        2.0,    # contrast_mean
    ]).astype(np.float64))

    def sample(self) -> np.ndarray:
        """Sample noise from process distribution."""
        return np.random.multivariate_normal(
            np.zeros(self.covariance.shape[0]),
            self.covariance,
        )


# ─── State Transition Model ──────────────────────────────────────────────────

class StateTransitionModel:
    """State transition model: x_t = f(x_{t-1}) + ε_t.

    The transition is a simple linear model with damping:
        x_t = A * x_{t-1} + ε_t

    Where A is a diagonal damping matrix (each component decays toward
    its equilibrium value).
    """

    def __init__(self, damping: Optional[np.ndarray] = None):
        # Damping factors (how fast each component returns to equilibrium)
        if damping is None:
            self.damping = np.array([
                0.95,   # yaw (slow decay)
                0.95,   # pitch
                0.95,   # roll
                0.99,   # identity_uncertainty (very slow decay)
                0.99,   # appearance_latent_norm
                0.98,   # temporal_confidence
                0.90,   # drift_score (faster decay toward 0)
                0.95,   # continuity_score
                1.0,    # recovery_state (no decay)
                0.99,   # brightness_mean
                0.99,   # contrast_mean
            ])
        else:
            self.damping = damping

        # Equilibrium values (what each component decays toward)
        self.equilibrium = np.array([
            0.0,    # yaw
            0.0,    # pitch
            0.0,    # roll
            0.5,    # identity_uncertainty
            0.0,    # appearance_latent_norm
            1.0,    # temporal_confidence
            0.0,    # drift_score
            1.0,    # continuity_score
            0.0,    # recovery_state
            128.0,  # brightness_mean
            50.0,   # contrast_mean
        ])

    def predict(
        self,
        state: LatentState,
        noise: Optional[ProcessNoiseModel] = None,
    ) -> LatentState:
        """Predict next state: x_t = A * x_{t-1} + ε_t.

        Args:
            state: Current latent state
            noise: Optional process noise model

        Returns:
            Predicted next state
        """
        x = state.to_vector()

        # Apply damping toward equilibrium
        x_pred = self.damping * x + (1 - self.damping) * self.equilibrium

        # Add process noise if provided
        if noise is not None:
            epsilon = noise.sample()
            x_pred += epsilon

        # Clip to valid ranges
        x_pred[3] = np.clip(x_pred[3], 0, 1)   # identity_uncertainty
        x_pred[5] = np.clip(x_pred[5], 0, 1)   # temporal_confidence
        x_pred[7] = np.clip(x_pred[7], 0, 1)   # continuity_score
        x_pred[8] = np.clip(x_pred[8], 0, 1)   # recovery_state
        x_pred[9] = np.clip(x_pred[9], 0, 255) # brightness_mean

        # Update covariance: P_pred = A * P * A^T + Q
        A = np.diag(self.damping)
        P = state.covariance
        Q = noise.covariance if noise is not None else np.zeros_like(P)
        P_pred = A @ P @ A.T + Q

        new_state = LatentState.from_vector(x_pred)
        new_state.covariance = P_pred
        return new_state


# ─── Observation Model ───────────────────────────────────────────────────────

class ObservationModel:
    """Observation model: z_t = h(x_t) + δ_t.

    Maps latent state to observable quantities.
    Now observes 9 of 11 state dimensions (added brightness, contrast, drift, continuity).
    """

    # Observation dimension
    OBS_DIM = 9

    def __init__(self):
        # Observation noise covariance (9x9)
        self.R = np.diag([
            1.0,    # yaw observation noise
            1.0,    # pitch
            0.5,    # roll
            0.05,   # identity_uncertainty
            0.05,   # temporal_confidence (confidence)
            5.0,    # brightness_mean (high noise — lighting varies)
            2.0,    # contrast_mean
            0.1,    # drift_score
            0.05,   # continuity_score
        ]).astype(np.float64)

    def observe(self, state: LatentState) -> Dict[str, float]:
        """Map latent state to observation space.

        Args:
            state: Latent state

        Returns:
            Observation dict
        """
        return {
            "yaw": state.yaw,
            "pitch": state.pitch,
            "roll": state.roll,
            "identity_uncertainty": state.identity_uncertainty,
            "confidence": state.temporal_confidence,
            "brightness_mean": state.brightness_mean,
            "contrast_mean": state.contrast_mean,
            "drift_score": state.drift_score,
            "continuity_score": state.continuity_score,
        }

    def residual(
        self,
        state: LatentState,
        observation: Dict[str, float],
    ) -> np.ndarray:
        """Compute observation residual: z - h(x).

        Args:
            state: Predicted state
            observation: Actual observation

        Returns:
            Residual vector
        """
        predicted = self.observe(state)
        return np.array([
            observation.get("yaw", 0) - predicted["yaw"],
            observation.get("pitch", 0) - predicted["pitch"],
            observation.get("roll", 0) - predicted["roll"],
            observation.get("identity_uncertainty", 0) - predicted["identity_uncertainty"],
            observation.get("confidence", 0) - predicted["confidence"],
            observation.get("brightness_mean", 128) - predicted["brightness_mean"],
            observation.get("contrast_mean", 50) - predicted["contrast_mean"],
            observation.get("drift_score", 0) - predicted["drift_score"],
            observation.get("continuity_score", 1) - predicted["continuity_score"],
        ])

    def jacobian(self, state: LatentState) -> np.ndarray:
        """Compute observation Jacobian H.

        For linear observation model, H is a selection matrix.

        Args:
            state: Latent state

        Returns:
            Jacobian matrix (obs_dim x state_dim)
        """
        # Observation maps: yaw(0), pitch(1), roll(2), id_unc(3), conf(5),
        #                    brightness(9), contrast(10), drift(6), continuity(7)
        H = np.zeros((self.OBS_DIM, LatentState.STATE_DIM))
        H[0, 0] = 1.0  # yaw
        H[1, 1] = 1.0  # pitch
        H[2, 2] = 1.0  # roll
        H[3, 3] = 1.0  # identity_uncertainty
        H[4, 5] = 1.0  # temporal_confidence (observation = confidence)
        H[5, 9] = 1.0  # brightness_mean
        H[6, 10] = 1.0  # contrast_mean
        H[7, 6] = 1.0  # drift_score
        H[8, 7] = 1.0  # continuity_score
        return H


# ─── State Evolution Report ──────────────────────────────────────────────────

@dataclass
class StateEvolutionReport:
    """Per-frame state evolution report."""
    state: LatentState = field(default_factory=LatentState)
    covariance_trace: float = 0.0
    covariance_eigenvalues: List[float] = field(default_factory=list)
    drift_score: float = 0.0
    continuity_score: float = 1.0
    uncertainty_growth: float = 0.0
    recovery_state: str = "stable"
    frame_idx: int = 0

    def __post_init__(self):
        if self.state.covariance is not None:
            self.covariance_trace = float(np.trace(self.state.covariance))
            eigvals = np.linalg.eigvalsh(self.state.covariance)
            self.covariance_eigenvalues = eigvals.tolist()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "state": {
                "yaw": self.state.yaw,
                "pitch": self.state.pitch,
                "roll": self.state.roll,
                "identity_uncertainty": self.state.identity_uncertainty,
                "appearance_latent_norm": self.state.appearance_latent_norm,
                "temporal_confidence": self.state.temporal_confidence,
                "drift_score": self.state.drift_score,
                "continuity_score": self.state.continuity_score,
                "recovery_state": self.state.recovery_state,
                "brightness_mean": self.state.brightness_mean,
                "contrast_mean": self.state.contrast_mean,
            },
            "covariance_trace": self.covariance_trace,
            "covariance_eigenvalues": self.covariance_eigenvalues,
            "drift_score": self.drift_score,
            "continuity_score": self.continuity_score,
            "uncertainty_growth": self.uncertainty_growth,
            "recovery_state": self.recovery_state,
        }


# ─── State-Space Estimator ───────────────────────────────────────────────────

class StateSpaceEstimator:
    """Kalman-like state-space estimator.

    Implements predict-update cycle:
        1. Predict: x_pred = A * x_{t-1} + ε_t
        2. Update: x_t = x_pred + K * (z_t - H * x_pred)

    Where K is the Kalman gain.
    """

    def __init__(self):
        self.transition = StateTransitionModel()
        self.process_noise = ProcessNoiseModel()
        self.observation = ObservationModel()
        self.state = LatentState()
        self._frame_count = 0
        self._last_trace = 0.0
        self._occlusion_count = 0

    def predict(self) -> LatentState:
        """Predict next state without observation.

        Returns:
            Predicted state
        """
        self.state = self.transition.predict(self.state, self.process_noise)
        self._frame_count += 1
        self._occlusion_count += 1

        # Update recovery state based on occlusion
        self._update_recovery_state()

        return self.state

    def update(self, observation: Dict[str, float]) -> LatentState:
        """Update state with observation.

        Args:
            observation: Observation dict with yaw, pitch, roll, confidence, identity_uncertainty,
                         brightness_mean, contrast_mean, drift_score, continuity_score

        Returns:
            Updated state
        """
        # Compute Kalman gain: K = P * H^T * (H * P * H^T + R)^{-1}
        H = self.observation.jacobian(self.state)
        P = self.state.covariance
        R = self.observation.R

        S = H @ P @ H.T + R  # Innovation covariance
        K = P @ H.T @ np.linalg.inv(S)

        # Compute residual using observation model
        residual = self.observation.residual(self.state, observation)

        # Update state
        x = self.state.to_vector()
        x_update = x + K @ residual
        self.state = LatentState.from_vector(x_update)

        # Update covariance: P = (I - K * H) * P
        I = np.eye(LatentState.STATE_DIM)
        self.state.covariance = (I - K @ H) @ P

        # Reset occlusion count
        self._occlusion_count = 0
        self._update_recovery_state()

        return self.state

    def step(self, observation: Dict[str, float]) -> StateEvolutionReport:
        """Full predict-update cycle.

        Args:
            observation: Observation dict

        Returns:
            State evolution report
        """
        # Predict
        self.predict()

        # Update
        self.update(observation)

        # Compute metrics
        return self.get_report()

    def get_report(self) -> StateEvolutionReport:
        """Get current state evolution report.

        Returns:
            StateEvolutionReport with all metrics
        """
        # Compute drift score (distance from equilibrium)
        equilibrium = LatentState()
        drift = np.linalg.norm(
            self.state.to_vector() - equilibrium.to_vector()
        )

        # Compute uncertainty growth
        current_trace = float(np.trace(self.state.covariance))
        uncertainty_growth = current_trace - self._last_trace
        self._last_trace = current_trace

        return StateEvolutionReport(
            state=self.state,
            drift_score=drift,
            continuity_score=self.state.continuity_score,
            uncertainty_growth=uncertainty_growth,
            recovery_state=self.state.recovery_state,
            frame_idx=self._frame_count,
        )

    def _update_recovery_state(self):
        """Update recovery state based on occlusion and uncertainty.

        Thresholds calibrated after process noise reduction (Q[4]: 100→1)
        and observation model expansion (5→9 dimensions).
        Initial trace ≈ 11.0, steady-state trace ≈ 15-25 with new parameters.
        """
        trace = float(np.trace(self.state.covariance))

        if self._occlusion_count > 20:
            self.state.recovery_state = "reset_required"
        elif self._occlusion_count > 10:
            self.state.recovery_state = "degraded"
        elif self._occlusion_count > 5:
            self.state.recovery_state = "uncertain"
        elif trace > 50.0:
            self.state.recovery_state = "recovering"
        else:
            self.state.recovery_state = "stable"
