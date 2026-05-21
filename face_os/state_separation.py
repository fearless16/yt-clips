"""
state_separation.py — State Separation for Face OS.

Phase 2D: State Separation.

Explicit separation of:
- Physical state (geometry, pose, lighting, appearance, motion)
- Belief state (covariance, uncertainty, confidence, innovation)
- Meta-state (recovery, degradation, reset, health)

This prevents mixing of:
- Confidence scalars into physical dynamics
- Recovery flags into latent geometry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from face_os.state_space import LatentState


# ─── Physical State ──────────────────────────────────────────────────────────

@dataclass
class PhysicalState:
    """Physical state: geometry, pose, lighting, appearance, motion.

    Contains ONLY physical quantities — no uncertainty or recovery.
    """
    # Geometry/pose
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    # Lighting
    brightness_mean: float = 128.0
    contrast_mean: float = 50.0
    # Appearance
    appearance_latent_norm: float = 0.0

    PHYSICAL_DIM = 6

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.yaw, self.pitch, self.roll,
            self.brightness_mean, self.contrast_mean,
            self.appearance_latent_norm,
        ], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> PhysicalState:
        return cls(
            yaw=float(vec[0]),
            pitch=float(vec[1]),
            roll=float(vec[2]),
            brightness_mean=float(np.clip(vec[3], 0, 255)),
            contrast_mean=float(max(0, vec[4])),
            appearance_latent_norm=float(max(0, vec[5])),
        )


# ─── Belief State ────────────────────────────────────────────────────────────

@dataclass
class BeliefState:
    """Belief state: covariance, uncertainty, confidence, innovation.

    Contains ONLY belief quantities — no physical or recovery.
    """
    # Covariance (6x6 for physical state)
    covariance: np.ndarray = field(
        default_factory=lambda: np.eye(PhysicalState.PHYSICAL_DIM) * 1.0
    )
    # Uncertainty
    uncertainty: float = 0.5
    # Confidence
    temporal_confidence: float = 1.0
    identity_uncertainty: float = 0.5
    # Innovation statistics
    innovation_norm: float = 0.0
    drift_score: float = 0.0
    continuity_score: float = 1.0

    BELIEF_DIM = 6  # uncertainty, temporal_conf, id_unc, innovation, drift, continuity

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.uncertainty,
            self.temporal_confidence,
            self.identity_uncertainty,
            self.innovation_norm,
            self.drift_score,
            self.continuity_score,
        ], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> BeliefState:
        return cls(
            uncertainty=float(np.clip(vec[0], 0, 1)),
            temporal_confidence=float(np.clip(vec[1], 0, 1)),
            identity_uncertainty=float(np.clip(vec[2], 0, 1)),
            innovation_norm=float(max(0, vec[3])),
            drift_score=float(max(0, vec[4])),
            continuity_score=float(np.clip(vec[5], 0, 1)),
        )


# ─── Meta State ──────────────────────────────────────────────────────────────

@dataclass
class MetaState:
    """Meta-state: recovery, degradation, reset, health.

    Contains ONLY meta quantities — no physical or belief.
    """
    # Recovery
    recovery_state: str = "stable"
    recovery_probability: float = 0.0
    # Degradation
    degradation_score: float = 0.0
    # Reset
    reset_probability: float = 0.0
    # Health
    system_health: float = 1.0

    META_DIM = 4  # recovery_prob, degradation, reset_prob, health

    def to_vector(self) -> np.ndarray:
        recovery_float = {
            "stable": 0.0,
            "uncertain": 0.25,
            "degraded": 0.5,
            "recovering": 0.75,
            "reset_required": 1.0,
        }.get(self.recovery_state, 0.0)

        return np.array([
            recovery_float,
            self.degradation_score,
            self.reset_probability,
            self.system_health,
        ], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> MetaState:
        recovery_map = {
            0.0: "stable",
            0.25: "uncertain",
            0.5: "degraded",
            0.75: "recovering",
            1.0: "reset_required",
        }
        recovery_float = float(np.clip(vec[0], 0, 1))
        closest = min(recovery_map.keys(), key=lambda x: abs(x - recovery_float))

        return cls(
            recovery_state=recovery_map[closest],
            recovery_probability=float(np.clip(vec[1], 0, 1)),
            degradation_score=float(np.clip(vec[2], 0, 1)),
            reset_probability=float(np.clip(vec[3], 0, 1)),
            system_health=float(np.clip(vec[3], 0, 1)),
        )


# ─── Separated State ─────────────────────────────────────────────────────────

@dataclass
class SeparatedState:
    """Separated state: physical + belief + meta."""
    physical: PhysicalState = field(default_factory=PhysicalState)
    belief: BeliefState = field(default_factory=BeliefState)
    meta: MetaState = field(default_factory=MetaState)

    def to_latent(self) -> LatentState:
        """Convert separated state to LatentState."""
        return LatentState(
            yaw=self.physical.yaw,
            pitch=self.physical.pitch,
            roll=self.physical.roll,
            identity_uncertainty=self.belief.identity_uncertainty,
            appearance_latent_norm=self.physical.appearance_latent_norm,
            temporal_confidence=self.belief.temporal_confidence,
            drift_score=self.belief.drift_score,
            continuity_score=self.belief.continuity_score,
            recovery_state=self.meta.recovery_state,
            brightness_mean=self.physical.brightness_mean,
            contrast_mean=self.physical.contrast_mean,
        )

    @classmethod
    def from_latent(cls, latent: LatentState) -> SeparatedState:
        """Construct separated state from LatentState."""
        physical = PhysicalState(
            yaw=latent.yaw,
            pitch=latent.pitch,
            roll=latent.roll,
            brightness_mean=latent.brightness_mean,
            contrast_mean=latent.contrast_mean,
            appearance_latent_norm=latent.appearance_latent_norm,
        )
        belief = BeliefState(
            uncertainty=latent.identity_uncertainty,
            temporal_confidence=latent.temporal_confidence,
            identity_uncertainty=latent.identity_uncertainty,
            drift_score=latent.drift_score,
            continuity_score=latent.continuity_score,
        )
        meta = MetaState(
            recovery_state=latent.recovery_state,
        )
        return cls(physical=physical, belief=belief, meta=meta)


# ─── State Separator ─────────────────────────────────────────────────────────

class StateSeparator:
    """Separates and merges LatentState and SeparatedState."""

    def separate(self, latent: LatentState) -> SeparatedState:
        """Separate LatentState into physical, belief, meta.

        Args:
            latent: Combined latent state

        Returns:
            Separated state
        """
        return SeparatedState.from_latent(latent)

    def merge(self, separated: SeparatedState) -> LatentState:
        """Merge separated state into LatentState.

        Args:
            separated: Separated state

        Returns:
            Combined latent state
        """
        return separated.to_latent()
