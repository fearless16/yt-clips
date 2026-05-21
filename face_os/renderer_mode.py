"""Renderer Mode Module.

Manages renderer mode state to prevent bifurcation issues.

Renderer Modes:
    PHYSICAL: Using PhysicalRenderer with intrinsic components
    ALPHA_FALLBACK: Using alpha compositing (legacy)
    HYBRID: Blending between physical and alpha based on confidence

The mode is determined by:
    - Availability of intrinsic components
    - Quality of intrinsic decomposition
    - Confidence thresholds
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class RendererMode(Enum):
    """Renderer mode state machine."""
    PHYSICAL = "physical"          # Full PhysicalRenderer
    ALPHA_FALLBACK = "alpha"       # Legacy alpha compositing
    HYBRID = "hybrid"              # Blended mode


@dataclass
class RendererModeState:
    """State tracking for renderer mode transitions."""

    current_mode: RendererMode = RendererMode.ALPHA_FALLBACK
    previous_mode: Optional[RendererMode] = None
    mode_confidence: float = 0.0
    frames_in_mode: int = 0
    transition_count: int = 0

    # Thresholds for mode transitions
    PHYSICAL_CONFIDENCE_THRESHOLD: float = 0.6
    HYBRID_CONFIDENCE_THRESHOLD: float = 0.3
    MIN_FRAMES_BEFORE_TRANSITION: int = 5  # Original value — prevents thrashing

    def update(
        self,
        intrinsic_available: bool,
        intrinsic_confidence: float,
        decomposition_error: float,
    ) -> RendererMode:
        """Update renderer mode based on intrinsic quality.

        Args:
            intrinsic_available: Whether intrinsic components exist
            intrinsic_confidence: Confidence of intrinsic decomposition [0, 1]
            decomposition_error: Reconstruction error of decomposition

        Returns:
            Current renderer mode
        """
        self.frames_in_mode += 1

        # Compute effective confidence
        effective_confidence = intrinsic_confidence * (1.0 - min(decomposition_error, 1.0))

        # Determine target mode
        if not intrinsic_available:
            target_mode = RendererMode.ALPHA_FALLBACK
        elif effective_confidence >= self.PHYSICAL_CONFIDENCE_THRESHOLD:
            target_mode = RendererMode.PHYSICAL
        elif effective_confidence >= self.HYBRID_CONFIDENCE_THRESHOLD:
            target_mode = RendererMode.HYBRID
        else:
            target_mode = RendererMode.ALPHA_FALLBACK

        # Apply hysteresis — only transition if in current mode long enough
        if target_mode != self.current_mode:
            if self.frames_in_mode >= self.MIN_FRAMES_BEFORE_TRANSITION:
                self.previous_mode = self.current_mode
                self.current_mode = target_mode
                self.frames_in_mode = 0
                self.transition_count += 1

        self.mode_confidence = effective_confidence
        return self.current_mode

    def get_blend_weight(self) -> float:
        """Get blend weight for hybrid mode.

        Returns:
            Weight for physical renderer [0, 1]
            0 = full alpha, 1 = full physical
        """
        if self.current_mode == RendererMode.PHYSICAL:
            return 1.0
        elif self.current_mode == RendererMode.ALPHA_FALLBACK:
            return 0.0
        else:  # HYBRID
            # Smooth blend based on confidence
            return float(np.clip(
                (self.mode_confidence - self.HYBRID_CONFIDENCE_THRESHOLD)
                / (self.PHYSICAL_CONFIDENCE_THRESHOLD - self.HYBRID_CONFIDENCE_THRESHOLD),
                0.0,
                1.0
            ))

    def is_stable(self) -> bool:
        """Check if renderer mode is stable (no recent transitions)."""
        return self.frames_in_mode >= self.MIN_FRAMES_BEFORE_TRANSITION

    def get_transition_rate(self) -> float:
        """Get mode transition rate (transitions per frame)."""
        total_frames = self.transition_count * self.MIN_FRAMES_BEFORE_TRANSITION + self.frames_in_mode
        if total_frames == 0:
            return 0.0
        return self.transition_count / total_frames


@dataclass
class RendererModeReport:
    """Report for renderer mode metrics."""

    mode: RendererMode
    confidence: float
    blend_weight: float
    frames_in_mode: int
    transition_count: int
    transition_rate: float
    is_stable: bool

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "confidence": self.confidence,
            "blend_weight": self.blend_weight,
            "frames_in_mode": self.frames_in_mode,
            "transition_count": self.transition_count,
            "transition_rate": self.transition_rate,
            "is_stable": self.is_stable,
        }
