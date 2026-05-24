"""Renderer Mode Module.

BEAST MODE FIXES:
- Fixed the Hysteresis Trap: Hard failures (intrinsic unavailable) now bypass hysteresis.
- Nuked the fake math in get_transition_rate. Added a real total_frames counter.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class RendererMode(Enum):
    """Renderer mode state machine."""
    PHYSICAL = "physical"
    ALPHA_FALLBACK = "alpha"
    HYBRID = "hybrid"


@dataclass
class RendererModeState:
    """State tracking for renderer mode transitions."""

    current_mode: RendererMode = RendererMode.ALPHA_FALLBACK
    previous_mode: Optional[RendererMode] = None
    mode_confidence: float = 0.0
    frames_in_mode: int = 0
    transition_count: int = 0
    total_frames: int = 0  # BEAST MODE: Real frame counter

    PHYSICAL_CONFIDENCE_THRESHOLD: float = 0.45
    HYBRID_CONFIDENCE_THRESHOLD: float = 0.20
    MIN_FRAMES_BEFORE_TRANSITION: int = 3

    def update(
        self,
        intrinsic_available: bool,
        intrinsic_confidence: float,
        decomposition_error: float,
    ) -> RendererMode:
        self.total_frames += 1
        self.frames_in_mode += 1

        effective_confidence = intrinsic_confidence * (1.0 - min(decomposition_error, 1.0))

        if not intrinsic_available:
            target_mode = RendererMode.ALPHA_FALLBACK
        elif effective_confidence >= self.PHYSICAL_CONFIDENCE_THRESHOLD:
            target_mode = RendererMode.PHYSICAL
        elif effective_confidence >= self.HYBRID_CONFIDENCE_THRESHOLD:
            target_mode = RendererMode.HYBRID
        else:
            target_mode = RendererMode.ALPHA_FALLBACK

        # BEAST MODE FIX: Hard failures bypass hysteresis.
        # If intrinsic is gone, we MUST drop to ALPHA immediately, no matter the frame count.
        # Hysteresis only applies to confidence flicker between PHYSICAL and HYBRID.
        is_hard_failure = (not intrinsic_available) and (self.current_mode != RendererMode.ALPHA_FALLBACK)
        
        if target_mode != self.current_mode:
            if is_hard_failure or self.frames_in_mode >= self.MIN_FRAMES_BEFORE_TRANSITION:
                self.previous_mode = self.current_mode
                self.current_mode = target_mode
                self.frames_in_mode = 0
                self.transition_count += 1

        self.mode_confidence = effective_confidence
        return self.current_mode

    def get_blend_weight(self) -> float:
        if self.current_mode == RendererMode.PHYSICAL:
            return 1.0
        elif self.current_mode == RendererMode.ALPHA_FALLBACK:
            return 0.0
        else:
            return float(np.clip(
                (self.mode_confidence - self.HYBRID_CONFIDENCE_THRESHOLD)
                / (self.PHYSICAL_CONFIDENCE_THRESHOLD - self.HYBRID_CONFIDENCE_THRESHOLD),
                0.0,
                1.0
            ))

    def is_stable(self) -> bool:
        return self.frames_in_mode >= self.MIN_FRAMES_BEFORE_TRANSITION

    def get_transition_rate(self) -> float:
        # BEAST MODE FIX: Use the real total_frames counter instead of fake math.
        if self.total_frames == 0:
            return 0.0
        return self.transition_count / self.total_frames


@dataclass
class RendererModeReport:
    mode: RendererMode
    confidence: float
    blend_weight: float
    frames_in_mode: int
    transition_count: int
    transition_rate: float
    is_stable: bool

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "confidence": self.confidence,
            "blend_weight": self.blend_weight,
            "frames_in_mode": self.frames_in_mode,
            "transition_count": self.transition_count,
            "transition_rate": self.transition_rate,
            "is_stable": self.is_stable,
        }