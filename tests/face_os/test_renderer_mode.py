"""Tests for Renderer Mode Module.

Tests for:
- RendererMode enum
- RendererModeState transitions
- Blend weight computation
- Mode stability
- Transition rate

Target: 20 tests
"""

import numpy as np
import pytest

from face_os.renderer_mode import (
    RendererMode,
    RendererModeState,
    RendererModeReport,
)


class TestRendererMode:
    """Test RendererMode enum."""

    def test_mode_values(self):
        """Mode values must be strings."""
        assert RendererMode.PHYSICAL.value == "physical"
        assert RendererMode.ALPHA_FALLBACK.value == "alpha"
        assert RendererMode.HYBRID.value == "hybrid"

    def test_mode_count(self):
        """Must have exactly 3 modes."""
        assert len(RendererMode) == 3


class TestRendererModeState:
    """Test RendererModeState."""

    def test_initial_mode(self):
        """Initial mode must be ALPHA_FALLBACK."""
        state = RendererModeState()
        assert state.current_mode == RendererMode.ALPHA_FALLBACK

    def test_transition_to_physical(self):
        """High confidence must transition to PHYSICAL."""
        state = RendererModeState()
        
        # Update with high confidence for enough frames
        for _ in range(10):
            mode = state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.9,
                decomposition_error=0.1,
            )
        
        assert mode == RendererMode.PHYSICAL

    def test_transition_to_hybrid(self):
        """Medium confidence must transition to HYBRID."""
        state = RendererModeState()
        
        # Update with medium confidence for enough frames
        for _ in range(10):
            mode = state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.5,
                decomposition_error=0.1,
            )
        
        assert mode == RendererMode.HYBRID

    def test_transition_to_alpha_fallback(self):
        """Low confidence must transition to ALPHA_FALLBACK."""
        state = RendererModeState()
        
        # Start with physical
        for _ in range(10):
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.9,
                decomposition_error=0.1,
            )
        
        # Transition to low confidence
        for _ in range(10):
            mode = state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.1,
                decomposition_error=0.5,
            )
        
        assert mode == RendererMode.ALPHA_FALLBACK

    def test_no_intrinsic_goes_to_alpha(self):
        """No intrinsic must go to ALPHA_FALLBACK."""
        state = RendererModeState()
        
        mode = state.update(
            intrinsic_available=False,
            intrinsic_confidence=0.0,
            decomposition_error=1.0,
        )
        
        assert mode == RendererMode.ALPHA_FALLBACK

    def test_hysteresis_prevents_rapid_transitions(self):
        """Must stay in mode for minimum frames before transitioning."""
        state = RendererModeState()
        
        # Oscillate confidence
        for i in range(20):
            confidence = 0.9 if i % 2 == 0 else 0.1
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=confidence,
                decomposition_error=0.1,
            )
        
        # Should not have transitioned many times
        assert state.transition_count <= 4

    def test_blend_weight_physical(self):
        """PHYSICAL mode must have blend weight 1.0."""
        state = RendererModeState()
        state.current_mode = RendererMode.PHYSICAL
        assert state.get_blend_weight() == 1.0

    def test_blend_weight_alpha(self):
        """ALPHA_FALLBACK mode must have blend weight 0.0."""
        state = RendererModeState()
        state.current_mode = RendererMode.ALPHA_FALLBACK
        assert state.get_blend_weight() == 0.0

    def test_blend_weight_hybrid(self):
        """HYBRID mode must have blend weight between 0 and 1."""
        state = RendererModeState()
        state.current_mode = RendererMode.HYBRID
        state.mode_confidence = 0.5
        
        weight = state.get_blend_weight()
        assert 0.0 <= weight <= 1.0

    def test_stability_after_frames(self):
        """Must be stable after minimum frames."""
        state = RendererModeState()
        
        # Stay in same mode for enough frames
        for _ in range(10):
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.9,
                decomposition_error=0.1,
            )
        
        assert state.is_stable()

    def test_transition_rate(self):
        """Transition rate must be bounded."""
        state = RendererModeState()
        
        # Do some transitions
        for i in range(100):
            confidence = 0.9 if i < 50 else 0.1
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=confidence,
                decomposition_error=0.1,
            )
        
        rate = state.get_transition_rate()
        assert 0.0 <= rate <= 1.0

    def test_frames_in_mode_increments(self):
        """Frames in mode must increment."""
        state = RendererModeState()
        
        state.update(
            intrinsic_available=True,
            intrinsic_confidence=0.9,
            decomposition_error=0.1,
        )
        
        assert state.frames_in_mode >= 1

    def test_transition_count_increments(self):
        """Transition count must increment on mode change."""
        state = RendererModeState()
        
        # Start with low confidence
        for _ in range(10):
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.1,
                decomposition_error=0.5,
            )
        
        initial_count = state.transition_count
        
        # Switch to high confidence
        for _ in range(10):
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.9,
                decomposition_error=0.1,
            )
        
        assert state.transition_count > initial_count

    def test_effective_confidence(self):
        """Effective confidence must account for decomposition error."""
        state = RendererModeState()
        
        state.update(
            intrinsic_available=True,
            intrinsic_confidence=0.8,
            decomposition_error=0.3,
        )
        
        # Effective = confidence * (1 - error) = 0.8 * 0.7 = 0.56
        assert abs(state.mode_confidence - 0.56) < 0.01


class TestRendererModeReport:
    """Test RendererModeReport."""

    def test_report_to_dict(self):
        """Report must convert to dict."""
        state = RendererModeState()
        state.update(
            intrinsic_available=True,
            intrinsic_confidence=0.9,
            decomposition_error=0.1,
        )
        
        report = RendererModeReport(
            mode=state.current_mode,
            confidence=state.mode_confidence,
            blend_weight=state.get_blend_weight(),
            frames_in_mode=state.frames_in_mode,
            transition_count=state.transition_count,
            transition_rate=state.get_transition_rate(),
            is_stable=state.is_stable(),
        )
        
        d = report.to_dict()
        assert d["mode"] in ["physical", "alpha", "hybrid"]
        assert 0.0 <= d["confidence"] <= 1.0
        assert 0.0 <= d["blend_weight"] <= 1.0

    def test_report_has_all_fields(self):
        """Report must have all required fields."""
        report = RendererModeReport(
            mode=RendererMode.PHYSICAL,
            confidence=0.9,
            blend_weight=1.0,
            frames_in_mode=10,
            transition_count=1,
            transition_rate=0.1,
            is_stable=True,
        )
        
        assert hasattr(report, "mode")
        assert hasattr(report, "confidence")
        assert hasattr(report, "blend_weight")
        assert hasattr(report, "frames_in_mode")
        assert hasattr(report, "transition_count")
        assert hasattr(report, "transition_rate")
        assert hasattr(report, "is_stable")


class TestRendererModeTransitions:
    """Test renderer mode transition stability."""

    def test_no_mode_thrashing(self):
        """Must not thrash between modes."""
        state = RendererModeState()
        
        # Oscillate confidence rapidly
        for i in range(100):
            if i % 3 == 0:
                confidence = 0.9
            elif i % 3 == 1:
                confidence = 0.5
            else:
                confidence = 0.1
            
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=confidence,
                decomposition_error=0.1,
            )
        
        # Should have limited transitions due to hysteresis
        # With MIN_FRAMES_BEFORE_TRANSITION=5, max ~20 transitions in 100 frames
        assert state.transition_count <= 25

    def test_smooth_transition_to_physical(self):
        """Transition to PHYSICAL must be smooth."""
        state = RendererModeState()
        
        # Gradually increase confidence
        for i in range(20):
            confidence = 0.3 + 0.04 * i  # 0.3 to 1.06
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=min(confidence, 1.0),
                decomposition_error=0.1,
            )
        
        assert state.current_mode == RendererMode.PHYSICAL

    def test_smooth_transition_to_alpha(self):
        """Transition to ALPHA_FALLBACK must be smooth."""
        state = RendererModeState()
        
        # Start with physical
        for _ in range(10):
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.9,
                decomposition_error=0.1,
            )
        
        # Gradually decrease confidence
        for i in range(20):
            confidence = 0.9 - 0.05 * i  # 0.9 to -0.1
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=max(confidence, 0.0),
                decomposition_error=0.1,
            )
        
        assert state.current_mode == RendererMode.ALPHA_FALLBACK
