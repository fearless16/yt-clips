"""Layer 2d: Renderer mode state machine tests.

Validates the ALPHA → HYBRID → PHYSICAL mode transition logic (D-02).

State machine invariants:
  - Initial mode is ALPHA
  - Transition requires sustained high confidence
  - Blend weight tracks mode (0 for alpha, 1 for physical)
  - Transition count increments on each mode change
"""
import pytest

from face_os.renderer_mode import RendererMode, RendererModeState


@pytest.fixture
def state():
    """Fresh renderer mode state."""
    return RendererModeState()


# ═══════════════════════════════════════════════════════════════════
# Initial State
# ═══════════════════════════════════════════════════════════════════

class TestInitialState:
    """Fresh state machine must start in ALPHA mode."""

    def test_initial_mode_is_alpha(self, state):
        """Default mode is ALPHA."""
        assert state.current_mode == RendererMode.ALPHA_FALLBACK

    def test_initial_transition_count_zero(self, state):
        """No transitions yet."""
        assert state.transition_count == 0

    def test_initial_blend_weight(self, state):
        """Alpha mode → blend_weight near 0."""
        w = state.get_blend_weight()
        assert w <= 0.1, f"Initial blend_weight={w}"


# ═══════════════════════════════════════════════════════════════════
# Mode Transitions
# ═══════════════════════════════════════════════════════════════════

class TestModeTransitions:
    """State machine transitions under various confidence streams."""

    def test_stays_alpha_with_no_intrinsic(self, state):
        """Without intrinsic components, stays ALPHA."""
        for _ in range(20):
            state.update(
                intrinsic_available=False,
                intrinsic_confidence=0.0,
                decomposition_error=1.0,
            )
        assert state.current_mode == RendererMode.ALPHA_FALLBACK

    def test_stays_alpha_with_low_confidence(self, state):
        """Low confidence intrinsic → stays ALPHA."""
        for _ in range(20):
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.1,  # Very low
                decomposition_error=0.5,
            )
        assert state.current_mode == RendererMode.ALPHA_FALLBACK

    def test_transition_to_physical_with_good_input(self, state):
        """Sustained good input → eventually reaches PHYSICAL or HYBRID."""
        for _ in range(30):
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.95,
                decomposition_error=0.05,
            )
        assert state.current_mode in (RendererMode.PHYSICAL, RendererMode.HYBRID), (
            f"Mode={state.current_mode} after 30 good frames"
        )

    def test_transition_count_increments(self, state):
        """Mode changes should increment transition_count."""
        initial = state.transition_count
        # Feed good data to trigger transition
        for _ in range(30):
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.95,
                decomposition_error=0.05,
            )
        if state.current_mode != RendererMode.ALPHA_FALLBACK:
            assert state.transition_count > initial


# ═══════════════════════════════════════════════════════════════════
# Blend Weight
# ═══════════════════════════════════════════════════════════════════

class TestBlendWeight:
    """Blend weight must track the current mode."""

    def test_alpha_mode_low_weight(self):
        """ALPHA mode → blend_weight ≤ 0.1."""
        s = RendererModeState()
        w = s.get_blend_weight()
        assert w <= 0.1, f"Alpha blend_weight={w}"

    def test_physical_mode_high_weight(self):
        """PHYSICAL mode → blend_weight ≥ 0.8."""
        s = RendererModeState()
        for _ in range(50):
            s.update(
                intrinsic_available=True,
                intrinsic_confidence=0.99,
                decomposition_error=0.02,
            )
        if s.current_mode == RendererMode.PHYSICAL:
            w = s.get_blend_weight()
            assert w >= 0.8, f"Physical blend_weight={w}"

    def test_blend_weight_in_range(self):
        """Blend weight always in [0, 1]."""
        s = RendererModeState()
        for _ in range(20):
            s.update(
                intrinsic_available=True,
                intrinsic_confidence=np.random.uniform(0, 1),
                decomposition_error=np.random.uniform(0, 1),
            )
            w = s.get_blend_weight()
            assert 0.0 <= w <= 1.0, f"blend_weight={w} out of [0,1]"


# ═══════════════════════════════════════════════════════════════════
# Stability
# ═══════════════════════════════════════════════════════════════════

class TestStability:
    """State machine should stabilize after convergence."""

    def test_is_stable_after_convergence(self):
        """is_stable() true after many consistent frames."""
        s = RendererModeState()
        for _ in range(50):
            s.update(
                intrinsic_available=True,
                intrinsic_confidence=0.95,
                decomposition_error=0.05,
            )
        # After 50 consistent frames, should be stable
        assert s.is_stable(), "Not stable after 50 consistent frames"

    def test_transition_rate_decreases(self):
        """Transition rate should decrease as state stabilizes."""
        s = RendererModeState()
        rates = []
        for i in range(50):
            s.update(
                intrinsic_available=True,
                intrinsic_confidence=0.95,
                decomposition_error=0.05,
            )
            if i % 10 == 9:
                rates.append(s.get_transition_rate())
        # Rate should not increase over time (may stay zero)
        if len(rates) >= 2 and rates[0] > 0:
            assert rates[-1] <= rates[0] + 0.01


# Need numpy for the blend weight randomized test
import numpy as np
