"""
test_phase2g_recovery_dynamics.py — Tests for Phase 2G Probabilistic Recovery Dynamics.

Tests:
- RecoveryTransitionMatrix: transition probabilities, row sums, steady state
- ProbabilisticRecoveryState: probabilities, entropy, dominant state
- RecoveryDynamics: predict, update, step, entropy tracking
"""

import numpy as np
import pytest

from face_os.recovery_dynamics import (
    RecoveryDynamics,
    RecoveryReport,
    RecoveryState,
    RecoveryTransitionMatrix,
    ProbabilisticRecoveryState,
)


# ─── Test Class 1: RecoveryTransitionMatrix ──────────────────────────────────

class TestRecoveryTransitionMatrix:
    """Test transition matrix properties."""

    def test_matrix_exists(self):
        """RecoveryTransitionMatrix must exist."""
        matrix = RecoveryTransitionMatrix()
        assert matrix is not None

    def test_matrix_shape(self):
        """Transition matrix must be 5x5."""
        matrix = RecoveryTransitionMatrix()
        assert matrix.matrix.shape == (5, 5)

    def test_rows_sum_to_one(self):
        """Each row must sum to 1.0."""
        matrix = RecoveryTransitionMatrix()
        row_sums = np.sum(matrix.matrix, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_non_negative(self):
        """All probabilities must be non-negative."""
        matrix = RecoveryTransitionMatrix()
        assert np.all(matrix.matrix >= 0)

    def test_stable_self_transition_high(self):
        """STABLE → STABLE should have high probability."""
        matrix = RecoveryTransitionMatrix()
        assert matrix.matrix[0, 0] >= 0.9

    def test_stable_to_uncertain_low(self):
        """STABLE → UNCERTAIN should have low probability."""
        matrix = RecoveryTransitionMatrix()
        assert matrix.matrix[0, 1] <= 0.1

    def test_step_preserves_sum(self):
        """Step must preserve probability sum."""
        matrix = RecoveryTransitionMatrix()
        state = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        next_state = matrix.step(state)
        np.testing.assert_allclose(np.sum(next_state), 1.0, atol=1e-10)

    def test_step_converges_to_steady_state(self):
        """Repeated steps must converge to steady state."""
        matrix = RecoveryTransitionMatrix()
        state = np.array([0.0, 0.0, 0.0, 0.0, 1.0])  # Start at RESET_REQUIRED
        for _ in range(100):
            state = matrix.step(state)
        steady = matrix.steady_state()
        np.testing.assert_allclose(state, steady, atol=1e-6)

    def test_steady_state_sums_to_one(self):
        """Steady state must sum to 1.0."""
        matrix = RecoveryTransitionMatrix()
        steady = matrix.steady_state()
        np.testing.assert_allclose(np.sum(steady), 1.0, atol=1e-10)

    def test_steady_state_non_negative(self):
        """Steady state must be non-negative."""
        matrix = RecoveryTransitionMatrix()
        steady = matrix.steady_state()
        assert np.all(steady >= -1e-10)


# ─── Test Class 2: ProbabilisticRecoveryState ────────────────────────────────

class TestProbabilisticRecoveryState:
    """Test probabilistic state properties."""

    def test_state_exists(self):
        """ProbabilisticRecoveryState must exist."""
        state = ProbabilisticRecoveryState()
        assert state is not None

    def test_initial_probabilities(self):
        """Initial state must be STABLE with probability 1.0."""
        state = ProbabilisticRecoveryState()
        np.testing.assert_allclose(state.probabilities[0], 1.0, atol=1e-10)

    def test_probabilities_sum_to_one(self):
        """Probabilities must sum to 1.0."""
        state = ProbabilisticRecoveryState()
        np.testing.assert_allclose(np.sum(state.probabilities), 1.0, atol=1e-10)

    def test_entropy_initial_zero(self):
        """Initial entropy must be zero (certain state)."""
        state = ProbabilisticRecoveryState()
        assert state.entropy < 1e-6

    def test_entropy_max_uniform(self):
        """Uniform distribution must have maximum entropy."""
        state = ProbabilisticRecoveryState(
            probabilities=np.ones(5) / 5.0
        )
        assert state.entropy > 1.0

    def test_dominant_state_initial(self):
        """Initial dominant state must be STABLE."""
        state = ProbabilisticRecoveryState()
        assert state.dominant_state == RecoveryState.STABLE

    def test_dominant_state_changes(self):
        """Dominant state must change with probabilities."""
        state = ProbabilisticRecoveryState(
            probabilities=np.array([0.1, 0.7, 0.1, 0.05, 0.05])
        )
        assert state.dominant_state == RecoveryState.UNCERTAIN

    def test_confidence_initial(self):
        """Initial confidence must be 1.0."""
        state = ProbabilisticRecoveryState()
        assert state.confidence == 1.0

    def test_confidence_decreases_with_uncertainty(self):
        """Confidence must decrease with uncertainty."""
        state = ProbabilisticRecoveryState(
            probabilities=np.ones(5) / 5.0
        )
        assert state.confidence < 0.5

    def test_to_dict(self):
        """to_dict must return all required fields."""
        state = ProbabilisticRecoveryState()
        d = state.to_dict()
        assert "probabilities" in d
        assert "entropy" in d
        assert "dominant_state" in d
        assert "confidence" in d

    def test_to_dict_probabilities(self):
        """to_dict must include all state probabilities."""
        state = ProbabilisticRecoveryState()
        d = state.to_dict()
        for state_name in ["STABLE", "UNCERTAIN", "DEGRADED", "RECOVERING", "RESET_REQUIRED"]:
            assert state_name in d["probabilities"]


# ─── Test Class 3: RecoveryDynamics ──────────────────────────────────────────

class TestRecoveryDynamics:
    """Test recovery dynamics engine."""

    def test_dynamics_exists(self):
        """RecoveryDynamics must exist."""
        dynamics = RecoveryDynamics()
        assert dynamics is not None

    def test_predict(self):
        """Predict must return updated state."""
        dynamics = RecoveryDynamics()
        state = dynamics.predict()
        assert state is not None
        assert state.probabilities.shape == (5,)

    def test_predict_preserves_sum(self):
        """Predict must preserve probability sum."""
        dynamics = RecoveryDynamics()
        state = dynamics.predict()
        np.testing.assert_allclose(np.sum(state.probabilities), 1.0, atol=1e-10)

    def test_update_with_low_trace(self):
        """Low trace must favor STABLE state."""
        dynamics = RecoveryDynamics()
        state = dynamics.update(trace=5.0, occlusion_count=0)
        assert state.probabilities[0] > 0.5  # STABLE

    def test_update_with_high_trace(self):
        """High trace must shift away from STABLE."""
        dynamics = RecoveryDynamics(observation_weight=1.0)
        state = dynamics.update(trace=100.0, occlusion_count=0)
        assert state.probabilities[0] < 0.5  # Not STABLE

    def test_update_with_occlusion(self):
        """High occlusion count must shift to DEGRADED/RESET."""
        dynamics = RecoveryDynamics(observation_weight=1.0)
        state = dynamics.update(trace=5.0, occlusion_count=25)
        # Should shift toward DEGRADED or RESET_REQUIRED
        assert state.probabilities[2] + state.probabilities[4] > 0.3

    def test_step(self):
        """Full step must return RecoveryReport."""
        dynamics = RecoveryDynamics()
        report = dynamics.step(trace=10.0, occlusion_count=0)
        assert isinstance(report, RecoveryReport)

    def test_step_has_state(self):
        """Report must have state."""
        dynamics = RecoveryDynamics()
        report = dynamics.step(trace=10.0, occlusion_count=0)
        assert report.state is not None

    def test_step_has_frame_idx(self):
        """Report must have frame index."""
        dynamics = RecoveryDynamics()
        report = dynamics.step(trace=10.0, occlusion_count=0)
        assert report.frame_idx == 1

    def test_entropy_increases_with_uncertainty(self):
        """Entropy must increase with uncertain observations."""
        dynamics = RecoveryDynamics()
        dynamics.step(trace=10.0, occlusion_count=0)  # Stable
        entropy_stable = dynamics.state.entropy

        dynamics2 = RecoveryDynamics()
        dynamics2.step(trace=100.0, occlusion_count=15)  # Uncertain
        entropy_uncertain = dynamics2.state.entropy

        assert entropy_uncertain > entropy_stable

    def test_get_state_vector(self):
        """get_state_vector must return 5-element vector."""
        dynamics = RecoveryDynamics()
        vec = dynamics.get_state_vector()
        assert vec.shape == (5,)

    def test_set_state_vector(self):
        """set_state_vector must update state."""
        dynamics = RecoveryDynamics()
        new_state = np.array([0.1, 0.7, 0.1, 0.05, 0.05])
        dynamics.set_state_vector(new_state)
        np.testing.assert_allclose(dynamics.state.probabilities, new_state, atol=1e-10)

    def test_recovery_report_to_dict(self):
        """RecoveryReport.to_dict must return all fields."""
        dynamics = RecoveryDynamics()
        report = dynamics.step(trace=10.0, occlusion_count=0)
        d = report.to_dict()
        assert "frame_idx" in d
        assert "state" in d
        assert "transition_applied" in d
        assert "observation_update_applied" in d


# ─── Test Class 4: Transition Dynamics ───────────────────────────────────────

class TestTransitionDynamics:
    """Test transition dynamics over multiple frames."""

    def test_stable_persists(self):
        """STABLE state must persist under low observations."""
        dynamics = RecoveryDynamics()
        for _ in range(50):
            dynamics.step(trace=5.0, occlusion_count=0)
        assert dynamics.state.dominant_state == RecoveryState.STABLE

    def test_recovery_after_degraded(self):
        """System must recover after degraded state."""
        dynamics = RecoveryDynamics()
        # Push to DEGRADED
        for _ in range(20):
            dynamics.step(trace=80.0, occlusion_count=5)
        # Now provide good observations
        for _ in range(50):
            dynamics.step(trace=5.0, occlusion_count=0)
        # Should recover to STABLE or UNCERTAIN
        assert dynamics.state.dominant_state in [RecoveryState.STABLE, RecoveryState.UNCERTAIN]

    def test_entropy_bounded(self):
        """Entropy must be bounded."""
        dynamics = RecoveryDynamics()
        for _ in range(100):
            dynamics.step(trace=50.0, occlusion_count=10)
        assert dynamics.state.entropy < 2.0  # ln(5) ≈ 1.61

    def test_probabilities_always_valid(self):
        """Probabilities must always be valid."""
        dynamics = RecoveryDynamics()
        for _ in range(100):
            dynamics.step(trace=50.0, occlusion_count=10)
            assert np.all(dynamics.state.probabilities >= 0)
            np.testing.assert_allclose(np.sum(dynamics.state.probabilities), 1.0, atol=1e-10)
