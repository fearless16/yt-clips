"""
test_phase2d_state_separation.py — Phase 2D: State Separation Tests.

Tests:
1. Covariance semantic consistency
2. State isolation
3. Belief propagation correctness
4. Meta-state transition consistency

TDD: These tests should FAIL before implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from face_os.state_space import LatentState
from face_os.state_separation import (
    PhysicalState,
    BeliefState,
    MetaState,
    SeparatedState,
    StateSeparator,
)


# ─── Test Class 1: PhysicalState ─────────────────────────────────────────────

class TestPhysicalState:
    """Test physical state (geometry, pose, lighting, appearance, motion)."""

    def test_physical_state_exists(self):
        """PhysicalState must exist."""
        state = PhysicalState()
        assert state is not None

    def test_physical_state_has_geometry(self):
        """PhysicalState must have geometry components."""
        state = PhysicalState()
        assert hasattr(state, "yaw")
        assert hasattr(state, "pitch")
        assert hasattr(state, "roll")

    def test_physical_state_has_lighting(self):
        """PhysicalState must have lighting components."""
        state = PhysicalState()
        assert hasattr(state, "brightness_mean")
        assert hasattr(state, "contrast_mean")

    def test_physical_state_has_appearance(self):
        """PhysicalState must have appearance components."""
        state = PhysicalState()
        assert hasattr(state, "appearance_latent_norm")

    def test_physical_state_no_belief(self):
        """PhysicalState must NOT contain belief components."""
        state = PhysicalState()
        assert not hasattr(state, "covariance")
        assert not hasattr(state, "uncertainty")

    def test_physical_state_no_meta(self):
        """PhysicalState must NOT contain meta-state components."""
        state = PhysicalState()
        assert not hasattr(state, "recovery_state")
        assert not hasattr(state, "degradation_score")

    def test_physical_state_to_vector(self):
        """PhysicalState must be convertible to vector."""
        state = PhysicalState()
        vec = state.to_vector()
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1


# ─── Test Class 2: BeliefState ───────────────────────────────────────────────

class TestBeliefState:
    """Test belief state (covariance, uncertainty, confidence, innovation)."""

    def test_belief_state_exists(self):
        """BeliefState must exist."""
        state = BeliefState()
        assert state is not None

    def test_belief_state_has_covariance(self):
        """BeliefState must have covariance."""
        state = BeliefState()
        assert hasattr(state, "covariance")
        assert state.covariance is not None

    def test_belief_state_has_uncertainty(self):
        """BeliefState must have uncertainty."""
        state = BeliefState()
        assert hasattr(state, "uncertainty")

    def test_belief_state_has_confidence(self):
        """BeliefState must have confidence."""
        state = BeliefState()
        assert hasattr(state, "temporal_confidence")
        assert hasattr(state, "identity_uncertainty")

    def test_belief_state_has_innovation(self):
        """BeliefState must have innovation statistics."""
        state = BeliefState()
        assert hasattr(state, "innovation_norm")
        assert hasattr(state, "drift_score")
        assert hasattr(state, "continuity_score")

    def test_belief_state_no_physical(self):
        """BeliefState must NOT contain physical components."""
        state = BeliefState()
        assert not hasattr(state, "yaw")
        assert not hasattr(state, "pitch")

    def test_belief_state_no_meta(self):
        """BeliefState must NOT contain meta-state components."""
        state = BeliefState()
        assert not hasattr(state, "recovery_state")

    def test_belief_state_covariance_positive_semidefinite(self):
        """BeliefState covariance must be positive semi-definite."""
        state = BeliefState()
        eigvals = np.linalg.eigvalsh(state.covariance)
        assert np.all(eigvals >= -1e-10)


# ─── Test Class 3: MetaState ─────────────────────────────────────────────────

class TestMetaState:
    """Test meta-state (recovery, degradation, reset, health)."""

    def test_meta_state_exists(self):
        """MetaState must exist."""
        state = MetaState()
        assert state is not None

    def test_meta_state_has_recovery(self):
        """MetaState must have recovery components."""
        state = MetaState()
        assert hasattr(state, "recovery_state")
        assert hasattr(state, "recovery_probability")

    def test_meta_state_has_degradation(self):
        """MetaState must have degradation components."""
        state = MetaState()
        assert hasattr(state, "degradation_score")

    def test_meta_state_has_reset(self):
        """MetaState must have reset components."""
        state = MetaState()
        assert hasattr(state, "reset_probability")

    def test_meta_state_has_health(self):
        """MetaState must have system health."""
        state = MetaState()
        assert hasattr(state, "system_health")

    def test_meta_state_no_physical(self):
        """MetaState must NOT contain physical components."""
        state = MetaState()
        assert not hasattr(state, "yaw")
        assert not hasattr(state, "brightness_mean")

    def test_meta_state_no_belief(self):
        """MetaState must NOT contain belief components."""
        state = MetaState()
        assert not hasattr(state, "covariance")

    def test_meta_state_recovery_probability_bounded(self):
        """Recovery probability must be in [0, 1]."""
        state = MetaState()
        assert 0.0 <= state.recovery_probability <= 1.0

    def test_meta_state_reset_probability_bounded(self):
        """Reset probability must be in [0, 1]."""
        state = MetaState()
        assert 0.0 <= state.reset_probability <= 1.0


# ─── Test Class 4: SeparatedState ────────────────────────────────────────────

class TestSeparatedState:
    """Test separated state (physical + belief + meta)."""

    def test_separated_state_exists(self):
        """SeparatedState must exist."""
        state = SeparatedState()
        assert state is not None

    def test_separated_state_has_physical(self):
        """SeparatedState must have physical state."""
        state = SeparatedState()
        assert hasattr(state, "physical")
        assert isinstance(state.physical, PhysicalState)

    def test_separated_state_has_belief(self):
        """SeparatedState must have belief state."""
        state = SeparatedState()
        assert hasattr(state, "belief")
        assert isinstance(state.belief, BeliefState)

    def test_separated_state_has_meta(self):
        """SeparatedState must have meta-state."""
        state = SeparatedState()
        assert hasattr(state, "meta")
        assert isinstance(state.meta, MetaState)

    def test_separated_state_to_latent(self):
        """SeparatedState must be convertible to LatentState."""
        state = SeparatedState()
        latent = state.to_latent()
        assert isinstance(latent, LatentState)

    def test_separated_state_from_latent(self):
        """SeparatedState must be constructible from LatentState."""
        latent = LatentState(yaw=5.0, pitch=-3.0)
        separated = SeparatedState.from_latent(latent)
        assert separated.physical.yaw == 5.0
        assert separated.physical.pitch == -3.0


# ─── Test Class 5: StateSeparator ────────────────────────────────────────────

class TestStateSeparator:
    """Test state separator."""

    def test_separator_exists(self):
        """StateSeparator must exist."""
        separator = StateSeparator()
        assert separator is not None

    def test_separator_separates(self):
        """Separator must separate LatentState into components."""
        separator = StateSeparator()
        latent = LatentState(yaw=5.0, pitch=-3.0, roll=1.0)
        separated = separator.separate(latent)
        assert isinstance(separated, SeparatedState)

    def test_separator_preserves_values(self):
        """Separator must preserve all values."""
        separator = StateSeparator()
        latent = LatentState(yaw=5.0, pitch=-3.0, roll=1.0)
        separated = separator.separate(latent)
        assert separated.physical.yaw == 5.0
        assert separated.physical.pitch == -3.0
        assert separated.physical.roll == 1.0

    def test_separator_roundtrip(self):
        """Separate then merge must preserve state."""
        separator = StateSeparator()
        latent1 = LatentState(yaw=5.0, pitch=-3.0, roll=1.0)
        separated = separator.separate(latent1)
        latent2 = separator.merge(separated)
        np.testing.assert_allclose(
            latent1.to_vector(), latent2.to_vector(), atol=1e-10
        )
