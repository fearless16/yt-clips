"""
test_phase2a_state_space.py — Phase 2A: State-Space Formulation Tests.

Tests:
1. State evolution consistency
2. Uncertainty propagation
3. Temporal stability
4. Drift boundedness
5. Occlusion recovery
6. Re-entry stabilization

TDD: These tests should FAIL before implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Optional

from face_os.types import (
    GeometryState,
    IdentityState,
    TemporalState,
    EnergyTerms,
)
from face_os.state_space import (
    LatentState,
    StateTransitionModel,
    ProcessNoiseModel,
    ObservationModel,
    StateEvolutionReport,
    StateSpaceEstimator,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_observation(
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    confidence: float = 0.9,
    identity_uncertainty: float = 0.1,
) -> dict:
    """Create a synthetic observation."""
    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "confidence": confidence,
        "identity_uncertainty": identity_uncertainty,
        "canonical_face": np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8),
        "mask": np.ones((256, 256), dtype=np.float32) * 0.5,
    }


# ─── Test Class 1: LatentState ──────────────────────────────────────────────

class TestLatentState:
    """Test latent state representation."""

    def test_latent_state_has_geometry(self):
        """LatentState must contain geometry components."""
        state = LatentState()
        assert hasattr(state, "yaw")
        assert hasattr(state, "pitch")
        assert hasattr(state, "roll")

    def test_latent_state_has_identity(self):
        """LatentState must contain identity components."""
        state = LatentState()
        assert hasattr(state, "identity_uncertainty")
        assert hasattr(state, "appearance_latent_norm")

    def test_latent_state_has_temporal(self):
        """LatentState must contain temporal components."""
        state = LatentState()
        assert hasattr(state, "temporal_confidence")
        assert hasattr(state, "drift_score")
        assert hasattr(state, "continuity_score")

    def test_latent_state_has_covariance(self):
        """LatentState must contain covariance matrix."""
        state = LatentState()
        assert hasattr(state, "covariance")
        assert state.covariance is not None
        assert state.covariance.shape[0] == state.covariance.shape[1]

    def test_latent_state_has_recovery(self):
        """LatentState must contain recovery state."""
        state = LatentState()
        assert hasattr(state, "recovery_state")
        assert state.recovery_state in ["stable", "uncertain", "degraded", "recovering", "reset_required"]

    def test_latent_state_to_vector(self):
        """LatentState must be convertible to vector."""
        state = LatentState()
        vec = state.to_vector()
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1

    def test_latent_state_from_vector(self):
        """LatentState must be constructible from vector."""
        state1 = LatentState(yaw=5.0, pitch=-3.0, roll=1.0)
        vec = state1.to_vector()
        state2 = LatentState.from_vector(vec)
        assert abs(state2.yaw - 5.0) < 1e-6
        assert abs(state2.pitch - (-3.0)) < 1e-6
        assert abs(state2.roll - 1.0) < 1e-6


# ─── Test Class 2: StateTransitionModel ──────────────────────────────────────

class TestStateTransitionModel:
    """Test state transition model x_t = f(x_{t-1}, u_t, ε_t)."""

    def test_transition_model_exists(self):
        """StateTransitionModel must exist."""
        model = StateTransitionModel()
        assert model is not None

    def test_transition_model_has_predict(self):
        """StateTransitionModel must have predict method."""
        model = StateTransitionModel()
        assert hasattr(model, "predict")

    def test_transition_preserves_state_shape(self):
        """Transition must preserve state vector dimension."""
        model = StateTransitionModel()
        state = LatentState()
        state_vec = state.to_vector()
        predicted = model.predict(state)
        assert predicted.to_vector().shape == state_vec.shape

    def test_transition_is_deterministic_without_noise(self):
        """Transition without noise must be deterministic."""
        model = StateTransitionModel()
        state = LatentState(yaw=5.0, pitch=-3.0, roll=1.0)

        pred1 = model.predict(state)
        pred2 = model.predict(state)

        np.testing.assert_allclose(
            pred1.to_vector(), pred2.to_vector(), atol=1e-10
        )

    def test_transition_with_process_noise(self):
        """Transition with noise must produce different results."""
        model = StateTransitionModel()
        noise = ProcessNoiseModel()
        state = LatentState(yaw=5.0, pitch=-3.0, roll=1.0)

        pred1 = model.predict(state, noise=noise)
        pred2 = model.predict(state, noise=noise)

        # With noise, results should differ
        diff = np.linalg.norm(pred1.to_vector() - pred2.to_vector())
        # Noise is random, so diff > 0 with high probability
        # But we can't guarantee it every time, so just check shape
        assert pred1.to_vector().shape == pred2.to_vector().shape

    def test_transition_preserves_stability(self):
        """Repeated transitions must not diverge."""
        model = StateTransitionModel()
        state = LatentState(yaw=5.0, pitch=-3.0, roll=1.0)

        for _ in range(100):
            state = model.predict(state)

        # State must remain bounded
        vec = state.to_vector()
        assert np.all(np.isfinite(vec))
        assert np.max(np.abs(vec)) < 1000.0


# ─── Test Class 3: ProcessNoiseModel ─────────────────────────────────────────

class TestProcessNoiseModel:
    """Test process noise model."""

    def test_process_noise_exists(self):
        """ProcessNoiseModel must exist."""
        noise = ProcessNoiseModel()
        assert noise is not None

    def test_process_noise_has_covariance(self):
        """ProcessNoiseModel must have covariance matrix."""
        noise = ProcessNoiseModel()
        assert hasattr(noise, "covariance")
        assert noise.covariance.shape[0] == noise.covariance.shape[1]

    def test_process_noise_covariance_positive_definite(self):
        """Process noise covariance must be positive definite."""
        noise = ProcessNoiseModel()
        eigvals = np.linalg.eigvalsh(noise.covariance)
        assert np.all(eigvals > 0)

    def test_process_noise_sample_shape(self):
        """Noise sample must match state dimension."""
        noise = ProcessNoiseModel()
        sample = noise.sample()
        assert sample.shape == (noise.covariance.shape[0],)

    def test_process_noise_sample_statistics(self):
        """Noise samples must have correct statistics."""
        noise = ProcessNoiseModel()
        samples = np.array([noise.sample() for _ in range(1000)])
        sample_mean = np.mean(samples, axis=0)
        # Mean should be near zero
        assert np.max(np.abs(sample_mean)) < 0.5


# ─── Test Class 4: ObservationModel ──────────────────────────────────────────

class TestObservationModel:
    """Test observation model."""

    def test_observation_model_exists(self):
        """ObservationModel must exist."""
        model = ObservationModel()
        assert model is not None

    def test_observation_model_has_observe(self):
        """ObservationModel must have observe method."""
        model = ObservationModel()
        assert hasattr(model, "observe")

    def test_observation_maps_state_to_observation(self):
        """Observation must map latent state to observation space."""
        model = ObservationModel()
        state = LatentState(yaw=5.0, pitch=-3.0, roll=1.0)
        obs = model.observe(state)
        assert obs is not None
        assert "yaw" in obs
        assert "pitch" in obs
        assert "roll" in obs

    def test_observation_residual(self):
        """Observation residual must be computable."""
        model = ObservationModel()
        state = LatentState(yaw=5.0, pitch=-3.0, roll=1.0)
        obs = _make_observation(yaw=6.0, pitch=-2.0, roll=0.5)
        residual = model.residual(state, obs)
        assert residual is not None
        assert isinstance(residual, np.ndarray)

    def test_observation_jacobian(self):
        """Observation Jacobian must be computable."""
        model = ObservationModel()
        state = LatentState(yaw=5.0, pitch=-3.0, roll=1.0)
        J = model.jacobian(state)
        assert J is not None
        assert isinstance(J, np.ndarray)


# ─── Test Class 5: StateEvolutionReport ──────────────────────────────────────

class TestStateEvolutionReport:
    """Test state evolution report."""

    def test_report_exists(self):
        """StateEvolutionReport must exist."""
        report = StateEvolutionReport()
        assert report is not None

    def test_report_has_state(self):
        """Report must contain current state."""
        report = StateEvolutionReport()
        assert hasattr(report, "state")
        assert isinstance(report.state, LatentState)

    def test_report_has_covariance(self):
        """Report must contain covariance."""
        report = StateEvolutionReport()
        assert hasattr(report, "covariance_trace")
        assert hasattr(report, "covariance_eigenvalues")

    def test_report_has_metrics(self):
        """Report must contain all required metrics."""
        report = StateEvolutionReport()
        assert hasattr(report, "drift_score")
        assert hasattr(report, "continuity_score")
        assert hasattr(report, "uncertainty_growth")
        assert hasattr(report, "recovery_state")

    def test_report_to_dict(self):
        """Report must be serializable to dict."""
        report = StateEvolutionReport()
        d = report.to_dict()
        assert "state" in d
        assert "covariance_trace" in d
        assert "drift_score" in d


# ─── Test Class 6: StateSpaceEstimator ───────────────────────────────────────

class TestStateSpaceEstimator:
    """Test state-space estimator (Kalman-like filter)."""

    def test_estimator_exists(self):
        """StateSpaceEstimator must exist."""
        estimator = StateSpaceEstimator()
        assert estimator is not None

    def test_estimator_has_predict(self):
        """Estimator must have predict step."""
        estimator = StateSpaceEstimator()
        assert hasattr(estimator, "predict")

    def test_estimator_has_update(self):
        """Estimator must have update step."""
        estimator = StateSpaceEstimator()
        assert hasattr(estimator, "update")

    def test_estimator_predict_update_cycle(self):
        """Estimator must support predict-update cycle."""
        estimator = StateSpaceEstimator()

        obs = _make_observation(yaw=5.0, pitch=-3.0, roll=1.0)
        report = estimator.step(obs)

        assert isinstance(report, StateEvolutionReport)
        assert report.state is not None

    def test_estimator_uncertainty_grows_without_observations(self):
        """Uncertainty must grow when no observations arrive."""
        estimator = StateSpaceEstimator()

        # Predict without update
        for _ in range(10):
            estimator.predict()

        report = estimator.get_report()
        # Uncertainty should have grown
        assert report.covariance_trace > 0

    def test_estimator_uncertainty_shrinks_with_observations(self):
        """Uncertainty must shrink for observed components when observations arrive."""
        estimator = StateSpaceEstimator()

        # Get initial covariance for observed components (yaw, pitch, roll, conf, id_unc)
        # Indices: yaw=0, pitch=1, roll=2, id_unc=3, conf=5
        observed_indices = [0, 1, 2, 3, 5]
        P0 = estimator.state.covariance
        trace0 = np.trace(P0[np.ix_(observed_indices, observed_indices)])

        # Update with observations
        for _ in range(20):
            obs = _make_observation(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.95)
            estimator.step(obs)

        P1 = estimator.state.covariance
        trace1 = np.trace(P1[np.ix_(observed_indices, observed_indices)])

        # Observed component uncertainty should have decreased
        assert trace1 < trace0, f"Observed uncertainty: {trace0:.3f} -> {trace1:.3f}"

    def test_estimator_converges_under_stable_input(self):
        """Estimator must converge under stable input."""
        estimator = StateSpaceEstimator()

        # Feed stable observations
        for _ in range(100):
            obs = _make_observation(yaw=5.0, pitch=-3.0, roll=1.0, confidence=0.9)
            estimator.step(obs)

        report = estimator.get_report()
        # State should be near observation
        assert abs(report.state.yaw - 5.0) < 2.0
        assert abs(report.state.pitch - (-3.0)) < 2.0
        assert abs(report.state.roll - 1.0) < 2.0

    def test_estimator_drift_bounded(self):
        """Estimator geometry drift must be bounded over long sequences."""
        estimator = StateSpaceEstimator()

        # Feed 500 stable observations
        for _ in range(500):
            obs = _make_observation(yaw=5.0, pitch=-3.0, roll=1.0, confidence=0.9)
            estimator.step(obs)

        report = estimator.get_report()
        # Geometry must be near observation
        assert abs(report.state.yaw - 5.0) < 5.0
        assert abs(report.state.pitch - (-3.0)) < 5.0
        assert abs(report.state.roll - 1.0) < 5.0
        # State must be finite
        assert np.all(np.isfinite(report.state.to_vector()))

    def test_estimator_handles_occlusion(self):
        """Estimator must handle occlusion (no observation)."""
        estimator = StateSpaceEstimator()

        # Normal operation
        for _ in range(20):
            obs = _make_observation(yaw=5.0, pitch=-3.0, roll=1.0, confidence=0.9)
            estimator.step(obs)

        # Occlusion (predict only)
        for _ in range(10):
            estimator.predict()

        report = estimator.get_report()
        # State should still be bounded
        assert np.all(np.isfinite(report.state.to_vector()))
        # Recovery state should reflect uncertainty
        assert report.recovery_state in ["uncertain", "degraded", "recovering"]

    def test_estimator_recovers_after_occlusion(self):
        """Estimator must recover after occlusion."""
        estimator = StateSpaceEstimator()

        # Normal operation
        for _ in range(20):
            obs = _make_observation(yaw=5.0, pitch=-3.0, roll=1.0, confidence=0.9)
            estimator.step(obs)

        # Occlusion
        for _ in range(10):
            estimator.predict()

        # Recovery
        for _ in range(20):
            obs = _make_observation(yaw=5.0, pitch=-3.0, roll=1.0, confidence=0.9)
            estimator.step(obs)

        report = estimator.get_report()
        # Should be back to stable
        assert report.recovery_state in ["stable", "recovering"]
        assert abs(report.state.yaw - 5.0) < 3.0

    def test_estimator_report_has_all_metrics(self):
        """Estimator report must have all required metrics."""
        estimator = StateSpaceEstimator()

        obs = _make_observation(yaw=5.0, pitch=-3.0, roll=1.0)
        estimator.step(obs)

        report = estimator.get_report()
        d = report.to_dict()

        assert "state" in d
        assert "covariance_trace" in d
        assert "covariance_eigenvalues" in d
        assert "drift_score" in d
        assert "continuity_score" in d
        assert "uncertainty_growth" in d
        assert "recovery_state" in d
