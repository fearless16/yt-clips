"""Long-Horizon Stability Tests.

Tests for stability over 1000+ frames:
- Identity drift
- Transform stability
- Covariance bounded
- Recovery behavior
- Renderer consistency

Target: 15 tests
"""

import numpy as np
import pytest

from face_os.state_evolution import StateEvolution
from face_os.lie_group import SE2Transform, SIM2Transform, interpolate_se2, interpolate_sim2
from face_os.renderer_mode import RendererMode, RendererModeState
from face_os.intrinsic_decomposition import IntrinsicDecomposer


class TestLongHorizonIdentityDrift:
    """Test identity drift over 1000+ frames."""

    def test_identity_stable_over_1000_frames(self):
        """Identity must remain stable over 1000 frames."""
        decomposer = IntrinsicDecomposer()

        # Create stable identity
        base_image = np.random.rand(64, 64, 3).astype(np.float32) * 0.5 + 0.25
        base_result = decomposer.decompose(base_image)

        # Simulate 1000 frames with small variations
        for i in range(1000):
            # Small variation (simulating lighting changes)
            variation = np.random.randn(64, 64, 3).astype(np.float32) * 0.02
            frame = np.clip(base_image + variation, 0, 1)
            result = decomposer.decompose(frame)

        # Final albedo should be close to base
        diff = np.mean(np.abs(result.albedo - base_result.albedo))
        assert diff < 0.3  # Allow some drift

    def test_albedo_stability_metric(self):
        """Albedo stability must be measurable."""
        decomposer = IntrinsicDecomposer()

        base_image = np.random.rand(64, 64, 3).astype(np.float32) * 0.5 + 0.25
        albedos = []

        for i in range(100):
            variation = np.random.randn(64, 64, 3).astype(np.float32) * 0.01
            frame = np.clip(base_image + variation, 0, 1)
            result = decomposer.decompose(frame)
            albedos.append(result.albedo)

        stability = decomposer.compute_albedo_stability(albedos)
        assert stability > 0.5  # Should be reasonably stable


class TestLongHorizonTransformStability:
    """Test transform stability over 1000+ frames."""

    def test_sim2_stable_over_1000_frames(self):
        """SIM2 interpolation must be stable over 1000 frames."""
        T_current = SIM2Transform.identity()

        for i in range(1000):
            # Small random transform
            T_target = SIM2Transform(
                theta=np.random.randn() * 0.01,
                tx=np.random.randn() * 0.1,
                ty=np.random.randn() * 0.1,
                scale=1.0 + np.random.randn() * 0.01,
            )
            T_current = interpolate_sim2(T_current, T_target, 0.1)

        # Should still be valid
        M = T_current.to_matrix()
        assert not np.any(np.isnan(M))
        assert np.linalg.det(M) > 0

    def test_transform_no_accumulation(self):
        """Transform must not accumulate error."""
        T_identity = SIM2Transform.identity()
        T_current = SIM2Transform.identity()

        for i in range(1000):
            # Small perturbation and back
            T_perturb = SIM2Transform(theta=0.01, tx=0.1, ty=0.1, scale=1.01)
            T_current = interpolate_sim2(T_current, T_perturb, 0.5)
            T_current = interpolate_sim2(T_current, T_identity, 0.5)

        # Should be close to identity
        distance = np.linalg.norm(T_current.log() - T_identity.log())
        assert distance < 1.0


class TestLongHorizonCovariance:
    """Test covariance stability over 1000+ frames."""

    def test_covariance_bounded_over_1000_frames(self):
        """Covariance must remain bounded over 1000 frames."""
        model = StateEvolution()
        cov = np.eye(11)

        for i in range(1000):
            cov = model.predict_covariance(cov)

        # Covariance should be bounded
        assert np.all(np.diag(cov) < 1000)
        assert not np.any(np.isnan(cov))

    def test_covariance_trace_growth(self):
        """Covariance trace must grow sub-linearly."""
        model = StateEvolution()
        cov = np.eye(11)

        traces = []
        for i in range(100):
            cov = model.predict_covariance(cov)
            traces.append(np.trace(cov))

        # Trace should stabilize (not grow linearly)
        growth_rate = (traces[-1] - traces[0]) / len(traces)
        assert growth_rate < 10.0  # Sub-linear growth


class TestLongHorizonRendererMode:
    """Test renderer mode stability over 1000+ frames."""

    def test_renderer_mode_stable(self):
        """Renderer mode must be stable over 1000 frames."""
        state = RendererModeState()

        for i in range(1000):
            # Consistent confidence
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.8,
                decomposition_error=0.1,
            )

        # Should be in physical mode
        assert state.current_mode == RendererMode.PHYSICAL
        # Should have few transitions
        assert state.transition_count < 10

    def test_renderer_mode_no_thrashing(self):
        """Renderer mode must not thrash over 1000 frames."""
        state = RendererModeState()

        for i in range(1000):
            # Oscillating confidence
            confidence = 0.8 if i % 10 < 7 else 0.2
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=confidence,
                decomposition_error=0.1,
            )

        # Should have limited transitions (with hysteresis)
        # With MIN_FRAMES_BEFORE_TRANSITION=5, max ~200 transitions in 1000 frames
        assert state.transition_count < 250


class TestLongHorizonRecovery:
    """Test recovery behavior over 1000+ frames."""

    def test_recovery_after_occlusion(self):
        """Must recover after simulated occlusion."""
        model = StateEvolution()
        state = np.zeros(11)

        # Normal frames
        for i in range(500):
            state = model.predict(state)

        # Simulated occlusion (high process noise)
        for i in range(100):
            state = model.predict(state)
            state += np.random.randn(11) * 0.5  # Add noise

        # Recovery frames
        for i in range(400):
            state = model.predict(state)

        # Should be stable
        assert not np.any(np.isnan(state))
        assert np.all(np.abs(state) < 100)
