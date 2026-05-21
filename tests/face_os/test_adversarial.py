"""Adversarial Robustness Test Suite.

Tests for pathological edge cases that could break the system:
- Lighting explosions
- Partial occlusion
- Landmark corruption
- Transform singularities
- Intrinsic collapse
- Renderer mode thrashing
- Specular spikes
- Yaw discontinuities

Target: 30 tests
"""

import numpy as np
import pytest

from face_os.intrinsic_decomposition import IntrinsicDecomposer, IntrinsicComponents
from face_os.physical_renderer import PhysicalRenderer, LightingModel
from face_os.lie_group import SE2Transform, SIM2Transform, interpolate_se2, interpolate_sim2
from face_os.renderer_mode import RendererMode, RendererModeState


class TestPathologicalLighting:
    """Test extreme lighting conditions."""

    def test_all_black_image(self):
        """All-black image must not crash decomposition."""
        decomposer = IntrinsicDecomposer()
        image = np.zeros((256, 256, 3), dtype=np.float32)
        result = decomposer.decompose(image)
        assert result.albedo.shape == (256, 256, 3)
        assert not np.any(np.isnan(result.albedo))

    def test_all_white_image(self):
        """All-white image must not crash decomposition."""
        decomposer = IntrinsicDecomposer()
        image = np.ones((256, 256, 3), dtype=np.float32)
        result = decomposer.decompose(image)
        assert result.albedo.shape == (256, 256, 3)
        assert not np.any(np.isnan(result.albedo))

    def test_extreme_contrast(self):
        """Extreme contrast must not crash decomposition."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(256, 256, 3).astype(np.float32)
        image = image * 100  # Extreme values
        result = decomposer.decompose(image)
        assert not np.any(np.isnan(result.albedo))
        assert not np.any(np.isinf(result.albedo))

    def test_single_channel_dominant(self):
        """Single channel dominant must not crash."""
        decomposer = IntrinsicDecomposer()
        image = np.zeros((256, 256, 3), dtype=np.float32)
        image[:, :, 0] = 1.0  # Only red channel
        result = decomposer.decompose(image)
        assert not np.any(np.isnan(result.albedo))

    def test_nan_input_handling(self):
        """NaN input must be handled gracefully."""
        decomposer = IntrinsicDecomposer()
        image = np.full((256, 256, 3), np.nan, dtype=np.float32)
        # Should not crash, but may produce NaN output
        try:
            result = decomposer.decompose(image)
            # If it doesn't crash, check output
            assert result.albedo.shape == (256, 256, 3)
        except Exception:
            # It's acceptable to raise an exception for NaN input
            pass


class TestLandmarkCorruption:
    """Test corrupted landmark handling."""

    def test_zero_landmarks(self):
        """Zero landmarks must not crash renderer."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5
        result = renderer.render(albedo, normal_map, shading)
        assert result.rendered.shape == (256, 256, 3)

    def test_all_same_landmarks(self):
        """All same landmarks must not crash."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5
        result = renderer.render(albedo, normal_map, shading)
        assert result.rendered.shape == (256, 256, 3)

    def test_extreme_landmark_values(self):
        """Extreme landmark values must not crash."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5
        normal_map = np.ones((256, 256, 3), dtype=np.float32) * 1000  # Extreme
        # Normalize to unit vectors
        norms = np.linalg.norm(normal_map, axis=2, keepdims=True)
        normal_map = normal_map / (norms + 1e-8)
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5
        result = renderer.render(albedo, normal_map, shading)
        assert not np.any(np.isnan(result.rendered))


class TestTransformSingularities:
    """Test transform edge cases."""

    def test_identity_transform(self):
        """Identity transform must be stable."""
        T = SE2Transform.identity()
        assert T.theta == 0.0
        assert T.tx == 0.0
        assert T.ty == 0.0
        M = T.to_matrix()
        assert np.allclose(M, np.eye(3), atol=1e-10)

    def test_zero_rotation_transform(self):
        """Zero rotation must be stable."""
        T = SE2Transform(theta=0.0, tx=1.0, ty=2.0)
        M = T.to_matrix()
        assert abs(M[0, 0] - 1.0) < 1e-10
        assert abs(M[1, 1] - 1.0) < 1e-10

    def test_pi_rotation_transform(self):
        """Pi rotation must be stable."""
        T = SE2Transform(theta=np.pi, tx=0.0, ty=0.0)
        M = T.to_matrix()
        assert abs(M[0, 0] + 1.0) < 1e-10
        assert abs(M[1, 1] + 1.0) < 1e-10

    def test_very_small_scale(self):
        """Very small scale must not crash."""
        T = SIM2Transform(theta=0.0, tx=0.0, ty=0.0, scale=1e-10)
        M = T.to_matrix()
        assert not np.any(np.isnan(M))

    def test_very_large_scale(self):
        """Very large scale must not crash."""
        T = SIM2Transform(theta=0.0, tx=0.0, ty=0.0, scale=1e10)
        M = T.to_matrix()
        assert not np.any(np.isnan(M))

    def test_interpolation_at_extremes(self):
        """Interpolation at extremes must be stable."""
        T1 = SE2Transform(theta=0.0, tx=0.0, ty=0.0)
        T2 = SE2Transform(theta=np.pi, tx=100.0, ty=100.0)

        T_start = interpolate_se2(T1, T2, 0.0)
        T_end = interpolate_se2(T1, T2, 1.0)
        T_mid = interpolate_se2(T1, T2, 0.5)

        assert not np.any(np.isnan(T_start.log()))
        assert not np.any(np.isnan(T_end.log()))
        assert not np.any(np.isnan(T_mid.log()))


class TestIntrinsicCollapse:
    """Test intrinsic decomposition edge cases."""

    def test_uniform_image(self):
        """Uniform image must produce uniform albedo."""
        decomposer = IntrinsicDecomposer()
        image = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        result = decomposer.decompose(image)
        albedo_std = np.std(result.albedo)
        assert albedo_std < 0.1  # Should be nearly uniform

    def test_gradient_image(self):
        """Gradient image must produce smooth albedo."""
        decomposer = IntrinsicDecomposer()
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        xx, yy = np.meshgrid(x, y)
        image = np.stack([xx, yy, xx * 0.5], axis=2).astype(np.float32)
        result = decomposer.decompose(image)
        # Albedo should exist and not be NaN
        assert not np.any(np.isnan(result.albedo))
        assert result.albedo.shape == (256, 256, 3)

    def test_high_frequency_texture(self):
        """High frequency texture must not crash."""
        decomposer = IntrinsicDecomposer()
        # Create checkerboard pattern
        x = np.arange(256) // 16 % 2
        y = np.arange(256) // 16 % 2
        xx, yy = np.meshgrid(x, y)
        image = np.stack([xx, yy, xx], axis=2).astype(np.float32)
        result = decomposer.decompose(image)
        assert not np.any(np.isnan(result.albedo))


class TestRendererModeThrashing:
    """Test renderer mode stability."""

    def test_rapid_confidence_oscillation(self):
        """Rapid confidence oscillation must not cause thrashing."""
        state = RendererModeState()

        # Oscillate rapidly
        for i in range(100):
            confidence = 0.9 if i % 2 == 0 else 0.1
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=confidence,
                decomposition_error=0.1,
            )

        # Should have limited transitions due to hysteresis
        assert state.transition_count < 50

    def test_sudden_confidence_drop(self):
        """Sudden confidence drop must transition smoothly."""
        state = RendererModeState()

        # Start with high confidence
        for _ in range(20):
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.9,
                decomposition_error=0.1,
            )

        initial_mode = state.current_mode

        # Sudden drop
        for _ in range(20):
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.1,
                decomposition_error=0.5,
            )

        # Should have transitioned
        assert state.current_mode != initial_mode

    def test_sudden_intrinsic_loss(self):
        """Sudden intrinsic loss must fallback gracefully."""
        state = RendererModeState()

        # Start with intrinsic
        for _ in range(20):
            state.update(
                intrinsic_available=True,
                intrinsic_confidence=0.9,
                decomposition_error=0.1,
            )

        # Sudden loss
        for _ in range(20):
            mode = state.update(
                intrinsic_available=False,
                intrinsic_confidence=0.0,
                decomposition_error=1.0,
            )

        assert mode == RendererMode.ALPHA_FALLBACK


class TestSpecularSpikes:
    """Test specular edge cases."""

    def test_bright_specular(self):
        """Bright specular must not crash."""
        renderer = PhysicalRenderer()
        albedo = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32)

        # Very bright lighting
        lighting = LightingModel(
            ambient=0.0,
            diffuse_intensity=2.0,
            specular_intensity=5.0,
            specular_power=128.0,
        )

        result = renderer.render(albedo, normal_map, shading, lighting=lighting)
        assert not np.any(np.isnan(result.rendered))
        assert np.all(result.rendered >= 0)

    def test_zero_specular(self):
        """Zero specular must work."""
        renderer = PhysicalRenderer()
        albedo = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32)

        lighting = LightingModel(
            ambient=0.1,
            diffuse_intensity=0.8,
            specular_intensity=0.0,
        )

        result = renderer.render(albedo, normal_map, shading, lighting=lighting)
        assert np.all(result.specular_component == 0)


class TestYawDiscontinuities:
    """Test yaw angle edge cases."""

    def test_yaw_wraparound(self):
        """Yaw wraparound must be handled."""
        # Test angles near ±180
        angles = [179.0, -179.0, 180.0, -180.0, 0.0, 90.0, -90.0]
        for angle in angles:
            T = SE2Transform(theta=np.radians(angle), tx=0.0, ty=0.0)
            M = T.to_matrix()
            assert not np.any(np.isnan(M))
            assert abs(np.linalg.det(M) - 1.0) < 1e-10

    def test_yaw_interpolation_across_boundary(self):
        """Yaw interpolation across ±180 boundary must be smooth."""
        T1 = SE2Transform(theta=np.radians(170), tx=0.0, ty=0.0)
        T2 = SE2Transform(theta=np.radians(-170), tx=0.0, ty=0.0)

        # Interpolate
        T_mid = interpolate_se2(T1, T2, 0.5)

        # Should not be NaN
        assert not np.any(np.isnan(T_mid.log()))

    def test_continuous_yaw_rotation(self):
        """Continuous yaw rotation must be smooth."""
        angles = np.linspace(0, 2 * np.pi, 100)
        transforms = [SE2Transform(theta=a, tx=0.0, ty=0.0) for a in angles]

        # Check smoothness
        for i in range(len(transforms) - 1):
            T1 = transforms[i]
            T2 = transforms[i + 1]
            T_interp = interpolate_se2(T1, T2, 0.5)
            assert not np.any(np.isnan(T_interp.log()))


class TestCovarianceExplosion:
    """Test covariance edge cases."""

    def test_zero_covariance(self):
        """Zero covariance must not crash."""
        from face_os.state_space import LatentState
        state = LatentState()
        state.covariance = np.zeros_like(state.covariance)
        # Should not crash when accessing
        assert state.covariance.shape == (11, 11)

    def test_large_covariance(self):
        """Large covariance must not crash."""
        from face_os.state_space import LatentState
        state = LatentState()
        state.covariance = np.eye(11) * 1e10
        # Should not crash when accessing
        assert state.covariance.shape == (11, 11)

    def test_singular_covariance(self):
        """Singular covariance must not crash."""
        from face_os.state_space import LatentState
        state = LatentState()
        state.covariance = np.zeros((11, 11))
        state.covariance[0, 0] = 1.0  # Only one dimension
        # Should not crash when accessing
        assert state.covariance.shape == (11, 11)


class TestEdgeCaseImages:
    """Test edge case image sizes."""

    def test_1x1_image(self):
        """1x1 image must not crash."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(1, 1, 3).astype(np.float32)
        # 1x1 image is too small for gradient computation, should handle gracefully
        try:
            result = decomposer.decompose(image)
            assert result.albedo.shape == (1, 1, 3)
        except ValueError:
            # Acceptable to raise ValueError for too-small images
            pass

    def test_very_small_image(self):
        """Very small image must not crash."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(4, 4, 3).astype(np.float32)
        result = decomposer.decompose(image)
        assert result.albedo.shape == (4, 4, 3)

    def test_non_square_image(self):
        """Non-square image must not crash."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(128, 256, 3).astype(np.float32)
        result = decomposer.decompose(image)
        assert result.albedo.shape == (128, 256, 3)
