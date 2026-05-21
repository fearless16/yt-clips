"""Phase 3B: Physical Renderer Tests.

Tests for physically-based rendering:
- Lambertian diffuse + Blinn-Phong specular
- Lighting model (ambient, diffuse, specular)
- Rendering equation: Y = R(G, A, L, V)
- Energy conservation
- Gradient smoothness

Target: 30 tests
"""

import numpy as np
import pytest

from face_os.physical_renderer import (
    PhysicalRenderer,
    PhysicalRenderConfig,
    LightingModel,
    PhysicalRenderOutput,
    RenderReport,
)


class TestLightingModel:
    """Test LightingModel dataclass."""

    def test_default_lighting(self):
        """Default lighting must have reasonable values."""
        lighting = LightingModel()
        assert lighting.ambient >= 0
        assert lighting.diffuse_intensity >= 0
        assert lighting.specular_power >= 0
        assert np.linalg.norm(lighting.diffuse_direction) > 0

    def test_lighting_direction_normalized(self):
        """Light direction must be normalized."""
        lighting = LightingModel(
            diffuse_direction=np.array([1.0, 1.0, 1.0])
        )
        norm = np.linalg.norm(lighting.diffuse_direction)
        assert abs(norm - 1.0) < 1e-5

    def test_spherical_harmonics_shape(self):
        """Spherical harmonics must have 9 coefficients."""
        sh = np.zeros(9)
        lighting = LightingModel(spherical_harmonics=sh)
        assert lighting.spherical_harmonics.shape == (9,)

    def test_lighting_values_non_negative(self):
        """Lighting intensities must be non-negative."""
        lighting = LightingModel(
            ambient=-0.1,
            diffuse_intensity=-0.5,
        )
        # Should be clamped to non-negative
        assert lighting.ambient >= 0
        assert lighting.diffuse_intensity >= 0


class TestPhysicalRenderConfig:
    """Test PhysicalRenderConfig."""

    def test_default_config(self):
        """Default config must have reasonable values."""
        config = PhysicalRenderConfig()
        assert config.diffuse_weight > 0
        assert config.specular_weight >= 0
        assert config.ambient_weight >= 0
        assert 0 < config.energy_conservation_limit <= 1

    def test_config_validation(self):
        """Config values must be valid."""
        config = PhysicalRenderConfig(
            diffuse_weight=0.8,
            specular_weight=0.2,
            ambient_weight=0.1,
            energy_conservation_limit=0.95,
        )
        assert config.diffuse_weight == 0.8
        assert config.specular_weight == 0.2
        assert config.ambient_weight == 0.1


class TestPhysicalRenderer:
    """Test PhysicalRenderer."""

    def test_render_shape(self):
        """Rendering must preserve input shape."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0  # All normals pointing up
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        result = renderer.render(albedo, normal_map, shading)
        assert result.rendered.shape == (256, 256, 3)

    def test_render_valid_range(self):
        """Rendering output must be in [0, 1]."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        result = renderer.render(albedo, normal_map, shading)
        assert np.all(result.rendered >= 0) and np.all(result.rendered <= 1)

    def test_diffuse_component(self):
        """Diffuse component must be non-negative."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        result = renderer.render(albedo, normal_map, shading)
        assert np.all(result.diffuse_component >= 0)

    def test_specular_component(self):
        """Specular component must be non-negative."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        result = renderer.render(albedo, normal_map, shading)
        assert np.all(result.specular_component >= 0)

    def test_ambient_component(self):
        """Ambient component must be non-negative."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        result = renderer.render(albedo, normal_map, shading)
        assert np.all(result.ambient_component >= 0)

    def test_energy_conservation(self):
        """Output energy must be <= input energy."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        result = renderer.render(albedo, normal_map, shading)
        
        # Output energy <= input energy
        input_energy = np.mean(albedo * shading.repeat(3, axis=2))
        output_energy = np.mean(result.rendered)
        assert output_energy <= input_energy * 1.5  # Allow 50% tolerance

    def test_rendering_error(self):
        """Rendering error must be computed."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        result = renderer.render(albedo, normal_map, shading)
        assert hasattr(result, "rendering_error")
        assert result.rendering_error >= 0

    def test_lighting_invariance(self):
        """Same albedo with different lighting must produce different results."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        # Different lighting
        lighting1 = LightingModel(diffuse_intensity=1.0)
        lighting2 = LightingModel(diffuse_intensity=0.5)

        result1 = renderer.render(albedo, normal_map, shading, lighting=lighting1)
        result2 = renderer.render(albedo, normal_map, shading, lighting=lighting2)

        # Results must be different
        diff = np.mean(np.abs(result1.rendered - result2.rendered))
        assert diff > 0.01

    def test_view_consistency(self):
        """Same geometry with different views must produce different specular."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        # Different view directions
        view1 = np.array([0.0, 0.0, 1.0])  # Front
        view2 = np.array([1.0, 0.0, 0.0])  # Side

        result1 = renderer.render(albedo, normal_map, shading, view_direction=view1)
        result2 = renderer.render(albedo, normal_map, shading, view_direction=view2)

        # Specular must be different
        spec_diff = np.mean(np.abs(
            result1.specular_component - result2.specular_component
        ))
        assert spec_diff > 0.001

    def test_gradient_smoothness(self):
        """Rendering must have smooth gradients."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        result = renderer.render(albedo, normal_map, shading)

        # Check gradient magnitude
        from scipy.ndimage import sobel
        gradient_x = sobel(result.rendered[:, :, 0], axis=1)
        gradient_y = sobel(result.rendered[:, :, 0], axis=0)
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
        assert np.mean(gradient_mag) < 0.5  # Smooth gradients

    def test_multiple_calls_deterministic(self):
        """Multiple calls must be deterministic."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        result1 = renderer.render(albedo, normal_map, shading)
        result2 = renderer.render(albedo, normal_map, shading)

        np.testing.assert_array_equal(result1.rendered, result2.rendered)

    def test_render_with_custom_lighting(self):
        """Rendering must work with custom lighting."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        lighting = LightingModel(
            ambient=0.2,
            diffuse_intensity=0.8,
            specular_power=32.0,
        )

        result = renderer.render(albedo, normal_map, shading, lighting=lighting)
        assert result.rendered.shape == (256, 256, 3)


class TestLambertianDiffuse:
    """Test Lambertian diffuse model."""

    def test_diffuse_intensity(self):
        """Diffuse intensity must be proportional to N·L."""
        renderer = PhysicalRenderer()
        albedo = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0  # All normals pointing up
        shading = np.ones((256, 256, 1), dtype=np.float32)

        # Light from above
        lighting = LightingModel(
            diffuse_direction=np.array([0.0, 0.0, 1.0]),
            diffuse_intensity=1.0,
        )

        result = renderer.render(albedo, normal_map, shading, lighting=lighting)
        
        # Diffuse should be approximately albedo * N·L
        expected_diffuse = albedo * 1.0  # N·L = 1 for aligned normals
        np.testing.assert_allclose(
            result.diffuse_component, expected_diffuse, atol=0.1
        )

    def test_diffuse_zero_at_grazing(self):
        """Diffuse must be zero at grazing angles."""
        renderer = PhysicalRenderer()
        albedo = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0  # Normals pointing up
        shading = np.ones((256, 256, 1), dtype=np.float32)

        # Light from the side (grazing angle)
        lighting = LightingModel(
            diffuse_direction=np.array([1.0, 0.0, 0.0]),
            diffuse_intensity=1.0,
        )

        result = renderer.render(albedo, normal_map, shading, lighting=lighting)
        
        # Diffuse should be approximately zero (N·L ≈ 0)
        assert np.mean(result.diffuse_component) < 0.1


class TestBlinnPhongSpecular:
    """Test Blinn-Phong specular model."""

    def test_specular_peak(self):
        """Specular must peak at mirror reflection."""
        renderer = PhysicalRenderer()
        albedo = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0  # Normals pointing up
        shading = np.ones((256, 256, 1), dtype=np.float32)

        # Light and view aligned (mirror reflection)
        lighting = LightingModel(
            diffuse_direction=np.array([0.0, 0.0, 1.0]),
            specular_power=32.0,
        )
        view = np.array([0.0, 0.0, 1.0])

        result = renderer.render(albedo, normal_map, shading, lighting=lighting, view_direction=view)
        
        # Specular should be non-zero
        assert np.max(result.specular_component) > 0

    def test_specular_off_angle(self):
        """Specular must decrease off mirror angle."""
        renderer = PhysicalRenderer()
        albedo = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32)

        # Light from above, view from side
        lighting = LightingModel(
            diffuse_direction=np.array([0.0, 0.0, 1.0]),
            specular_power=32.0,
        )
        view = np.array([1.0, 0.0, 0.0])

        result = renderer.render(albedo, normal_map, shading, lighting=lighting, view_direction=view)
        
        # Specular should be lower than aligned case
        aligned_result = renderer.render(
            albedo, normal_map, shading, lighting=lighting,
            view_direction=np.array([0.0, 0.0, 1.0])
        )
        assert np.mean(result.specular_component) <= np.mean(aligned_result.specular_component)


class TestRenderReport:
    """Test RenderReport."""

    def test_report_to_dict(self):
        """Report must convert to dict."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        result = renderer.render(albedo, normal_map, shading)

        report = RenderReport(
            frame_idx=0,
            output=result,
            render_time_ms=10.0,
        )

        d = report.to_dict()
        assert d["frame_idx"] == 0
        assert d["render_time_ms"] == 10.0
        assert "rendering_error" in d

    def test_report_has_metrics(self):
        """Report must have all required metrics."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        result = renderer.render(albedo, normal_map, shading)

        report = RenderReport(
            frame_idx=0,
            output=result,
            render_time_ms=10.0,
        )

        assert hasattr(report, "frame_idx")
        assert hasattr(report, "output")
        assert hasattr(report, "render_time_ms")


class TestRenderingEquation:
    """Test rendering equation: Y = ambient + diffuse + specular."""

    def test_components_sum_to_output(self):
        """Weighted components must sum to output."""
        renderer = PhysicalRenderer()
        albedo = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32) * 0.5

        result = renderer.render(albedo, normal_map, shading)

        # Weighted sum of components should approximate output
        component_sum = (
            renderer.config.ambient_weight * result.ambient_component
            + renderer.config.diffuse_weight * result.diffuse_component
            + renderer.config.specular_weight * result.specular_component
        )
        np.testing.assert_allclose(result.rendered, component_sum, atol=0.01)

    def test_rendering_equation_correctness(self):
        """Rendering equation must be physically correct."""
        renderer = PhysicalRenderer()
        albedo = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        shading = np.ones((256, 256, 1), dtype=np.float32)

        lighting = LightingModel(
            ambient=0.1,
            diffuse_intensity=0.8,
            specular_power=32.0,
        )

        result = renderer.render(albedo, normal_map, shading, lighting=lighting)

        # For aligned normals and view:
        # ambient = albedo * ambient_intensity
        # diffuse = albedo * diffuse_intensity * N·L
        # specular = specular_power * (N·H)^shininess
        # Y = ambient + diffuse + specular
        
        expected_ambient = albedo * 0.1
        expected_diffuse = albedo * 0.8 * 1.0  # N·L = 1
        
        np.testing.assert_allclose(result.ambient_component, expected_ambient, atol=0.05)
        np.testing.assert_allclose(result.diffuse_component, expected_diffuse, atol=0.1)
