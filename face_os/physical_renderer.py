"""Physically-Inspired Renderer Module.

NOTE: This is a physically-INSPIRED renderer, NOT a fully physically-based renderer.
It uses Lambertian diffuse + Blinn-Phong specular models, which are approximations.
Full PBR would require: BRDF correctness, energy conservation, full illumination model.

Rendering Equation (approximate):
    Y = ambient + diffuse + specular

where:
    ambient = albedo * ambient_intensity
    diffuse = albedo * diffuse_intensity * max(0, N·L)
    specular = specular_power * max(0, N·H)^shininess

Components:
    N = surface normal (from normal map, estimated from shading gradients)
    L = light direction
    V = view direction
    H = normalize(L + V) (half-vector)

Limitations:
    - Normals estimated from shading, not actual geometry
    - No BRDF correctness
    - No energy conservation
    - Simplified illumination model

References:
    - Lambertian diffuse model
    - Blinn-Phong specular model
    - Physically-based rendering (Pharr, Jakob, Humphreys)
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LightingModel:
    """Lighting model configuration."""

    # Ambient intensity
    ambient: float = 0.1

    # Main light direction (normalized)
    diffuse_direction: np.ndarray = field(
        default_factory=lambda: np.array([0.3, 0.3, 0.9])
    )

    # Diffuse intensity
    diffuse_intensity: float = 0.8

    # Specular power (shininess)
    specular_power: float = 32.0

    # Specular intensity
    specular_intensity: float = 0.3

    # Spherical harmonics coefficients (9 coefficients)
    spherical_harmonics: np.ndarray = field(
        default_factory=lambda: np.zeros(9)
    )

    def __post_init__(self):
        """Normalize light direction."""
        norm = np.linalg.norm(self.diffuse_direction)
        if norm > 0:
            self.diffuse_direction = self.diffuse_direction / norm
        else:
            self.diffuse_direction = np.array([0.0, 0.0, 1.0])

        # Clamp negative values
        self.ambient = max(0, self.ambient)
        self.diffuse_intensity = max(0, self.diffuse_intensity)
        self.specular_power = max(0, self.specular_power)
        self.specular_intensity = max(0, self.specular_intensity)


@dataclass
class PhysicalRenderConfig:
    """Configuration for physical renderer."""

    # Diffuse weight
    diffuse_weight: float = 0.7

    # Specular weight
    specular_weight: float = 0.2

    # Ambient weight
    ambient_weight: float = 0.1

    # Energy conservation limit (output <= input * limit)
    energy_conservation_limit: float = 0.95

    # Shininess for specular
    shininess: float = 32.0

    # Use spherical harmonics for ambient
    use_spherical_harmonics: bool = False

    # Output clamping
    clamp_output: bool = True


@dataclass
class PhysicalRenderOutput:
    """Physical rendering output."""

    # Rendered image: (H, W, 3)
    rendered: np.ndarray

    # Diffuse component: (H, W, 3)
    diffuse_component: np.ndarray

    # Specular component: (H, W, 3)
    specular_component: np.ndarray

    # Ambient component: (H, W, 3)
    ambient_component: np.ndarray

    # Rendering error: ||Y_observed - Y_rendered||
    rendering_error: float


@dataclass
class RenderReport:
    """Per-frame render metrics."""

    frame_idx: int
    output: PhysicalRenderOutput
    render_time_ms: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "frame_idx": self.frame_idx,
            "render_time_ms": self.render_time_ms,
            "rendering_error": self.output.rendering_error,
            "mean_diffuse": float(np.mean(self.output.diffuse_component)),
            "mean_specular": float(np.mean(self.output.specular_component)),
            "mean_ambient": float(np.mean(self.output.ambient_component)),
        }


class PhysicalRenderer:
    """Physically-based renderer.

    Rendering equation:
        Y = ambient + diffuse + specular

    where:
        ambient = albedo * ambient_intensity
        diffuse = albedo * diffuse_intensity * max(0, N·L)
        specular = specular_power * max(0, N·H)^shininess
    """

    def __init__(self, config: Optional[PhysicalRenderConfig] = None):
        """Initialize renderer.

        Args:
            config: Renderer configuration
        """
        self.config = config or PhysicalRenderConfig()

    def render(
        self,
        albedo: np.ndarray,
        normal_map: np.ndarray,
        shading: np.ndarray,
        lighting: Optional[LightingModel] = None,
        view_direction: Optional[np.ndarray] = None,
    ) -> PhysicalRenderOutput:
        """Render face with physical lighting model.

        Args:
            albedo: Albedo map (H, W, 3), [0, 1]
            normal_map: Normal map (H, W, 3), unit vectors
            shading: Shading map (H, W, 1), [0, 1]
            lighting: Lighting model (default: standard lighting)
            view_direction: View direction (default: [0, 0, 1])

        Returns:
            PhysicalRenderOutput with rendered image and components
        """
        if albedo.ndim != 3 or albedo.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) albedo, got {albedo.shape}")

        start_time = time.time()

        # Default lighting
        if lighting is None:
            lighting = LightingModel()

        # Default view direction (camera looking at face)
        if view_direction is None:
            view_direction = np.array([0.0, 0.0, 1.0])
        else:
            view_direction = np.array(view_direction, dtype=np.float64)
            norm = np.linalg.norm(view_direction)
            if norm > 0:
                view_direction = view_direction / norm
            else:
                view_direction = np.array([0.0, 0.0, 1.0])

        H, W, _ = albedo.shape

        # Compute lighting components
        ambient = self._compute_ambient(albedo, lighting, shading)
        diffuse = self._compute_diffuse(albedo, normal_map, lighting, shading)
        specular = self._compute_specular(
            normal_map, lighting, view_direction, shading
        )

        # Combine components
        rendered = (
            self.config.ambient_weight * ambient
            + self.config.diffuse_weight * diffuse
            + self.config.specular_weight * specular
        )

        # Energy conservation: clamp output
        if self.config.clamp_output:
            rendered = np.clip(rendered, 0, 1)

        # Compute rendering error (self-reconstruction)
        reconstruction = (
            self.config.ambient_weight * ambient
            + self.config.diffuse_weight * diffuse
            + self.config.specular_weight * specular
        )
        rendering_error = float(np.mean(np.abs(rendered - reconstruction)))

        render_time = (time.time() - start_time) * 1000

        return PhysicalRenderOutput(
            rendered=rendered.astype(np.float32),
            diffuse_component=diffuse.astype(np.float32),
            specular_component=specular.astype(np.float32),
            ambient_component=ambient.astype(np.float32),
            rendering_error=rendering_error,
        )

    def _compute_ambient(
        self,
        albedo: np.ndarray,
        lighting: LightingModel,
        shading: np.ndarray,
    ) -> np.ndarray:
        """Compute ambient component.

        ambient = albedo * ambient_intensity * shading

        Args:
            albedo: Albedo map (H, W, 3)
            lighting: Lighting model
            shading: Shading map (H, W, 1)

        Returns:
            Ambient component (H, W, 3)
        """
        # Ambient = albedo * ambient_intensity
        ambient = albedo * lighting.ambient

        # Modulate by shading
        shading_3ch = np.repeat(shading, 3, axis=2)
        ambient = ambient * shading_3ch

        return ambient

    def _compute_diffuse(
        self,
        albedo: np.ndarray,
        normal_map: np.ndarray,
        lighting: LightingModel,
        shading: np.ndarray,
    ) -> np.ndarray:
        """Compute Lambertian diffuse component.

        diffuse = albedo * diffuse_intensity * max(0, N·L)

        Args:
            albedo: Albedo map (H, W, 3)
            normal_map: Normal map (H, W, 3), unit vectors
            lighting: Lighting model
            shading: Shading map (H, W, 1)

        Returns:
            Diffuse component (H, W, 3)
        """
        # N·L (dot product of normal and light direction)
        N_dot_L = np.sum(normal_map * lighting.diffuse_direction, axis=2)
        N_dot_L = np.maximum(N_dot_L, 0)  # Clamp negative

        # Diffuse = albedo * intensity * N·L
        diffuse = albedo * lighting.diffuse_intensity * N_dot_L[:, :, np.newaxis]

        # Modulate by shading
        shading_3ch = np.repeat(shading, 3, axis=2)
        diffuse = diffuse * shading_3ch

        return diffuse

    def _compute_specular(
        self,
        normal_map: np.ndarray,
        lighting: LightingModel,
        view_direction: np.ndarray,
        shading: np.ndarray,
    ) -> np.ndarray:
        """Compute Blinn-Phong specular component.

        H = normalize(L + V)
        specular = specular_power * max(0, N·H)^shininess

        Args:
            normal_map: Normal map (H, W, 3), unit vectors
            lighting: Lighting model
            view_direction: View direction (3,)
            shading: Shading map (H, W, 1)

        Returns:
            Specular component (H, W, 3)
        """
        # Half-vector: H = normalize(L + V)
        H = lighting.diffuse_direction + view_direction
        H_norm = np.linalg.norm(H)
        if H_norm > 0:
            H = H / H_norm
        else:
            H = np.array([0.0, 0.0, 1.0])

        # N·H (dot product of normal and half-vector)
        N_dot_H = np.sum(normal_map * H, axis=2)
        N_dot_H = np.maximum(N_dot_H, 0)  # Clamp negative

        # Specular = intensity * (N·H)^shininess
        specular_2d = lighting.specular_intensity * np.power(
            N_dot_H, self.config.shininess
        )

        # Expand to 3 channels
        specular = specular_2d[:, :, np.newaxis] * np.ones((1, 1, 3))

        # Modulate by shading
        shading_3ch = np.repeat(shading, 3, axis=2)
        specular = specular * shading_3ch

        return specular

    def render_with_intrinsic(
        self,
        intrinsic_components: 'IntrinsicComponents',
        lighting: Optional[LightingModel] = None,
        view_direction: Optional[np.ndarray] = None,
    ) -> PhysicalRenderOutput:
        """Render using intrinsic decomposition components.

        Args:
            intrinsic_components: IntrinsicComponents from decomposition
            lighting: Lighting model
            view_direction: View direction

        Returns:
            PhysicalRenderOutput
        """
        return self.render(
            albedo=intrinsic_components.albedo,
            normal_map=intrinsic_components.normal_map,
            shading=intrinsic_components.shading,
            lighting=lighting,
            view_direction=view_direction,
        )

    def compute_rendering_error(
        self,
        observed: np.ndarray,
        rendered: np.ndarray,
    ) -> float:
        """Compute rendering error: ||Y_observed - Y_rendered||.

        Args:
            observed: Observed image (H, W, 3)
            rendered: Rendered image (H, W, 3)

        Returns:
            Mean absolute error
        """
        return float(np.mean(np.abs(observed - rendered)))

    def compute_energy_conservation(
        self,
        albedo: np.ndarray,
        rendered: np.ndarray,
    ) -> float:
        """Compute energy conservation ratio.

        ratio = mean(rendered) / mean(albedo)

        Args:
            albedo: Albedo map (H, W, 3)
            rendered: Rendered image (H, W, 3)

        Returns:
            Energy conservation ratio [0, 1]
        """
        input_energy = np.mean(albedo)
        output_energy = np.mean(rendered)

        if input_energy > 0:
            return float(output_energy / input_energy)
        else:
            return 0.0
