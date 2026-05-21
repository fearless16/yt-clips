
"""Physically-inspired renderer module for Face OS.

This version keeps the public API but fixes the most damaging drift:

- removes per-pixel shading re-multiplication of all terms
- uses shading only as a global irradiance prior, not a second lighting pass
- computes rendering error against an optional observed target (or zero if absent)
- keeps output in linear-light float space until final consumer converts if needed
- preserves a clean separation between albedo, normals, lighting, and diagnostics

NOTE:
This is still physically-inspired, not full physically-based rendering.
It is intended to be stable, testable, and pipeline-friendly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _as_float_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image, got {arr.shape}")
    return np.clip(arr, 0.0, 1.0) if arr.max(initial=0.0) <= 1.5 else np.clip(arr / 255.0, 0.0, 1.0)


def _normalize_vec(v: np.ndarray, default: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(arr))
    if n <= 1e-8:
        if default is None:
            default = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return default.astype(np.float64)
    return (arr / n).astype(np.float64)


def _ensure_normal_map(normal_map: np.ndarray) -> np.ndarray:
    arr = np.asarray(normal_map, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) normal map, got {arr.shape}")
    nrm = np.linalg.norm(arr, axis=2, keepdims=True)
    nrm = np.where(nrm > 1e-8, nrm, 1.0)
    return arr / nrm


def _ensure_shading(shading: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(shading, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim != 3 or arr.shape[2] not in (1, 3):
        raise ValueError(f"Expected shading map (H, W, 1/3), got {arr.shape}")
    if arr.shape[:2] != shape_hw:
        raise ValueError(f"Shading shape {arr.shape[:2]} does not match expected {shape_hw}")
    if arr.shape[2] == 3:
        arr = np.mean(arr, axis=2, keepdims=True)
    return np.clip(arr, 0.0, 1.0)


@dataclass
class LightingModel:
    """Lighting model configuration."""

    ambient: float = 0.1
    diffuse_direction: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.3, 0.9], dtype=np.float64))
    diffuse_intensity: float = 0.8
    specular_power: float = 32.0
    specular_intensity: float = 0.3
    spherical_harmonics: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.float32))

    def __post_init__(self):
        self.diffuse_direction = _normalize_vec(self.diffuse_direction)
        self.ambient = max(0.0, float(self.ambient))
        self.diffuse_intensity = max(0.0, float(self.diffuse_intensity))
        self.specular_power = max(0.0, float(self.specular_power))
        self.specular_intensity = max(0.0, float(self.specular_intensity))


@dataclass
class PhysicalRenderConfig:
    """Configuration for the physically-inspired renderer."""

    diffuse_weight: float = 0.7
    specular_weight: float = 0.2
    ambient_weight: float = 0.1
    energy_conservation_limit: float = 0.95
    shininess: float = 32.0
    use_spherical_harmonics: bool = False
    clamp_output: bool = True


@dataclass
class PhysicalRenderOutput:
    """Physical rendering output."""

    rendered: np.ndarray
    diffuse_component: np.ndarray
    specular_component: np.ndarray
    ambient_component: np.ndarray
    rendering_error: float
    render_time_ms: float = 0.0


@dataclass
class RenderReport:
    """Per-frame render metrics."""

    frame_idx: int
    output: PhysicalRenderOutput
    render_time_ms: float

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "render_time_ms": self.render_time_ms,
            "rendering_error": self.output.rendering_error,
            "mean_diffuse": float(np.mean(self.output.diffuse_component)),
            "mean_specular": float(np.mean(self.output.specular_component)),
            "mean_ambient": float(np.mean(self.output.ambient_component)),
        }


class PhysicalRenderer:
    """Physically-inspired renderer.

    Rendering equation (approximate):
        Y = ambient + diffuse + specular

    Important correction:
    - shading is NOT multiplied into every term pixel-by-pixel.
    - shading is used as a global irradiance prior so the renderer
      does not double-apply illumination from the decomposition stage.
    """

    def __init__(self, config: Optional[PhysicalRenderConfig] = None):
        self.config = config or PhysicalRenderConfig()
        self._last_report: Optional[RenderReport] = None

    def render(
        self,
        albedo: np.ndarray,
        normal_map: np.ndarray,
        shading: np.ndarray,
        lighting: Optional[LightingModel] = None,
        view_direction: Optional[np.ndarray] = None,
        observed: Optional[np.ndarray] = None,
        frame_idx: Optional[int] = None,
    ) -> PhysicalRenderOutput:
        """Render face with a physically-inspired lighting model."""

        start_time = time.perf_counter()

        albedo = _as_float_image(albedo)
        normal_map = _ensure_normal_map(normal_map)
        shading = _ensure_shading(shading, albedo.shape[:2])

        if albedo.shape[:2] != normal_map.shape[:2]:
            raise ValueError(
                f"Albedo shape {albedo.shape[:2]} does not match normal map shape {normal_map.shape[:2]}"
            )

        if lighting is None:
            lighting = LightingModel()

        if view_direction is None:
            view_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        view_direction = _normalize_vec(view_direction)

        # Global irradiance prior from shading, not per-pixel re-lighting.
        irradiance = float(np.mean(shading))
        irradiance = float(np.clip(irradiance, 0.25, 1.50))

        ambient = self._compute_ambient(albedo, lighting, irradiance)
        diffuse = self._compute_diffuse(albedo, normal_map, lighting, irradiance)
        specular = self._compute_specular(normal_map, lighting, view_direction, irradiance)

        rendered = (
            self.config.ambient_weight * ambient
            + self.config.diffuse_weight * diffuse
            + self.config.specular_weight * specular
        )

        if self.config.clamp_output:
            rendered = np.clip(rendered, 0.0, 1.0)

        rendered = self._apply_energy_conservation(albedo, rendered)

        if observed is not None:
            observed_f = _as_float_image(observed)
            if observed_f.shape != rendered.shape:
                raise ValueError(
                    f"Observed image shape {observed_f.shape} does not match rendered shape {rendered.shape}"
                )
            rendering_error = self.compute_rendering_error(observed_f, rendered)
        else:
            rendering_error = 0.0

        render_time_ms = (time.perf_counter() - start_time) * 1000.0

        output = PhysicalRenderOutput(
            rendered=rendered.astype(np.float32),
            diffuse_component=diffuse.astype(np.float32),
            specular_component=specular.astype(np.float32),
            ambient_component=ambient.astype(np.float32),
            rendering_error=float(rendering_error),
            render_time_ms=float(render_time_ms),
        )

        if frame_idx is not None:
            self._last_report = RenderReport(
                frame_idx=int(frame_idx),
                output=output,
                render_time_ms=float(render_time_ms),
            )

        return output

    def _compute_ambient(
        self,
        albedo: np.ndarray,
        lighting: LightingModel,
        irradiance: float,
    ) -> np.ndarray:
        """Compute ambient component.

        Ambient is a global term; it should not re-apply the full shading map.
        """
        ambient = albedo * lighting.ambient * irradiance

        if lighting.spherical_harmonics is not None and self.config.use_spherical_harmonics:
            # Soft low-order bias only, not a second lighting field.
            sh = np.asarray(lighting.spherical_harmonics, dtype=np.float32).reshape(-1)
            sh_scale = float(np.clip(np.mean(np.abs(sh)), 0.0, 1.0))
            ambient = ambient * (0.9 + 0.1 * sh_scale)

        return ambient

    def _compute_diffuse(
        self,
        albedo: np.ndarray,
        normal_map: np.ndarray,
        lighting: LightingModel,
        irradiance: float,
    ) -> np.ndarray:
        """Compute Lambertian diffuse component."""
        N_dot_L = np.sum(normal_map * lighting.diffuse_direction[np.newaxis, np.newaxis, :], axis=2)
        N_dot_L = np.maximum(N_dot_L, 0.0)
        diffuse = albedo * lighting.diffuse_intensity * N_dot_L[:, :, np.newaxis] * irradiance
        return diffuse

    def _compute_specular(
        self,
        normal_map: np.ndarray,
        lighting: LightingModel,
        view_direction: np.ndarray,
        irradiance: float,
    ) -> np.ndarray:
        """Compute Blinn-Phong specular component."""
        half_vec = _normalize_vec(lighting.diffuse_direction + view_direction)
        N_dot_H = np.sum(normal_map * half_vec[np.newaxis, np.newaxis, :], axis=2)
        N_dot_H = np.maximum(N_dot_H, 0.0)

        shininess = float(self.config.shininess if self.config.shininess > 0 else lighting.specular_power)
        spec_scalar = lighting.specular_intensity * np.power(N_dot_H, shininess) * irradiance
        specular = np.repeat(spec_scalar[:, :, np.newaxis], 3, axis=2)
        return specular

    def _apply_energy_conservation(self, albedo: np.ndarray, rendered: np.ndarray) -> np.ndarray:
        """Clamp/suppress output when output energy exceeds the configured limit."""
        input_energy = float(np.mean(albedo))
        output_energy = float(np.mean(rendered))

        if input_energy <= 1e-8 or output_energy <= 1e-8:
            return rendered

        ratio = output_energy / input_energy
        if ratio > self.config.energy_conservation_limit:
            scale = self.config.energy_conservation_limit / ratio
            rendered = rendered * scale

        if self.config.clamp_output:
            rendered = np.clip(rendered, 0.0, 1.0)
        return rendered

    def render_with_intrinsic(
        self,
        intrinsic_components: "IntrinsicComponents",
        lighting: Optional[LightingModel] = None,
        view_direction: Optional[np.ndarray] = None,
        observed: Optional[np.ndarray] = None,
        frame_idx: Optional[int] = None,
    ) -> PhysicalRenderOutput:
        """Render using intrinsic decomposition components."""
        return self.render(
            albedo=intrinsic_components.albedo,
            normal_map=intrinsic_components.normal_map,
            shading=intrinsic_components.shading,
            lighting=lighting,
            view_direction=view_direction,
            observed=observed,
            frame_idx=frame_idx,
        )

    def compute_rendering_error(self, observed: np.ndarray, rendered: np.ndarray) -> float:
        """Compute mean absolute error against an observed frame."""
        observed_f = _as_float_image(observed)
        rendered_f = _as_float_image(rendered)
        if observed_f.shape != rendered_f.shape:
            raise ValueError(
                f"Observed image shape {observed_f.shape} does not match rendered shape {rendered_f.shape}"
            )
        return float(np.mean(np.abs(observed_f - rendered_f)))

    def compute_energy_conservation(self, albedo: np.ndarray, rendered: np.ndarray) -> float:
        """Compute output/input energy ratio."""
        albedo_f = _as_float_image(albedo)
        rendered_f = _as_float_image(rendered)
        input_energy = float(np.mean(albedo_f))
        output_energy = float(np.mean(rendered_f))
        if input_energy <= 1e-8:
            return 0.0
        return float(output_energy / input_energy)

    def get_last_report(self) -> Optional[RenderReport]:
        """Return the last per-frame report if available."""
        return self._last_report
