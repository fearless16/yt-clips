"""Physically-inspired renderer module for Face OS.

Beast-mode fix goals:
- keep the public API stable
- remove shading scalar-collapse as the only lighting prior
- preserve a high-frequency residual path
- keep energy conservation on the *base* render, not the whole image
- keep output in linear-light float space until final conversion
- stay stable, testable, and pipeline-friendly

This is still physically-inspired, not full physically-based rendering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


def _as_float_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
        
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got {arr.ndim}D with shape {arr.shape}")
        
    # Safeguard against accidental concatenation with high-dim feature maps
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
        
    if arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image, got {arr.shape}")
        
    if arr.size == 0:
        raise ValueError("Empty image")
        
    mx = float(np.max(arr))
    if mx <= 1.5:
        return np.clip(arr, 0.0, 1.0)
    return np.clip(arr / 255.0, 0.0, 1.0)


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
    
    # Handle 1D embeddings or weird dimensions
    if arr.ndim == 1:
        arr = np.full((shape_hw[0], shape_hw[1], 1), np.mean(arr) if arr.size > 0 else 0.0, dtype=np.float32)
    elif arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    elif arr.ndim > 3:
        arr = np.squeeze(arr)
        if arr.ndim == 2: arr = arr[:, :, np.newaxis]
        elif arr.ndim != 3: arr = np.zeros((shape_hw[0], shape_hw[1], 1), dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected shading map (H, W, C), got {arr.shape}")

    # KILL THE 256-CHANNEL BULLSHIT
    if arr.shape[2] > 3:
        arr = np.mean(arr, axis=2, keepdims=True)
        
    if arr.shape[:2] != shape_hw:
        arr = cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)
        if arr.ndim == 2: arr = arr[:, :, np.newaxis]
        
    if arr.shape[2] == 3:
        arr = np.mean(arr, axis=2, keepdims=True)
        
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _gaussian_blur_float(img: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.ascontiguousarray(img, dtype=np.float32)
    if sigma <= 0:
        return arr
    k = int(max(3, 2 * round(3 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    out = cv2.GaussianBlur(arr, (k, k), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT101)
    # OpenCV squeezes single-channel input — restore 3D if needed
    if out.ndim < arr.ndim:
        out = out[..., np.newaxis]
    return out


def _luminance(img: np.ndarray) -> np.ndarray:
    arr = _as_float_image(img)
    return np.mean(arr, axis=2).astype(np.float32)


def _high_frequency_component(img: np.ndarray, sigma: float) -> np.ndarray:
    arr = _as_float_image(img)
    low = _gaussian_blur_float(arr, sigma)
    return arr - low


def _edge_strength_mask(img: np.ndarray) -> np.ndarray:
    lum = _luminance(img)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mx = float(np.max(mag)) if mag.size else 0.0
    if mx <= 1e-8:
        return np.full_like(mag, 0.25, dtype=np.float32)
    mag = mag / mx
    return np.clip(0.25 + 0.75 * mag, 0.25, 1.0).astype(np.float32)


@dataclass
class LightingModel:
    """Lighting model configuration."""

    ambient: float = 0.10
    diffuse_direction: np.ndarray = field(
        default_factory=lambda: np.array([0.3, 0.3, 0.9], dtype=np.float64)
    )
    diffuse_intensity: float = 0.80
    specular_power: float = 32.0
    specular_intensity: float = 0.30
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

    diffuse_weight: float = 0.70
    specular_weight: float = 0.20
    ambient_weight: float = 0.10

    energy_conservation_limit: float = 0.95
    clamp_output: bool = True

    detail_strength: float = 0.65
    detail_sigma: float = 2.0
    observed_detail_mix: float = 0.20
    use_detail_residual: bool = True
    preserve_edges: bool = True

    shininess: float = 32.0
    use_spherical_harmonics: bool = False


@dataclass
class PhysicalRenderOutput:
    """Physical rendering output."""

    rendered: np.ndarray
    diffuse_component: np.ndarray
    specular_component: np.ndarray
    ambient_component: np.ndarray
    detail_component: np.ndarray
    rendering_error: float
    high_frequency_retention: float = 0.0
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
            "high_frequency_retention": self.output.high_frequency_retention,
            "mean_diffuse": float(np.mean(self.output.diffuse_component)),
            "mean_specular": float(np.mean(self.output.specular_component)),
            "mean_ambient": float(np.mean(self.output.ambient_component)),
            "mean_detail": float(np.mean(self.output.detail_component)),
        }


class PhysicalRenderer:
    """Physically-inspired renderer with HF detail preservation."""

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

        irradiance = float(np.mean(shading))
        irradiance = float(np.clip(irradiance, 0.25, 1.50))

        ambient = self._compute_ambient(albedo, lighting, irradiance)
        diffuse = self._compute_diffuse(albedo, normal_map, lighting, irradiance)
        specular = self._compute_specular(normal_map, lighting, view_direction, irradiance)

        base_render = (
            self.config.ambient_weight * ambient
            + self.config.diffuse_weight * diffuse
            + self.config.specular_weight * specular
        )

        base_render = self._apply_energy_conservation(albedo, base_render)

        detail = self._compute_detail_component(
            albedo=albedo,
            shading=shading,
            observed=observed,
            normal_map=normal_map,
        )

        if self.config.use_detail_residual:
            detail_mask = self._compute_detail_mask(albedo, normal_map) if self.config.preserve_edges else 1.0
            rendered = base_render + self.config.detail_strength * detail_mask[..., np.newaxis] * detail
        else:
            rendered = base_render

        if self.config.clamp_output:
            rendered = np.clip(rendered, 0.0, 1.0)

        if observed is not None:
            observed_f = _as_float_image(observed)
            if observed_f.shape != rendered.shape:
                raise ValueError(
                    f"Observed image shape {observed_f.shape} does not match rendered shape {rendered.shape}"
                )
            rendering_error = self.compute_rendering_error(observed_f, rendered)
            reference_for_hf = observed_f
        else:
            rendering_error = 0.0
            reference_for_hf = albedo

        hf_retention = self._compute_high_frequency_retention(reference_for_hf, rendered)

        render_time_ms = (time.perf_counter() - start_time) * 1000.0

        output = PhysicalRenderOutput(
            rendered=rendered.astype(np.float32),
            diffuse_component=diffuse.astype(np.float32),
            specular_component=specular.astype(np.float32),
            ambient_component=ambient.astype(np.float32),
            detail_component=detail.astype(np.float32),
            rendering_error=float(rendering_error),
            high_frequency_retention=float(hf_retention),
            render_time_ms=float(render_time_ms),
        )

        if frame_idx is not None:
            self._last_report = RenderReport(
                frame_idx=int(frame_idx),
                output=output,
                render_time_ms=float(render_time_ms),
            )

        return output

    def _compute_ambient(self, albedo: np.ndarray, lighting: LightingModel, irradiance: float) -> np.ndarray:
        ambient = albedo * lighting.ambient * irradiance
        if lighting.spherical_harmonics is not None and self.config.use_spherical_harmonics:
            sh = np.asarray(lighting.spherical_harmonics, dtype=np.float32).reshape(-1)
            sh_scale = float(np.clip(np.mean(np.abs(sh)), 0.0, 1.0))
            ambient = ambient * (0.9 + 0.1 * sh_scale)
        return ambient

    def _compute_diffuse(self, albedo: np.ndarray, normal_map: np.ndarray, lighting: LightingModel, irradiance: float) -> np.ndarray:
        N_dot_L = np.sum(normal_map * lighting.diffuse_direction[np.newaxis, np.newaxis, :], axis=2)
        N_dot_L = np.maximum(N_dot_L, 0.0)
        diffuse = albedo * lighting.diffuse_intensity * N_dot_L[:, :, np.newaxis] * irradiance
        return diffuse

    def _compute_specular(self, normal_map: np.ndarray, lighting: LightingModel, view_direction: np.ndarray, irradiance: float) -> np.ndarray:
        half_vec = _normalize_vec(lighting.diffuse_direction + view_direction)
        N_dot_H = np.sum(normal_map * half_vec[np.newaxis, np.newaxis, :], axis=2)
        N_dot_H = np.maximum(N_dot_H, 0.0)
        shininess = float(self.config.shininess if self.config.shininess > 0 else lighting.specular_power)
        spec_scalar = lighting.specular_intensity * np.power(N_dot_H, shininess) * irradiance
        specular = np.repeat(spec_scalar[:, :, np.newaxis], 3, axis=2)
        return specular

    def _compute_detail_component(self, albedo: np.ndarray, shading: np.ndarray, observed: Optional[np.ndarray], normal_map: np.ndarray) -> np.ndarray:
        detail_sigma = max(0.5, float(self.config.detail_sigma))
        albedo_hp = _high_frequency_component(albedo, detail_sigma)

        # BULLETPROOF SHADING SANITIZATION (Kills the 256->768 broadcast nuke)
        shd = np.asarray(shading, dtype=np.float32)
        if shd.ndim == 1:
            shd = np.full((albedo.shape[0], albedo.shape[1], 1), np.mean(shd) if shd.size > 0 else 0.0, dtype=np.float32)
        elif shd.ndim == 2:
            shd = shd[:, :, np.newaxis]
            
        if shd.ndim == 3 and shd.shape[2] > 1:
            shd = np.mean(shd, axis=2, keepdims=True)
        elif shd.ndim != 3:
            shd = np.zeros((albedo.shape[0], albedo.shape[1], 1), dtype=np.float32)
            
        if shd.shape[:2] != albedo.shape[:2]:
            shd = cv2.resize(shd, (albedo.shape[1], albedo.shape[0]), interpolation=cv2.INTER_LINEAR)
            shd = shd[:, :, np.newaxis]

        shading_lp = _gaussian_blur_float(shd, max(1.5, detail_sigma * 2.5))
        shading_residual = np.repeat(shd - shading_lp, 3, axis=2)

        detail = albedo_hp + 0.15 * shading_residual

        if observed is not None:
            observed_f = _as_float_image(observed)
            observed_hp = _high_frequency_component(observed_f, detail_sigma)
            detail = (1.0 - self.config.observed_detail_mix) * detail + self.config.observed_detail_mix * observed_hp

        edge_mask = self._compute_detail_mask(albedo, normal_map)[..., np.newaxis]
        detail = detail * edge_mask

        detail = detail - np.mean(detail, axis=(0, 1), keepdims=True)
        return detail.astype(np.float32)

    def _compute_detail_mask(self, albedo: np.ndarray, normal_map: np.ndarray) -> np.ndarray:
        edge = _edge_strength_mask(albedo)
        n_blur = _gaussian_blur_float(normal_map, 1.5)
        n_var = np.linalg.norm(normal_map - n_blur, axis=2)
        n_max = float(np.max(n_var)) if n_var.size else 0.0
        if n_max > 1e-8:
            n_var = n_var / n_max
        else:
            n_var = np.zeros_like(edge, dtype=np.float32)
        mask = 0.55 * edge + 0.45 * n_var
        mask = np.clip(mask, 0.20, 1.0).astype(np.float32)
        return mask

    def _apply_energy_conservation(self, albedo: np.ndarray, rendered: np.ndarray) -> np.ndarray:
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

    def render_with_intrinsic(self, intrinsic_components: "IntrinsicComponents", lighting: Optional[LightingModel] = None, view_direction: Optional[np.ndarray] = None, observed: Optional[np.ndarray] = None, frame_idx: Optional[int] = None) -> PhysicalRenderOutput:
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
        observed_f = _as_float_image(observed)
        rendered_f = _as_float_image(rendered)
        if observed_f.shape != rendered_f.shape:
            raise ValueError(f"Observed image shape {observed_f.shape} does not match rendered shape {rendered_f.shape}")
        return float(np.mean(np.abs(observed_f - rendered_f)))

    def compute_energy_conservation(self, albedo: np.ndarray, rendered: np.ndarray) -> float:
        albedo_f = _as_float_image(albedo)
        rendered_f = _as_float_image(rendered)
        input_energy = float(np.mean(albedo_f))
        output_energy = float(np.mean(rendered_f))
        if input_energy <= 1e-8:
            return 0.0
        return float(output_energy / input_energy)

    def _compute_high_frequency_retention(self, reference: np.ndarray, rendered: np.ndarray) -> float:
        ref = _as_float_image(reference)
        out = _as_float_image(rendered)
        def lap_var(img: np.ndarray) -> float:
            gray = np.mean(img, axis=2).astype(np.float32)
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            return float(np.var(lap))
        ref_hf = lap_var(ref)
        out_hf = lap_var(out)
        if ref_hf <= 1e-8:
            return 0.0
        return float(out_hf / ref_hf)

    def render_with_mesh(self, albedo: np.ndarray, mesh_vertices: np.ndarray, mesh_faces: np.ndarray, shading: np.ndarray, lighting: Optional[LightingModel] = None, image_size: Optional[tuple] = None, view_direction: Optional[np.ndarray] = None, observed: Optional[np.ndarray] = None, frame_idx: Optional[int] = None) -> PhysicalRenderOutput:
        albedo = _as_float_image(albedo)
        h, w = albedo.shape[:2]
        if image_size is None:
            image_size = (h, w)
        if mesh_vertices is not None and mesh_faces is not None:
            normal_map = self._rasterize_mesh_normals(mesh_vertices, mesh_faces, image_size)
        else:
            normal_map = self._face_prior_normals(image_size)
        if normal_map.shape[:2] != albedo.shape[:2]:
            normal_map = cv2.resize(normal_map, (albedo.shape[1], albedo.shape[0]), interpolation=cv2.INTER_LINEAR)
            norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
            normal_map = normal_map / (norm + 1e-8)
        return self.render(
            albedo=albedo, normal_map=normal_map, shading=shading, lighting=lighting,
            view_direction=view_direction, observed=observed, frame_idx=frame_idx,
        )

    def _compute_per_face_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-8)
        return normals.astype(np.float32)

    def _rasterize_mesh_normals(self, vertices: np.ndarray, faces: np.ndarray, image_size: tuple) -> np.ndarray:
        h, w = image_size
        face_normals = self._compute_per_face_normals(vertices, faces)
        v_xy = vertices[:, :2].astype(np.float32)
        v_min = v_xy.min(axis=0)
        v_max = v_xy.max(axis=0)
        v_range = v_max - v_min
        v_range = np.where(v_range > 1e-8, v_range, 1.0)
        px = (v_xy[:, 0] - v_min[0]) / v_range[0] * (w - 1)
        py = (v_xy[:, 1] - v_min[1]) / v_range[1] * (h - 1)
        normal_map = np.zeros((h, w, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        weight_map = np.zeros((h, w), dtype=np.float32)
        for fi in range(len(faces)):
            face = faces[fi]
            fn = face_normals[fi]
            sx = px[face]
            sy = py[face]
            x_min = max(0, int(np.floor(np.min(sx))))
            x_max = min(w - 1, int(np.ceil(np.max(sx))))
            y_min = max(0, int(np.floor(np.min(sy))))
            y_max = min(h - 1, int(np.ceil(np.max(sy))))
            if x_max < x_min or y_max < y_min:
                continue
            yi, xi = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
            yi = yi.astype(np.float32)
            xi = xi.astype(np.float32)
            v0x, v0y = sx[0], sy[0]
            v1x, v1y = sx[1], sy[1]
            v2x, v2y = sx[2], sy[2]
            denom = (v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y)
            if abs(denom) < 1e-10:
                continue
            u = ((v1y - v2y) * (xi - v2x) + (v2x - v1x) * (yi - v2y)) / denom
            v = ((v2y - v0y) * (xi - v2x) + (v0x - v2x) * (yi - v2y)) / denom
            bw = 1.0 - u - v
            inside = (u >= 0) & (v >= 0) & (bw >= 0)
            if not np.any(inside):
                continue
            area = abs((v1x - v0x) * (v2y - v0y) - (v2x - v0x) * (v1y - v0y))
            weight = min(1.0 / (area + 1e-6), 10.0)
            mask_slice = np.zeros((h, w), dtype=bool)
            mask_slice[y_min:y_max + 1, x_min:x_max + 1] = inside
            normal_map[mask_slice] += fn * weight
            weight_map[mask_slice] += weight
        valid = weight_map > 0
        for c in range(3):
            normal_map[:, :, c][valid] /= weight_map[valid]
        norms = np.linalg.norm(normal_map, axis=2, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        normal_map /= norms
        return normal_map

    @staticmethod
    def _barycentric_coords(px: float, py: float, x0: float, y0: float, x1: float, y1: float, x2: float, y2: float) -> Optional[tuple]:
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-10:
            return None
        u = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        v = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        w = 1.0 - u - v
        return (u, v, w)

    def _face_prior_normals(self, image_size: tuple) -> np.ndarray:
        h, w = image_size
        y = np.linspace(-1, 1, h, dtype=np.float32)
        x = np.linspace(-1, 1, w, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        a, b = 1.2, 1.5
        r2 = (xx / a) ** 2 + (yy / b) ** 2
        r2 = np.clip(r2, 0, 1)
        z = np.sqrt(1 - r2)
        nx = xx / (a * a)
        ny = yy / (b * b)
        nz = z
        normal_map = np.stack([nx, ny, nz], axis=2)
        norms = np.linalg.norm(normal_map, axis=2, keepdims=True)
        normal_map = normal_map / (norms + 1e-8)
        return normal_map.astype(np.float32)

    def get_last_report(self) -> Optional[RenderReport]:
        return self._last_report
