"""Intrinsic Decomposition Module — Beast Mode.

This version keeps the public API stable but aggressively improves
detail preservation and decomposition quality.

Goals:
- preserve high-frequency identity detail
- keep normals geometry-based whenever possible
- keep shading as a smooth illumination prior, not a texture killer
- expose confidence, uncertainty, and optional detail residuals
- stay pipeline-friendly and deterministic

Mathematical model:
    Y = A * S + specular

Where:
    A = albedo (identity-intrinsic, lighting-invariant reflectance)
    S = shading (smooth illumination field)
    specular = sparse view-dependent highlight residual

Beast-mode additions:
- detail_residual output for downstream reconstruction
- edge-aware albedo refinement
- less destructive smoothing
- stronger specular separation
- geometry-first normal estimation
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------

def _max_value(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.max(arr))


def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if _max_value(arr) > 1.5:
        arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0)
    mask = arr <= 0.04045
    lin = np.where(mask, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    return lin.astype(np.float32)


def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
    lin = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    mask = lin <= 0.0031308
    srgb = np.where(mask, lin * 12.92, 1.055 * (lin ** (1.0 / 2.4)) - 0.055)
    return (np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _ensure_linear_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image, got {arr.shape}")
    if _max_value(arr) > 1.5:
        return _srgb_to_linear(arr)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _luminance_linear(image_lin: np.ndarray) -> np.ndarray:
    b = image_lin[:, :, 0]
    g = image_lin[:, :, 1]
    r = image_lin[:, :, 2]
    return (0.0722 * b + 0.7152 * g + 0.2126 * r).astype(np.float32)


def _gaussian_blur_float(img: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if sigma <= 0:
        return arr
    k = int(max(3, 2 * round(3 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(
        arr,
        (k, k),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT101,
    )


def _edge_strength_mask(img: np.ndarray) -> np.ndarray:
    lum = _luminance_linear(_ensure_linear_image(img))
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mx = _max_value(mag)
    if mx <= 1e-8:
        return np.full_like(mag, 0.35, dtype=np.float32)
    mag = mag / mx
    return np.clip(0.30 + 0.70 * mag, 0.30, 1.0).astype(np.float32)


def _normalize_map_to_unit_vectors(normal_map: np.ndarray) -> np.ndarray:
    arr = np.asarray(normal_map, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) normal map, got {arr.shape}")
    nrm = np.linalg.norm(arr, axis=2, keepdims=True)
    nrm = np.where(nrm > 1e-8, nrm, 1.0)
    return arr / nrm


# ---------------------------------------------------------------------------
# Configuration and output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DecompositionConfig:
    """Configuration for intrinsic decomposition."""
    albedo_smoothness: float = 0.25
    shading_smoothness: float = 0.25
    specular_threshold: float = 0.05
    min_albedo: float = 0.02
    bilateral_sigma_spatial: float = 9.0
    bilateral_sigma_intensity: float = 0.06
    max_specular_ratio: float = 0.35
    confidence_threshold: float = 0.50
    detail_preservation: float = 0.80
    observed_detail_mix: float = 0.20
    detail_sigma: float = 1.8
    preserve_edges: bool = True


@dataclass
class IntrinsicComponents:
    """Intrinsic components of a face image."""
    albedo: np.ndarray
    shading: np.ndarray
    specular: np.ndarray
    normal_map: np.ndarray
    confidence: np.ndarray
    reconstruction_error: float

    albedo_uncertainty: Optional[np.ndarray] = None
    shading_uncertainty: Optional[np.ndarray] = None
    specular_uncertainty: Optional[np.ndarray] = None
    decomposition_quality: float = 0.0

    detail_residual: Optional[np.ndarray] = None
    low_frequency_albedo: Optional[np.ndarray] = None
    normal_source: str = "face_prior"

    def reconstruct(self) -> np.ndarray:
        shading_3ch = np.repeat(np.clip(self.shading, 0.0, 1.0), 3, axis=2)
        recon = self.albedo * shading_3ch + self.specular
        if self.detail_residual is not None:
            recon = recon + self.detail_residual
        return np.clip(recon, 0.0, 1.0).astype(np.float32)


@dataclass
class DecompositionReport:
    """Per-frame intrinsic decomposition metrics."""
    frame_idx: int
    components: IntrinsicComponents
    albedo_stability: float
    shading_smoothness: float
    specular_sparsity: float
    decomposition_time_ms: float

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "albedo_stability": self.albedo_stability,
            "shading_smoothness": self.shading_smoothness,
            "specular_sparsity": self.specular_sparsity,
            "decomposition_time_ms": self.decomposition_time_ms,
            "reconstruction_error": self.components.reconstruction_error,
            "decomposition_quality": self.components.decomposition_quality,
            "normal_source": self.components.normal_source,
        }


# ---------------------------------------------------------------------------
# Main decomposer
# ---------------------------------------------------------------------------

class IntrinsicDecomposer:
    def __init__(
        self,
        config: Optional[DecompositionConfig] = None,
        use_mesh_normals: bool = True,
    ):
        self.config = config or DecompositionConfig()
        self.use_mesh_normals = use_mesh_normals
        self._normal_source = "face_prior"

    def decompose(
        self,
        image: np.ndarray,
        mesh_478: Optional[np.ndarray] = None,
        warp_M: Optional[np.ndarray] = None,
    ) -> IntrinsicComponents:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got {image.shape}")

        start_time = time.perf_counter()
        image_lin = _ensure_linear_image(image)

        shading = self._estimate_shading(image_lin)
        albedo, low_freq_albedo, detail_residual = self._extract_albedo(image_lin, shading)
        specular = self._compute_specular(image_lin, albedo, shading)

        if (
            self.use_mesh_normals
            and mesh_478 is not None
            and warp_M is not None
            and np.asarray(mesh_478).ndim == 2
            and np.asarray(mesh_478).shape[0] >= 468
        ):
            try:
                normal_map = self._estimate_normals_from_mesh(mesh_478, warp_M, image_lin.shape[:2])
                normal_source = "mesh"
                self._normal_source = "mesh"
            except Exception:
                normal_map = self._estimate_normals_from_face_prior(shading)
                normal_source = "face_prior"
                self._normal_source = "face_prior"
        else:
            normal_map = self._estimate_normals_from_face_prior(shading)
            normal_source = "face_prior"
            self._normal_source = "face_prior"

        confidence = self._compute_confidence(image_lin, albedo, shading, specular, detail_residual)
        reconstruction_error = self._compute_reconstruction_error(image_lin, albedo, shading, specular, detail_residual)

        albedo_uncertainty = self._compute_albedo_uncertainty(albedo, shading, detail_residual)
        shading_uncertainty = self._compute_shading_uncertainty(shading)
        specular_uncertainty = self._compute_specular_uncertainty(specular)

        decomposition_quality = self._compute_decomposition_quality(
            reconstruction_error=reconstruction_error,
            confidence=confidence,
            albedo_uncertainty=albedo_uncertainty,
            detail_residual=detail_residual,
        )

        return IntrinsicComponents(
            albedo=albedo.astype(np.float32),
            shading=shading.astype(np.float32),
            specular=specular.astype(np.float32),
            normal_map=normal_map.astype(np.float32),
            confidence=confidence.astype(np.float32),
            reconstruction_error=float(reconstruction_error),
            albedo_uncertainty=albedo_uncertainty.astype(np.float32),
            shading_uncertainty=shading_uncertainty.astype(np.float32),
            specular_uncertainty=specular_uncertainty.astype(np.float32),
            decomposition_quality=float(decomposition_quality),
            detail_residual=detail_residual.astype(np.float32) if detail_residual is not None else None,
            low_frequency_albedo=low_freq_albedo.astype(np.float32) if low_freq_albedo is not None else None,
            normal_source=normal_source,
        )

    def _estimate_normals_from_mesh(self, mesh_478: np.ndarray, warp_M: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        mesh_478 = np.asarray(mesh_478, dtype=np.float32)
        warp_M = np.asarray(warp_M, dtype=np.float32)

        if mesh_478.ndim != 2 or mesh_478.shape[1] < 3:
            raise ValueError(f"Expected mesh_478 with shape (N, 3+), got {mesh_478.shape}")
        if warp_M.shape != (2, 3):
            raise ValueError(f"Expected warp_M with shape (2, 3), got {warp_M.shape}")

        from face_os.landmarks import mesh_normal_map
        normal_map = mesh_normal_map(mesh_478, warp_M, target_shape[::-1])
        normal_map = _normalize_map_to_unit_vectors(normal_map)
        return normal_map

    @staticmethod
    @lru_cache(maxsize=4)
    def _get_cached_face_prior(h: int, w: int) -> np.ndarray:
        """BEAST MODE: Cache the static geometry math so we don't recompute grids every frame."""
        cy, cx = h / 2.0, w / 2.0
        ry, rx = h * 0.47, w * 0.42
        yy, xx = np.ogrid[:h, :w]
        nx = (xx - cx) / max(rx, 1.0)
        ny = (yy - cy) / max(ry, 1.0)
        r2 = nx * nx + ny * ny
        nz = np.sqrt(np.maximum(0.0, 1.0 - np.clip(r2, 0.0, 1.0)))
        normal_x = nx / max(rx, 1.0)
        normal_y = ny / max(ry, 1.0)
        normal_z = nz / max(min(rx, ry), 1.0)
        norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2) + 1e-8
        normal_map = np.stack([normal_x / norm, normal_y / norm, normal_z / norm], axis=2)
        return normal_map.astype(np.float32)

    def _estimate_normals_from_face_prior(self, shading: np.ndarray) -> np.ndarray:
        h, w = shading.shape[:2]
        return self._get_cached_face_prior(h, w)

    def _estimate_shading(self, image_lin: np.ndarray) -> np.ndarray:
        gray = _luminance_linear(image_lin)
        h, w = gray.shape[:2]
        
        # BEAST MODE FPS FIX: Downscale -> Bilateral -> Upscale
        scale = 0.25
        small = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        
        try:
            shading_small = cv2.bilateralFilter(
                small, d=0,
                sigmaColor=max(self.config.bilateral_sigma_intensity, 1e-6),
                sigmaSpace=max(self.config.bilateral_sigma_spatial * scale, 1e-6),
            )
        except Exception:
            sigma = max(self.config.bilateral_sigma_spatial * scale / 5.0, 1e-6)
            ksize = max(3, int(sigma * 6) | 1)
            shading_small = cv2.GaussianBlur(small, (ksize, ksize), sigma)
            
        shading_2d = cv2.resize(shading_small, (w, h), interpolation=cv2.INTER_LINEAR)

        lp_sigma = max(self.config.shading_smoothness * 12.0, 1.0)
        lp = _gaussian_blur_float(gray, lp_sigma)
        shading_2d = 0.65 * shading_2d + 0.35 * lp

        shading_2d = np.clip(shading_2d, self.config.min_albedo, 1.0)
        return shading_2d[:, :, np.newaxis].astype(np.float32)

    def _extract_albedo(self, image_lin: np.ndarray, shading: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shading_3ch = np.repeat(np.clip(shading, self.config.min_albedo, 1.0), 3, axis=2)
        albedo_raw = image_lin / (shading_3ch + 1e-8)
        albedo_raw = np.clip(albedo_raw, 0.0, 1.0).astype(np.float32)

        sigma = max(self.config.albedo_smoothness * 1.2, 0.35)
        # BEAST MODE: Native 3-channel blur, killed the python loop
        low_freq = _gaussian_blur_float(albedo_raw, sigma)

        high_freq = albedo_raw - low_freq

        edge_mask = _edge_strength_mask(image_lin) if self.config.preserve_edges else np.ones(albedo_raw.shape[:2], dtype=np.float32)
        detail_gain = float(np.clip(self.config.detail_preservation, 0.0, 1.0))

        albedo = low_freq + detail_gain * high_freq * edge_mask[:, :, np.newaxis]
        albedo = np.clip(albedo, 0.0, 1.0).astype(np.float32)

        if self.config.albedo_smoothness > 0.0:
            stabil_sigma = max(self.config.albedo_smoothness * 0.6, 0.25)
            # BEAST MODE: Native 3-channel blur, killed the python loop
            stabilized = _gaussian_blur_float(albedo, stabil_sigma)
            albedo = 0.80 * albedo + 0.20 * stabilized
            albedo = np.clip(albedo, 0.0, 1.0).astype(np.float32)

        detail_residual = high_freq * edge_mask[:, :, np.newaxis]
        detail_residual = detail_residual - np.mean(detail_residual, axis=(0, 1), keepdims=True)

        return albedo, low_freq.astype(np.float32), detail_residual.astype(np.float32)

    def _compute_specular(self, image_lin: np.ndarray, albedo: np.ndarray, shading: np.ndarray) -> np.ndarray:
        shading_3ch = np.repeat(shading, 3, axis=2)
        diffuse = albedo * shading_3ch
        residual = np.maximum(0.0, image_lin - diffuse)

        residual_mag = np.mean(residual, axis=2, keepdims=True)
        edge_mask = _edge_strength_mask(image_lin)[:, :, np.newaxis]

        luma = np.mean(image_lin, axis=2, keepdims=True)
        adaptive_thr = self.config.specular_threshold * (0.75 + 0.25 * np.clip(luma, 0.0, 1.0))
        keep = (residual_mag >= adaptive_thr) & (edge_mask >= 0.35) & (luma >= 0.04) & (luma <= 0.96)

        specular = np.where(keep, residual, 0.0)

        max_spec = np.clip(image_lin * self.config.max_specular_ratio, 0.0, 1.0)
        specular = np.minimum(specular, max_spec)

        if self.config.shading_smoothness > 0.0:
            sigma = max(self.config.shading_smoothness * 0.8, 0.4)
            # BEAST MODE: Native 3-channel blur, killed the python loop
            spec_smooth = _gaussian_blur_float(specular, sigma)
            specular = 0.85 * specular + 0.15 * spec_smooth

        return specular.astype(np.float32)

    def _compute_confidence(self, image_lin: np.ndarray, albedo: np.ndarray, shading: np.ndarray, specular: np.ndarray, detail_residual: np.ndarray) -> np.ndarray:
        reconstructed = albedo * np.repeat(shading, 3, axis=2) + specular + detail_residual
        error = np.mean(np.abs(image_lin - reconstructed), axis=2, keepdims=True)
        confidence = np.exp(-12.0 * error)

        shading_grad_x = cv2.Sobel(shading[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
        shading_grad_y = cv2.Sobel(shading[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)
        shading_grad = np.sqrt(shading_grad_x**2 + shading_grad_y**2)[:, :, np.newaxis]
        confidence *= np.exp(-2.5 * np.clip(shading_grad, 0.0, 1.0))

        detail_mag = np.mean(np.abs(detail_residual), axis=2, keepdims=True)
        confidence *= np.exp(-2.0 * np.clip(detail_mag, 0.0, 1.0))

        return np.clip(confidence, 0.0, 1.0).astype(np.float32)

    def _compute_reconstruction_error(self, image_lin: np.ndarray, albedo: np.ndarray, shading: np.ndarray, specular: np.ndarray, detail_residual: np.ndarray) -> float:
        reconstructed = albedo * np.repeat(shading, 3, axis=2) + specular + detail_residual
        return float(np.mean(np.abs(image_lin - reconstructed)))

    def _compute_albedo_uncertainty(self, albedo: np.ndarray, shading: np.ndarray, detail_residual: np.ndarray) -> np.ndarray:
        shading_uncertainty = 1.0 - np.clip(shading, 0.0, 1.0)
        albedo_gray = np.mean(albedo, axis=2).astype(np.float32)
        gx = cv2.Sobel(albedo_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(albedo_gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        edge_uncertainty = np.clip(gradient_mag * 4.0, 0.0, 1.0)
        detail_mag = np.mean(np.abs(detail_residual), axis=2)
        detail_uncertainty = np.clip(detail_mag * 3.0, 0.0, 1.0)
        uncertainty = np.maximum(shading_uncertainty[:, :, 0], edge_uncertainty)
        uncertainty = np.maximum(uncertainty, detail_uncertainty)
        return uncertainty[:, :, np.newaxis].astype(np.float32)

    def _compute_shading_uncertainty(self, shading: np.ndarray) -> np.ndarray:
        s = shading[:, :, 0].astype(np.float32)
        gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        uncertainty = np.clip(gradient_mag * 3.5, 0.0, 1.0)
        return uncertainty[:, :, np.newaxis].astype(np.float32)

    def _compute_specular_uncertainty(self, specular: np.ndarray) -> np.ndarray:
        mag = np.linalg.norm(specular, axis=2)
        uncertainty = np.clip(mag * 1.8, 0.0, 1.0)
        return uncertainty[:, :, np.newaxis].astype(np.float32)

    def _compute_decomposition_quality(self, reconstruction_error: float, confidence: np.ndarray, albedo_uncertainty: np.ndarray, detail_residual: np.ndarray) -> float:
        error_quality = 1.0 - min(reconstruction_error * 5.0, 1.0)
        confidence_quality = float(np.mean(confidence))
        uncertainty_quality = 1.0 - float(np.mean(albedo_uncertainty))
        detail_mag = float(np.mean(np.abs(detail_residual)))
        detail_quality = 1.0 - min(detail_mag * 0.8, 1.0)
        quality = (0.35 * error_quality + 0.30 * confidence_quality + 0.20 * uncertainty_quality + 0.15 * detail_quality)
        return float(np.clip(quality, 0.0, 1.0))

    def get_normal_source(self) -> str:
        return self._normal_source

    def decompose_batch(self, images: list[np.ndarray]) -> list[IntrinsicComponents]:
        return [self.decompose(img) for img in images]

    def compute_albedo_stability(self, albedos: list[np.ndarray]) -> float:
        if len(albedos) < 2:
            return 1.0
        stacked = np.stack(albedos, axis=0)
        mean_albedo = np.mean(stacked, axis=0)
        std_albedo = np.std(stacked, axis=0)
        cv = np.mean(std_albedo) / (np.mean(mean_albedo) + 1e-8)
        stability = 1.0 - cv
        return float(np.clip(stability, 0.0, 1.0))

    def compute_high_frequency_retention(self, reference: np.ndarray, reconstructed: np.ndarray) -> float:
        ref = _ensure_linear_image(reference)
        out = _ensure_linear_image(reconstructed)
        def lap_var(img: np.ndarray) -> float:
            gray = _luminance_linear(img).astype(np.float32)
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            return float(np.var(lap))
        ref_hf = lap_var(ref)
        out_hf = lap_var(out)
        if ref_hf <= 1e-8:
            return 0.0
        return float(out_hf / ref_hf)