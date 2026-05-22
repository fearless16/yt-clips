"""
Intrinsic Decomposition Module.

This module decomposes a face crop into intrinsic components:

- Albedo (A): identity-intrinsic, lighting-invariant reflectance
- Shading (S): smooth illumination field
- Specular: sparse view-dependent highlight residual
- Normal map: surface orientation estimate

Mathematical model:
    Y = A * S + specular

Architecture rules:
- Input should be treated as linear-light as early as possible.
- Normals should come from mesh geometry when available.
- Shading-gradient normals are only a fallback prior.
- Confidence and uncertainty are explicit outputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------

def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """
    Convert sRGB/BGR image data to linear-light float32 in [0, 1].

    This uses the standard sRGB transfer curve, not a crude power-law shortcut.
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0)
    mask = arr <= 0.04045
    lin = np.where(mask, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    return lin.astype(np.float32)


def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Convert linear-light float32 in [0, 1] back to uint8 sRGB/BGR."""
    lin = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    mask = lin <= 0.0031308
    srgb = np.where(mask, lin * 12.92, 1.055 * (lin ** (1.0 / 2.4)) - 0.055)
    return (np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _ensure_linear_image(img: np.ndarray) -> np.ndarray:
    """
    Ensure a 3-channel image is in linear-light float32 [0, 1].

    Accepts uint8 [0,255] or float images in either [0,1] or [0,255].
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image, got {arr.shape}")
    if arr.max(initial=0.0) > 1.5:
        return _srgb_to_linear(arr)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Configuration and output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DecompositionConfig:
    """Configuration for intrinsic decomposition."""

    # Higher values give smoother albedo estimates.
    albedo_smoothness: float = 0.5

    # Higher values give smoother shading estimates.
    shading_smoothness: float = 0.3

    # Threshold above which residual pixels may be considered specular.
    specular_threshold: float = 0.08

    # Minimum shading value to prevent divide-by-zero during albedo extraction.
    min_albedo: float = 0.01

    # Bilateral filter parameters for shading estimation.
    bilateral_sigma_spatial: float = 15.0
    bilateral_sigma_intensity: float = 0.1

    # Upper bound for how much specular energy we keep.
    max_specular_ratio: float = 0.30

    # Confidence floor for valid decomposition.
    confidence_threshold: float = 0.50


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
        }


# ---------------------------------------------------------------------------
# Main decomposer
# ---------------------------------------------------------------------------

class IntrinsicDecomposer:
    """
    Retinex-inspired intrinsic decomposition.

    The module is intentionally conservative:
    - use smooth shading as illumination prior,
    - extract albedo by division in linear-light space,
    - keep specular sparse,
    - use geometry-derived normals when available.
    """

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
        """
        Decompose a face image into intrinsic components.

        Args:
            image: Input image, uint8 or float, shape (H, W, 3)
            mesh_478: Optional MediaPipe 478-point mesh in 3D-ish coordinates
            warp_M: Optional forward similarity transform (source -> canonical),
                    shape (2, 3)

        Returns:
            IntrinsicComponents
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got {image.shape}")

        image_lin = _ensure_linear_image(image)
        start_time = time.time()

        # 1) Smooth shading estimate in linear-light space.
        shading = self._estimate_shading(image_lin)

        # 2) Retinex-style albedo.
        albedo = self._extract_albedo(image_lin, shading)

        # 3) Sparse specular residual.
        specular = self._compute_specular(image_lin, albedo, shading)

        # 4) Normals from mesh geometry if available, otherwise face-prior fallback.
        if (
            self.use_mesh_normals
            and mesh_478 is not None
            and warp_M is not None
            and np.asarray(mesh_478).ndim == 2
            and np.asarray(mesh_478).shape[0] >= 468
        ):
            try:
                normal_map = self._estimate_normals_from_mesh(mesh_478, warp_M, image_lin.shape[:2])
                self._normal_source = "mesh"
            except Exception:
                normal_map = self._estimate_normals_from_face_prior(shading)
                self._normal_source = "face_prior"
        else:
            normal_map = self._estimate_normals_from_face_prior(shading)
            self._normal_source = "face_prior"

        # 5) Confidence + uncertainty.
        confidence = self._compute_confidence(image_lin, albedo, shading, specular)
        reconstruction_error = self._compute_reconstruction_error(image_lin, albedo, shading, specular)

        albedo_uncertainty = self._compute_albedo_uncertainty(albedo, shading)
        shading_uncertainty = self._compute_shading_uncertainty(shading)
        specular_uncertainty = self._compute_specular_uncertainty(specular)

        decomposition_quality = self._compute_decomposition_quality(
            reconstruction_error=reconstruction_error,
            confidence=confidence,
            albedo_uncertainty=albedo_uncertainty,
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
        )

    # -----------------------------------------------------------------------
    # Geometry / normals
    # -----------------------------------------------------------------------

    def _estimate_normals_from_mesh(
        self,
        mesh_478: np.ndarray,
        warp_M: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Estimate normals from mesh geometry instead of shading gradients.

        This keeps normals tied to geometry, not photometry.
        """
        mesh_478 = np.asarray(mesh_478, dtype=np.float32)
        warp_M = np.asarray(warp_M, dtype=np.float32)

        if mesh_478.ndim != 2 or mesh_478.shape[1] < 3:
            raise ValueError(f"Expected mesh_478 with shape (N, 3+), got {mesh_478.shape}")
        if warp_M.shape != (2, 3):
            raise ValueError(f"Expected warp_M with shape (2, 3), got {warp_M.shape}")

        # Keep this as a narrow integration point so the decomposer
        # does not depend on any shading-derived fallback.
        from face_os.landmarks import mesh_normal_map  # local import to avoid cycles

        return mesh_normal_map(mesh_478, warp_M, target_shape[::-1])

    def _estimate_normals_from_face_prior(self, shading: np.ndarray) -> np.ndarray:
        """
        Deterministic face-prior normal map.

        This is a geometry prior, not a photometric derivative.
        It exists only as a fallback when mesh normals are not available.
        """
        h, w = shading.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        ry, rx = h * 0.45, w * 0.40

        yy, xx = np.ogrid[:h, :w]
        nx = (xx - cx) / max(rx, 1.0)
        ny = (yy - cy) / max(ry, 1.0)

        r2 = nx * nx + ny * ny
        nz = np.sqrt(np.maximum(0.0, 1.0 - r2))

        # Ellipsoid-ish normal field.
        normal_x = nx / max(rx, 1.0)
        normal_y = ny / max(ry, 1.0)
        normal_z = nz / max(min(rx, ry), 1.0)

        norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2) + 1e-8
        normal_map = np.stack([normal_x / norm, normal_y / norm, normal_z / norm], axis=2)
        return normal_map.astype(np.float32)

    # -----------------------------------------------------------------------
    # Shading / albedo / specular
    # -----------------------------------------------------------------------

    def _estimate_shading(self, image_lin: np.ndarray) -> np.ndarray:
        """
        Estimate smooth illumination from linear-light image.

        Uses a bilateral filter over luminance to preserve structure while
        removing local texture detail.
        """
        # Linear-light luminance from BGR input.
        b = image_lin[:, :, 0]
        g = image_lin[:, :, 1]
        r = image_lin[:, :, 2]
        gray = (0.0722 * b + 0.7152 * g + 0.2126 * r).astype(np.float32)

        try:
            shading_2d = cv2.bilateralFilter(
                gray,
                d=0,
                sigmaColor=max(self.config.bilateral_sigma_intensity, 1e-6),
                sigmaSpace=max(self.config.bilateral_sigma_spatial, 1e-6),
            )
        except Exception:
            sigma = max(self.config.bilateral_sigma_spatial / 5.0, 1e-6)
            ksize = max(3, int(sigma * 6) | 1)
            shading_2d = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

        shading_2d = np.clip(shading_2d, self.config.min_albedo, 1.0)
        return shading_2d[:, :, np.newaxis].astype(np.float32)

    def _extract_albedo(self, image_lin: np.ndarray, shading: np.ndarray) -> np.ndarray:
        """
        Extract albedo by division in linear-light space.

        A = Y / S
        """
        shading_3ch = np.repeat(np.clip(shading, self.config.min_albedo, 1.0), 3, axis=2)
        albedo = image_lin / (shading_3ch + 1e-8)

        # Light smoothing only; do not erase identity texture.
        sigma = max(self.config.albedo_smoothness * 2.0, 1e-6)
        ksize = max(3, int(sigma * 6) | 1)
        for c in range(3):
            albedo[:, :, c] = cv2.GaussianBlur(albedo[:, :, c], (ksize, ksize), sigma)

        return np.clip(albedo, 0.0, 1.0).astype(np.float32)

    def _compute_specular(
        self,
        image_lin: np.ndarray,
        albedo: np.ndarray,
        shading: np.ndarray,
    ) -> np.ndarray:
        """
        Compute sparse specular residual.

        Specular should be the positive residual left after diffuse reconstruction,
        not a broad hand-tuned correction field.
        """
        shading_3ch = np.repeat(shading, 3, axis=2)
        diffuse = albedo * shading_3ch
        residual = np.maximum(0.0, image_lin - diffuse)

        # Keep only the strong residuals.
        residual_mag = np.mean(residual, axis=2, keepdims=True)
        keep = residual_mag >= self.config.specular_threshold

        # Suppress noise in very dark or overexposed zones.
        mean_luma = np.mean(image_lin, axis=2, keepdims=True)
        keep = keep & (mean_luma >= 0.08) & (mean_luma <= 0.95)

        specular = np.where(keep, residual, 0.0)

        # Respect a maximum specular energy budget.
        max_spec = np.clip(image_lin * self.config.max_specular_ratio, 0.0, 1.0)
        specular = np.minimum(specular, max_spec)
        return specular.astype(np.float32)

    # -----------------------------------------------------------------------
    # Confidence / uncertainty / quality
    # -----------------------------------------------------------------------

    def _compute_confidence(
        self,
        image_lin: np.ndarray,
        albedo: np.ndarray,
        shading: np.ndarray,
        specular: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-pixel confidence from reconstruction consistency.

        High confidence = low residual, smooth shading, stable albedo.
        """
        reconstructed = albedo * np.repeat(shading, 3, axis=2) + specular
        error = np.mean(np.abs(image_lin - reconstructed), axis=2, keepdims=True)

        # Map error into confidence in [0, 1].
        confidence = np.exp(-10.0 * error)

        # Slightly reduce confidence in regions with unstable shading.
        shading_grad_x = cv2.Sobel(shading[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
        shading_grad_y = cv2.Sobel(shading[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)
        shading_grad = np.sqrt(shading_grad_x**2 + shading_grad_y**2)[:, :, np.newaxis]
        confidence *= np.exp(-3.0 * np.clip(shading_grad, 0.0, 1.0))

        return np.clip(confidence, 0.0, 1.0).astype(np.float32)

    def _compute_reconstruction_error(
        self,
        image_lin: np.ndarray,
        albedo: np.ndarray,
        shading: np.ndarray,
        specular: np.ndarray,
    ) -> float:
        reconstructed = albedo * np.repeat(shading, 3, axis=2) + specular
        return float(np.mean(np.abs(image_lin - reconstructed)))

    def _compute_albedo_uncertainty(self, albedo: np.ndarray, shading: np.ndarray) -> np.ndarray:
        """
        Albedo uncertainty rises where illumination is weak or texture is unstable.
        """
        shading_uncertainty = 1.0 - np.clip(shading, 0.0, 1.0)

        albedo_gray = np.mean(albedo, axis=2).astype(np.float32)
        gx = cv2.Sobel(albedo_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(albedo_gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        edge_uncertainty = np.clip(gradient_mag * 5.0, 0.0, 1.0)

        uncertainty = np.maximum(shading_uncertainty[:, :, 0], edge_uncertainty)
        return uncertainty[:, :, np.newaxis].astype(np.float32)

    def _compute_shading_uncertainty(self, shading: np.ndarray) -> np.ndarray:
        """
        Shading uncertainty rises where the shading field is not smooth.
        """
        s = shading[:, :, 0].astype(np.float32)
        gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        uncertainty = np.clip(gradient_mag * 4.0, 0.0, 1.0)
        return uncertainty[:, :, np.newaxis].astype(np.float32)

    def _compute_specular_uncertainty(self, specular: np.ndarray) -> np.ndarray:
        """
        Specular uncertainty rises where specular energy is large and potentially ambiguous.
        """
        mag = np.linalg.norm(specular, axis=2)
        uncertainty = np.clip(mag * 2.0, 0.0, 1.0)
        return uncertainty[:, :, np.newaxis].astype(np.float32)

    def _compute_decomposition_quality(
        self,
        reconstruction_error: float,
        confidence: np.ndarray,
        albedo_uncertainty: np.ndarray,
    ) -> float:
        """
        Overall decomposition quality in [0, 1].
        """
        error_quality = 1.0 - min(reconstruction_error * 5.0, 1.0)
        confidence_quality = float(np.mean(confidence))
        uncertainty_quality = 1.0 - float(np.mean(albedo_uncertainty))

        quality = 0.4 * error_quality + 0.3 * confidence_quality + 0.3 * uncertainty_quality
        return float(np.clip(quality, 0.0, 1.0))

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def get_normal_source(self) -> str:
        """Return the last normal source used by the decomposer."""
        return self._normal_source

    def decompose_batch(self, images: list[np.ndarray]) -> list[IntrinsicComponents]:
        """Decompose a batch of images."""
        return [self.decompose(img) for img in images]

    def compute_albedo_stability(self, albedos: list[np.ndarray]) -> float:
        """
        Compute a simple temporal stability score for a sequence of albedo maps.
        """
        if len(albedos) < 2:
            return 1.0

        stacked = np.stack(albedos, axis=0)
        mean_albedo = np.mean(stacked, axis=0)
        std_albedo = np.std(stacked, axis=0)

        cv = np.mean(std_albedo) / (np.mean(mean_albedo) + 1e-8)
        stability = 1.0 - cv
        return float(np.clip(stability, 0.0, 1.0))