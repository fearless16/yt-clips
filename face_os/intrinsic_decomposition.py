"""Intrinsic Decomposition Module.

Decomposes face images into intrinsic components:
- Albedo (A): identity-intrinsic, lighting-invariant
- Shading (S): lighting-dependent, identity-invariant
- Specular: view-dependent specular response
- Normal map: surface orientation

Mathematical Model:
    Y = A * S + specular

where:
    Y = observed image
    A = albedo (reflectance)
    S = shading (illumination)
    specular = view-dependent highlights

Approach:
    Retinex-inspired decomposition with face priors:
    1. Estimate illumination via bilateral filtering (smooth shading)
    2. Extract albedo via Retinex: A = Y / S
    3. Compute specular as residual: specular = max(0, Y - A * S)
    4. Estimate normals from shading gradient

References:
    - Retinex theory (Land & McCann, 1971)
    - Intrinsic images (Barrow & Tenenbaum, 1978)
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, laplace


@dataclass
class DecompositionConfig:
    """Configuration for intrinsic decomposition."""

    # Albedo smoothness (higher = smoother albedo)
    albedo_smoothness: float = 0.5

    # Shading smoothness (higher = smoother shading)
    shading_smoothness: float = 0.3

    # Specular threshold (pixels above this are considered specular)
    specular_threshold: float = 0.8

    # Confidence threshold
    confidence_threshold: float = 0.5

    # Bilateral filter sigma for Retinex
    bilateral_sigma_spatial: float = 15.0
    bilateral_sigma_intensity: float = 0.1

    # Normal estimation scale
    normal_scale: float = 1.0

    # Minimum albedo (prevent division by zero)
    min_albedo: float = 0.01

    # Maximum specular ratio
    max_specular_ratio: float = 0.3


@dataclass
class IntrinsicComponents:
    """Intrinsic components of a face image."""

    # Albedo: (H, W, 3) — identity-intrinsic, lighting-invariant
    albedo: np.ndarray

    # Shading: (H, W, 1) — lighting-dependent, identity-invariant
    shading: np.ndarray

    # Specular: (H, W, 3) — view-dependent highlights
    specular: np.ndarray

    # Normal map: (H, W, 3) — surface normals (unit vectors)
    normal_map: np.ndarray

    # Confidence: (H, W, 1) — decomposition confidence [0, 1]
    confidence: np.ndarray

    # Reconstruction error: ||Y - (A * S + specular)||
    reconstruction_error: float

    # V3: Uncertainty propagation
    # Albedo uncertainty: (H, W, 1) — uncertainty in albedo estimate
    albedo_uncertainty: Optional[np.ndarray] = None

    # Shading uncertainty: (H, W, 1) — uncertainty in shading estimate
    shading_uncertainty: Optional[np.ndarray] = None

    # Specular uncertainty: (H, W, 1) — uncertainty in specular estimate
    specular_uncertainty: Optional[np.ndarray] = None

    # Overall decomposition quality [0, 1]
    decomposition_quality: float = 0.0


@dataclass
class DecompositionReport:
    """Per-frame decomposition metrics."""

    frame_idx: int
    components: IntrinsicComponents
    albedo_stability: float
    shading_smoothness: float
    specular_sparsity: float
    decomposition_time_ms: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "frame_idx": self.frame_idx,
            "albedo_stability": self.albedo_stability,
            "shading_smoothness": self.shading_smoothness,
            "specular_sparsity": self.specular_sparsity,
            "decomposition_time_ms": self.decomposition_time_ms,
            "reconstruction_error": self.components.reconstruction_error,
        }


class IntrinsicDecomposer:
    """Retinex-inspired intrinsic decomposition.

    Decomposes image Y into:
        Y = A * S + specular

    where:
        A = albedo (reflectance, identity-intrinsic)
        S = shading (illumination, smooth)
        specular = max(0, Y - A * S)
    """

    def __init__(self, config: Optional[DecompositionConfig] = None, use_mesh_normals: bool = True):
        """Initialize decomposer.

        Args:
            config: Decomposition configuration
            use_mesh_normals: When True and mesh_478 is provided, derive normals
                              from mesh geometry instead of shading gradients
        """
        self.config = config or DecompositionConfig()
        self.use_mesh_normals = use_mesh_normals
        self._normal_source = "mesh"  # "mesh" or "shading_gradient"

    def decompose(
        self,
        image: np.ndarray,
        mesh_478: Optional[np.ndarray] = None,
        warp_M: Optional[np.ndarray] = None,
    ) -> IntrinsicComponents:
        """Decompose image into intrinsic components.

        When mesh_478 is provided and use_mesh_normals is True, surface normals
        are derived from mesh geometry instead of shading gradients, breaking
        the circular shading→normals→shading dependency.

        Args:
            image: Input image (H, W, 3), float32, [0, 1]
            mesh_478: Optional (478, 3) MediaPipe mesh for geometry normals
            warp_M: Optional (2, 3) forward similarity warp (source→canonical)

        Returns:
            IntrinsicComponents with albedo, shading, specular, normals, confidence
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got {image.shape}")

        start_time = time.time()

        # All paths need shading for albedo/specular estimation
        shading = self._estimate_shading(image)
        albedo = self._extract_albedo(image, shading)
        specular = self._compute_specular(image, albedo, shading)

        # Step 4: Estimate normals — from mesh geometry or shading gradient
        if (self.use_mesh_normals
            and mesh_478 is not None
            and len(mesh_478) >= 468
            and warp_M is not None):
            normal_map = self._estimate_normals_from_mesh(mesh_478, warp_M, image.shape[:2])
            self._normal_source = "mesh"
        else:
            normal_map = self._estimate_normals(shading)
            self._normal_source = "shading_gradient"

        # Step 5: Compute confidence
        confidence = self._compute_confidence(image, albedo, shading, specular)

        # Step 6: Compute reconstruction error
        reconstruction_error = self._compute_reconstruction_error(
            image, albedo, shading, specular
        )

        # Step 7: Compute uncertainty propagation
        albedo_uncertainty = self._compute_albedo_uncertainty(albedo, shading)
        shading_uncertainty = self._compute_shading_uncertainty(shading)
        specular_uncertainty = self._compute_specular_uncertainty(specular)

        # Step 8: Compute overall decomposition quality
        decomposition_quality = self._compute_decomposition_quality(
            reconstruction_error, confidence, albedo_uncertainty
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

    def _estimate_normals_from_mesh(
        self,
        mesh_478: np.ndarray,
        warp_M: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Estimate surface normals from MediaPipe 478-point mesh geometry.

        Breaks the circular shading→normals→shading dependency by computing
        normals from the 3D mesh topology (852 triangles) instead of shading gradients.

        Args:
            mesh_478: (478, 3) MediaPipe mesh in pixel+depth coordinates
            warp_M: (2, 3) forward similarity transform (source→canonical)
            target_shape: (H, W) of the canonical image

        Returns:
            (H, W, 3) normal map, unit vectors
        """
        from face_os.landmarks import mesh_normal_map
        return mesh_normal_map(mesh_478, warp_M, target_shape[::-1])

    def _estimate_shading(self, image: np.ndarray) -> np.ndarray:
        """Estimate illumination via bilateral filtering.

        Shading is assumed to be smooth (low-frequency).
        We use Gaussian filtering as a proxy for bilateral filtering.

        Args:
            image: Input image (H, W, 3)

        Returns:
            Shading estimate (H, W, 1)
        """
        # Convert to grayscale for shading estimation
        gray = np.mean(image, axis=2)

        # Estimate shading via Gaussian blur (smooth illumination)
        sigma = self.config.bilateral_sigma_spatial / 5.0
        shading_2d = gaussian_filter(gray, sigma=sigma)

        # Clip to valid range
        shading_2d = np.clip(shading_2d, self.config.min_albedo, 1.0)

        # Expand to (H, W, 1)
        shading = shading_2d[:, :, np.newaxis]

        return shading

    def _extract_albedo(
        self, image: np.ndarray, shading: np.ndarray
    ) -> np.ndarray:
        """Extract albedo via Retinex: A = Y / S.

        Uses edge-preserving smoothing to preserve texture edges.

        Args:
            image: Input image (H, W, 3)
            shading: Shading estimate (H, W, 1)

        Returns:
            Albedo estimate (H, W, 3)
        """
        # Retinex: A = Y / S
        shading_3ch = np.repeat(shading, 3, axis=2)
        albedo = image / (shading_3ch + 1e-8)

        # Edge-preserving smoothing: use median filter then Gaussian
        # Median preserves edges better than Gaussian alone
        from scipy.ndimage import median_filter
        
        sigma = self.config.albedo_smoothness * 3  # Reduced from 5
        for c in range(3):
            # Apply median filter first (edge-preserving)
            albedo[:, :, c] = median_filter(albedo[:, :, c], size=5)
            # Then light Gaussian smoothing
            albedo[:, :, c] = gaussian_filter(albedo[:, :, c], sigma=sigma)

        # Clip to valid range
        albedo = np.clip(albedo, 0, 1)

        return albedo

    def _compute_specular(
        self,
        image: np.ndarray,
        albedo: np.ndarray,
        shading: np.ndarray,
    ) -> np.ndarray:
        """Compute specular as residual: specular = max(0, Y - A * S).

        Specular highlights are sparse and bright.
        We threshold more aggressively to ensure sparsity.

        Args:
            image: Input image (H, W, 3)
            albedo: Albedo estimate (H, W, 3)
            shading: Shading estimate (H, W, 1)

        Returns:
            Specular estimate (H, W, 3)
        """
        # Reconstruct diffuse component
        shading_3ch = np.repeat(shading, 3, axis=2)
        diffuse = albedo * shading_3ch

        # Specular = max(0, Y - diffuse)
        specular = np.maximum(0, image - diffuse)

        # Very aggressive thresholding for sparsity
        # Only keep pixels that are significantly above diffuse
        threshold = self.config.specular_threshold * 0.5  # Increased further
        specular[specular < threshold] = 0

        # Additional: suppress specular in dark regions (likely noise)
        dark_mask = np.mean(image, axis=2, keepdims=True) < 0.4
        specular[dark_mask.repeat(3, axis=2)] = 0

        # Additional: suppress specular in very bright regions (likely overexposure)
        bright_mask = np.mean(image, axis=2, keepdims=True) > 0.9
        specular[bright_mask.repeat(3, axis=2)] = 0

        return specular

    def _estimate_normals(self, shading: np.ndarray) -> np.ndarray:
        """Estimate surface normals from shading gradient.

        Normal = normalize([-dS/dx, -dS/dy, 1])

        Args:
            shading: Shading estimate (H, W, 1)

        Returns:
            Normal map (H, W, 3), unit vectors
        """
        shading_2d = shading[:, :, 0]

        # Compute gradients
        dy, dx = np.gradient(shading_2d)

        # Normal = [-dx, -dy, 1] (scaled)
        scale = self.config.normal_scale
        nx = -dx * scale
        ny = -dy * scale
        nz = np.ones_like(dx)

        # Normalize to unit vectors
        norms = np.sqrt(nx**2 + ny**2 + nz**2)
        nx = nx / (norms + 1e-8)
        ny = ny / (norms + 1e-8)
        nz = nz / (norms + 1e-8)

        # Stack to (H, W, 3)
        normal_map = np.stack([nx, ny, nz], axis=2)

        return normal_map

    def _compute_confidence(
        self,
        image: np.ndarray,
        albedo: np.ndarray,
        shading: np.ndarray,
        specular: np.ndarray,
    ) -> np.ndarray:
        """Compute decomposition confidence.

        Confidence = 1 - reconstruction_error / input_energy

        Args:
            image: Input image (H, W, 3)
            albedo: Albedo estimate (H, W, 3)
            shading: Shading estimate (H, W, 1)
            specular: Specular estimate (H, W, 3)

        Returns:
            Confidence map (H, W, 1), [0, 1]
        """
        # Reconstruct
        shading_3ch = np.repeat(shading, 3, axis=2)
        reconstructed = albedo * shading_3ch + specular

        # Per-pixel error
        error = np.mean(np.abs(image - reconstructed), axis=2)

        # Normalize by input energy
        input_energy = np.mean(np.abs(image), axis=2) + 1e-8

        # Confidence = 1 - relative_error
        confidence = 1.0 - error / input_energy
        confidence = np.clip(confidence, 0, 1)

        # Expand to (H, W, 1)
        confidence = confidence[:, :, np.newaxis]

        return confidence

    def _compute_reconstruction_error(
        self,
        image: np.ndarray,
        albedo: np.ndarray,
        shading: np.ndarray,
        specular: np.ndarray,
    ) -> float:
        """Compute reconstruction error: ||Y - (A * S + specular)||.

        Args:
            image: Input image (H, W, 3)
            albedo: Albedo estimate (H, W, 3)
            shading: Shading estimate (H, W, 1)
            specular: Specular estimate (H, W, 3)

        Returns:
            Mean absolute error
        """
        shading_3ch = np.repeat(shading, 3, axis=2)
        reconstructed = albedo * shading_3ch + specular
        error = np.mean(np.abs(image - reconstructed))
        return float(error)

    def _compute_albedo_uncertainty(
        self,
        albedo: np.ndarray,
        shading: np.ndarray,
    ) -> np.ndarray:
        """Compute uncertainty in albedo estimate.

        Uncertainty is higher where:
        - Shading is dark (division by small number)
        - Albedo has high variance (ambiguous regions)
        - Edges (gradient discontinuities)

        Args:
            albedo: Albedo estimate (H, W, 3)
            shading: Shading estimate (H, W, 1)

        Returns:
            Albedo uncertainty (H, W, 1)
        """
        # Shading-based uncertainty: darker shading = higher uncertainty
        shading_uncertainty = 1.0 - shading

        # Gradient-based uncertainty: edges = higher uncertainty
        from scipy.ndimage import sobel
        albedo_gray = np.mean(albedo, axis=2)
        gradient_x = sobel(albedo_gray, axis=1)
        gradient_y = sobel(albedo_gray, axis=0)
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
        edge_uncertainty = np.clip(gradient_mag * 10, 0, 1)

        # Combined uncertainty
        uncertainty = np.maximum(shading_uncertainty[:, :, 0], edge_uncertainty)
        return uncertainty[:, :, np.newaxis]

    def _compute_shading_uncertainty(
        self,
        shading: np.ndarray,
    ) -> np.ndarray:
        """Compute uncertainty in shading estimate.

        Uncertainty is higher where:
        - Shading changes rapidly (gradient)
        - Shading is near boundaries

        Args:
            shading: Shading estimate (H, W, 1)

        Returns:
            Shading uncertainty (H, W, 1)
        """
        from scipy.ndimage import sobel
        shading_2d = shading[:, :, 0]
        gradient_x = sobel(shading_2d, axis=1)
        gradient_y = sobel(shading_2d, axis=0)
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
        uncertainty = np.clip(gradient_mag * 5, 0, 1)
        return uncertainty[:, :, np.newaxis]

    def _compute_specular_uncertainty(
        self,
        specular: np.ndarray,
    ) -> np.ndarray:
        """Compute uncertainty in specular estimate.

        Uncertainty is higher where:
        - Specular is bright (could be texture)
        - Specular is near edges

        Args:
            specular: Specular estimate (H, W, 3)

        Returns:
            Specular uncertainty (H, W, 1)
        """
        specular_mag = np.linalg.norm(specular, axis=2)
        uncertainty = np.clip(specular_mag * 2, 0, 1)
        return uncertainty[:, :, np.newaxis]

    def _compute_decomposition_quality(
        self,
        reconstruction_error: float,
        confidence: np.ndarray,
        albedo_uncertainty: np.ndarray,
    ) -> float:
        """Compute overall decomposition quality.

        Quality = f(reconstruction_error, confidence, uncertainty)

        Args:
            reconstruction_error: Reconstruction error
            confidence: Confidence map (H, W, 1)
            albedo_uncertainty: Albedo uncertainty (H, W, 1)

        Returns:
            Decomposition quality [0, 1]
        """
        # Error quality: lower error = higher quality
        error_quality = 1.0 - min(reconstruction_error * 5, 1.0)

        # Confidence quality: higher confidence = higher quality
        confidence_quality = float(np.mean(confidence))

        # Uncertainty quality: lower uncertainty = higher quality
        uncertainty_quality = 1.0 - float(np.mean(albedo_uncertainty))

        # Weighted combination
        quality = (
            0.4 * error_quality
            + 0.3 * confidence_quality
            + 0.3 * uncertainty_quality
        )

        return float(np.clip(quality, 0, 1))

    def decompose_batch(
        self, images: list[np.ndarray]
    ) -> list[IntrinsicComponents]:
        """Decompose batch of images.

        Args:
            images: List of input images (H, W, 3)

        Returns:
            List of IntrinsicComponents
        """
        return [self.decompose(img) for img in images]

    def compute_albedo_stability(
        self, albedos: list[np.ndarray]
    ) -> float:
        """Compute albedo stability across frames.

        Stability = 1 - std(albedo) / mean(albedo)

        Args:
            albedos: List of albedo maps (H, W, 3)

        Returns:
            Stability score [0, 1], higher = more stable
        """
        if len(albedos) < 2:
            return 1.0

        # Stack albedos
        stacked = np.stack(albedos, axis=0)

        # Compute mean and std across frames
        mean_albedo = np.mean(stacked, axis=0)
        std_albedo = np.std(stacked, axis=0)

        # Stability = 1 - coefficient of variation
        cv = np.mean(std_albedo) / (np.mean(mean_albedo) + 1e-8)
        stability = 1.0 - cv
        stability = np.clip(stability, 0, 1)

        return float(stability)
