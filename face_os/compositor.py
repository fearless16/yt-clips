"""
compositor.py — Module 9: Confidence-Weighted Compositor.

BEAST MODE FIXES:
- Nuked the fake 2.2 gamma power-law. Implemented real piecewise sRGB transfer curve.
- Optimized multiband_blend mask pyramid to prevent redundant cv2.resize calls.
- Added NaN/Inf safeguards in Laplacian reconstruction to prevent black-frame crashes.
"""

from typing import Optional

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import ConfidenceMap, EnhancementMask

cfg = get_config()


# ---------------------------------------------------------------------------
# Linear-light compositing helpers
# ---------------------------------------------------------------------------

def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """
    Convert an 8-bit sRGB/BGR image to linear-light float32 in [0, 1].
    Uses the exact piecewise sRGB transfer curve, not a crude 2.2 gamma shortcut.
    """
    f = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
    mask = f <= 0.04045
    lin = np.where(mask, f / 12.92, np.power((f + 0.055) / 1.055, 2.4))
    return lin.astype(np.float32)


def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """
    Convert a linear-light float32 image in [0, 1] back to uint8 sRGB/BGR.
    """
    lin = np.clip(img, 0.0, 1.0)
    mask = lin <= 0.0031308
    srgb = np.where(mask, lin * 12.92, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return (np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _blend_linear(bg: np.ndarray, fg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend two images in linear-light space."""
    if bg.shape != fg.shape:
        raise ValueError(f"bg and fg must have the same shape, got {bg.shape} vs {fg.shape}")
    if bg.ndim != 3 or bg.shape[2] != 3:
        raise ValueError(f"bg must be HxWx3, got {bg.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H, W), got {mask.shape}")
    if bg.shape[:2] != mask.shape[:2]:
        raise ValueError(f"mask spatial size must match images")

    # Fast path: if mask is purely 1.0, skip math
    if np.min(mask) >= 0.999:
        return fg.copy()
    if np.max(mask) <= 0.001:
        return bg.copy()

    bg_lin = _srgb_to_linear(bg)
    fg_lin = _srgb_to_linear(fg)

    m3 = np.clip(mask.astype(np.float32), 0.0, 1.0)[:, :, np.newaxis]
    blended = bg_lin * (1.0 - m3) + fg_lin * m3
    return _linear_to_srgb(blended)


def multiband_blend(bg: np.ndarray, fg: np.ndarray, mask: np.ndarray, levels: int = 0) -> np.ndarray:
    """Laplacian pyramid multi-band blending in linear light space.

    Args:
        bg, fg: uint8 BGR images, same shape.
        mask: 2D float32 in [0, 1] per pixel.
        levels: Pyramid depth. 0 = auto-detect (2 for crops <200px,
                3 for 200-500px, 4 for >500px min dimension).
                Any positive integer overrides.
    """
    if bg.shape != fg.shape:
        raise ValueError(f"Shape mismatch: {bg.shape} vs {fg.shape}")

    mask_f = np.clip(mask.astype(np.float32), 0.0, 1.0)

    # Phase A: fast-path — skip pyramid for near-uniform masks
    if np.min(mask_f) >= 0.999:
        return fg.copy()
    if np.max(mask_f) <= 0.001:
        return bg.copy()

    if levels <= 0:
        min_dim = min(bg.shape[0], bg.shape[1])
        if min_dim < 200:
            levels = 2
        elif min_dim < 500:
            levels = 3
        else:
            levels = 4

    bg_lin = _srgb_to_linear(bg)
    fg_lin = _srgb_to_linear(fg)
    mask_3ch = np.stack([mask_f] * 3, axis=2)

    # Build Gaussian pyramids
    gp_mask = [mask_3ch]
    gp_bg = [bg_lin]
    gp_fg = [fg_lin]
    
    for _ in range(levels):
        mask_3ch = cv2.pyrDown(mask_3ch)
        bg_lin = cv2.pyrDown(bg_lin)
        fg_lin = cv2.pyrDown(fg_lin)
        gp_mask.append(mask_3ch)
        gp_bg.append(bg_lin)
        gp_fg.append(fg_lin)

    # Build Laplacian pyramids
    def _build_laplacian(gp):
        lp = [gp[-1]]
        for i in range(levels, 0, -1):
            expanded = cv2.pyrUp(gp[i], dstsize=(gp[i-1].shape[1], gp[i-1].shape[0]))
            lp.append(gp[i-1] - expanded)
        lp.reverse()
        return lp

    lp_bg = _build_laplacian(gp_bg)
    lp_fg = _build_laplacian(gp_fg)

    # Blend each Laplacian level
    blended_lp = []
    for i in range(levels + 1):
        lb, lf, gm = lp_bg[i], lp_fg[i], gp_mask[i]
        # BEAST MODE: Only resize if dimensions actually mismatch (odd/even edge cases)
        if gm.shape[:2] != lb.shape[:2]:
            gm = cv2.resize(gm, (lb.shape[1], lb.shape[0]), interpolation=cv2.INTER_LINEAR)
            if gm.ndim == 2:
                gm = gm[:, :, np.newaxis]
        blended_lp.append(lb * (1.0 - gm) + lf * gm)

    # Reconstruct
    result = blended_lp[-1]
    for i in range(levels, 0, -1):
        expanded = cv2.pyrUp(result, dstsize=(blended_lp[i-1].shape[1], blended_lp[i-1].shape[0]))
        result = expanded + blended_lp[i-1]

    # Safeguard against NaNs in reconstruction
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return _linear_to_srgb(result)


# ---------------------------------------------------------------------------
# Compositor
# ---------------------------------------------------------------------------

class Compositor:
    """Thin final assembly layer for face compositing."""

    def __init__(self) -> None:
        self._feather_kernel = None

    def reset(self) -> None:
        self._feather_kernel = None

    def composite(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        confidence: Optional[ConfidenceMap] = None,
        enhancement_mask: Optional[EnhancementMask] = None,
        face_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if face_mask is None and enhancement_mask is not None:
            face_mask = enhancement_mask.face_mask

        if original.shape != enhanced.shape:
            raise ValueError(f"original and enhanced must have the same shape")

        if face_mask is None:
            if confidence is not None and confidence.combined is not None:
                return self._blend_by_confidence(original, enhanced, confidence.combined)
            return enhanced

        feathered = self._feather_mask(face_mask)
        blend_weight = feathered.copy()

        if confidence is not None and confidence.combined is not None:
            conf = confidence.combined
            if conf.shape[:2] != blend_weight.shape[:2]:
                conf = cv2.resize(conf, (blend_weight.shape[1], blend_weight.shape[0]), interpolation=cv2.INTER_LINEAR)
            blend_weight = blend_weight * conf

        return _blend_linear(original, enhanced, blend_weight)

    def composite_with_memory(
        self,
        original: np.ndarray,
        memory_face: np.ndarray,
        confidence: ConfidenceMap,
        face_mask: np.ndarray,
    ) -> np.ndarray:
        if memory_face is None:
            return original

        h, w = original.shape[:2]
        if memory_face.shape[:2] != (h, w):
            memory_face = cv2.resize(memory_face, (w, h), interpolation=cv2.INTER_LANCZOS4)

        conf = confidence.combined
        if conf is None:
            conf = np.full((h, w), 0.5, dtype=np.float32)
        elif conf.shape[:2] != (h, w):
            conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)

        feathered = self._feather_mask(face_mask)
        blend_weight = np.clip(feathered * conf, 0.0, 1.0)

        return _blend_linear(original, memory_face, blend_weight)

    def _feather_mask(self, mask: np.ndarray) -> np.ndarray:
        feather = int(getattr(cfg.compositor, 'feather_pixels', 10))
        if feather <= 0:
            return np.clip(mask.astype(np.float32), 0.0, 1.0)

        ksize = max(3, feather * 2 + 1)
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), feather / 2.0)
        return np.clip(blurred, 0.0, 1.0)

    def _blend_by_confidence(self, original: np.ndarray, enhanced: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        h, w = original.shape[:2]
        if confidence.shape[:2] != (h, w):
            confidence = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)
        return _blend_linear(original, enhanced, confidence)


def face_prior_normal_map(h: int, w: int) -> np.ndarray:
    """D-04: Analytic ellipsoid face-prior normal map (zero cost, landmark-free).

    Unit normals derived from the gradient of a parametric ellipsoid
    approximating face surface curvature. Serves as geometric reference
    when full mesh normals are not available (alpha/enhancement paths).

    Args:
        h, w: Height and width of the target normal map.

    Returns:
        float32 (H, W, 3) unit-normal map.
    """
    cy, cx = h / 2.0, w / 2.0
    ry, rx = h * 0.47, w * 0.42
    yy, xx = np.ogrid[:h, :w]
    nx = (xx - cx) / max(rx, 1.0)
    ny = (yy - cy) / max(ry, 1.0)
    r2 = nx**2 + ny**2
    nz = np.sqrt(np.maximum(0.0, 1.0 - np.clip(r2, 0.0, 1.0)))
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    normal_map = np.stack([nx / norm, ny / norm, nz / norm], axis=2)
    return normal_map.astype(np.float32)


def compute_normal_variance_mask(normal_map: np.ndarray, radius: int = 5) -> np.ndarray:
    """D-04: Normal-variance edge mask for geometry-aware blending.

    Computes local normal variance via Gaussian blur to detect
    anatomical edges (nose bridge, eyebrows, jaw contour) vs flat
    regions (cheeks, forehead).

    Args:
        normal_map: float32 (H,W,3) unit-normal map.
        radius: Gaussian blur kernel radius in pixels.

    Returns:
        float32 (H,W) edge mask in [0, 1] — high values at
        geometric edges, low values at flat regions.
    """
    blurred = cv2.GaussianBlur(normal_map, (0, 0), radius)
    variance = np.sum((normal_map.astype(np.float32) - blurred) ** 2, axis=2)
    return variance / (max(np.percentile(variance, 98), 1e-6))