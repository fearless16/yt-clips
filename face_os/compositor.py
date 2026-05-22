"""
compositor.py — Module 9: Confidence-Weighted Compositor.

This module is intentionally kept thin.

Architecture rule:
- Geometry, identity, temporal, and photometric correction happen upstream.
- The compositor performs only the final blend into output space.

Core behavior:
- High-confidence face pixels come from the enhanced face.
- Low-confidence pixels come from the original crop.
- Mask edges are feathered for a smooth transition.
- Blending happens in linear-light space, not gamma space.
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

    Compositing in gamma space is visually incorrect, because the blend
    weights do not correspond to physical light intensities.
    """
    f = img.astype(np.float32) / 255.0
    return np.power(np.clip(f, 0.0, 1.0), 2.2)


def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """
    Convert a linear-light float32 image in [0, 1] back to uint8 sRGB/BGR.
    """
    g = np.power(np.clip(img, 0.0, 1.0), 1.0 / 2.2)
    return (g * 255.0).astype(np.uint8)


def _blend_linear(bg: np.ndarray, fg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blend two images in linear-light space.

    Args:
        bg: Background image, shape (H, W, 3), uint8 BGR.
        fg: Foreground image, shape (H, W, 3), uint8 BGR.
        mask: Blend mask, shape (H, W), float in [0, 1].

    Returns:
        Blended image, shape (H, W, 3), uint8 BGR.
    """
    if bg.shape != fg.shape:
        raise ValueError(f"bg and fg must have the same shape, got {bg.shape} vs {fg.shape}")

    if bg.ndim != 3 or bg.shape[2] != 3:
        raise ValueError(f"bg must be HxWx3, got {bg.shape}")

    if fg.ndim != 3 or fg.shape[2] != 3:
        raise ValueError(f"fg must be HxWx3, got {fg.shape}")

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H, W), got {mask.shape}")

    if bg.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"mask spatial size must match images, got mask={mask.shape[:2]} "
            f"and images={bg.shape[:2]}"
        )

    bg_lin = _srgb_to_linear(bg)
    fg_lin = _srgb_to_linear(fg)

    # Expand mask to 3 channels and clamp it to a valid blend range.
    m3 = np.clip(mask.astype(np.float32), 0.0, 1.0)[:, :, np.newaxis]

    blended = bg_lin * (1.0 - m3) + fg_lin * m3
    return _linear_to_srgb(blended)


# ---------------------------------------------------------------------------
# Compositor
# ---------------------------------------------------------------------------

class Compositor:
    """
    Thin final assembly layer for face compositing.

    Responsibilities:
    - Feather the face mask.
    - Blend enhanced face into original crop.
    - Use confidence as a per-pixel mixing weight.
    - Keep all photometric stabilization upstream.

    Non-responsibilities:
    - No temporal EMA state.
    - No lighting matching.
    - No multi-band restoration.
    - No geometry correction.
    """

    def __init__(self) -> None:
        self._feather_kernel = None

    def reset(self) -> None:
        """
        Reset compositor-local state between clips.

        This compositor does not maintain temporal luminance memory.
        Any photometric locking should happen upstream in a dedicated module.
        """
        self._feather_kernel = None

    def composite(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        confidence: Optional[ConfidenceMap] = None,
        enhancement_mask: Optional[EnhancementMask] = None,
        face_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Composite an enhanced face back onto the original crop.

        Blend strategy:
        1. Use the face mask to define the face region.
        2. Feather mask edges for a smooth transition.
        3. Modulate by confidence when available.
        4. Blend in linear-light space.

        Args:
            original: Original cropped frame, shape (H, W, 3), uint8 BGR.
            enhanced: Enhanced cropped frame, shape (H, W, 3), uint8 BGR.
            confidence: Per-pixel confidence map from memory atlas.
            enhancement_mask: Optional container for region masks.
            face_mask: Optional explicit face mask, shape (H, W), float [0, 1].

        Returns:
            Composited frame, shape (H, W, 3), uint8 BGR.
        """
        if face_mask is None and enhancement_mask is not None:
            face_mask = enhancement_mask.face_mask

        if original.shape != enhanced.shape:
            raise ValueError(
                f"original and enhanced must have the same shape, got "
                f"{original.shape} vs {enhanced.shape}"
            )

        # No mask means there is no local face region to composite.
        # In that case, either use global confidence blending or return the enhanced crop.
        if face_mask is None:
            if confidence is not None and confidence.combined is not None:
                return self._blend_by_confidence(original, enhanced, confidence.combined)
            return enhanced

        # Build a smooth face-region mask.
        feathered = self._feather_mask(face_mask)
        blend_weight = feathered.copy()

        # Confidence further modulates the face region if it exists.
        if confidence is not None and confidence.combined is not None:
            conf = confidence.combined
            if conf.shape[:2] != blend_weight.shape[:2]:
                conf = cv2.resize(
                    conf,
                    (blend_weight.shape[1], blend_weight.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            blend_weight = blend_weight * conf

        # Final assembly step only.
        return _blend_linear(original, enhanced, blend_weight)

    def composite_with_memory(
        self,
        original: np.ndarray,
        memory_face: np.ndarray,
        confidence: ConfidenceMap,
        face_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Composite using a stable memory face.

        The memory face is assumed to be an upstream product:
        identity estimation, temporal stabilization, and photometric correction
        must already have been handled before this method is called.

        Args:
            original: Original cropped frame, shape (H, W, 3), uint8 BGR.
            memory_face: Stable face image from memory atlas, shape (H, W, 3).
            confidence: Per-pixel confidence map.
            face_mask: Face region mask, shape (H, W), float [0, 1].

        Returns:
            Composited frame, shape (H, W, 3), uint8 BGR.
        """
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
        """
        Feather mask edges using Gaussian blur.

        Feathering is used only to soften the transition boundary.
        It should not be used as a substitute for topology or geometry.
        """
        feather = int(cfg.compositor.feather_pixels)
        if feather <= 0:
            return np.clip(mask.astype(np.float32), 0.0, 1.0)

        # Gaussian kernel size must be odd and at least 3.
        ksize = max(3, feather * 2 + 1)
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), feather / 2)
        return np.clip(blurred, 0.0, 1.0)

    def _blend_by_confidence(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        confidence: np.ndarray,
    ) -> np.ndarray:
        """
        Blend the entire frame using a confidence map.

        This is the fallback path when no explicit face mask is available.
        """
        h, w = original.shape[:2]
        if confidence.shape[:2] != (h, w):
            confidence = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)

        return _blend_linear(original, enhanced, confidence)
