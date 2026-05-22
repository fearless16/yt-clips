"""
compositor.py — Module 9: Confidence-Weighted Compositor.

Composites the enhanced face back onto the cropped frame using
per-pixel confidence from the Identity Memory Atlas.

Core principle:
  - High confidence pixels: use the accumulated/enhanced face (stable, clean)
  - Low confidence pixels: use the original frame (noisy but authentic)
  - Edge blending: smooth transition between face and background

This prevents artifacts where enhancement creates visible seams
between the face region and the background.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import ConfidenceMap, EnhancementMask


cfg = get_config()


# ─── D-01: Linear-light compositing helpers ─────────────────────────────────

def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 image to linear-light float32 [0,1].

    D-01: Gamma-space compositing is physically incorrect.
    Blending must happen in linear-light space.
    """
    f = img.astype(np.float32) / 255.0
    return np.power(f, 2.2)


def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Convert linear-light float32 [0,1] back to sRGB uint8."""
    g = np.power(np.clip(img, 0, 1), 1.0 / 2.2)
    return (g * 255).astype(np.uint8)


def _blend_linear(bg: np.ndarray, fg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend two sRGB images in linear-light space.

    D-01: Fixes gamma-space compositing.
    bg, fg: (H,W,3) uint8 BGR
    mask: (H,W) float32 [0,1]
    Returns: (H,W,3) uint8 BGR
    """
    bg_lin = _srgb_to_linear(bg)
    fg_lin = _srgb_to_linear(fg)
    m3 = mask[:, :, np.newaxis] if mask.ndim == 2 else mask
    blended = bg_lin * (1 - m3) + fg_lin * m3
    return _linear_to_srgb(blended)


# ─── D-01c: Multi-band compositing (Laplacian pyramid) ──────────────────────

def multiband_blend(
    bg: np.ndarray,
    fg: np.ndarray,
    mask: np.ndarray,
    levels: int = 4,
) -> np.ndarray:
    """Multi-band compositing via Laplacian pyramid blending.

    D-01c: Single-band alpha blending destroys frequency bands.
    Laplacian pyramid blends each frequency band independently.

    Args:
        bg: Background (H, W, 3) uint8 BGR
        fg: Foreground (H, W, 3) uint8 BGR
        mask: Blend mask (H, W) float32 [0,1]
        levels: Number of pyramid levels

    Returns:
        Blended result (H, W, 3) uint8 BGR
    """
    bg_f = bg.astype(np.float32)
    fg_f = fg.astype(np.float32)

    # Build Gaussian pyramids
    gp_bg = [bg_f]
    gp_fg = [fg_f]
    gp_mask = [mask.astype(np.float32)]

    for _i in range(levels):
        gp_bg.append(cv2.pyrDown(gp_bg[-1]))
        gp_fg.append(cv2.pyrDown(gp_fg[-1]))
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))

    # Build Laplacian pyramids for bg and fg
    lp_bg = [gp_bg[levels]]
    lp_fg = [gp_fg[levels]]

    for i in range(levels, 0, -1):
        up_bg = cv2.pyrUp(gp_bg[i], dstsize=(gp_bg[i - 1].shape[1], gp_bg[i - 1].shape[0]))
        up_fg = cv2.pyrUp(gp_fg[i], dstsize=(gp_fg[i - 1].shape[1], gp_fg[i - 1].shape[0]))
        lp_bg.append(gp_bg[i - 1] - up_bg)
        lp_fg.append(gp_fg[i - 1] - up_fg)

    # Blend each Laplacian level using the corresponding mask level
    blended_lp = []
    for i in range(levels + 1):
        m = gp_mask[levels - i] if i < levels else gp_mask[0]
        # Ensure mask has 3 channels
        m3 = m[:, :, np.newaxis] if m.ndim == 2 else m
        # Ensure spatial dimensions match
        if m3.shape[:2] != lp_bg[i].shape[:2]:
            m3 = cv2.resize(m3, (lp_bg[i].shape[1], lp_bg[i].shape[0]), interpolation=cv2.INTER_LINEAR)
            if m3.ndim == 2:
                m3 = m3[:, :, np.newaxis]
        blended_lp.append(lp_bg[i] * (1 - m3) + lp_fg[i] * m3)

    # Reconstruct from blended pyramid
    result = blended_lp[0]
    for i in range(1, levels + 1):
        result = cv2.pyrUp(result, dstsize=(blended_lp[i].shape[1], blended_lp[i].shape[0]))
        result = result + blended_lp[i]

    return np.clip(result, 0, 255).astype(np.uint8)


# Module-level state for photometric locking
_prev_luminance: Optional[float] = None
_luminance_ema_alpha: float = 0.5
_luminance_clamp: float = 8.0  # LAB units


def reset_photometric_lock() -> None:
    """Reset photometric lock state between clips."""
    global _prev_luminance
    _prev_luminance = None


def photometric_lock(frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Temporal photometric locking via luminance EMA.

    D-01: Prevents temporal photometric instability by locking
    frame luminance to a running average.

    Args:
        frame: Input frame (H, W, 3) uint8 BGR
        mask: Optional face mask (H, W) float32 [0,1]. If provided,
              luminance is computed only within the mask.

    Returns:
        Photometrically locked frame (H, W, 3) uint8 BGR
    """
    global _prev_luminance

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)

    if mask is not None and mask.max() > 0.01:
        face_pixels = mask > 0.5
        if face_pixels.sum() < 100:
            return frame
        current_luma = float(l_channel[face_pixels].mean())
    else:
        current_luma = float(l_channel.mean())

    if _prev_luminance is None:
        _prev_luminance = current_luma
        return frame

    _prev_luminance = (
        _luminance_ema_alpha * current_luma
        + (1 - _luminance_ema_alpha) * _prev_luminance
    )

    delta = _prev_luminance - current_luma
    if abs(delta) < 2.0:
        return frame

    delta = np.clip(delta, -_luminance_clamp, _luminance_clamp)

    lab_out = lab.astype(np.float32)
    lab_out[:, :, 0] = np.clip(l_channel + delta, 0, 255)
    return cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2BGR)


class Compositor:
    """Composites enhanced face onto the frame using confidence blending."""

    def __init__(self):
        self._feather_kernel = None
        # D-01: Temporal photometric locking — EMA of face luminance
        self._luma_ema: Optional[float] = None
        self._luma_alpha: float = 0.5  # EMA smoothing factor (higher = more responsive)

    def reset(self) -> None:
        """Reset compositor state between clips."""
        self._luma_ema = None
        self._feather_kernel = None

    def composite(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        confidence: Optional[ConfidenceMap] = None,
        enhancement_mask: Optional[EnhancementMask] = None,
        face_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Composite enhanced face onto original frame.

        Strategy:
        1. Use face_mask to define the blending region
        2. Use confidence to weight enhanced vs original
        3. Feather edges for smooth transition
        4. Match lighting between face and background

        Args:
            original: Original cropped frame (BGR)
            enhanced: Enhanced frame (BGR)
            confidence: Per-pixel confidence from memory atlas
            enhancement_mask: Region masks for enhancement
            face_mask: Binary face mask (H, W) float [0, 1]

        Returns:
            Composited frame (BGR)
        """
        if face_mask is None and enhancement_mask is not None:
            face_mask = enhancement_mask.face_mask

        if face_mask is None:
            # No face mask — blend globally based on confidence
            if confidence and confidence.combined is not None:
                return self._blend_by_confidence(original, enhanced, confidence.combined)
            return enhanced

        # Feather the face mask edges
        feathered = self._feather_mask(face_mask)

        # Compute blend weight
        blend_weight = feathered.copy()

        # Modulate by confidence if available
        if confidence and confidence.combined is not None:
            conf_resized = cv2.resize(
                confidence.combined,
                (blend_weight.shape[1], blend_weight.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            blend_weight = blend_weight * conf_resized

        # Optional: match lighting between face region and background
        if cfg.compositor.use_light_matching:
            enhanced = self._match_lighting(original, enhanced, face_mask)

        # D-01: Temporal photometric locking — reduce frame-to-frame luminance flicker
        # Uses EMA to smooth face luminance, preventing side-screen light reflections
        face_pixels = face_mask > 0.5
        if face_pixels.sum() > 100:
            face_lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            cur_luma = float(face_lab[:, :, 0][face_pixels].mean())
            if self._luma_ema is None:
                self._luma_ema = cur_luma
            else:
                # Gentle EMA: smooth toward running average
                self._luma_ema = self._luma_alpha * cur_luma + (1 - self._luma_alpha) * self._luma_ema
                delta = self._luma_ema - cur_luma
                # Only adjust if delta > 2 LAB units (ignore small variations)
                if abs(delta) > 2.0:
                    # Clamp adjustment to ±8 LAB units to prevent darkening
                    delta = np.clip(delta, -8.0, 8.0)
                    lab = face_lab.copy().astype(np.float32)
                    lab[:, :, 0] = np.clip(lab[:, :, 0] + delta, 0, 255)
                    enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # D-01: Linear-light compositing (physically correct gamma handling)
        result = _blend_linear(original, enhanced, blend_weight)

        return result

    def composite_with_memory(
        self,
        original: np.ndarray,
        memory_face: np.ndarray,
        confidence: ConfidenceMap,
        face_mask: np.ndarray,
    ) -> np.ndarray:
        """Composite using the stable memory face.

        The memory face is the accumulated appearance from the Identity Memory Atlas.
        It's cleaner and more stable than any single frame.

        Args:
            original: Original cropped frame
            memory_face: Accumulated stable face from memory atlas
            confidence: Per-pixel confidence
            face_mask: Face region mask

        Returns:
            Composited frame
        """
        if memory_face is None:
            return original

        # Resize memory face to match frame
        h, w = original.shape[:2]
        if memory_face.shape[:2] != (h, w):
            memory_face = cv2.resize(memory_face, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Get confidence map
        conf = confidence.combined
        if conf is None:
            conf = np.ones((h, w), dtype=np.float32) * 0.5
        elif conf.shape[:2] != (h, w):
            conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)

        # Feather face mask
        feathered = self._feather_mask(face_mask)

        # Blend weight: face_mask * confidence
        blend_weight = feathered * conf

        # Match lighting
        if cfg.compositor.use_light_matching:
            memory_face = self._match_lighting(original, memory_face, face_mask)

        # D-01: Linear-light compositing (physically correct gamma handling)
        result = _blend_linear(original, memory_face, blend_weight)

        return result

    def _feather_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply Gaussian feathering to mask edges."""
        feather = cfg.compositor.feather_pixels
        if feather <= 0:
            return mask

        ksize = max(3, feather * 2 + 1)
        return cv2.GaussianBlur(mask, (ksize, ksize), feather / 2)

    def _blend_by_confidence(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        confidence: np.ndarray,
    ) -> np.ndarray:
        """Blend using per-pixel confidence."""
        h, w = original.shape[:2]
        if confidence.shape[:2] != (h, w):
            confidence = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)

        # D-01: Linear-light compositing (physically correct gamma handling)
        return _blend_linear(original, enhanced, confidence)

    def _match_lighting(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Match lighting between face region and background.

        Computes the mean brightness difference between the face region
        in the original and enhanced frames, and adjusts the enhanced
        frame to match.
        """
        if mask.max() < 0.01:
            return target

        # Compute mean brightness in face region for both frames
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        mask_bool = mask > 0.5
        if mask_bool.sum() < 100:
            return target

        ref_mean = float(np.mean(ref_lab[:, :, 0][mask_bool]))
        tgt_mean = float(np.mean(tgt_lab[:, :, 0][mask_bool]))

        # Adjust target brightness to match reference
        diff = ref_mean - tgt_mean
        if abs(diff) > 5:  # Only adjust if significant difference
            adjustment = diff * 0.3  # Partial adjustment
            tgt_lab[:, :, 0] += adjustment
            tgt_lab = np.clip(tgt_lab, 0, 255).astype(np.uint8)
            return cv2.cvtColor(tgt_lab, cv2.COLOR_LAB2BGR)

        return target
