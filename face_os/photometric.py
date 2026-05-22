"""
photometric.py — Temporal Photometric Stabilization.

Upstream module that locks frame-to-frame luminance via EMA.
Compositor does NOT handle this — it only does final blend.

D-01e: Prevents temporal photometric instability from side-screen
light reflections and per-frame luminance jitter.
"""

from typing import Optional

import cv2
import numpy as np


# Module-level temporal state
_prev_luminance: Optional[float] = None
_luminance_ema_alpha: float = 0.5
_luminance_clamp: float = 8.0  # LAB units


def reset_photometric_lock() -> None:
    """Reset photometric lock temporal state between clips."""
    global _prev_luminance
    _prev_luminance = None


def photometric_lock(frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Temporal photometric locking via luminance EMA.

    Converts to LAB, computes mean L (optionally within mask),
    applies EMA smoothing with ±8 LAB clamp, returns locked frame.

    Args:
        frame: Input frame, shape (H, W, 3), uint8 BGR.
        mask: Optional face mask, shape (H, W), float [0, 1].

    Returns:
        Photometrically locked frame, shape (H, W, 3), uint8 BGR.
    """
    global _prev_luminance

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

    if mask is not None and mask.sum() > 100:
        luma = float(lab[:, :, 0][mask > 0.5].mean())
    else:
        luma = float(lab[:, :, 0].mean())

    if _prev_luminance is None:
        _prev_luminance = luma
        return frame

    _prev_luminance = _luminance_ema_alpha * luma + (1 - _luminance_ema_alpha) * _prev_luminance
    delta = _prev_luminance - luma

    if abs(delta) < 2.0:
        return frame

    delta = float(np.clip(delta, -_luminance_clamp, _luminance_clamp))
    lab[:, :, 0] = np.clip(lab[:, :, 0] + delta, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
