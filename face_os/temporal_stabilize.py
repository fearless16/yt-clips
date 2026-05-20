"""
temporal_stabilize.py — Module 6: Temporal Stabilizer.

Implements "Identity Inertia" — the principle that:
  ΔI_identity ≪ ΔI_source

The source video may have wild exposure swings, colored light reflections,
and compression artifacts. But the person's identity should remain stable.

This module:
  - Tracks frame-to-frame appearance changes
  - Suppresses flicker (sudden L/a/b shifts not caused by real motion)
  - Uses temporal window averaging for stable output
  - Applies motion compensation to avoid smearing during real movement
"""

from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import FaceTrack, FrameData


cfg = get_config()


class TemporalStabilizer:
    """Maintains temporal consistency across frames.

    Core algorithm:
    1. Convert each frame to LAB
    2. Track per-frame LAB statistics (mean, std)
    3. If a frame deviates too much from the rolling average,
       pull it back toward the average (flicker suppression)
    4. If motion is detected (optical flow), reduce stabilization
       to avoid smearing real movement
    """

    def __init__(self):
        self._window = deque(maxlen=cfg.temporal.temporal_window)
        self._lab_history = deque(maxlen=30)
        self._prev_gray: Optional[np.ndarray] = None
        self._motion_score = 0.0

    def stabilize(
        self,
        frame: np.ndarray,
        face_track: Optional[FaceTrack] = None,
    ) -> np.ndarray:
        """Apply temporal stabilization to a frame.

        Returns the stabilized frame (may be identical to input if stable).
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Compute per-channel statistics
        l_mean = float(np.mean(lab[:, :, 0]))
        a_mean = float(np.mean(lab[:, :, 1]))
        b_mean = float(np.mean(lab[:, :, 2]))

        current_stats = np.array([l_mean, a_mean, b_mean])
        self._lab_history.append(current_stats)

        # Compute motion score (optical flow magnitude)
        self._motion_score = self._compute_motion(frame)

        # Need at least 3 frames to stabilize
        if len(self._lab_history) < 3:
            return frame

        # Rolling average of LAB stats
        history = np.array(list(self._lab_history))
        avg_stats = np.mean(history, axis=0)
        std_stats = np.std(history, axis=0)

        # How far is this frame from the average?
        deviation = current_stats - avg_stats
        deviation_magnitude = np.sqrt(np.sum(deviation ** 2))

        # Threshold: if deviation is large, it's likely flicker
        threshold = cfg.temporal.flicker_threshold

        if deviation_magnitude > threshold:
            # Apply correction — pull toward average
            # But reduce correction if there's real motion
            motion_factor = 1.0 - min(self._motion_score / 50.0, 0.8)
            inertia = cfg.temporal.identity_inertia

            correction = deviation * inertia * motion_factor
            corrected_stats = current_stats - correction

            # Apply correction to LAB channels
            lab[:, :, 0] += (corrected_stats[0] - current_stats[0])
            lab[:, :, 1] += (corrected_stats[1] - current_stats[1])
            lab[:, :, 2] += (corrected_stats[2] - current_stats[2])

            lab = np.clip(lab, 0, 255).astype(np.uint8)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return frame

    def stabilize_face_region(
        self,
        frame: np.ndarray,
        face_mask: np.ndarray,
        face_track: Optional[FaceTrack] = None,
    ) -> np.ndarray:
        """Stabilize only the face region (background left untouched).

        This is more precise — only the face gets identity inertia,
        while the background retains its natural variation.
        """
        if face_mask is None or face_mask.max() < 0.01:
            return self.stabilize(frame, face_track)

        # Stabilize full frame
        stabilized = self.stabilize(frame, face_track)

        # Blend: face region gets stabilization, background stays original
        mask_3ch = face_mask[:, :, np.newaxis]
        result = frame.astype(np.float32) * (1 - mask_3ch) + stabilized.astype(np.float32) * mask_3ch
        return np.clip(result, 0, 255).astype(np.uint8)

    def _compute_motion(self, frame: np.ndarray) -> float:
        """Compute motion score from optical flow magnitude."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0

        # Resize for speed
        h, w = gray.shape
        small_h, small_w = h // 4, w // 4
        small_curr = cv2.resize(gray, (small_w, small_h))
        small_prev = cv2.resize(self._prev_gray, (small_w, small_h))

        # Dense optical flow (Farneback)
        try:
            flow = cv2.calcOpticalFlowFarneback(
                small_prev, small_curr,
                None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )
            magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
            motion_score = float(np.mean(magnitude))
        except Exception:
            motion_score = 0.0

        self._prev_gray = gray
        return motion_score

    def reset(self) -> None:
        """Reset temporal state (call at start of each clip)."""
        self._window.clear()
        self._lab_history.clear()
        self._prev_gray = None
        self._motion_score = 0.0
