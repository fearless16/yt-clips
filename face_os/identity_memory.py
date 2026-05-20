"""
identity_memory.py — Module 8: Identity Memory Atlas (Photic Memory).

This is the "Photic Memory" concept — treating video pixels as noisy
photon observations and accumulating a better model over time.

Instead of processing each frame independently:
  - Each frame is a partial, noisy observation of the face
  - Over time, confidence accumulates for well-observed pixels
  - Hidden appearance details (pores, beard texture, skin tone) refine
  - Lighting response becomes more accurate

The memory atlas is a per-pixel confidence-weighted accumulation:
  - High confidence pixels: use accumulated appearance (stable, clean)
  - Low confidence pixels: use current frame (noisy but fresh)

This makes the face engine EVOLVE — getting better with every frame.
"""

from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import ConfidenceMap, FaceTrack, Landmarks


cfg = get_config()


class IdentityMemoryAtlas:
    """Per-pixel confidence accumulation for face identity.

    The atlas stores:
    - Accumulated appearance (RGB + LAB)
    - Per-pixel observation count
    - Per-pixel temporal stability score
    - Pose-weighted confidence

    Memory decays without new observations — pixels that aren't
    seen recently lose confidence over time.
    """

    def __init__(self, atlas_size: Tuple[int, int] = (256, 256)):
        self.atlas_size = atlas_size
        self._accumulated: Optional[np.ndarray] = None  # (H, W, 3) float64
        self._observation_count: Optional[np.ndarray] = None  # (H, W) float64
        self._temporal_stability: Optional[np.ndarray] = None  # (H, W) float64
        self._last_observation: Optional[np.ndarray] = None  # (H, W) int64 (frame idx)
        self._frame_count = 0
        self._pose_history = deque(maxlen=30)

    def initialize(self, first_observation: np.ndarray) -> None:
        """Initialize atlas with first observation.

        Args:
            first_observation: Face region (H, W, 3) in BGR
        """
        h, w = first_observation.shape[:2]
        self.atlas_size = (w, h)

        self._accumulated = cv2.cvtColor(first_observation, cv2.COLOR_BGR2LAB).astype(np.float64)
        self._observation_count = np.ones((h, w), dtype=np.float64)
        self._temporal_stability = np.ones((h, w), dtype=np.float64) * 0.5
        self._last_observation = np.full((h, w), self._frame_count, dtype=np.int64)

    def update(
        self,
        face_region: np.ndarray,
        confidence: float = 1.0,
        pose: Optional[Tuple[float, float, float]] = None,
    ) -> Optional[ConfidenceMap]:
        """Update memory with a new face observation.

        Args:
            face_region: Face region (H, W, 3) in BGR, aligned to canonical
            confidence: Detection/tracking confidence [0, 1]
            pose: Head pose (yaw, pitch, roll) for pose weighting

        Returns:
            ConfidenceMap with per-pixel confidence scores
        """
        self._frame_count += 1

        if self._accumulated is None:
            self.initialize(face_region)
            return self._make_confidence_map()

        # Resize if needed
        h, w = self.atlas_size[1], self.atlas_size[0]
        if face_region.shape[:2] != (h, w):
            face_region = cv2.resize(face_region, (w, h), interpolation=cv2.INTER_LANCZOS4)

        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB).astype(np.float64)

        # Compute per-pixel quality
        quality = self._compute_quality(face_region, confidence)

        # Pose weighting: prefer frontal observations for atlas
        pose_weight = self._compute_pose_weight(pose)

        # Accumulation rate (slower for well-observed pixels)
        rate = cfg.memory.accumulation_rate
        obs_factor = 1.0 / (1.0 + self._observation_count * 0.01)
        effective_rate = rate * obs_factor * quality * pose_weight

        # EMA update
        self._accumulated = self._accumulated * (1 - effective_rate) + lab * effective_rate

        # Update observation count
        self._observation_count += quality

        # Update temporal stability (how similar is this frame to recent history)
        if self._frame_count > 1:
            diff = np.abs(lab - self._accumulated)
            stability = np.exp(-np.mean(diff, axis=2) / 20.0)
            alpha = 0.1
            self._temporal_stability = self._temporal_stability * (1 - alpha) + stability * alpha

        # Decay unobserved pixels
        frames_since = self._frame_count - self._last_observation
        decay_mask = (frames_since > cfg.memory.max_age_frames).astype(np.float64)
        self._observation_count *= (1 - decay_mask * 0.5)

        # Update last observation
        self._last_observation = np.full((h, w), self._frame_count, dtype=np.int64)

        return self._make_confidence_map()

    def get_stable_face(self) -> Optional[np.ndarray]:
        """Get the accumulated stable face appearance.

        Returns:
            BGR image (H, W, 3) of the accumulated face, or None
        """
        if self._accumulated is None:
            return None

        lab = np.clip(self._accumulated, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def get_confidence(self) -> Optional[np.ndarray]:
        """Get per-pixel confidence map.

        Returns:
            Float32 array (H, W) with confidence [0, 1]
        """
        if self._observation_count is None:
            return None

        # Confidence = normalized observation count * temporal stability
        obs_conf = np.clip(self._observation_count / cfg.memory.min_observations, 0, 1)
        if self._temporal_stability is not None:
            return (obs_conf * self._temporal_stability).astype(np.float32)
        return obs_conf.astype(np.float32)

    def _compute_quality(self, face_region: np.ndarray, confidence: float) -> np.ndarray:
        """Compute per-pixel quality for weighting observations."""
        h, w = face_region.shape[:2]

        # Sharpness
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
        sharpness = np.clip(lap / 50.0, 0, 1)

        # Brightness (prefer well-lit)
        brightness = gray.astype(np.float32) / 255.0
        brightness_weight = 1.0 - np.abs(brightness - 0.5) * 2
        brightness_weight = np.clip(brightness_weight, 0.1, 1.0)

        return sharpness * brightness_weight * confidence

    def _compute_pose_weight(self, pose: Optional[Tuple[float, float, float]]) -> float:
        """Compute weight based on pose similarity to history.

        Frontal poses get higher weight for building the canonical atlas.
        """
        if not cfg.memory.use_pose_weighting or pose is None:
            return 1.0

        self._pose_history.append(pose)

        if len(self._pose_history) < 2:
            return 1.0

        # Average pose
        avg_yaw = np.mean([p[0] for p in self._pose_history])
        avg_pitch = np.mean([p[1] for p in self._pose_history])

        # Distance from average
        yaw_diff = abs(pose[0] - avg_yaw)
        pitch_diff = abs(pose[1] - avg_pitch)
        pose_dist = np.sqrt(yaw_diff ** 2 + pitch_diff ** 2)

        # Frontal bonus: prefer poses close to (0, 0, 0)
        frontal_dist = np.sqrt(pose[0] ** 2 + pose[1] ** 2)

        # Weight: higher for consistent, frontal poses
        consistency_weight = np.exp(-pose_dist / 30.0)
        frontal_weight = np.exp(-frontal_dist / 45.0)

        return float(consistency_weight * frontal_weight)

    def _make_confidence_map(self) -> ConfidenceMap:
        """Create a ConfidenceMap from current state."""
        confidence = self.get_confidence()
        if confidence is None:
            return ConfidenceMap()

        # Per-region quality scores
        h, w = confidence.shape

        # Eye region (top 40% of face, middle 60%)
        eye_region = confidence[int(h * 0.2):int(h * 0.4), int(w * 0.2):int(w * 0.8)]
        eye_quality = float(np.mean(eye_region)) if eye_region.size > 0 else 0.0

        # Skin region (lower face)
        skin_region = confidence[int(h * 0.4):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
        skin_quality = float(np.mean(skin_region)) if skin_region.size > 0 else 0.0

        # Contour region (edges of face)
        contour_region = confidence[int(h * 0.1):int(h * 0.9), :]
        contour_quality = float(np.mean(contour_region)) if contour_region.size > 0 else 0.0

        return ConfidenceMap(
            spatial_confidence=confidence,
            temporal_confidence=self._temporal_stability.astype(np.float32) if self._temporal_stability is not None else None,
            combined=confidence,
            eye_quality=eye_quality,
            skin_quality=skin_quality,
            contour_quality=contour_quality,
        )

    def reset(self) -> None:
        """Reset memory (call at start of each clip)."""
        self._accumulated = None
        self._observation_count = None
        self._temporal_stability = None
        self._last_observation = None
        self._frame_count = 0
        self._pose_history.clear()
