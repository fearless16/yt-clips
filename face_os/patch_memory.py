"""
patch_memory.py — Per-Region Patch Memory with Independent Dynamics.

CORE IDEA:
  Different face regions have DIFFERENT temporal dynamics:
  - Eyes: blink frequently, but structure is very stable between blinks
  - Beard: moves with jaw, but texture is extremely stable
  - Skin: changes with lighting, but tone is consistent
  - Forehead: almost never changes (least expressive region)
  - Lips: highly dynamic (speech, expressions)

  Each region maintains INDEPENDENT:
  - Best observation (pose-conditioned)
  - Confidence trajectory
  - Update rate
  - Freeze conditions

POSE-CONDITIONED PATCH DATABASE:
  Store best patches per pose bin:
  - frontal neutral
  - frontal smile
  - left yaw 15°
  - right yaw 15°
  - slight up/down

  When reconstructing, query the patch database for the closest pose.
  This kills: smear, fake texture, temporal instability.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config

cfg = get_config()


# Canonical region definitions (in 256x256 canonical space)
REGION_DEFS = {
    'left_eye': {
        'bounds': (0.20, 0.28, 0.42, 0.42),
        'update_rate': 0.05,      # Slow update (preserve structure)
        'freeze_on_blink': True,
        'priority': 'critical',   # Never hallucinate
    },
    'right_eye': {
        'bounds': (0.58, 0.28, 0.80, 0.42),
        'update_rate': 0.05,
        'freeze_on_blink': True,
        'priority': 'critical',
    },
    'beard': {
        'bounds': (0.25, 0.55, 0.75, 0.85),
        'update_rate': 0.08,
        'freeze_on_blink': False,
        'priority': 'high',       # Texture matters
    },
    'skin': {
        'bounds': (0.20, 0.40, 0.80, 0.65),
        'update_rate': 0.12,
        'freeze_on_blink': False,
        'priority': 'medium',
    },
    'forehead': {
        'bounds': (0.25, 0.10, 0.75, 0.30),
        'update_rate': 0.03,      # Very slow (almost never changes)
        'freeze_on_blink': False,
        'priority': 'low',
    },
    'nose': {
        'bounds': (0.40, 0.38, 0.60, 0.55),
        'update_rate': 0.10,
        'freeze_on_blink': False,
        'priority': 'medium',
    },
    'lips': {
        'bounds': (0.35, 0.60, 0.65, 0.72),
        'update_rate': 0.15,      # Fast update (highly dynamic)
        'freeze_on_blink': False,
        'priority': 'high',
    },
}


def _pose_bin(pose: Tuple[float, float, float]) -> str:
    """Quantize pose to a bin label.

    Returns: e.g., 'frontal_neutral', 'left_15', 'right_15', etc.
    """
    yaw, pitch, roll = pose

    if abs(yaw) < 10 and abs(pitch) < 10:
        return 'frontal_neutral'
    elif yaw < -10:
        return f'left_{int(abs(yaw) // 10) * 10}'
    elif yaw > 10:
        return f'right_{int(yaw // 10) * 10}'
    elif pitch < -10:
        return f'down_{int(abs(pitch) // 10) * 10}'
    elif pitch > 10:
        return f'up_{int(pitch // 10) * 10}'
    else:
        return 'frontal_neutral'


class RegionPatch:
    """Memory for a single face region.

    Stores:
    - best_patch: highest quality observation (NEVER averaged)
    - best_quality: quality of best observation
    - best_pose: pose when best was captured
    - pose_patches: dict of pose_bin → best patch for that pose
    - observation_count: total observations
    - current_confidence: current confidence level
    """

    def __init__(self, region_name: str, region_def: dict):
        self.name = region_name
        self.bounds = region_def['bounds']
        self.update_rate = region_def['update_rate']
        self.freeze_on_blink = region_def['freeze_on_blink']
        self.priority = region_def['priority']

        # Best observation (never averaged)
        self.best_patch: Optional[np.ndarray] = None
        self.best_quality: float = 0.0
        self.best_pose: Optional[Tuple[float, float, float]] = None

        # Pose-conditioned patch database
        self.pose_patches: Dict[str, np.ndarray] = {}
        self.pose_qualities: Dict[str, float] = {}

        # Stats
        self.observation_count: int = 0
        self.current_confidence: float = 0.0
        self.last_update_frame: int = -1

    def extract_region(
        self,
        canonical_face: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Extract this region from a canonical face image.

        Args:
            canonical_face: (H, W, 3) BGR in canonical space

        Returns:
            Region patch (h_r, w_r, 3) or None
        """
        h, w = canonical_face.shape[:2]
        x1f, y1f, x2f, y2f = self.bounds
        x1, y1 = int(x1f * w), int(y1f * h)
        x2, y2 = int(x2f * w), int(y2f * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return canonical_face[y1:y2, x1:x2].copy()

    def update(
        self,
        canonical_face: np.ndarray,
        quality_map: np.ndarray,
        pose: Optional[Tuple[float, float, float]] = None,
        is_blink: bool = False,
        frame_idx: int = 0,
    ) -> float:
        """Update this region's memory.

        Rules:
        1. If blink and freeze_on_blink → skip update
        2. Extract region patch
        3. Compute region quality
        4. If quality > best_quality → update best (NEVER average)
        5. Update pose-conditioned database
        6. Compute confidence

        Returns:
            Updated confidence for this region
        """
        self.observation_count += 1

        # Freeze during blinks for eye regions
        if is_blink and self.freeze_on_blink:
            return self.current_confidence

        # Extract region
        patch = self.extract_region(canonical_face)
        if patch is None:
            return self.current_confidence

        # Extract quality for this region
        h, w = quality_map.shape
        x1f, y1f, x2f, y2f = self.bounds
        x1, y1 = int(x1f * w), int(y1f * h)
        x2, y2 = int(x2f * w), int(y2f * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        region_quality_map = quality_map[y1:y2, x1:x2]
        if region_quality_map.size == 0:
            return self.current_confidence

        region_quality = float(np.mean(region_quality_map))

        # Update best observation (NEVER average — only replace if better)
        if region_quality > self.best_quality:
            self.best_patch = patch.copy()
            self.best_quality = region_quality
            self.best_pose = pose
            self.last_update_frame = frame_idx

        # Update pose-conditioned database
        if pose is not None:
            pbin = _pose_bin(pose)
            if pbin not in self.pose_qualities or region_quality > self.pose_qualities[pbin]:
                self.pose_patches[pbin] = patch.copy()
                self.pose_qualities[pbin] = region_quality

        # Confidence = quality * observation_factor
        obs_factor = min(self.observation_count / 10.0, 1.0)
        self.current_confidence = region_quality * obs_factor

        return self.current_confidence

    def query(
        self,
        pose: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Query the best patch for this region.

        If pose is available, returns the best patch for that pose bin.
        Otherwise, returns the overall best patch.

        Returns:
            (patch, confidence) or (None, 0.0)
        """
        if self.best_patch is None:
            return None, 0.0

        # Try pose-specific patch first
        if pose is not None and self.pose_patches:
            pbin = _pose_bin(pose)
            if pbin in self.pose_patches:
                return self.pose_patches[pbin], self.current_confidence

            # Find closest pose bin
            closest = self._find_closest_pose(pbin)
            if closest and closest in self.pose_patches:
                return self.pose_patches[closest], self.current_confidence * 0.8

        return self.best_patch, self.current_confidence

    def _find_closest_pose(self, target_bin: str) -> Optional[str]:
        """Find the closest available pose bin."""
        if not self.pose_patches:
            return None

        # Simple distance: prefer same category
        if target_bin.startswith('frontal'):
            for b in self.pose_patches:
                if b.startswith('frontal'):
                    return b
        elif target_bin.startswith('left'):
            for b in self.pose_patches:
                if b.startswith('left'):
                    return b
        elif target_bin.startswith('right'):
            for b in self.pose_patches:
                if b.startswith('right'):
                    return b

        # Fallback: any available
        return next(iter(self.pose_patches), None)


class PatchMemory:
    """Per-region patch memory for the entire face.

    Each region has independent dynamics:
    - Eyes: preserve structure, freeze on blink
    - Beard: preserve texture
    - Skin: preserve tone
    - Forehead: almost never changes
    - Lips: fast update (speech/expression)

    The patch database stores pose-conditioned best patches.
    When reconstructing, we query the closest pose bin.
    """

    def __init__(self):
        self.regions: Dict[str, RegionPatch] = {}
        self._initialized = False

    def initialize(self, canonical_face: np.ndarray, quality_map: np.ndarray) -> None:
        """Initialize all regions from first observation."""
        for name, rdef in REGION_DEFS.items():
            self.regions[name] = RegionPatch(name, rdef)

        # Update all with first observation
        for name, region in self.regions.items():
            region.update(canonical_face, quality_map, pose=None, frame_idx=0)

        self._initialized = True

    def update(
        self,
        canonical_face: np.ndarray,
        quality_map: np.ndarray,
        pose: Optional[Tuple[float, float, float]] = None,
        is_blink: bool = False,
        frame_idx: int = 0,
    ) -> Dict[str, float]:
        """Update all regions with new observation.

        Returns:
            Dict of region_name → confidence
        """
        if not self._initialized:
            self.initialize(canonical_face, quality_map)
            return {name: 0.1 for name in self.regions}

        confidences = {}
        for name, region in self.regions.items():
            conf = region.update(canonical_face, quality_map, pose, is_blink, frame_idx)
            confidences[name] = conf

        return confidences

    def query_region(
        self,
        region_name: str,
        pose: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Query best patch for a specific region.

        Returns:
            (patch, confidence) or (None, 0.0)
        """
        if region_name not in self.regions:
            return None, 0.0

        return self.regions[region_name].query(pose)

    def query_all(
        self,
        canonical_shape: Tuple[int, int],
        pose: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct full face from patch memory.

        Composites all region patches back into a full canonical face.

        Returns:
            (reconstructed_face, per_pixel_confidence)
        """
        h, w = canonical_shape
        result = np.zeros((h, w, 3), dtype=np.float32)
        conf_map = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for name, region in self.regions.items():
            patch, conf = region.query(pose)
            if patch is None or conf < 0.01:
                continue

            # Get region bounds
            x1f, y1f, x2f, y2f = region.bounds
            x1, y1 = int(x1f * w), int(y1f * h)
            x2, y2 = int(x2f * w), int(y2f * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Resize patch to fit
            patch_resized = cv2.resize(
                patch, (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_LANCZOS4,
            )

            # Gaussian weight (center of region = higher weight)
            rh, rw = y2 - y1, x2 - x1
            yy, xx = np.mgrid[0:rh, 0:rw]
            center_y, center_x = rh / 2, rw / 2
            sigma_y, sigma_x = rh / 3, rw / 3
            gaussian = np.exp(
                -((yy - center_y) ** 2 / (2 * sigma_y ** 2) +
                  (xx - center_x) ** 2 / (2 * sigma_x ** 2))
            ).astype(np.float32)

            weight = gaussian * conf

            result[y1:y2, x1:x2] += patch_resized.astype(np.float32) * weight[:, :, np.newaxis]
            conf_map[y1:y2, x1:x2] += weight
            weight_map[y1:y2, x1:x2] += weight

        # Normalize
        valid = weight_map > 0
        if valid.any():
            for c in range(3):
                result[:, :, c][valid] /= weight_map[valid]
            conf_map[valid] /= weight_map[valid]

        result = np.clip(result, 0, 255).astype(np.uint8)
        return result, conf_map.astype(np.float32)

    def get_region_confidences(self) -> Dict[str, float]:
        """Get current confidence for all regions."""
        return {
            name: region.current_confidence
            for name, region in self.regions.items()
        }

    def reset(self) -> None:
        """Reset all patch memory."""
        self.regions.clear()
        self._initialized = False
