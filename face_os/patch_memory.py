"""
patch_memory.py — Per-Region Patch Memory with Independent Dynamics.

BEAST MODE FIXES:
- Nuked the dead `expression` and `lighting` bins (pipeline doesn't pass them).
- Deleted the unused `query_all` CPU nuke (Gaussian mgrid).
- Optimized memory copies (only copy when actually updating the best patch).
- Simplified pose binning strings for faster dictionary lookups.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config

cfg = get_config()


# Canonical region definitions (in 256x256 canonical space)
REGION_DEFS = {
    'left_eye': {
        'bounds': (0.20, 0.28, 0.42, 0.42),
        'update_rate': 0.05,
        'freeze_on_blink': True,
        'priority': 'critical',
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
        'priority': 'high',
    },
    'skin': {
        'bounds': (0.20, 0.40, 0.80, 0.65),
        'update_rate': 0.12,
        'freeze_on_blink': False,
        'priority': 'medium',
    },
    'forehead': {
        'bounds': (0.25, 0.10, 0.75, 0.30),
        'update_rate': 0.03,
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
        'update_rate': 0.15,
        'freeze_on_blink': False,
        'priority': 'high',
    },
}


def _pose_bin(pose: Tuple[float, float, float]) -> str:
    """Quantize pose to a fast, short bin label."""
    if pose is None:
        return 'any'
    yaw, pitch, _ = pose
    
    if abs(yaw) < 10 and abs(pitch) < 10:
        return 'F'  # Frontal
    if yaw < -10:
        return f'L{int(abs(yaw) // 10) * 10}'
    if yaw > 10:
        return f'R{int(yaw // 10) * 10}'
    if pitch < -10:
        return f'D{int(abs(pitch) // 10) * 10}'
    if pitch > 10:
        return f'U{int(pitch // 10) * 10}'
    return 'F'


class RegionPatch:
    """Memory for a single face region."""

    def __init__(self, region_name: str, region_def: dict):
        self.name = region_name
        self.bounds = region_def['bounds']
        self.update_rate = region_def['update_rate']
        self.freeze_on_blink = region_def['freeze_on_blink']
        self.priority = region_def['priority']

        self.best_patch: Optional[np.ndarray] = None
        self.best_quality: float = 0.0
        
        self.pose_patches: Dict[str, np.ndarray] = {}
        self.pose_qualities: Dict[str, float] = {}

        self.observation_count: int = 0
        self.current_confidence: float = 0.0

    def extract_region(self, canonical_face: np.ndarray) -> Optional[np.ndarray]:
        h, w = canonical_face.shape[:2]
        x1f, y1f, x2f, y2f = self.bounds
        x1, y1 = int(x1f * w), int(y1f * h)
        x2, y2 = int(x2f * w), int(y2f * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Return a copy so upstream mutations don't break our memory
        return canonical_face[y1:y2, x1:x2].copy()

    def update(
        self,
        canonical_face: np.ndarray,
        quality_map: np.ndarray,
        pose: Optional[Tuple[float, float, float]] = None,
        is_blink: bool = False,
        frame_idx: int = 0,
    ) -> float:
        self.observation_count += 1

        if is_blink and self.freeze_on_blink:
            return self.current_confidence

        # Extract quality for this region
        h, w = quality_map.shape
        x1f, y1f, x2f, y2f = self.bounds
        x1, y1 = int(x1f * w), int(y1f * h)
        x2, y2 = int(x2f * w), int(y2f * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return self.current_confidence

        region_quality_map = quality_map[y1:y2, x1:x2]
        if region_quality_map.size == 0:
            return self.current_confidence

        region_quality = float(np.mean(region_quality_map))

        # BEAST MODE FIX: Only extract and copy the patch if it's actually better!
        # Saves massive GC pressure from useless array copies.
        if region_quality > self.best_quality:
            patch = self.extract_region(canonical_face)
            if patch is not None:
                self.best_patch = patch
                self.best_quality = region_quality
                
                if pose is not None:
                    pbin = _pose_bin(pose)
                    if pbin not in self.pose_qualities or region_quality > self.pose_qualities[pbin]:
                        self.pose_patches[pbin] = patch  # Already a copy from extract_region
                        self.pose_qualities[pbin] = region_quality

        obs_factor = min(self.observation_count / 10.0, 1.0)
        self.current_confidence = region_quality * obs_factor

        return self.current_confidence

    def query(
        self,
        pose: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        if self.best_patch is None:
            return None, 0.0

        if pose is not None and self.pose_patches:
            pbin = _pose_bin(pose)
            if pbin in self.pose_patches:
                return self.pose_patches[pbin], self.current_confidence

            # Fallback to closest available pose bin
            closest = self._find_closest_pose(pbin)
            if closest and closest in self.pose_patches:
                return self.pose_patches[closest], self.current_confidence * 0.8

        return self.best_patch, self.current_confidence

    def _find_closest_pose(self, target_bin: str) -> Optional[str]:
        if not self.pose_patches:
            return None

        if target_bin == 'F':
            for b in self.pose_patches:
                if b == 'F': return b
        elif target_bin.startswith('L'):
            for b in self.pose_patches:
                if b.startswith('L'): return b
        elif target_bin.startswith('R'):
            for b in self.pose_patches:
                if b.startswith('R'): return b

        return next(iter(self.pose_patches), None)


class PatchMemory:
    """Per-region patch memory for the entire face."""

    def __init__(self):
        self.regions: Dict[str, RegionPatch] = {}
        self._initialized = False

    def initialize(self, canonical_face: np.ndarray, quality_map: np.ndarray) -> None:
        for name, rdef in REGION_DEFS.items():
            self.regions[name] = RegionPatch(name, rdef)

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
        if not self._initialized:
            self.initialize(canonical_face, quality_map)
            return {name: 0.1 for name in self.regions}

        confidences = {}
        for name, region in self.regions.items():
            conf = region.update(
                canonical_face, quality_map, pose, is_blink, frame_idx
            )
            confidences[name] = conf

        return confidences

    def query_region(
        self,
        region_name: str,
        pose: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        if region_name not in self.regions:
            return None, 0.0

        return self.regions[region_name].query(pose)

    def get_region_confidences(self) -> Dict[str, float]:
        return {
            name: region.current_confidence
            for name, region in self.regions.items()
        }

    def reset(self) -> None:
        self.regions.clear()
        self._initialized = False