"""
temporal_solve.py — Bidirectional Temporal Solver.

THE OFFLINE PIPELINE'S SUPERPOWER:
  Real-time systems can only look at past frames.
  We can look at BOTH past AND future.

  Current: past → present
  Needed:  past + future → present

  Example:
    - Frame 100 is blurry (motion blur)
    - Frame 103 is sharp
    - Frame 103's sharp data should REPAIR frame 100

  This is CINEMA RESTORATION territory.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config

cfg = get_config()


class FrameQuality:
    """Quality metrics for a single frame."""

    def __init__(
        self,
        frame_idx: int,
        sharpness: float = 0.0,
        motion_blur: float = 0.0,
        pose: Optional[Tuple[float, float, float]] = None,
        detection_confidence: float = 0.0,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.frame_idx = frame_idx
        self.sharpness = sharpness
        self.motion_blur = motion_blur
        self.pose = pose
        self.detection_confidence = detection_confidence
        self.face_bbox = face_bbox

    @property
    def overall_quality(self) -> float:
        """Combined quality score."""
        q = self.sharpness * 0.6
        q += (1.0 - min(self.motion_blur / 50.0, 1.0)) * 0.2
        q += self.detection_confidence * 0.2
        return float(np.clip(q, 0, 1))


class BidirectionalSolver:
    """Solves identity reconstruction using both past and future frames."""

    def __init__(self, lookback_frames: int = 10, lookahead_frames: int = 10):
        self.lookback = lookback_frames
        self.lookahead = lookahead_frames

        self._canonical_faces: Dict[int, np.ndarray] = {}
        self._quality_maps: Dict[int, np.ndarray] = {}
        self._frame_qualities: Dict[int, FrameQuality] = {}
        self._poses: Dict[int, Tuple[float, float, float]] = {}

        self._hq_frames: List[int] = []

    def add_frame(
        self,
        frame_idx: int,
        canonical_face: np.ndarray,
        quality_map: np.ndarray,
        frame_quality: FrameQuality,
    ) -> None:
        """Store a frame during the forward pass."""
        # Stop doing .copy() everywhere, save the RAM from getting raped
        self._canonical_faces[frame_idx] = canonical_face
        self._quality_maps[frame_idx] = quality_map
        self._frame_qualities[frame_idx] = frame_quality

        if frame_quality.pose is not None:
            self._poses[frame_idx] = frame_quality.pose

    def identify_hq_frames(self, quality_threshold: float = 0.3) -> List[int]:
        """Identify high-quality frames for propagation."""
        hq = []
        last_hq = -100
        sorted_indices = sorted(self._frame_qualities.keys())

        for idx in sorted_indices:
            fq = self._frame_qualities[idx]
            if fq.overall_quality >= quality_threshold:
                if idx - last_hq >= 5:
                    hq.append(idx)
                    last_hq = idx

        # BHENCHOD SAFETY NET: If the whole video is shit and no frame passes threshold,
        # pick the top 10% least-shitty frames so the solver doesn't sit idle.
        if not hq and sorted_indices:
            sorted_by_quality = sorted(
                sorted_indices, 
                key=lambda i: self._frame_qualities[i].overall_quality, 
                reverse=True
            )
            hq = sorted_by_quality[:max(1, len(sorted_by_quality) // 10)]

        self._hq_frames = hq
        return hq

    def _score_reference_frame(self, frame_idx: int, ref_idx: int) -> float:
        ref_quality = self._frame_qualities.get(ref_idx)
        if ref_quality is None:
            return float("-inf")

        dist = abs(ref_idx - frame_idx)
        temporal_weight = float(np.exp(-dist / max(self.lookahead, self.lookback, 1)))
        pose_sim = self._pose_similarity(frame_idx, ref_idx)
        quality = float(ref_quality.overall_quality)
        return float(quality * temporal_weight * pose_sim)

    def _select_best_reference(self, frame_idx: int) -> Optional[int]:
        if frame_idx not in self._canonical_faces:
            return None

        candidates = [frame_idx]

        forward_hq = self._find_nearest_hq(frame_idx, direction='forward')
        backward_hq = self._find_nearest_hq(frame_idx, direction='backward')
        if forward_hq is not None:
            candidates.append(forward_hq)
        if backward_hq is not None:
            candidates.append(backward_hq)

        best_idx = None
        best_score = float("-inf")
        for idx in candidates:
            score = self._score_reference_frame(frame_idx, idx)
            if idx == frame_idx:
                current_quality = self._frame_qualities.get(idx)
                score = current_quality.overall_quality if current_quality is not None else score
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def solve_frame(self, frame_idx: int, canonical_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        h, w = canonical_shape

        if frame_idx not in self._canonical_faces:
            return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w), dtype=np.float32)

        current = self._canonical_faces[frame_idx]
        current_quality = self._quality_maps.get(frame_idx, np.ones((h, w), dtype=np.float32) * 0.5)
        fq = self._frame_qualities.get(frame_idx)

        if fq and fq.overall_quality >= 0.7:
            return current, np.clip(current_quality, 0, 1).astype(np.float32)

        best_ref = self._select_best_reference(frame_idx)
        if best_ref is None or best_ref == frame_idx:
            return current, np.clip(current_quality, 0, 1).astype(np.float32)

        ref_face = self._canonical_faces.get(best_ref)
        ref_quality = self._quality_maps.get(best_ref)
        if ref_face is None or ref_quality is None:
            return current, np.clip(current_quality, 0, 1).astype(np.float32)

        best_quality_scalar = float(self._frame_qualities.get(best_ref).overall_quality) if self._frame_qualities.get(best_ref) else float(np.mean(ref_quality))
        current_quality_scalar = float(fq.overall_quality) if fq is not None else float(np.mean(current_quality))
        dist = abs(best_ref - frame_idx)
        temporal_weight = float(np.exp(-dist / max(self.lookahead, self.lookback, 1)))
        pose_sim = self._pose_similarity(frame_idx, best_ref)
        repair_strength = np.clip(best_quality_scalar * temporal_weight * pose_sim, 0.0, 1.0)

        current_f = current.astype(np.float32)
        ref_f = ref_face.astype(np.float32)

        current_low = cv2.GaussianBlur(current_f, (0, 0), 2.0)
        ref_low = cv2.GaussianBlur(ref_f, (0, 0), 2.0)
        ref_high = ref_f - ref_low

        low_mix = np.clip(0.2 + 0.8 * repair_strength, 0.0, 1.0)
        repaired_low = current_low * (1.0 - low_mix) + ref_low * low_mix

        # BEAST MODE GHOSTING FIX: 
        # Only inject reference HF if pose is almost identical (>0.85).
        # Otherwise, temporal ghosting will make the face look like it has two noses.
        if pose_sim > 0.85:
            high_mix = np.clip(0.65 + 0.35 * repair_strength, 0.65, 1.0)
            repaired_high = ref_high * high_mix
        else:
            # Keep current frame's high freq to avoid double-vision ghosting
            current_high = current_f - current_low
            repaired_high = current_high

        result = repaired_low + repaired_high
        result = np.clip(result, 0, 255).astype(np.uint8)

        confidence = np.maximum(current_quality, ref_quality * (0.5 + 0.5 * temporal_weight * pose_sim))
        confidence = np.clip(confidence, 0, 1).astype(np.float32)

        return result, confidence

    def solve_all(self, canonical_shape: Tuple[int, int]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        results = {}
        for idx in sorted(self._canonical_faces.keys()):
            results[idx] = self.solve_frame(idx, canonical_shape)
        return results

    def _find_nearest_hq(self, frame_idx: int, direction: str = 'forward') -> Optional[int]:
        if not self._hq_frames:
            return None

        max_dist = self.lookahead if direction == 'forward' else self.lookback

        if direction == 'forward':
            candidates = [f for f in self._hq_frames if f > frame_idx and f - frame_idx <= max_dist]
            return candidates[0] if candidates else None
        else:
            candidates = [f for f in self._hq_frames if f < frame_idx and frame_idx - f <= max_dist]
            return candidates[-1] if candidates else None

    def _pose_similarity(self, idx1: int, idx2: int) -> float:
        pose1 = self._poses.get(idx1)
        pose2 = self._poses.get(idx2)

        if pose1 is None or pose2 is None:
            return 0.8 

        yaw_diff = abs(pose1[0] - pose2[0])
        pitch_diff = abs(pose1[1] - pose2[1])
        dist = np.sqrt(yaw_diff ** 2 + pitch_diff ** 2)

        return float(np.exp(-dist / 30.0))

    def get_hq_frame_count(self) -> int:
        return len(self._hq_frames)

    def clear(self) -> None:
        self._canonical_faces.clear()
        self._quality_maps.clear()
        self._frame_qualities.clear()
        self._poses.clear()
        self._hq_frames.clear()


class TemporalRepairEngine:
    """Combines bidirectional solving with identity state."""

    def __init__(self, lookback: int = 10, lookahead: int = 10):
        self.solver = BidirectionalSolver(lookback, lookahead)
        self._frame_buffer: Dict[int, np.ndarray] = {}
        self._quality_buffer: Dict[int, np.ndarray] = {}

    def collect_frame(
        self,
        frame_idx: int,
        canonical_face: np.ndarray,
        quality_map: np.ndarray,
        sharpness: float = 0.0,
        motion_blur: float = 0.0,
        pose: Optional[Tuple[float, float, float]] = None,
        detection_confidence: float = 0.5,
    ) -> None:
        # Stop doing .copy() here as well, let the solver handle references
        self._frame_buffer[frame_idx] = canonical_face
        self._quality_buffer[frame_idx] = quality_map

        fq = FrameQuality(
            frame_idx=frame_idx,
            sharpness=sharpness,
            motion_blur=motion_blur,
            pose=pose,
            detection_confidence=detection_confidence,
        )

        self.solver.add_frame(frame_idx, canonical_face, quality_map, fq)

    def solve(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        hq = self.solver.identify_hq_frames()
        print(f"  Bidirectional solver: {len(hq)} HQ frames identified")

        if not self._frame_buffer:
            return {}

        return self.solver.solve_all(
            canonical_shape=next(iter(self._frame_buffer.values())).shape[:2]
        )

    def clear(self) -> None:
        self.solver.clear()
        self._frame_buffer.clear()
        self._quality_buffer.clear()