"""
temporal_solve.py — confidence-only bidirectional temporal propagation.

The temporal estimator may look backward and forward through the clip, but it
must not repair a frame by copying or blending RGB texture from another frame.
Its job is to propagate belief: confidence, drift, continuity, and motion
signals. The canonical face returned for a solved frame is therefore the
original/current canonical face unchanged.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


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
        """Combined quality score in [0, 1]."""
        q = self.sharpness * 0.6
        q += (1.0 - min(self.motion_blur / 50.0, 1.0)) * 0.2
        q += self.detection_confidence * 0.2
        return float(np.clip(q, 0.0, 1.0))


class BidirectionalSolver:
    """Propagates temporal confidence using past and future frame quality."""

    def __init__(self, lookback_frames: int = 10, lookahead_frames: int = 10):
        self.lookback = lookback_frames
        self.lookahead = lookahead_frames

        self._canonical_faces: Dict[int, np.ndarray] = {}
        self._quality_maps: Dict[int, np.ndarray] = {}
        self._frame_qualities: Dict[int, FrameQuality] = {}
        self._poses: Dict[int, Tuple[float, float, float]] = {}

        self._hq_frames: List[int] = []
        self._temporal_metrics: Dict[int, Dict[str, float]] = {}
        self._motion_fields: Dict[int, np.ndarray] = {}

    def add_frame(
        self,
        frame_idx: int,
        canonical_face: np.ndarray,
        quality_map: np.ndarray,
        frame_quality: FrameQuality,
    ) -> None:
        """Store a frame during the forward pass.

        Frames are intentionally stored by reference. The temporal solver only
        reads RGB data and returns the current frame unchanged.
        """
        self._canonical_faces[frame_idx] = canonical_face
        self._quality_maps[frame_idx] = quality_map
        self._frame_qualities[frame_idx] = frame_quality

        if frame_quality.pose is not None:
            self._poses[frame_idx] = frame_quality.pose

    def identify_hq_frames(self, quality_threshold: float = 0.3) -> List[int]:
        """Identify high-quality frames useful for confidence propagation."""
        hq: List[int] = []
        last_hq = -100
        sorted_indices = sorted(self._frame_qualities.keys())

        for idx in sorted_indices:
            fq = self._frame_qualities[idx]
            if fq.overall_quality >= quality_threshold and idx - last_hq >= 5:
                hq.append(idx)
                last_hq = idx

        if not hq and sorted_indices:
            sorted_by_quality = sorted(
                sorted_indices,
                key=lambda i: self._frame_qualities[i].overall_quality,
                reverse=True,
            )
            hq = sorted_by_quality[: max(1, len(sorted_by_quality) // 10)]

        self._hq_frames = hq
        return hq

    def _quality_map_for_frame(self, frame_idx: int, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        quality = self._quality_maps.get(frame_idx)

        if quality is None:
            fq = self._frame_qualities.get(frame_idx)
            fill = fq.overall_quality if fq is not None else 0.0
            return np.full((h, w), fill, dtype=np.float32)

        quality_arr = np.asarray(quality, dtype=np.float32)
        if quality_arr.ndim == 3:
            quality_arr = np.mean(quality_arr, axis=2)

        if quality_arr.shape != (h, w):
            fill = float(np.mean(quality_arr)) if quality_arr.size else 0.0
            return np.full((h, w), fill, dtype=np.float32)

        return np.clip(quality_arr, 0.0, 1.0).astype(np.float32, copy=False)

    def _reference_candidates(self, frame_idx: int) -> List[int]:
        candidates = []
        for idx in self._frame_qualities:
            if idx == frame_idx:
                continue
            if idx < frame_idx and frame_idx - idx <= self.lookback:
                candidates.append(idx)
            elif idx > frame_idx and idx - frame_idx <= self.lookahead:
                candidates.append(idx)

        if self._hq_frames:
            hq_in_window = [
                idx
                for idx in self._hq_frames
                if idx != frame_idx
                and (
                    (idx < frame_idx and frame_idx - idx <= self.lookback)
                    or (idx > frame_idx and idx - frame_idx <= self.lookahead)
                )
            ]
            candidates.extend(hq_in_window)

        return sorted(set(candidates), key=lambda idx: (abs(idx - frame_idx), idx))

    def _score_reference_frame(self, frame_idx: int, ref_idx: int) -> float:
        ref_quality = self._frame_qualities.get(ref_idx)
        if ref_quality is None:
            return float("-inf")

        dist = abs(ref_idx - frame_idx)
        temporal_weight = self._temporal_weight(dist)
        pose_sim = self._pose_similarity(frame_idx, ref_idx)
        quality = float(ref_quality.overall_quality)
        return float(quality * temporal_weight * pose_sim)

    def _select_best_reference(self, frame_idx: int) -> Optional[int]:
        best_idx = None
        best_score = float("-inf")

        for idx in self._reference_candidates(frame_idx):
            score = self._score_reference_frame(frame_idx, idx)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _temporal_weight(self, distance: int) -> float:
        window = max(self.lookahead, self.lookback, 1)
        return float(np.exp(-float(distance) / float(window)))

    def _propagated_confidence(
        self,
        frame_idx: int,
        canonical_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, Optional[int], float]:
        confidence = self._quality_map_for_frame(frame_idx, canonical_shape)
        best_ref: Optional[int] = None
        best_score = 0.0

        for ref_idx in self._reference_candidates(frame_idx):
            ref_quality = self._frame_qualities.get(ref_idx)
            if ref_quality is None:
                continue

            dist = abs(ref_idx - frame_idx)
            temporal_weight = self._temporal_weight(dist)
            pose_sim = self._pose_similarity(frame_idx, ref_idx)
            propagation = float(ref_quality.overall_quality * temporal_weight * pose_sim)

            ref_confidence = self._quality_map_for_frame(ref_idx, canonical_shape)
            confidence = np.maximum(confidence, ref_confidence * propagation)

            if propagation > best_score:
                best_score = propagation
                best_ref = ref_idx

        return np.clip(confidence, 0.0, 1.0).astype(np.float32), best_ref, best_score

    def _update_temporal_fields(
        self,
        frame_idx: int,
        ref_idx: Optional[int],
        propagation_score: float,
        confidence: np.ndarray,
        canonical_shape: Tuple[int, int],
    ) -> None:
        h, w = canonical_shape

        if ref_idx is None:
            pose_sim = 1.0
            quality_gap = 0.0
            motion_vector = (0.0, 0.0)
            temporal_weight = 1.0
        else:
            pose_sim = self._pose_similarity(frame_idx, ref_idx)
            current_q = self._frame_qualities.get(frame_idx)
            ref_q = self._frame_qualities.get(ref_idx)
            current_scalar = current_q.overall_quality if current_q is not None else float(np.mean(confidence))
            ref_scalar = ref_q.overall_quality if ref_q is not None else float(np.mean(confidence))
            quality_gap = abs(float(ref_scalar) - float(current_scalar))
            temporal_weight = self._temporal_weight(abs(ref_idx - frame_idx))
            motion_vector = self._pose_delta(frame_idx, ref_idx)

        drift_score = float(np.clip((1.0 - pose_sim) * 0.75 + quality_gap * 0.25, 0.0, 1.0))
        continuity_score = float(np.clip(pose_sim * temporal_weight, 0.0, 1.0))
        motion_norm = float(np.sqrt(motion_vector[0] ** 2 + motion_vector[1] ** 2))

        self._temporal_metrics[frame_idx] = {
            "reference_frame": float(ref_idx) if ref_idx is not None else -1.0,
            "propagation_score": float(np.clip(propagation_score, 0.0, 1.0)),
            "confidence_mean": float(np.mean(confidence)),
            "drift_score": drift_score,
            "continuity_score": continuity_score,
            "motion_field_norm": motion_norm,
            "rgb_texture_propagated": 0.0,
        }

        motion_field = np.zeros((h, w, 2), dtype=np.float32)
        motion_field[..., 0] = motion_vector[0]
        motion_field[..., 1] = motion_vector[1]
        self._motion_fields[frame_idx] = motion_field

    def solve_frame(self, frame_idx: int, canonical_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Return the current canonical face unchanged plus temporal confidence."""
        h, w = canonical_shape

        if frame_idx not in self._canonical_faces:
            confidence = np.zeros((h, w), dtype=np.float32)
            self._temporal_metrics[frame_idx] = {
                "reference_frame": -1.0,
                "propagation_score": 0.0,
                "confidence_mean": 0.0,
                "drift_score": 1.0,
                "continuity_score": 0.0,
                "motion_field_norm": 0.0,
                "rgb_texture_propagated": 0.0,
            }
            self._motion_fields[frame_idx] = np.zeros((h, w, 2), dtype=np.float32)
            return np.zeros((h, w, 3), dtype=np.uint8), confidence

        current = self._canonical_faces[frame_idx]
        confidence, best_ref, propagation_score = self._propagated_confidence(frame_idx, canonical_shape)
        self._update_temporal_fields(frame_idx, best_ref, propagation_score, confidence, canonical_shape)
        return current, confidence

    def solve_all(self, canonical_shape: Tuple[int, int]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        results = {}
        for idx in sorted(self._canonical_faces.keys()):
            results[idx] = self.solve_frame(idx, canonical_shape)
        return results

    def _find_nearest_hq(self, frame_idx: int, direction: str = "forward") -> Optional[int]:
        if not self._hq_frames:
            return None

        max_dist = self.lookahead if direction == "forward" else self.lookback

        if direction == "forward":
            candidates = [f for f in self._hq_frames if f > frame_idx and f - frame_idx <= max_dist]
            return candidates[0] if candidates else None

        candidates = [f for f in self._hq_frames if f < frame_idx and frame_idx - f <= max_dist]
        return candidates[-1] if candidates else None

    def _pose_similarity(self, idx1: int, idx2: int) -> float:
        pose1 = self._poses.get(idx1)
        pose2 = self._poses.get(idx2)

        if pose1 is None or pose2 is None:
            return 0.8

        yaw_diff = abs(pose1[0] - pose2[0])
        pitch_diff = abs(pose1[1] - pose2[1])
        dist = np.sqrt(yaw_diff**2 + pitch_diff**2)

        return float(np.exp(-dist / 30.0))

    def _pose_delta(self, idx1: int, idx2: int) -> Tuple[float, float]:
        pose1 = self._poses.get(idx1)
        pose2 = self._poses.get(idx2)

        if pose1 is None or pose2 is None:
            return (0.0, 0.0)

        return (
            float((pose2[0] - pose1[0]) / 30.0),
            float((pose2[1] - pose1[1]) / 30.0),
        )

    def get_hq_frame_count(self) -> int:
        return len(self._hq_frames)

    def get_temporal_metrics(self, frame_idx: int) -> Dict[str, float]:
        return dict(self._temporal_metrics.get(frame_idx, {}))

    def get_motion_field(self, frame_idx: int) -> Optional[np.ndarray]:
        return self._motion_fields.get(frame_idx)

    def clear(self) -> None:
        self._canonical_faces.clear()
        self._quality_maps.clear()
        self._frame_qualities.clear()
        self._poses.clear()
        self._hq_frames.clear()
        self._temporal_metrics.clear()
        self._motion_fields.clear()


class TemporalRepairEngine:
    """Pipeline-compatible wrapper for confidence-only temporal propagation."""

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
