"""
SUBSYSTEM C — TEMPORAL ESTIMATOR

Purpose:
Maintain temporal consistency.

Outputs:

temporal_state_t = {
    motion_field,
    temporal_confidence,
    drift_score,
    continuity_score,
    smoothing_constraints
}

Responsibilities:
- bidirectional smoothing
- confidence propagation
- optical-flow consistency
- identity continuity
- geometry continuity

Critical rule:
Temporal system updates CONFIDENCE, not raw texture.

Forbidden:
- backward texture injection
- frame averaging
- temporal blur accumulation
"""

from typing import Optional, Dict, Tuple
import numpy as np

from face_os.types import GeometryState, IdentityState, TemporalState
from face_os.temporal_solve import TemporalRepairEngine


class TemporalEstimator:
    """Temporal Estimator subsystem - maintains temporal consistency."""
    
    def __init__(self, lookback: int = 10, lookahead: int = 10):
        self.temporal_solver = TemporalRepairEngine(lookback=lookback, lookahead=lookahead)
        self.frame_count = 0
        
    def estimate(
        self,
        geometry_state: GeometryState,
        identity_state: IdentityState,
        previous_temporal_state: Optional[TemporalState] = None
    ) -> TemporalState:
        """
        Estimate temporal state from current geometry and identity states.
        
        Args:
            geometry_state: Current geometry state
            identity_state: Current identity state  
            previous_temporal_state: Previous temporal state for continuity
            
        Returns:
            TemporalState containing temporal consistency information
        """
        self.frame_count += 1
        
        # Extract motion field (simplified - in practice would use optical flow)
        motion_field = self._compute_motion_field(geometry_state, previous_temporal_state)
        
        # Compute temporal confidence
        temporal_confidence = self._compute_temporal_confidence(
            geometry_state, identity_state, previous_temporal_state
        )
        
        # Compute drift score (how much identity has drifted from anchor)
        drift_score = self._compute_drift_score(identity_state)
        
        # Compute continuity score (temporal smoothness)
        continuity_score = self._compute_continuity_score(
            geometry_state, previous_temporal_state
        )
        
        # Compute smoothing constraints
        smoothing_constraints = self._compute_smoothing_constraints(
            geometry_state, motion_field
        )
        
        return TemporalState(
            motion_field=motion_field,
            temporal_confidence=temporal_confidence,
            drift_score=drift_score,
            continuity_score=continuity_score,
            smoothing_constraints=smoothing_constraints
        )
        
    def collect_frame_for_bidirectional_solve(
        self,
        frame_idx: int,
        canonical_face: np.ndarray,
        quality_map: np.ndarray,
        sharpness: float,
        pose: Tuple[float, float, float],
        detection_confidence: float
    ) -> None:
        """Collect frame data for bidirectional temporal solving."""
        self.temporal_solver.collect_frame(
            frame_idx, canonical_face, quality_map,
            sharpness=sharpness,
            pose=pose,
            detection_confidence=detection_confidence
        )
        
    def solve_bidirectional(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Perform bidirectional temporal solve and return solved faces."""
        return self.temporal_solver.solve()
        
    def _compute_motion_field(
        self,
        geometry_state: GeometryState,
        previous_temporal_state: Optional[TemporalState]
    ) -> np.ndarray:
        """Compute motion field between current and previous frame."""
        if previous_temporal_state is None or geometry_state.landmarks is None:
            h, w = geometry_state.mask.shape if geometry_state.mask is not None else (256, 256)
            return np.zeros((h, w, 2), dtype=np.float32)
            
        # Simplified motion field based on pose change
        if previous_temporal_state.pose is not None:
            pose_change = (
                geometry_state.pose[0] - previous_temporal_state.pose[0],
                geometry_state.pose[1] - previous_temporal_state.pose[1],
                geometry_state.pose[2] - previous_temporal_state.pose[2]
            )
            h, w = geometry_state.mask.shape if geometry_state.mask is not None else (256, 256)
            motion_field = np.full((h, w, 2), pose_change[:2], dtype=np.float32)
            return motion_field
            
        h, w = geometry_state.mask.shape if geometry_state.mask is not None else (256, 256)
        return np.zeros((h, w, 2), dtype=np.float32)
        
    def _compute_temporal_confidence(
        self,
        geometry_state: GeometryState,
        identity_state: IdentityState,
        previous_temporal_state: Optional[TemporalState]
    ) -> float:
        """Compute temporal confidence based on consistency."""
        base_conf = geometry_state.geometry_confidence * (1.0 - identity_state.identity_uncertainty)
        
        if previous_temporal_state is None:
            return base_conf
            
        # Penalize large pose changes
        pose_change_penalty = 1.0
        if previous_temporal_state.pose is not None:
            pose_diff = sum(abs(a - b) for a, b in zip(geometry_state.pose, previous_temporal_state.pose))
            pose_change_penalty = max(0.0, 1.0 - pose_diff / 90.0)  # Normalize by 90 degrees
            
        return base_conf * pose_change_penalty
        
    def _compute_drift_score(self, identity_state: IdentityState) -> float:
        """Compute how much identity has drifted from anchor."""
        # In practice, this would compare to reference anchor
        # For now, use identity uncertainty as proxy
        return identity_state.identity_uncertainty
        
    def _compute_continuity_score(
        self,
        geometry_state: GeometryState,
        previous_temporal_state: Optional[TemporalState]
    ) -> float:
        """Compute temporal continuity score."""
        if previous_temporal_state is None:
            return 1.0
            
        # Higher score means more continuous
        pose_continuity = 1.0
        if previous_temporal_state.pose is not None:
            pose_diff = sum(abs(a - b) for a, b in zip(geometry_state.pose, previous_temporal_state.pose))
            pose_continuity = max(0.0, 1.0 - pose_diff / 45.0)  # Normalize by 45 degrees
            
        return pose_continuity
        
    def _compute_smoothing_constraints(
        self,
        geometry_state: GeometryState,
        motion_field: np.ndarray
    ) -> Dict[str, float]:
        """Compute constraints for temporal smoothing."""
        constraints = {
            'max_pose_velocity': 30.0,  # degrees per frame
            'max_landmark_acceleration': 10.0,  # pixels per frame squared
            'min_temporal_confidence': 0.3,
            'max_drift_threshold': 25.0,  # LAB distance
        }
        return constraints