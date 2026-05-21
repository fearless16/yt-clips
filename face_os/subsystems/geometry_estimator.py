"""
SUBSYSTEM A — GEOMETRY ESTIMATOR

Purpose:
Estimate all spatial structure.

Inputs:
- frame_t
- previous_geometry_state

Outputs:
geometry_state_t

Structure:

geometry_state_t = {
    landmarks_478,
    pose,
    canonical_transform,
    crop_transform,
    mesh,
    semantic_regions,
    mask,
    geometry_confidence
}

Responsibilities:
- landmark extraction
- head pose estimation  
- canonical UV mapping
- semantic region construction
- crop optimization
- warp transform generation

Forbidden:
- identity logic
- lighting logic
- RGB blending
"""

from typing import Optional, Tuple, Dict
import cv2
import numpy as np

from face_os.types import Landmarks, CropPlan, FaceTrack, GeometryState
from face_os.landmarks import extract_landmarks, create_region_masks
from face_os.crop_planner import CropPlanner
from face_os.canonical_map import warp_to_canonical


class GeometryEstimator:
    """Geometry Estimator subsystem - estimates all spatial structure."""
    
    def __init__(self, crop_planner: CropPlanner):
        self.crop_planner = crop_planner
        
    def estimate(
        self,
        frame: np.ndarray,
        face_track: Optional[FaceTrack],
        previous_geometry_state: Optional['GeometryState'] = None
    ) -> 'GeometryState':
        """
        Estimate geometry state from input frame and face track.
        
        Args:
            frame: Input frame (H, W, 3) BGR
            face_track: Detected face track with 478-point mesh
            previous_geometry_state: Previous geometry state for temporal continuity
            
        Returns:
            GeometryState containing all spatial information
        """
        # Extract landmarks and pose
        landmarks = None
        if face_track and face_track.mesh_478 is not None:
            landmarks = extract_landmarks(frame, face_track.mesh_478)
            
        # Plan crop
        crop_plan = self.crop_planner.plan_crop(
            frame.shape[:2], 
            face_track, 
            landmarks
        )
        
        # Create semantic regions
        region_masks = None
        if landmarks:
            region_masks = create_region_masks(landmarks, frame.shape[:2])
            
        # Create geometry-based mask (brightness-invariant)
        geometry_mask = self._create_geometry_mask(frame.shape[:2], landmarks)
        
        # Compute geometry confidence
        geometry_confidence = self._compute_geometry_confidence(
            face_track, landmarks, crop_plan
        )
        
        # Create canonical transform if possible
        canonical_transform = None
        inverse_transform = None
        canonical_face = None
        if landmarks:
            try:
                _, _, canonical_transform = warp_to_canonical(frame, landmarks)
                inverse_transform = np.linalg.inv(canonical_transform)[:2]
                # Warp frame to canonical space for downstream use
                canonical_face = cv2.warpAffine(
                    frame, canonical_transform[:2], (256, 256),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT
                )
            except Exception:
                canonical_transform = None
                inverse_transform = None
                
        return GeometryState(
            landmarks_478=face_track.mesh_478 if face_track else None,
            landmarks=landmarks,
            pose=(landmarks.yaw, landmarks.pitch, landmarks.roll) if landmarks else (0.0, 0.0, 0.0),
            canonical_transform=canonical_transform,
            inverse_transform=inverse_transform,
            crop_transform=crop_plan,
            mesh=face_track.mesh_478 if face_track else None,
            semantic_regions=region_masks,
            mask=geometry_mask,
            geometry_confidence=geometry_confidence,
            canonical_face=canonical_face
        )
        
    def _create_geometry_mask(
        self,
        frame_shape: Tuple[int, int],
        landmarks: Optional[Landmarks]
    ) -> np.ndarray:
        """Create brightness-invariant geometry-based face mask."""
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.float32)
        
        if landmarks is None or landmarks.points is None:
            return mask
            
        # Use convex hull of face oval points
        face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 50, 101, 323, 361, 288, 
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 
            132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        try:
            face_pts = landmarks.points[face_oval_indices].astype(np.int32)
            hull = cv2.convexHull(face_pts)
            cv2.fillConvexPoly(mask, hull, 1.0)
            
            # Extend upward to include forehead
            brow_top = int(min(landmarks.points[[10, 67, 109, 338, 297], 1]))
            jaw_top = int(min(face_pts[:, 1]))
            forehead_height = jaw_top - brow_top
            forehead_top = max(0, brow_top - forehead_height)
            face_left = max(0, int(min(face_pts[:, 0])) - 10)
            face_right = min(w, int(max(face_pts[:, 0])) + 10)
            mask[forehead_top:brow_top, face_left:face_right] = 1.0
            
            # Apply light feathering
            mask = cv2.GaussianBlur(mask, (11, 11), 3)
            mask = np.clip(mask, 0, 1)
        except Exception:
            pass
            
        return mask
        
    def _compute_geometry_confidence(
        self,
        face_track: Optional[FaceTrack],
        landmarks: Optional[Landmarks],
        crop_plan: CropPlan
    ) -> float:
        """Compute overall geometry confidence."""
        if face_track is None or landmarks is None:
            return 0.0
            
        # Detection confidence
        detection_conf = face_track.detection.confidence if face_track.detection else 0.0
        
        # Landmark confidence  
        landmark_conf = landmarks.landmark_confidence
        
        # Crop plan confidence
        crop_conf = crop_plan.confidence
        
        # Pose stability (lower angles = higher confidence)
        yaw_abs = abs(landmarks.yaw)
        pitch_abs = abs(landmarks.pitch)
        roll_abs = abs(landmarks.roll)
        pose_stability = max(0.0, 1.0 - (yaw_abs + pitch_abs + roll_abs) / 180.0)
        
        # Combine confidences
        total_conf = (
            detection_conf * 0.4 +
            landmark_conf * 0.3 +
            crop_conf * 0.2 +
            pose_stability * 0.1
        )
        
        return max(0.0, min(1.0, total_conf))