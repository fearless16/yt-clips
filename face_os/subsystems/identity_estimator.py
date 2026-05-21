"""
SUBSYSTEM B — IDENTITY ESTIMATOR

Purpose:
Estimate stable identity representation independent of lighting and pose.

Current V0.5 flaw:
RGB EMA identity memory.

Replace with:

identity_state_t = {
    anchor_basis,
    anchor_weights,
    appearance_latent,
    region_confidence,
    identity_uncertainty
}

Identity representation:

a_t = Σ(w_k * a_k)

Where:
- a_k are learned/selected anchor states
- w_k are confidence-normalized interpolation weights

Required anchor dimensions:
- frontal neutral
- left yaw
- right yaw
- smile
- low-light
- high-light
- blink
- beard-shadow

Forbidden:
- RGB EMA blending
- raw frame accumulation
- frame-space averaging
"""

from typing import Optional, Tuple, Dict
import cv2
import numpy as np

from face_os.types import GeometryState, IdentityState as IdentityStateType
from face_os.identity_state import IdentityState


class IdentityEstimator:
    """Identity Estimator subsystem - estimates stable identity representation."""
    
    def __init__(self):
        self.identity_belief = IdentityState()
        self.initialized = False
        
    def estimate(
        self,
        geometry_state: GeometryState,
        quality_map: Optional[np.ndarray] = None,
        face_track: Optional['FaceTrack'] = None
    ) -> IdentityStateType:
        """
        Estimate identity state from geometry state and quality information.
        
        Args:
            geometry_state: Geometry state containing canonical face and transforms
            quality_map: Per-pixel quality map for the canonical face
            face_track: Face track with verification information
            
        Returns:
            IdentityState containing stable identity representation
        """
        if geometry_state.canonical_face is None:
            # Return empty identity state
            return IdentityStateType(
                anchor_basis=[],
                anchor_weights=[],
                appearance_latent=None,
                region_confidence={},
                identity_uncertainty=1.0,
                initialized=False
            )
            
        # Ensure identity belief is initialized
        if not self.initialized and self.identity_belief._anchor_low is not None:
            # Initialize from anchor if available
            h, w = geometry_state.canonical_face.shape[:2]
            quality_init = np.ones((h, w), dtype=np.float32) * 0.9
            self.identity_belief.update(
                geometry_state.canonical_face, 
                quality_init, 
                pose=geometry_state.pose
            )
            self.initialized = True
            
        # Update identity belief with new observation
        if quality_map is None:
            # Create default quality map
            quality_map = self._compute_quality_map(geometry_state.canonical_face)
            
        # Extract verification information from face track
        face_bbox = None
        landmarks_pts = None
        embedding = None
        if face_track:
            if face_track.smooth_bbox:
                face_bbox = face_track.smooth_bbox
            if face_track.mesh_478 is not None:
                landmarks_pts = face_track.mesh_478[:, :2]
            if face_track.detection and face_track.detection.embedding is not None:
                embedding = face_track.detection.embedding
                
        # Update identity belief (this includes verification gating)
        update_applied = self.identity_belief.update(
            canonical_face=geometry_state.canonical_face,
            quality_map=quality_map,
            pose=geometry_state.pose,
            face_bbox=face_bbox,
            landmarks_pts=landmarks_pts,
            embedding=embedding
        )
        
        # Query identity representation
        if update_applied or self.initialized:
            identity_face, confidence_map = self.identity_belief.query_identity(quality_map)
        else:
            identity_face = geometry_state.canonical_face.copy()
            confidence_map = np.ones_like(quality_map) * 0.5
            
        # Compute region confidence
        region_confidence = self.identity_belief.compute_region_confidence()
        
        # Compute identity uncertainty (inverse of overall confidence)
        identity_uncertainty = 1.0 - float(np.mean(confidence_map))
        
        # Get anchor basis (for now, just the current identity)
        anchor_basis = [identity_face]
        anchor_weights = [1.0]
        
        return IdentityStateType(
            anchor_basis=anchor_basis,
            anchor_weights=anchor_weights,
            appearance_latent=identity_face,
            region_confidence=region_confidence,
            identity_uncertainty=identity_uncertainty,
            initialized=True
        )
        
    def _compute_quality_map(self, canonical_face: np.ndarray) -> np.ndarray:
        """Compute per-pixel quality map for canonical face."""
        h, w = canonical_face.shape[:2]
        
        # Sharpness
        gray = cv2.cvtColor(canonical_face, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
        sharpness = np.clip(lap / 50.0, 0, 1)
        
        # Brightness (prefer well-lit)
        brightness = gray.astype(np.float32) / 255.0
        brightness_weight = 1.0 - np.abs(brightness - 0.5) * 2
        brightness_weight = np.clip(brightness_weight, 0.1, 1.0)
        
        # Combine
        quality = sharpness * brightness_weight
        return quality.astype(np.float32)
        
    def set_anchor(self, reference_face: np.ndarray) -> None:
        """Set identity anchor from reference face."""
        self.identity_belief.set_anchor(reference_face)
        
    def is_initialized(self) -> bool:
        """Check if identity estimator is initialized."""
        return self.initialized and self.identity_belief.is_initialized()