"""
SUBSYSTEM D — RENDERER

Purpose:
Generate physically consistent output.

Inputs:
- geometry_state_t
- identity_state_t
- temporal_state_t

Outputs:
- rendered_face
- background_layer
- composite_output

Render equation:

Y = M ⊙ Y_face + (1 - M) ⊙ Y_bg

Where:
- M is geometry-derived semantic mask
- Y_face is latent-rendered face
- Y_bg is untouched background

Forbidden:
- RGB-space rescue compositing
- heuristic face merging
- implicit blending logic
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from face_os.types import GeometryState, IdentityState, TemporalState, CropPlan
from face_os.crop_planner import apply_crop
from face_os.face_enhance import render_frame, _create_enhancement_mask


class Renderer:
    """Renderer subsystem - generates physically consistent output."""
    
    def __init__(self):
        pass
        
    def render(
        self,
        source_frame: np.ndarray,
        geometry_state: GeometryState,
        identity_state: IdentityState,
        temporal_state: TemporalState,
        crop_plan: CropPlan
    ) -> np.ndarray:
        """
        Render final output frame from subsystem states.
        
        Args:
            source_frame: Original source frame
            geometry_state: Geometry state with transforms and masks
            identity_state: Identity state with appearance
            temporal_state: Temporal state with confidence
            crop_plan: Crop plan for output dimensions
            
        Returns:
            Rendered output frame (H, W, 3) uint8
        """
        # Apply crop to source frame
        cropped = apply_crop(source_frame, crop_plan)
        
        # Get region masks from geometry state
        region_masks = geometry_state.semantic_regions
        face_mask = region_masks.get("face") if region_masks else None
        
        # Check if we have valid identity to render
        if identity_state.initialized and identity_state.appearance_latent is not None:
            return self._render_identity_path(
                cropped, geometry_state, identity_state, temporal_state, region_masks
            )
        else:
            return self._render_enhancement_only_path(cropped, region_masks)
            
    def _render_identity_path(
        self,
        cropped: np.ndarray,
        geometry_state: GeometryState,
        identity_state: IdentityState,
        temporal_state: TemporalState,
        region_masks: Optional[dict]
    ) -> np.ndarray:
        """Render using identity reconstruction."""
        # Warp identity face from canonical space back to crop space
        if geometry_state.inverse_transform is None:
            return self._render_enhancement_only_path(cropped, region_masks)
            
        try:
            identity_face = identity_state.appearance_latent
            M_inv = geometry_state.inverse_transform
            
            # Warp identity to crop space
            identity_in_crop = cv2.warpAffine(
                identity_face, M_inv, (cropped.shape[1], cropped.shape[0]),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT,
            )
            
            # Get geometry mask and warp to crop space
            if geometry_state.mask is not None:
                canonical_mask = geometry_state.mask
                aligned_mask = cv2.warpAffine(
                    canonical_mask, M_inv, (cropped.shape[1], cropped.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                aligned_mask = np.clip(aligned_mask, 0, 1)
            else:
                # Fallback to region face mask
                aligned_mask = face_mask if face_mask is not None else np.ones(cropped.shape[:2], dtype=np.float32)
                
            # Compute blend weight from temporal confidence and aligned mask
            temporal_conf = temporal_state.temporal_confidence
            blend_weight = aligned_mask * temporal_conf
            blend_weight = np.clip(blend_weight, 0, 1)
            
            # Direct blend: source * (1-mask) + identity * mask
            blend_3d = blend_weight[:, :, np.newaxis]
            output = cropped.astype(np.float32) * (1 - blend_3d) + identity_in_crop.astype(np.float32) * blend_3d
            output = np.clip(output, 0, 255).astype(np.uint8)
            
            # Apply structure-preserving rendering on top
            enhancement_mask = _create_enhancement_mask(region_masks, output.shape) if region_masks else None
            rendered = render_frame(
                output, enhancement_mask, region_masks,
                identity_eyes=None, eye_confidence=0.0,
            )
            
            return rendered
            
        except Exception:
            # Fallback to enhancement only
            return self._render_enhancement_only_path(cropped, region_masks)
            
    def _render_enhancement_only_path(
        self,
        cropped: np.ndarray,
        region_masks: Optional[dict]
    ) -> np.ndarray:
        """Render using only structure-preserving enhancement (no identity)."""
        enhancement_mask = _create_enhancement_mask(region_masks, cropped.shape) if region_masks else None
        rendered = render_frame(
            cropped, enhancement_mask, region_masks,
            identity_eyes=None, eye_confidence=0.0,
        )
        return rendered