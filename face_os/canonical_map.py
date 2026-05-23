"""
canonical_map.py — Module 4: Canonical Face Mapping + Appearance Field.

BEAST MODE FIXES:
- Nuked the fake 68-point sin/cos mathematical template.
- Direct 5-anchor MediaPipe mapping for bulletproof similarity transforms.
- Shifted accumulation from float64 to float32 to save RAM/CPU.
- Fixed LAB space uint8 wrap-around (neon green face artifacts).
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import (
    AppearanceField,
    CanonicalMapping,
    FaceDetection,
    IdentityProfile,
    Landmarks,
)

cfg = get_config()

# ─── Canonical alignment ────────────────────────────────────────────────────

# BEAST MODE: Direct 5-anchor canonical positions in 256x256 space.
# MediaPipe indices: 1 (Nose), 33 (Left Eye Inner), 263 (Right Eye Inner), 61 (Mouth Left), 291 (Mouth Right)
# Note: Subject's left is on the right side of the image (X > 128).
_CANONICAL_5_ANCHORS = np.array([
    [128.0, 145.0],  # Nose tip
    [150.0, 105.0],  # Left eye inner (image right)
    [106.0, 105.0],  # Right eye inner (image left)
    [142.0, 185.0],  # Mouth left (image right)
    [114.0, 185.0],  # Mouth right (image left)
], dtype=np.float32)


# ─── Alignment transform ────────────────────────────────────────────────────

def compute_alignment(
    source_landmarks: Landmarks,
    canonical_size: Tuple[int, int] = (256, 256),
    mode: str = "similarity",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute transform from source face to canonical atlas."""
    src_pts = source_landmarks.points.astype(np.float32)
    num_points = len(src_pts)

    # Map MediaPipe 478 or dlib 68 to our 5 robust anchors
    if num_points >= 468:
        anchor_indices_src = [1, 33, 263, 61, 291]
    else:
        # Fallback for legacy 68-point: Nose(30), L-Eye(39), R-Eye(42), M-Left(48), M-Right(54)
        anchor_indices_src = [30, 39, 42, 48, 54]

    src_anchor = src_pts[anchor_indices_src]
    
    # Scale anchors if canonical_size is not 256x256
    w, h = canonical_size
    scale_x, scale_y = w / 256.0, h / 256.0
    dst_anchor = _CANONICAL_5_ANCHORS.copy()
    dst_anchor[:, 0] *= scale_x
    dst_anchor[:, 1] *= scale_y

    if mode == "similarity":
        M, inliers = cv2.estimateAffinePartial2D(src_anchor, dst_anchor)
        if M is None:
            M = np.eye(2, 3, dtype=np.float32)
        M_3x3 = np.vstack([M, [0, 0, 1]]).astype(np.float32)
        M_inv = cv2.invertAffineTransform(M).astype(np.float32)
        M_inv_3x3 = np.vstack([M_inv, [0, 0, 1]]).astype(np.float32)

    elif mode == "affine":
        M, inliers = cv2.estimateAffine2D(src_anchor, dst_anchor)
        if M is None:
            M = np.eye(2, 3, dtype=np.float32)
        M_3x3 = np.vstack([M, [0, 0, 1]]).astype(np.float32)
        M_inv = cv2.invertAffineTransform(M).astype(np.float32)
        M_inv_3x3 = np.vstack([M_inv, [0, 0, 1]]).astype(np.float32)

    else:
        # Perspective fallback to anchors if counts differ
        M_3x3, _ = cv2.findHomography(src_anchor, dst_anchor, cv2.RANSAC, 5.0)
        if M_3x3 is None:
            M_3x3 = np.eye(3, dtype=np.float32)
        M_inv_3x3 = np.linalg.inv(M_3x3).astype(np.float32)

    return M_3x3, M_inv_3x3


# ─── Atlas operations ───────────────────────────────────────────────────────

def warp_to_canonical(
    frame: np.ndarray,
    landmarks: Landmarks,
    canonical_size: Tuple[int, int] = (256, 256),
    mode: str = "similarity",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp a face region from source frame to canonical atlas space."""
    M, M_inv = compute_alignment(landmarks, canonical_size, mode)

    warped = cv2.warpAffine(
        frame, M[:2], canonical_size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT,
    )

    warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    return warped_rgb, warped_lab, M


def warp_from_canonical(
    atlas: np.ndarray,
    transform_matrix: np.ndarray,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Warp from canonical atlas back to source frame space."""
    M_inv = np.linalg.inv(transform_matrix)[:2]
    return cv2.warpAffine(
        atlas, M_inv, (target_shape[1], target_shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
    )


# ─── Appearance Field accumulation ──────────────────────────────────────────

class AppearanceFieldBuilder:
    """Builds and maintains the Appearance Field — the canonical face atlas."""

    def __init__(self, canonical_size: Tuple[int, int] = (256, 256)):
        self.canonical_size = canonical_size
        self.atlas: Optional[AppearanceField] = None
        self._accumulated_rgb: Optional[np.ndarray] = None
        self._accumulated_lab: Optional[np.ndarray] = None
        self._observation_count: Optional[np.ndarray] = None
        self._frame_count = 0

    def initialize(self, first_frame: np.ndarray, landmarks: Landmarks) -> None:
        """Initialize atlas from first observation."""
        w, h = self.canonical_size
        warped_rgb, warped_lab, M = warp_to_canonical(
            first_frame, landmarks, self.canonical_size
        )

        # BEAST MODE: float32 instead of float64 to save RAM/CPU
        self._accumulated_rgb = warped_rgb.astype(np.float32)
        self._accumulated_lab = warped_lab.astype(np.float32)
        self._observation_count = np.ones((h, w), dtype=np.float32)

        self.atlas = AppearanceField(
            atlas_rgb=warped_rgb.astype(np.uint8),
            atlas_lab=np.clip(warped_lab, 0, 255).astype(np.uint8),
            atlas_confidence=np.ones((h, w), dtype=np.float32) * 0.1,
            enrollment_frames=1,
            last_update_frame=0,
        )
        self._frame_count = 1

    def update(
        self,
        frame: np.ndarray,
        landmarks: Landmarks,
        detection_confidence: float = 1.0,
        pose_weight: float = 1.0,
    ) -> Optional[AppearanceField]:
        """Update atlas with a new frame observation."""
        if self.atlas is None or self._accumulated_rgb is None:
            self.initialize(frame, landmarks)
            return self.atlas

        warped_rgb, warped_lab, M = warp_to_canonical(
            frame, landmarks, self.canonical_size
        )

        quality = self._compute_quality(warped_rgb, detection_confidence, pose_weight)

        rate = cfg.memory.accumulation_rate
        weight = rate * quality * detection_confidence
        weight_3d = weight[:, :, np.newaxis].astype(np.float32)

        self._accumulated_rgb = (
            self._accumulated_rgb * (1.0 - weight_3d) + warped_rgb.astype(np.float32) * weight_3d
        )
        self._accumulated_lab = (
            self._accumulated_lab * (1.0 - weight_3d) + warped_lab * weight_3d
        )
        self._observation_count = self._observation_count * (1.0 - cfg.memory.decay_rate) + quality

        self.atlas.atlas_rgb = np.clip(self._accumulated_rgb, 0, 255).astype(np.uint8)
        # BEAST MODE FIX: Clip LAB before uint8 cast to prevent neon wrap-around artifacts
        self.atlas.atlas_lab = np.clip(self._accumulated_lab, 0, 255).astype(np.uint8)

        max_obs = cfg.memory.min_observations * 10
        self.atlas.atlas_confidence = np.clip(
            self._observation_count / max_obs, 0, 1
        ).astype(np.float32)

        self.atlas.enrollment_frames += 1
        self.atlas.last_update_frame = self._frame_count
        self._frame_count += 1

        return self.atlas

    def _compute_quality(
        self,
        warped_rgb: np.ndarray,
        detection_confidence: float,
        pose_weight: float,
    ) -> np.ndarray:
        """Compute per-pixel quality score for weighting observations."""
        gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
        lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
        sharpness = np.clip(lap / 50.0, 0, 1)

        brightness = gray.astype(np.float32) / 255.0
        brightness_weight = 1.0 - np.abs(brightness - 0.5) * 2
        brightness_weight = np.clip(brightness_weight, 0.1, 1.0)

        quality = sharpness * brightness_weight * detection_confidence * pose_weight
        quality = cv2.GaussianBlur(quality, (5, 5), 1.0)

        return quality

    def get_atlas(self) -> Optional[AppearanceField]:
        return self.atlas

    def is_enrolled(self, min_frames: int = 5) -> bool:
        return (
            self.atlas is not None
            and self.atlas.enrollment_frames >= min_frames
        )


# ─── Identity profile management ────────────────────────────────────────────

def build_identity_profile(
    reference_images: List[np.ndarray],
    reference_paths: List[str] = None,
) -> IdentityProfile:
    """Build an IdentityProfile from reference images."""
    profile = IdentityProfile(reference_paths=reference_paths or [])

    from face_os.detect_track import _compute_embedding, detect_faces, extract_face_mesh
    for ref in reference_images:
        detections = detect_faces(ref)
        if detections:
            for det in detections:
                if det.smooth_bbox is None:
                    continue
                emb = _compute_embedding(ref, det.smooth_bbox)
                if emb is not None:
                    profile.embeddings.append(emb)
                    break
        else:
            emb = _compute_embedding(ref, (0, 0, ref.shape[1], ref.shape[0]))
            if emb is not None:
                profile.embeddings.append(emb)

    if reference_images:
        builder = AppearanceFieldBuilder()
        best_img = reference_images[0]
        
        detections = detect_faces(best_img)
        
        if detections:
            track = detections[0]
            mesh = extract_face_mesh(best_img)
            if mesh is not None:
                track.mesh_478 = mesh
            from face_os.landmarks import extract_landmarks
            lm = extract_landmarks(best_img, mesh) if mesh is not None else None
            if lm is not None:
                builder.initialize(best_img, lm)
                profile.appearance = builder.get_atlas()
                profile.enrolled = True
                profile.enrollment_frames = 1

    return profile