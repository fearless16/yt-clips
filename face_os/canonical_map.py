"""
canonical_map.py — Module 4: Canonical Face Mapping + Appearance Field.

This is where the "Appearance Field" concept lives.

Instead of rendering face from mesh+texture, we:
  1. Align detected face to a canonical UV space (frontal, neutral pose)
  2. Accumulate pixel observations in this canonical space over time
  3. Each pixel has a confidence score based on observation count + quality
  4. The canonical atlas evolves — getting better with more frames

This is "Photic Memory" — treating each frame as a noisy photon observation
and accumulating a better model of the face over time.

Key insight: For overfit mode (one person), the canonical atlas can be
extremely high quality because we have hundreds/thousands of observations
of the same face under slightly different conditions.
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

# Standard 68-point canonical positions (frontal, neutral expression)
# These define where each landmark should be in the canonical atlas
_CANONICAL_68 = None


def _get_canonical_points(size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Get canonical landmark positions for the atlas."""
    global _CANONICAL_68
    if _CANONICAL_68 is not None:
        return _CANONICAL_68

    w, h = size
    pts = np.zeros((68, 2), dtype=np.float32)

    # Jaw line (0-16): U-shape at bottom
    for i in range(17):
        t = i / 16.0
        pts[i] = [
            w * (0.15 + 0.70 * t),
            h * (0.70 + 0.20 * np.sin(t * np.pi)),
        ]

    # Right eyebrow (17-21)
    for i in range(5):
        t = i / 4.0
        pts[17 + i] = [w * (0.20 + 0.12 * t), h * 0.28]

    # Left eyebrow (22-26)
    for i in range(5):
        t = i / 4.0
        pts[22 + i] = [w * (0.68 + 0.12 * t), h * 0.28]

    # Nose bridge (27-30)
    for i in range(4):
        pts[27 + i] = [w * 0.5, h * (0.35 + 0.10 * i / 3)]

    # Nose bottom (31-35)
    nose_y = h * 0.50
    pts[31] = [w * 0.42, nose_y]
    pts[32] = [w * 0.45, nose_y + h * 0.02]
    pts[33] = [w * 0.50, nose_y + h * 0.04]
    pts[34] = [w * 0.55, nose_y + h * 0.02]
    pts[35] = [w * 0.58, nose_y]

    # Right eye (36-41): elliptical
    for i in range(6):
        angle = i * np.pi / 3
        pts[36 + i] = [
            w * 0.32 + 15 * np.cos(angle),
            h * 0.35 + 8 * np.sin(angle),
        ]

    # Left eye (42-47): elliptical
    for i in range(6):
        angle = i * np.pi / 3
        pts[42 + i] = [
            w * 0.68 + 15 * np.cos(angle),
            h * 0.35 + 8 * np.sin(angle),
        ]

    # Outer lip (48-59)
    for i in range(12):
        angle = i * np.pi / 6
        pts[48 + i] = [
            w * 0.5 + 25 * np.cos(angle),
            h * 0.68 + 8 * np.sin(angle),
        ]

    # Inner lip (60-67)
    for i in range(8):
        angle = i * np.pi / 4
        pts[60 + i] = [
            w * 0.5 + 15 * np.cos(angle),
            h * 0.68 + 4 * np.sin(angle),
        ]

    _CANONICAL_68 = pts
    return pts


# ─── Alignment transform ────────────────────────────────────────────────────

def compute_alignment(
    source_landmarks: Landmarks,
    canonical_size: Tuple[int, int] = (256, 256),
    mode: str = "similarity",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute transform from source face to canonical atlas.

    Args:
        source_landmarks: Detected landmarks in source frame
        canonical_size: (width, height) of canonical atlas
        mode: similarity (rigid+scale), affine, or perspective

    Returns:
        (forward_matrix, inverse_matrix)
    """
    src_pts = source_landmarks.points.astype(np.float32)
    dst_pts = _get_canonical_points(canonical_size)

    if mode == "similarity":
        # Similarity transform (rotation + scale + translation)
        # Uses eyes, nose, mouth as anchor points
        anchor_indices = [30, 36, 45, 48, 54]  # Nose, eyes, mouth corners
        src_anchor = src_pts[anchor_indices]
        dst_anchor = dst_pts[anchor_indices]

        M = cv2.estimateAffinePartial2D(src_anchor, dst_anchor)[0]
        if M is None:
            M = np.eye(2, 3, dtype=np.float32)
        M_3x3 = np.vstack([M, [0, 0, 1]]).astype(np.float32)
        M_inv = cv2.invertAffineTransform(M).astype(np.float32)
        M_inv_3x3 = np.vstack([M_inv, [0, 0, 1]]).astype(np.float32)

    elif mode == "affine":
        # Full affine (6 DOF)
        anchor_indices = [30, 36, 45, 48, 54, 8]
        src_anchor = src_pts[anchor_indices]
        dst_anchor = dst_pts[anchor_indices]

        M = cv2.estimateAffine2D(src_anchor, dst_anchor)[0]
        if M is None:
            M = np.eye(2, 3, dtype=np.float32)
        M_3x3 = np.vstack([M, [0, 0, 1]]).astype(np.float32)
        M_inv = cv2.invertAffineTransform(M).astype(np.float32)
        M_inv_3x3 = np.vstack([M_inv, [0, 0, 1]]).astype(np.float32)

    else:
        # Perspective (8 DOF) — uses all 68 points
        M_3x3, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
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
    """Warp a face region from source frame to canonical atlas space.

    Returns:
        (warped_rgb, warped_lab, transform_matrix)
    """
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
    """Warp from canonical atlas back to source frame space.

    Args:
        atlas: Canonical atlas (H, W, C)
        transform_matrix: Forward transform (source -> canonical)
        target_shape: (height, width) of output frame

    Returns:
        Warped image in source frame space
    """
    M_inv = np.linalg.inv(transform_matrix)[:2]
    return cv2.warpAffine(
        atlas, M_inv, (target_shape[1], target_shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
    )


# ─── Appearance Field accumulation ──────────────────────────────────────────

class AppearanceFieldBuilder:
    """Builds and maintains the Appearance Field — the canonical face atlas.

    This is the core of "Photic Memory":
    - Each frame contributes a noisy observation of the face
    - Observations are accumulated in canonical space
    - Confidence increases with more observations
    - Recent observations are weighted more heavily

    The atlas improves over time — like a long-exposure photograph.
    """

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

        self._accumulated_rgb = warped_rgb.astype(np.float64)
        self._accumulated_lab = warped_lab.astype(np.float64)
        self._observation_count = np.ones((h, w), dtype=np.float64)

        self.atlas = AppearanceField(
            atlas_rgb=warped_rgb.astype(np.uint8),
            atlas_lab=warped_lab.astype(np.uint8),
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
        """Update atlas with a new frame observation.

        Args:
            frame: Source BGR frame
            landmarks: Detected landmarks
            detection_confidence: How confident the detection is
            pose_weight: Weight based on pose similarity to enrolled pose

        Returns:
            Updated AppearanceField
        """
        if self.atlas is None or self._accumulated_rgb is None:
            self.initialize(frame, landmarks)
            return self.atlas

        # Warp to canonical space
        warped_rgb, warped_lab, M = warp_to_canonical(
            frame, landmarks, self.canonical_size
        )

        # Compute per-pixel quality mask
        quality = self._compute_quality(warped_rgb, detection_confidence, pose_weight)

        # Accumulate with exponential moving average
        rate = cfg.memory.accumulation_rate
        weight = rate * quality * detection_confidence
        weight_3d = weight[:, :, np.newaxis]  # (H, W) -> (H, W, 1) for broadcasting

        self._accumulated_rgb = (
            self._accumulated_rgb * (1 - weight_3d) + warped_rgb.astype(np.float64) * weight_3d
        )
        self._accumulated_lab = (
            self._accumulated_lab * (1 - weight_3d) + warped_lab.astype(np.float64) * weight_3d
        )
        self._observation_count = self._observation_count * (1 - cfg.memory.decay_rate) + quality

        # Update atlas
        self.atlas.atlas_rgb = np.clip(self._accumulated_rgb, 0, 255).astype(np.uint8)
        self.atlas.atlas_lab = self._accumulated_lab.astype(np.uint8)

        # Confidence = normalized observation count
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
        h, w = warped_rgb.shape[:2]

        # Sharpness (Laplacian magnitude)
        gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
        lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
        sharpness = np.clip(lap / 50.0, 0, 1)

        # Brightness (prefer well-lit observations)
        brightness = gray.astype(np.float32) / 255.0
        brightness_weight = 1.0 - np.abs(brightness - 0.5) * 2  # Peak at mid-gray
        brightness_weight = np.clip(brightness_weight, 0.1, 1.0)

        # Combined quality
        quality = sharpness * brightness_weight * detection_confidence * pose_weight

        # Smooth to avoid sharp transitions
        quality = cv2.GaussianBlur(quality, (5, 5), 1.0)

        return quality

    def get_atlas(self) -> Optional[AppearanceField]:
        """Get the current appearance field."""
        return self.atlas

    def is_enrolled(self, min_frames: int = 5) -> bool:
        """Check if atlas has enough observations to be useful."""
        return (
            self.atlas is not None
            and self.atlas.enrollment_frames >= min_frames
        )


# ─── Identity profile management ────────────────────────────────────────────

def build_identity_profile(
    reference_images: List[np.ndarray],
    reference_paths: List[str] = None,
) -> IdentityProfile:
    """Build an IdentityProfile from reference images.

    Extracts embeddings and optionally builds initial appearance atlas.
    """
    profile = IdentityProfile(reference_paths=reference_paths or [])

    # Extract embeddings from face regions (not full images)
    try:
        import face_recognition
        from face_os.detect_track import detect_faces
        for img in reference_images:
            # Detect face first, then compute embedding from face region
            detections = detect_faces(img)
            if detections:
                track = detections[0]
                x, y, w, h = track.smooth_bbox
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # face_recognition expects (top, right, bottom, left) format
                locations = [(y, x + w, y + h, x)]
                encodings = face_recognition.face_encodings(rgb, locations)
                if encodings:
                    profile.embeddings.extend(encodings)
    except ImportError:
        pass

    # Build initial atlas from best reference
    if reference_images:
        builder = AppearanceFieldBuilder()
        best_img = reference_images[0]  # Primary reference
        
        # Use MediaPipe instead of Haar cascade
        from face_os.detect_track import detect_faces
        detections = detect_faces(best_img)
        
        if detections:
            track = detections[0]
            x, y, fw, fh = track.smooth_bbox
            from face_os.landmarks import extract_landmarks
            lm = extract_landmarks(best_img, (x, y, fw, fh))
            if lm is not None:
                builder.initialize(best_img, lm)
                profile.appearance = builder.get_atlas()
                profile.enrolled = True
                profile.enrollment_frames = 1

    return profile
