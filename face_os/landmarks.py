"""
landmarks.py — Module 3: Landmark Detection + Head Pose Estimation (V4).

Extracts facial landmarks and derives:
  - Eye centers, nose tip, mouth center
  - Head pose (yaw, pitch, roll) via PnP
  - Face contour points (for masking)

V4: Uses MediaPipe 478-point mesh directly. NO DLIB.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from face_os.types import Landmarks


# ─── MediaPipe 478-point indices ────────────────────────────────────────────

# Key points for pose estimation (PnP)
MPI_NOSE_TIP = 1
MPI_CHIN = 152
MPI_LEFT_EYE = 33      # Inner corner
MPI_RIGHT_EYE = 263    # Inner corner
MPI_LEFT_MOUTH = 61
MPI_RIGHT_MOUTH = 291

# Contours for region masks
MPI_LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MPI_RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MPI_LIPS_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
MPI_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def extract_landmarks(
    frame: np.ndarray,
    mesh_478: np.ndarray,
) -> Optional[Landmarks]:
    """Extract landmarks and pose from MediaPipe 478-point mesh.

    V4: Replaces dlib 68-point extraction.
    """
    if mesh_478 is None or len(mesh_478) < 468:
        return None

    # Use x, y coordinates for 2D operations
    pts_2d = mesh_478[:, :2].astype(np.float32)

    landmarks = Landmarks(
        points=pts_2d,
        landmark_confidence=1.0,
    )

    # Derive key regions
    landmarks.nose_tip = tuple(pts_2d[MPI_NOSE_TIP].tolist())
    landmarks.left_eye_center = tuple(pts_2d[MPI_LEFT_EYE].tolist())
    landmarks.right_eye_center = tuple(pts_2d[MPI_RIGHT_EYE].tolist())
    landmarks.mouth_center = tuple(((pts_2d[MPI_LEFT_MOUTH] + pts_2d[MPI_RIGHT_MOUTH]) / 2).tolist())

    # Estimate head pose
    _estimate_pose_478(landmarks, frame.shape[:2])

    return landmarks


def _estimate_pose_478(landmarks: Landmarks, frame_shape: Tuple[int, int]) -> None:
    """Estimate head pose from 478-point 2D landmarks using PnP."""
    pts = landmarks.points
    h, w = frame_shape

    # 3D model points (generic face, approximate)
    model_points = np.array([
        [0.0, 0.0, 0.0],             # Nose tip
        [0.0, -63.6, -12.5],         # Chin
        [-43.3, 32.7, -26.0],        # Left eye
        [43.3, 32.7, -26.0],         # Right eye
        [-28.9, -28.9, -24.1],       # Left mouth
        [28.9, -28.9, -24.1],        # Right mouth
    ], dtype=np.float64)

    # 2D image points (MediaPipe 478 indices)
    image_points = np.array([
        pts[MPI_NOSE_TIP],
        pts[MPI_CHIN],
        pts[MPI_LEFT_EYE],
        pts[MPI_RIGHT_EYE],
        pts[MPI_LEFT_MOUTH],
        pts[MPI_RIGHT_MOUTH],
    ], dtype=np.float64)

    # Camera matrix (approximate)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success:
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
            landmarks.yaw = float(angles[1])
            landmarks.pitch = float(angles[0])
            landmarks.roll = float(angles[2])
    except Exception:
        pass  # Keep default zeros


# ─── Region mask generation ─────────────────────────────────────────────────

def create_region_masks(
    landmarks: Landmarks,
    frame_shape: Tuple[int, int],
) -> dict:
    """Create smooth per-region masks from 478-point landmarks."""
    h, w = frame_shape
    pts = landmarks.points.astype(np.int32)
    masks = {}

    # Eye masks
    for name, indices in [("left_eye", MPI_LEFT_EYE_CONTOUR), ("right_eye", MPI_RIGHT_EYE_CONTOUR)]:
        eye_pts = pts[indices]
        center = np.mean(eye_pts, axis=0).astype(int)
        rx = int(np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0])) // 2 + 5
        ry = int(np.max(eye_pts[:, 1]) - np.min(eye_pts[:, 1])) // 2 + 5
        masks[name] = _elliptical_mask(h, w, center[1], center[0], ry, rx)

    # Brow masks (approximate above eyes)
    for name, eye_name in [("left_brow", "left_eye"), ("right_brow", "right_eye")]:
        if eye_name in masks:
            eye_mask = masks[eye_name]
            shift = int(masks[eye_name].shape[0] * 0.08) if masks[eye_name].shape[0] > 0 else 15
            brow_mask = np.zeros_like(eye_mask)
            if shift < h:
                brow_mask[:-shift, :] = eye_mask[shift:, :]
            masks[name] = cv2.GaussianBlur(brow_mask, (11, 11), 3)

    # Nose mask (approximate bridge and tip)
    nose_pts = pts[[MPI_NOSE_TIP, 168, 6, 197, 195, 5, 4]]
    center = np.mean(nose_pts, axis=0).astype(int)
    rx = int(np.max(nose_pts[:, 0]) - np.min(nose_pts[:, 0])) // 2 + 5
    ry = int(np.max(nose_pts[:, 1]) - np.min(nose_pts[:, 1])) // 2 + 5
    masks["nose"] = _elliptical_mask(h, w, center[1], center[0], ry, rx)

    # Mouth mask
    mouth_pts = pts[MPI_LIPS_CONTOUR]
    center = np.mean(mouth_pts, axis=0).astype(int)
    rx = int(np.max(mouth_pts[:, 0]) - np.min(mouth_pts[:, 0])) // 2 + 5
    ry = int(np.max(mouth_pts[:, 1]) - np.min(mouth_pts[:, 1])) // 2 + 5
    masks["mouth"] = _elliptical_mask(h, w, center[1], center[0], ry, rx)

    # Face contour (convex hull of face oval)
    face_pts = pts[MPI_FACE_OVAL]
    hull = cv2.convexHull(face_pts)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 255)

    # Extend upward to include forehead
    brow_top = int(np.min(pts[[10, 67, 109, 338, 297], 1]))
    jaw_top = int(np.min(face_pts[:, 1]))
    forehead_height = jaw_top - brow_top
    forehead_top = max(0, brow_top - forehead_height)
    face_left = max(0, int(np.min(face_pts[:, 0])) - 10)
    face_right = min(w, int(np.max(face_pts[:, 0])) + 10)
    face_mask[forehead_top:brow_top, face_left:face_right] = 255

    face_mask = cv2.GaussianBlur(face_mask.astype(np.float32) / 255.0, (11, 11), 3)
    face_mask = np.clip(face_mask, 0, 1)  # GaussianBlur can produce values > 1
    masks["face"] = face_mask

    # Skin = face minus eyes, nose, mouth
    skin = face_mask.copy()
    for name in ("left_eye", "right_eye", "nose", "mouth"):
        if name in masks:
            skin = np.clip(skin - masks[name], 0, 1)
    masks["skin"] = skin

    return masks


def _elliptical_mask(h: int, w: int, cy: int, cx: int, ry: int, rx: int) -> np.ndarray:
    """Create a smooth elliptical mask."""
    Y, X = np.ogrid[:h, :w]
    d = ((X - cx) / max(rx, 1)) ** 2 + ((Y - cy) / max(ry, 1)) ** 2
    mask = np.clip(1.0 - d, 0, 1)
    k = max(int(min(rx, ry) * 0.4) | 1, 3)
    mask = cv2.GaussianBlur(mask, (k, k), max(k / 3, 1.0))
    return np.clip(mask, 0, 1)  # GaussianBlur can produce values > 1
