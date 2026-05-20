"""
landmarks.py — Module 3: Landmark Detection + Head Pose Estimation.

Extracts 68-point facial landmarks and derives:
  - Eye centers (for eye-dominant rendering)
  - Nose tip, mouth center (for pose estimation)
  - Head pose (yaw, pitch, roll)
  - Face contour points (for masking)

Uses dlib 68-point model (fast, reliable for frontal faces).
Falls back to geometric estimation from bbox if dlib unavailable.
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import FaceDetection, FaceTrack, Landmarks


cfg = get_config()

# ─── dlib predictor (lazy loaded) ───────────────────────────────────────────

_PREDICTOR = None
_DLIB_AVAILABLE = False


def _get_predictor():
    """Lazy-load dlib shape predictor."""
    global _PREDICTOR, _DLIB_AVAILABLE
    if _PREDICTOR is not None:
        return _PREDICTOR

    try:
        import dlib
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            # Try common locations
            for candidate in [
                "models/shape_predictor_68_face_landmarks.dat",
                os.path.expanduser("~/.dlib/shape_predictor_68_face_landmarks.dat"),
            ]:
                if os.path.exists(candidate):
                    predictor_path = candidate
                    break

        if os.path.exists(predictor_path):
            _PREDICTOR = dlib.shape_predictor(predictor_path)
            _DLIB_AVAILABLE = True
            return _PREDICTOR
    except ImportError:
        pass

    return None


# ─── Landmark extraction ────────────────────────────────────────────────────

def extract_landmarks(
    frame: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
) -> Optional[Landmarks]:
    """Extract 68-point landmarks for a face.

    Falls back to geometric estimation if dlib unavailable.
    """
    x, y, w, h = face_bbox
    predictor = _get_predictor()

    if predictor is not None:
        return _extract_dlib(frame, x, y, w, h, predictor)
    else:
        return _estimate_geometric(frame, x, y, w, h)


def _extract_dlib(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    predictor,
) -> Optional[Landmarks]:
    """Extract landmarks using dlib shape predictor."""
    import dlib

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect = dlib.rectangle(x, y, x + w, y + h)

    try:
        shape = predictor(gray, rect)
    except Exception:
        return None

    # Extract 68 points
    points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)],
                       dtype=np.float32)

    landmarks = Landmarks(points=points, landmark_confidence=1.0)

    # Derive regions
    _compute_derived(landmarks)

    # Estimate head pose
    _estimate_pose(landmarks, frame.shape[:2])

    return landmarks


def _estimate_geometric(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
) -> Landmarks:
    """Estimate landmarks geometrically from face bounding box.

    Uses standard facial proportions:
      - Eyes at 35% from top
      - Nose at 55% from top
      - Mouth at 70% from top
    """
    # Generate approximate 68 points
    points = np.zeros((68, 2), dtype=np.float32)

    # Jaw line (points 0-16)
    for i in range(17):
        t = i / 16.0
        px = x + w * (0.1 + 0.8 * t)
        py = y + h * (0.85 + 0.15 * np.sin(t * np.pi))
        points[i] = [px, py]

    # Right eyebrow (points 17-21)
    for i in range(5):
        t = i / 4.0
        points[17 + i] = [x + w * (0.15 + 0.15 * t), y + h * (0.28 + 0.02 * np.sin(t * np.pi))]

    # Left eyebrow (points 22-26)
    for i in range(5):
        t = i / 4.0
        points[22 + i] = [x + w * (0.70 + 0.15 * t), y + h * (0.28 + 0.02 * np.sin(t * np.pi))]

    # Nose bridge (points 27-30)
    for i in range(4):
        t = i / 3.0
        points[27 + i] = [x + w * 0.5, y + h * (0.35 + 0.15 * t)]

    # Nose bottom (points 31-35)
    nose_w = w * 0.2
    nose_cx = x + w * 0.5
    nose_y = y + h * 0.52
    points[31] = [nose_cx - nose_w * 0.4, nose_y]
    points[32] = [nose_cx - nose_w * 0.2, nose_y + h * 0.03]
    points[33] = [nose_cx, nose_y + h * 0.05]
    points[34] = [nose_cx + nose_w * 0.2, nose_y + h * 0.03]
    points[35] = [nose_cx + nose_w * 0.4, nose_y]

    # Right eye (points 36-41)
    eye_cy = y + h * 0.35
    eye_rx = w * 0.12
    for i in range(6):
        angle = i * np.pi / 3 - np.pi / 2
        points[36 + i] = [
            x + w * 0.30 + eye_rx * np.cos(angle),
            eye_cy + eye_rx * 0.6 * np.sin(angle),
        ]

    # Left eye (points 42-47)
    for i in range(6):
        angle = i * np.pi / 3 - np.pi / 2
        points[42 + i] = [
            x + w * 0.70 + eye_rx * np.cos(angle),
            eye_cy + eye_rx * 0.6 * np.sin(angle),
        ]

    # Outer lip (points 48-59)
    lip_cy = y + h * 0.70
    lip_rx = w * 0.20
    lip_ry = h * 0.06
    for i in range(12):
        angle = i * np.pi / 6 - np.pi / 2
        points[48 + i] = [
            x + w * 0.5 + lip_rx * np.cos(angle),
            lip_cy + lip_ry * np.sin(angle),
        ]

    # Inner lip (points 60-67)
    inner_rx = w * 0.12
    inner_ry = h * 0.03
    for i in range(8):
        angle = i * np.pi / 4 - np.pi / 2
        points[60 + i] = [
            x + w * 0.5 + inner_rx * np.cos(angle),
            lip_cy + inner_ry * np.sin(angle),
        ]

    landmarks = Landmarks(
        points=points,
        landmark_confidence=0.5,  # Geometric estimate — lower confidence
    )

    _compute_derived(landmarks)
    return landmarks


def _compute_derived(landmarks: Landmarks) -> None:
    """Compute derived regions from landmark points."""
    pts = landmarks.points

    # Eye centers (average of eye contour points)
    # Right eye: points 36-41
    landmarks.right_eye_center = tuple(np.mean(pts[36:42], axis=0).tolist())
    # Left eye: points 42-47
    landmarks.left_eye_center = tuple(np.mean(pts[42:48], axis=0).tolist())

    # Nose tip (point 30)
    landmarks.nose_tip = tuple(pts[30].tolist())

    # Mouth center (average of outer lip)
    landmarks.mouth_center = tuple(np.mean(pts[48:60], axis=0).tolist())


def _estimate_pose(landmarks: Landmarks, frame_shape: Tuple[int, int]) -> None:
    """Estimate head pose from 2D landmarks.

    Uses the PnP approach with a generic 3D face model.
    """
    pts = landmarks.points
    h, w = frame_shape

    # 3D model points (generic face, approximate)
    model_points = np.array([
        [0.0, 0.0, 0.0],             # Nose tip (30)
        [0.0, -63.6, -12.5],          # Chin (8)
        [-43.3, 32.7, -26.0],         # Left eye (36)
        [43.3, 32.7, -26.0],          # Right eye (45)
        [-28.9, -28.9, -24.1],        # Left mouth (48)
        [28.9, -28.9, -24.1],         # Right mouth (54)
    ], dtype=np.float64)

    # 2D image points
    image_points = np.array([
        pts[30],  # Nose tip
        pts[8],   # Chin
        pts[36],  # Left eye
        pts[45],  # Right eye
        pts[48],  # Left mouth
        pts[54],  # Right mouth
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
    """Create smooth per-region masks from landmarks.

    Returns dict of region_name -> (H, W) float mask [0, 1].
    """
    h, w = frame_shape
    pts = landmarks.points.astype(np.int32)

    masks = {}

    # Eye masks (elliptical)
    for name, indices in [("left_eye", range(42, 48)), ("right_eye", range(36, 42))]:
        eye_pts = pts[list(indices)]
        center = np.mean(eye_pts, axis=0).astype(int)
        rx = int(np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0])) // 2 + 5
        ry = int(np.max(eye_pts[:, 1]) - np.min(eye_pts[:, 1])) // 2 + 5
        mask = _elliptical_mask(h, w, center[1], center[0], ry, rx)
        masks[name] = mask

    # Brow masks (above eyes)
    for name, indices in [("left_brow", range(22, 27)), ("right_brow", range(17, 22))]:
        brow_pts = pts[list(indices)]
        center = np.mean(brow_pts, axis=0).astype(int)
        rx = int(np.max(brow_pts[:, 0]) - np.min(brow_pts[:, 0])) // 2 + 8
        ry = 15
        mask = _elliptical_mask(h, w, center[1] - 10, center[0], ry, rx)
        masks[name] = mask

    # Nose mask
    nose_pts = pts[27:36]
    center = np.mean(nose_pts, axis=0).astype(int)
    rx = int(np.max(nose_pts[:, 0]) - np.min(nose_pts[:, 0])) // 2 + 5
    ry = int(np.max(nose_pts[:, 1]) - np.min(nose_pts[:, 1])) // 2 + 5
    masks["nose"] = _elliptical_mask(h, w, center[1], center[0], ry, rx)

    # Mouth mask
    mouth_pts = pts[48:60]
    center = np.mean(mouth_pts, axis=0).astype(int)
    rx = int(np.max(mouth_pts[:, 0]) - np.min(mouth_pts[:, 0])) // 2 + 5
    ry = int(np.max(mouth_pts[:, 1]) - np.min(mouth_pts[:, 1])) // 2 + 5
    masks["mouth"] = _elliptical_mask(h, w, center[1], center[0], ry, rx)

    # Face contour (convex hull of jaw points)
    jaw_pts = pts[0:17]
    hull = cv2.convexHull(jaw_pts)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 255)
    face_mask = cv2.GaussianBlur(face_mask.astype(np.float32) / 255.0, (11, 11), 3)
    masks["face"] = face_mask

    # Skin = face minus eyes, nose, mouth
    skin = face_mask.copy()
    for name in ("left_eye", "right_eye", "nose", "mouth"):
        skin = np.clip(skin - masks[name], 0, 1)
    masks["skin"] = skin

    return masks


def _elliptical_mask(h: int, w: int, cy: int, cx: int, ry: int, rx: int) -> np.ndarray:
    """Create a smooth elliptical mask."""
    Y, X = np.ogrid[:h, :w]
    d = ((X - cx) / max(rx, 1)) ** 2 + ((Y - cy) / max(ry, 1)) ** 2
    mask = np.clip(1.0 - d, 0, 1)
    k = max(int(min(rx, ry) * 0.4) | 1, 3)
    return cv2.GaussianBlur(mask, (k, k), max(k / 3, 1.0))
