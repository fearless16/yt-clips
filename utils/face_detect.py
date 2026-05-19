"""
face_detect.py — Shared face detection for the enhancement pipeline.

Provides a single Haar Cascade instance that all modules can reuse
instead of each loading their own copy.

Usage:
    from utils.face_detect import detect_face, detect_faces
    face = detect_face(frame)        # largest face, (x, y, w, h) or None
    faces = detect_faces(frame)      # all faces, list of (x, y, w, h)
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

# Module-level singleton
_CASCADE = None


def _get_cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _CASCADE


def detect_face(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect the largest face in a BGR frame. Returns (x, y, w, h) or None."""
    cascade = _get_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(rects) == 0:
        return None
    return max(rects, key=lambda r: r[2] * r[3])


def detect_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect all faces in a BGR frame. Returns list of (x, y, w, h)."""
    cascade = _get_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]


def crop_face_with_padding(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.3,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop face region with padding. Returns (crop, (x1, y1, x2, y2))."""
    x, y, w, h = bbox
    pad = int(max(w, h) * padding)
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_frame, x + w + pad)
    y2 = min(h_frame, y + h + pad)
    if y2 <= y1 or x2 <= x1:
        return frame, (0, 0, w_frame, h_frame)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
