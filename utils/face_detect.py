"""
face_detect.py — Shared face detection for the enhancement pipeline.
Uses MediaPipe FaceDetector with face_detector.tflite.
Model instances are cached (Singleton pattern) to avoid reloading.
"""

import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("face_detect", cfg.get("logging", {}).get("log_file", "logs/pipeline.log"), cfg.get("logging", {}).get("level", "INFO"))

_MP_INSTANCE: Optional[tuple] = None  # (mp_module, detector)
_MP_LOCK = threading.Lock()
_MP_UNAVAILABLE: bool = False  # True if mediapipe import failed once


def _get_mp() -> Optional[tuple]:
    """Singleton getter for (mediapipe_module, FaceDetector).

    Returns None if mediapipe is not installed (after logging a single warning).
    """
    global _MP_INSTANCE, _MP_UNAVAILABLE
    if _MP_UNAVAILABLE:
        return None
    if _MP_INSTANCE is None:
        with _MP_LOCK:
            if _MP_INSTANCE is None:
                try:
                    import mediapipe as mp
                    from mediapipe.tasks import python
                    from mediapipe.tasks.python import vision
                    model_path = Path(__file__).resolve().parent.parent / "face_detector.tflite"
                    if not model_path.exists():
                        model_path = Path("face_detector.tflite")
                    if not model_path.exists():
                        raise FileNotFoundError(f"MediaPipe face detector model file not found: {model_path}")
                    log.info(f"Initializing MediaPipe FaceDetector with model: {model_path}")
                    base_options = python.BaseOptions(model_asset_path=str(model_path))
                    options = vision.FaceDetectorOptions(base_options=base_options)
                    _MP_INSTANCE = (mp, vision.FaceDetector.create_from_options(options))
                except ImportError:
                    _MP_UNAVAILABLE = True
                    log.warning(
                        "MediaPipe not installed — face detection disabled. "
                        "Install with: pip install mediapipe"
                    )
                    return None
    return _MP_INSTANCE


def detect_face(frame: np.ndarray, score_threshold: float = 0.5) -> Optional[Tuple[int, int, int, int]]:
    """Detect the largest face in a BGR frame. Returns (x, y, w, h) or None.

    Parameters
    ----------
    frame : np.ndarray
        BGR image array.
    score_threshold : float
        Minimum confidence (0-1). Lower = more detections.

    Returns
    -------
    (x, y, w, h) tuple or None
    """
    faces = detect_faces(frame, score_threshold)
    if not faces:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


def detect_faces(frame: np.ndarray, score_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """Detect all faces in a BGR frame. Returns list of (x, y, w, h).

    Parameters
    ----------
    frame : np.ndarray
        BGR image array.
    score_threshold : float
        Minimum confidence (0-1).

    Returns
    -------
    List of (x, y, w, h) tuples, empty list if none found.
    """
    if frame is None or frame.size == 0:
        return []

    try:
        mp_detector = _get_mp()
        if mp_detector is None:
            return []
        mp, detector = mp_detector
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        res = detector.detect(mp_image)
        faces = []
        h, w = frame.shape[:2]
        
        for d in res.detections:
            score = d.categories[0].score if d.categories else 1.0
            if score < score_threshold:
                continue
                
            box = d.bounding_box
            x1 = max(0, int(box.origin_x))
            y1 = max(0, int(box.origin_y))
            x2 = min(w, x1 + int(box.width))
            y2 = min(h, y1 + int(box.height))
            
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2 - x1, y2 - y1))
                
        return faces
    except Exception as e:
        log.warning(f"MediaPipe face detection failed: {e}")
        return []


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
