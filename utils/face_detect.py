"""
face_detect.py — Shared face detection for the enhancement pipeline.

Uses OpenCV DNN SSD face detector (ResNet-10, Caffe backend).
Model auto-downloads on first use — no manual setup required.

Model: res10_300x300_ssd_iter_140000.caffemodel (~10 MB)
Accuracy: ~90% mAP on constrained video (vs ~60-70% for Haar/HOG).
Runs comfortably on CPU for batch/automation workflows.

Usage:
    from utils.face_detect import detect_face, detect_faces
    face = detect_face(frame)        # largest face, (x, y, w, h) or None
    faces = detect_faces(frame)      # all faces, list of (x, y, w, h)
"""

import sys
import ssl
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

_DNN_URLS = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}

_DNN_CACHE = Path.home() / ".cache" / "opencv_face_dnn"
_NET: Optional[cv2.dnn.Net] = None


def _ensure_model() -> Tuple[str, str]:
    """Download DNN model files if not cached. Returns (prototxt, caffemodel) paths."""
    _DNN_CACHE.mkdir(parents=True, exist_ok=True)

    prototxt = str(_DNN_CACHE / "deploy.prototxt")
    caffemodel = str(_DNN_CACHE / "res10_300x300_ssd_iter_140000.caffemodel")

    for name, url in _DNN_URLS.items():
        dest = _DNN_CACHE / name
        if not dest.exists():
            print(f"[face_detect] Downloading {name}...", file=sys.stderr)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            try:
                opener = urllib.request.build_opener(
                    urllib.request.HTTPSHandler(context=ctx)
                )
                with opener.open(url) as resp, open(dest, "wb") as f:
                    f.write(resp.read())
                print(f"[face_detect] Downloaded {name} ({dest.stat().st_size / 1024:.0f} KB)", file=sys.stderr)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download face detection model {name} from {url}: {e}"
                ) from e
    return prototxt, caffemodel


def _get_net() -> cv2.dnn.Net:
    global _NET
    if _NET is None:
        prototxt, caffemodel = _ensure_model()
        _NET = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    return _NET


def _dnn_detect(frame: np.ndarray, score_threshold: float = 0.7) -> List[Tuple[int, int, int, int]]:
    """Run DNN face detection. Returns list of (x, y, w, h) in original frame coords."""
    net = _get_net()
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < score_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        faces.append((x1, y1, x2 - x1, y2 - y1))

    return faces


def detect_face(frame: np.ndarray, score_threshold: float = 0.7) -> Optional[Tuple[int, int, int, int]]:
    """Detect the largest face in a BGR frame. Returns (x, y, w, h) or None.

    Parameters
    ----------
    frame : np.ndarray
        BGR image array.
    score_threshold : float
        Minimum confidence (0-1). Lower = more detections but more false positives.

    Returns
    -------
    (x, y, w, h) tuple or None
    """
    faces = _dnn_detect(frame, score_threshold)
    if not faces:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


def detect_faces(frame: np.ndarray, score_threshold: float = 0.7) -> List[Tuple[int, int, int, int]]:
    """Detect all faces in a BGR frame. Returns list of (x, y, w, h).

    Parameters
    ----------
    frame : np.ndarray
        BGR image array.
    score_threshold : float
        Minimum confidence (0-1). Lower = more detections but more false positives.

    Returns
    -------
    List of (x, y, w, h) tuples, empty list if none found.
    """
    return _dnn_detect(frame, score_threshold)


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
