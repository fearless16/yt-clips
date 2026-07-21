"""
face_detect.py — Shared face detection for the enhancement pipeline.
Primary (GPU): SCRFD via ONNX Runtime DirectML (32GB GPU, ~212 FPS)
Fallback (CPU): YuNet via OpenCV contrib
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

SCRFD_MODEL_PATH = Path(__file__).resolve().parent.parent / "scrfd_10g_bnkps.onnx"
YUNET_MODEL_PATH = Path(__file__).resolve().parent.parent / "face_detection_yunet_2023mar.onnx"

_LOCK = threading.Lock()

_ONNX_SESSION: Optional[object] = None
_ONNX_UNAVAILABLE: bool = False
_ONNX_AVAILABLE: bool = False

_YUNET_INSTANCE: Optional[cv2.FaceDetectorYN] = None
_YUNET_UNAVAILABLE: bool = False

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = "DmlExecutionProvider" in ort.get_available_providers()
except ImportError:
    _ONNX_AVAILABLE = False


def _download_model(url: str, dest: Path) -> bool:
    import urllib.request
    try:
        log.info(f"Downloading face detection model to {dest} ({dest.name})...")
        urllib.request.urlretrieve(url, str(dest))
        sz = dest.stat().st_size
        log.info(f"Downloaded {dest.name} ({sz / 1024:.0f} KB)")
        return True
    except Exception as e:
        log.warning(f"Failed to download {dest.name}: {e}")
        return False


def _ensure_scrfd_model() -> bool:
    if SCRFD_MODEL_PATH.exists():
        return True
    url = ("https://github.com/cysin/scrfd_onnx/raw/refs/heads/main/"
           "scrfd_10g_bnkps.onnx")
    return _download_model(url, SCRFD_MODEL_PATH)


def _ensure_yunet_model() -> bool:
    if YUNET_MODEL_PATH.exists():
        return True
    url = ("https://github.com/opencv/opencv_zoo/raw/main/"
           "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
    return _download_model(url, YUNET_MODEL_PATH)


def _get_onnx_session() -> Optional[object]:
    global _ONNX_SESSION, _ONNX_UNAVAILABLE, _ONNX_AVAILABLE
    if _ONNX_UNAVAILABLE or not _ONNX_AVAILABLE:
        return None
    if _ONNX_SESSION is None:
        with _LOCK:
            if _ONNX_SESSION is None:
                if not _ensure_scrfd_model():
                    _ONNX_UNAVAILABLE = True
                    return None
                try:
                    import onnxruntime as ort
                    _ONNX_SESSION = ort.InferenceSession(
                        str(SCRFD_MODEL_PATH),
                        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
                    )
                    log.info(f"SCRFD GPU face detector initialized (DirectML, 212 FPS)")
                except Exception as e:
                    _ONNX_UNAVAILABLE = True
                    log.warning(f"Failed to create ONNX GPU session: {e}")
                    return None
    return _ONNX_SESSION


def _get_yunet(input_size: Tuple[int, int]) -> Optional[cv2.FaceDetectorYN]:
    global _YUNET_INSTANCE, _YUNET_UNAVAILABLE
    if _YUNET_UNAVAILABLE:
        return None
    if _YUNET_INSTANCE is None:
        with _LOCK:
            if _YUNET_INSTANCE is None:
                if not _ensure_yunet_model():
                    _YUNET_UNAVAILABLE = True
                    return None
                try:
                    _YUNET_INSTANCE = cv2.FaceDetectorYN.create(
                        model=str(YUNET_MODEL_PATH),
                        config="",
                        input_size=input_size,
                        score_threshold=0.7,
                        nms_threshold=0.3,
                        top_k=5000,
                    )
                except Exception as e:
                    _YUNET_UNAVAILABLE = True
                    log.warning(f"Failed to create YuNet detector: {e}")
                    return None
    _YUNET_INSTANCE.setInputSize(input_size)
    return _YUNET_INSTANCE


def _decode_scrfd(outputs: List[np.ndarray], input_size: int, scale: float, score_threshold: float) -> List[Tuple[int, int, int, int]]:
    strides = [8, 16, 32]
    dets = []
    for idx, stride in enumerate(strides):
        scores = outputs[idx][0]
        bboxes = outputs[idx + 3][0]
        fm_h = input_size // stride
        fm_w = input_size // stride
        for i in range(fm_h * fm_w):
            for a in range(2):
                aidx = i * 2 + a
                score = float(scores[aidx][0])
                if score < score_threshold:
                    continue
                col = i % fm_w
                row = i // fm_w
                cx = col * stride + stride // 2
                cy = row * stride + stride // 2
                x1 = cx - float(bboxes[aidx][0])
                y1 = cy - float(bboxes[aidx][1])
                x2 = cx + float(bboxes[aidx][2])
                y2 = cy + float(bboxes[aidx][3])
                dets.append([x1, y1, x2, y2, score])
    if not dets:
        return []
    dets = np.array(dets)
    keep = _nms(dets, 0.4)
    dets = dets[keep]
    results = []
    ow = int(input_size / scale)
    oh = int(input_size / scale)
    for d in dets:
        x1 = max(0, int(d[0] / scale))
        y1 = max(0, int(d[1] / scale))
        x2 = min(ow, int(d[2] / scale))
        y2 = min(oh, int(d[3] / scale))
        if x2 > x1 and y2 > y1:
            results.append((x1, y1, x2 - x1, y2 - y1))
    return results


def _nms(dets: np.ndarray, thresh: float) -> List[int]:
    if len(dets) == 0:
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[order[1:]]
        order = order[1:][overlap <= thresh]
    return keep


def _detect_onnx(frame: np.ndarray, score_threshold: float) -> Optional[List[Tuple[int, int, int, int]]]:
    session = _get_onnx_session()
    if session is None:
        return None
    h, w = frame.shape[:2]
    input_size = 640
    scale = min(input_size / w, input_size / h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    try:
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: blob})
    except Exception as e:
        log.warning(f"ONNX GPU inference failed: {e}")
        return []
    faces = _decode_scrfd(outputs, input_size, scale, score_threshold)
    if faces:
        log.debug(f"GPU detected {len(faces)} face(s)")
    return faces


def _detect_yunet(frame: np.ndarray, score_threshold: float) -> List[Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    yunet = _get_yunet((w, h))
    if yunet is None:
        return []
    try:
        yunet.setScoreThreshold(score_threshold)
        _, results = yunet.detect(frame)
    except Exception as e:
        log.warning(f"YuNet detection failed: {e}")
        return []
    if results is None:
        return []
    faces = []
    for r in results:
        score = float(r[-1])
        if score < score_threshold:
            continue
        x1 = max(0, int(r[0]))
        y1 = max(0, int(r[1]))
        x2 = min(w, int(r[2]))
        y2 = min(h, int(r[3]))
        if x2 > x1 and y2 > y1:
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces


def detect_face(frame: np.ndarray, score_threshold: float = 0.5, backend: str = "auto") -> Optional[Tuple[int, int, int, int]]:
    """Detect the largest face in a BGR frame. Returns (x, y, w, h) or None.

    Parameters
    ----------
    frame : np.ndarray
        BGR image array.
    score_threshold : float
        Minimum confidence (0-1). Lower = more detections.
    backend : str
        "auto" (GPU if available), "yunet", or "onnx".

    Returns
    -------
    (x, y, w, h) tuple or None
    """
    faces = detect_faces(frame, score_threshold, backend)
    if not faces:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


def detect_faces(frame: np.ndarray, score_threshold: float = 0.5, backend: str = "auto") -> List[Tuple[int, int, int, int]]:
    """Detect all faces in a BGR frame. Returns list of (x, y, w, h).

    If GPU (DirectML) is available, uses SCRFD for best accuracy at 200+ FPS.
    Falls back to YuNet (OpenCV, CPU) otherwise.

    Parameters
    ----------
    frame : np.ndarray
        BGR image array.
    score_threshold : float
        Minimum confidence (0-1).
    backend : str
        "auto" (GPU if available), "yunet", or "onnx".

    Returns
    -------
    List of (x, y, w, h) tuples, empty list if none found.
    """
    if frame is None or frame.size == 0:
        return []

    if backend == "onnx" or (backend == "auto" and _ONNX_AVAILABLE and not _ONNX_UNAVAILABLE):
        result = _detect_onnx(frame, score_threshold)
        if result is not None:
            return result
        log.debug("ONNX GPU unavailable, falling back to YuNet (CPU)")
    return _detect_yunet(frame, score_threshold)


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
