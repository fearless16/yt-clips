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

_LOCK = threading.Lock()
_SESSION: Optional[object] = None

_INPUT_SIZE = 640
_CANVAS: Optional[np.ndarray] = None

try:
    import onnxruntime as ort
    _HAS_DML = "DmlExecutionProvider" in ort.get_available_providers()
except ImportError:
    _HAS_DML = False

if not _HAS_DML:
    raise RuntimeError(
        "DirectML GPU not available for ONNX Runtime. "
        "Install onnxruntime-directml and ensure your AMD GPU supports DirectML."
    )


def _download_model(url: str, dest: Path) -> bool:
    import urllib.request
    try:
        log.info("Downloading face detection model to %s (%s)...", dest, dest.name)
        urllib.request.urlretrieve(url, str(dest))
        sz = dest.stat().st_size
        log.info("Downloaded %s (%d KB)", dest.name, sz // 1024)
        return True
    except Exception as e:
        log.warning("Failed to download %s: %s", dest.name, e)
        return False


def _ensure_scrfd_model() -> bool:
    if SCRFD_MODEL_PATH.exists():
        return True
    url = ("https://github.com/cysin/scrfd_onnx/raw/refs/heads/main/"
           "scrfd_10g_bnkps.onnx")
    return _download_model(url, SCRFD_MODEL_PATH)


def get_session() -> object:
    global _SESSION
    if _SESSION is None:
        with _LOCK:
            if _SESSION is None:
                if not _ensure_scrfd_model():
                    raise RuntimeError("SCRFD model not found and download failed")
                import onnxruntime as ort
                _SESSION = ort.InferenceSession(
                    str(SCRFD_MODEL_PATH),
                    providers=["DmlExecutionProvider", "CPUExecutionProvider"],
                )
                log.info("SCRFD face detector initialized (DirectML GPU)")
    return _SESSION


def _decode_scrfd(outputs, input_size: int, scale: float, score_threshold: float) -> List[Tuple[int, int, int, int]]:
    strides = [8, 16, 32]
    dets_list = []
    for idx, stride in enumerate(strides):
        scores = outputs[idx][0]          # (n*2, 1)
        bboxes = outputs[idx + 3][0]      # (n*2, 4)
        fm_h = input_size // stride
        fm_w = input_size // stride
        n = fm_h * fm_w
        scores_flat = scores[:, 0]        # (n*2,)
        mask = scores_flat >= score_threshold
        if not np.any(mask):
            continue
        aidx = np.arange(n * 2, dtype=np.intp)[mask]
        selected_scores = scores_flat[mask]
        selected_bbox = bboxes[mask]
        i_vals = aidx // 2
        col = (i_vals % fm_w).astype(np.float32)
        row = (i_vals // fm_w).astype(np.float32)
        cx = col * stride + stride // 2
        cy = row * stride + stride // 2
        candidates = np.column_stack([
            cx - selected_bbox[:, 0],
            cy - selected_bbox[:, 1],
            cx + selected_bbox[:, 2],
            cy + selected_bbox[:, 3],
            selected_scores,
        ])
        dets_list.append(candidates)
    if not dets_list:
        return []
    dets = np.concatenate(dets_list)
    keep = _nms(dets, 0.4)
    dets = dets[keep]
    if len(dets) == 0:
        return []
    ow = int(input_size / scale)
    oh = int(input_size / scale)
    x1 = np.maximum(0, (dets[:, 0] / scale).astype(np.int32))
    y1 = np.maximum(0, (dets[:, 1] / scale).astype(np.int32))
    x2 = np.minimum(ow, (dets[:, 2] / scale).astype(np.int32))
    y2 = np.minimum(oh, (dets[:, 3] / scale).astype(np.int32))
    valid = (x2 > x1) & (y2 > y1)
    return [
        (int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i]))
        for i in np.where(valid)[0]
    ]


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


def _detect_onnx(frame: np.ndarray, score_threshold: float) -> List[Tuple[int, int, int, int]]:
    global _CANVAS
    session = get_session()
    input_size = _INPUT_SIZE
    h, w = frame.shape[:2]
    scale = min(input_size / w, input_size / h)
    nw, nh = int(w * scale), int(h * scale)

    if _CANVAS is None or _CANVAS.shape[0] != input_size:
        _CANVAS = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    elif _CANVAS.shape[1] != input_size:
        _CANVAS = np.zeros((input_size, input_size, 3), dtype=np.uint8)

    _CANVAS[:nh, :nw] = cv2.resize(frame, (nw, nh))
    rgb = cv2.cvtColor(_CANVAS, cv2.COLOR_BGR2RGB)
    blob = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, :]

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    faces = _decode_scrfd(outputs, input_size, scale, score_threshold)
    if faces:
        log.debug("GPU detected %d face(s)", len(faces))
    return faces


def detect_face(frame: np.ndarray, score_threshold: float = 0.5) -> Optional[Tuple[int, int, int, int]]:
    faces = detect_faces(frame, score_threshold)
    if not faces:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


def detect_faces(frame: np.ndarray, score_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    if frame is None or frame.size == 0:
        return []
    return _detect_onnx(frame, score_threshold)


def crop_face_with_padding(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.3,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
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
