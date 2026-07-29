import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from utils.face_detect import detect_faces

log = logging.getLogger("face_matcher")

_HOST_ENCODINGS: Optional[List[np.ndarray]] = None
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PHOTOS_DIR = _PROJECT_ROOT / "photos"


def _extract_embedding(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    face = frame_bgr[y:y + h, x:x + w]
    if face.size == 0:
        return None
    face = cv2.resize(face, (112, 112))
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    hist_l = cv2.calcHist([l], [0], None, [32], [0, 256]).flatten()
    hist_a = cv2.calcHist([a], [0], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([b], [0], None, [32], [0, 256]).flatten()
    hist = np.concatenate([hist_l, hist_a, hist_b])
    hist = hist / max(hist.sum(), 1)
    return hist


def _compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-6)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-6)
    return float(np.dot(emb1, emb2))


def get_host_encodings() -> List[np.ndarray]:
    global _HOST_ENCODINGS
    if _HOST_ENCODINGS is not None:
        return _HOST_ENCODINGS

    import cv2

    encodings = []
    photo_paths = sorted(
        str(p) for p in _PHOTOS_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not photo_paths:
        log.warning("No reference photos found in '%s' directory.", _PHOTOS_DIR)
        _HOST_ENCODINGS = []
        return []

    log.info("Encoding %d reference photos from %s using SCRFD GPU + LAB histogram...",
             len(photo_paths), _PHOTOS_DIR)
    for path in photo_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            faces = detect_faces(img, score_threshold=0.5)
            if faces:
                best = max(faces, key=lambda r: r[2] * r[3])
                emb = _extract_embedding(img, best)
                if emb is not None:
                    encodings.append(emb)
                    log.info("  Encoded face from %s", path)
            else:
                log.warning("No faces found in reference photo: %s", path)
        except Exception as e:
            log.error("Failed to load reference photo %s: %s", path, e)

    _HOST_ENCODINGS = encodings
    log.info("Successfully loaded %d host face encodings.", len(encodings))
    return _HOST_ENCODINGS


def find_host_in_frame(frame_bgr: np.ndarray, facecam_bounds: dict = None) -> Optional[dict]:
    host_encs = get_host_encodings()
    if not host_encs:
        return None

    import cv2

    try:
        faces = detect_faces(frame_bgr, score_threshold=0.5)

        if not faces and facecam_bounds:
            fc_x = facecam_bounds.get("x", 0)
            fc_y = facecam_bounds.get("y", 0)
            fc_w = facecam_bounds.get("width", 320)
            fc_h = facecam_bounds.get("height", 180)
            fh, fw = frame_bgr.shape[:2]
            x1, y1 = max(0, fc_x), max(0, fc_y)
            x2, y2 = min(fw, fc_x + fc_w), min(fh, fc_y + fc_h)
            if x2 > x1 and y2 > y1:
                facecam_crop = frame_bgr[y1:y2, x1:x2]
                faces = detect_faces(facecam_crop, score_threshold=0.5)
                if faces:
                    faces = [(x + x1, y + y1, w, h) for (x, y, w, h) in faces]
                    log.debug("Face found via facecam-region crop (%d,%d %dx%d)",
                              fc_x, fc_y, fc_w, fc_h)

        if not faces:
            return None

        best_match_idx = -1
        best_similarity = -1.0

        for i, (x, y, w, h) in enumerate(faces):
            emb = _extract_embedding(frame_bgr, (x, y, w, h))
            if emb is None:
                continue
            sims = [_compute_similarity(emb, he) for he in host_encs]
            max_sim = max(sims) if sims else 0.0
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_match_idx = i

        if best_match_idx != -1 and best_similarity >= 0.35:
            x, y, w, h = faces[best_match_idx]
            return {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "confidence": best_similarity,
            }

    except Exception as e:
        log.error("Face matching failed on frame: %s", e)

    return None
