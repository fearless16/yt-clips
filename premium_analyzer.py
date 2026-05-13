"""
premium_analyzer.py — YOLOv8-face + ByteTrack + Kalman filter + bezier crop.
Replaces frame_analyzer.py's cheap Haar Cascade + EMA with proper tracking.

Usage:
    from premium_analyzer import PremiumAnalyzer
    pa = PremiumAnalyzer()
    result = pa.analyze_clip(video_path, start, end)
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("premium", cfg["logging"]["log_file"], cfg["logging"]["level"])

# ─── GPU / Backend Detection ──────────────────────────────────────────────

HAS_TORCH = False
HAS_YOLO = False
HAS_FILTERPY = False
HAS_SCIPY = False
try:
    import torch
    HAS_TORCH = True
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    pass
try:
    from filterpy.kalman import KalmanFilter
    HAS_FILTERPY = True
except ImportError:
    pass
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    pass

# ─── ByteTrack helpers ────────────────────────────────────────────────────

class KalmanBoxTracker:

    def __init__(self, bbox: np.ndarray):
        if not HAS_FILTERPY:
            raise ImportError("filterpy required for Kalman tracking — pip install filterpy")
        self.id = KalmanBoxTracker._next_id()
        self.hits = 1
        self.no_loss = 0
        self.bbox = bbox.astype(np.float32)
        self.kf = self._init_kf(bbox)

    @staticmethod
    def _next_id():
        id_ = getattr(KalmanBoxTracker, "_counter", 0)
        KalmanBoxTracker._counter = id_ + 1
        return id_

    @staticmethod
    def _init_kf(bbox: np.ndarray):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=np.float32)
        kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=np.float32)
        kf.R[2:,2:] *= 10.
        kf.P[4:,4:] *= 1000.
        kf.P *= 10.
        kf.Q[-1,-1] *= 0.01
        kf.Q[4:,4:] *= 0.01
        kf.x[:4] = bbox.reshape(4, 1)
        return kf

    def update(self, bbox: np.ndarray):
        self.hits += 1
        self.no_loss = 0
        self.kf.update(bbox.reshape(4, 1))

    def predict(self) -> np.ndarray:
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.no_loss += 1
        return self.kf.x[:4].flatten()

    @property
    def state(self) -> np.ndarray:
        return self.kf.x[:4].flatten()


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _linear_assignment(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    if not HAS_SCIPY:
        # Fallback: greedy matching (no SciPy available)
        matched = []
        used_rows, used_cols = set(), set()
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if i not in used_rows and j not in used_cols:
                    matched.append((i, j))
                    used_rows.add(i)
                    used_cols.add(j)
        return matched
    row, col = linear_sum_assignment(cost_matrix)
    return list(zip(row, col))


class ByteTrack:
    """Simple ByteTrack — Kalman filter + IoU matching with two thresholds."""

    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1, max_lost: int = 30):
        self.high = high_thresh
        self.low = low_thresh
        self.max_lost = max_lost
        self.trackers: List[KalmanBoxTracker] = []
        self._frame_count = 0

    def update(self, detections: np.ndarray, scores: np.ndarray) -> List[Dict]:
        if detections.size == 0:
            detections = np.empty((0, 4))

        self._frame_count += 1

        # Predict
        for t in self.trackers:
            t.predict()

        # Split detections by score
        high_mask = scores >= self.high
        high_dets = detections[high_mask]
        high_scores = scores[high_mask]
        low_dets = detections[~high_mask]
        low_scores = scores[~high_mask]

        # First match: high score detections with trackers — UPDATE matched trackers
        matched, unmatched_dets, unmatched_trks = self._match(high_dets, self.trackers)
        for dt_idx, trk_idx in matched:
            self.trackers[trk_idx].update(high_dets[dt_idx])

        # Second match: low score detections with remaining unmatched trackers
        if len(unmatched_trks) > 0 and len(low_dets) > 0:
            remaining_trks = [self.trackers[i] for i in unmatched_trks]
            matched2, _, still_unmatched = self._match(low_dets, remaining_trks)
            for dt_idx, trk_idx in matched2:
                actual_trk = unmatched_trks[trk_idx]
                self.trackers[actual_trk].update(low_dets[dt_idx])
            unmatched_trks = [unmatched_trks[i] for i in still_unmatched if i < len(unmatched_trks)]

        # Create new trackers for unmatched high-score detections
        for dt_idx in unmatched_dets:
            if dt_idx < len(high_dets):
                self.trackers.append(KalmanBoxTracker(high_dets[dt_idx]))

        # Remove lost trackers
        self.trackers = [t for t in self.trackers if t.no_loss < self.max_lost]

        # Return active tracks
        results = []
        for t in self.trackers:
            if t.hits >= 1:
                bbox = t.state
                results.append({
                    "id": t.id,
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "hits": t.hits,
                })
        return results

    def _match(self, detections: np.ndarray, trackers: List[KalmanBoxTracker]):
        if len(detections) == 0 or len(trackers) == 0:
            return [], list(range(len(detections))), list(range(len(trackers)))

        cost = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for i, det in enumerate(detections):
            for j, trk in enumerate(trackers):
                cost[i, j] = 1 - _iou(det, trk.state)

        matches = _linear_assignment(cost)
        matched_dt = set()
        matched_trk = set()
        result = []
        for i, j in matches:
            if cost[i, j] < 1 - self.low:
                result.append((i, j))
                matched_dt.add(i)
                matched_trk.add(j)
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dt]
        unmatched_trks = [j for j in range(len(trackers)) if j not in matched_trk]
        return result, unmatched_dets, unmatched_trks


# ─── Face Detection ────────────────────────────────────────────────────────

class FaceDetector:
    def __init__(self):
        self.backend = "haar"
        self.yolo_model = None
        if HAS_YOLO:
            try:
                self.yolo_model = YOLO("yolov8n-face.pt")
                self.backend = "yolo"
                log.info("Premium: YOLOv8-face loaded")
            except Exception as e:
                log.warning("YOLOv8-face load failed: %s — falling back to Haar", e)
        if self.backend == "haar":
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.backend == "yolo" and self.yolo_model:
            xyxy, conf = self._detect_yolo(frame_bgr)
        else:
            xyxy, conf = self._detect_haar(frame_bgr)
        # Filter out non-facecam faces (poster players in background)
        return self._filter_by_facecam_region(xyxy, conf, frame_bgr.shape[1], frame_bgr.shape[0])

    def _detect_yolo(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        results = self.yolo_model(frame_bgr, verbose=False, conf=0.3, iou=0.5)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 4)), np.empty((0,))
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        return xyxy, conf

    def _detect_haar(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30),
        )
        if len(faces) == 0:
            return np.empty((0, 4)), np.empty((0,))
        xyxy = np.array([[x, y, x + w, y + h] for (x, y, w, h) in faces], dtype=np.float32)
        conf = np.ones(len(faces), dtype=np.float32) * 0.5
        return xyxy, conf

    @staticmethod
    def _filter_by_facecam_region(xyxy: np.ndarray, conf: np.ndarray, fw: int, fh: int) -> Tuple[np.ndarray, np.ndarray]:
        """Keep only faces in the bottom-left facecam region. Poster faces in background get filtered."""
        layout_cfg = cfg.get("layout", {})
        fc = layout_cfg.get("facecam", {})
        if not layout_cfg.get("has_facecam", False):
            return xyxy, conf
        fx, fy, fw_cfg, fh_cfg = fc.get("x", 0), fc.get("y", 540), fc.get("width", 320), fc.get("height", 180)
        margin_w, margin_h = fw_cfg * 0.3, fh_cfg * 0.3
        keep = []
        for i in range(len(xyxy)):
            cx = (xyxy[i, 0] + xyxy[i, 2]) / 2
            cy = (xyxy[i, 1] + xyxy[i, 3]) / 2
            if (fx - margin_w) <= cx <= (fx + fw_cfg + margin_w) and (fy - margin_h) <= cy <= (fy + fh_cfg + margin_h):
                keep.append(i)
        if not keep:
            return np.empty((0, 4)), np.empty((0,))
        return xyxy[keep], conf[keep]


# ─── Crop Trajectory (Bezier / Kalman) ────────────────────────────────────

def _bezier_interpolate(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """Cubic bezier: smooth ease-in-out."""
    u = 1 - t
    return u*u*u * p0 + 3*u*u*t * p1 + 3*u*t*t * p2 + t*t*t * p3


class SmoothCrop:
    """Produces buttery-smooth 9:16 crop windows from face trajectories. Centers face in frame."""

    def __init__(self, frame_w: int, frame_h: int, smooth_factor: float = 0.15):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.crop_w = int(frame_h * 9 / 16)
        self.crop_h = frame_h
        self.x_history: List[float] = []
        self.y_history: List[float] = []
        self.smooth_factor = smooth_factor

    def get_crop(self, face_center_x: float, face_center_y: float, frame_idx: int = 0) -> Dict:
        # Center X: put face in horizontal center of 9:16 crop
        target_x = face_center_x - self.crop_w // 2
        target_x = max(0, min(target_x, self.frame_w - self.crop_w))

        # Center Y: put face in vertical center of 9:16 frame (face at ~40% from top = talking head framing)
        face_in_frame_h = int(self.crop_h * 0.40)  # Face at 40% from top for natural framing
        target_y = int(face_center_y - face_in_frame_h)
        target_y = max(0, min(target_y, max(0, self.frame_h - self.crop_h)))

        self.x_history.append(target_x)
        self.y_history.append(target_y)
        if len(self.x_history) > 10:
            self.x_history.pop(0)
        if len(self.y_history) > 10:
            self.y_history.pop(0)

        smooth_x = self._smooth_bezier(target_x, self.x_history)
        smooth_y = self._smooth_bezier(target_y, self.y_history)
        return {"x": int(smooth_x), "y": int(smooth_y), "width": self.crop_w, "height": self.crop_h}

    def _smooth_bezier(self, target: float, history: List[float]) -> float:
        if len(history) < 3:
            return target
        p0 = history[-3]
        p3 = target
        mid = (p0 + p3) / 2
        cp1 = p0 + (mid - p0) * self.smooth_factor
        cp2 = p3 - (p3 - mid) * self.smooth_factor
        t = 1.0 - 1.0 / max(1, len(history))
        return _bezier_interpolate(p0, cp1, cp2, p3, t)


# ─── Layout Classifier (Lightweight CNN) ──────────────────────────────────

def _classify_layout(frame_bgr: np.ndarray) -> str:
    """Heuristic-based layout classification (CNN replacement).
    Returns: solo | split_both | split_guest_off | screen_share | blank
    """
    h, w = frame_bgr.shape[:2]
    if h < 4 or w < 4:
        return "solo"
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Check blank / black
    if gray.mean() < 25 and gray.var() < 100:
        return "blank"

    # Check vertical divider (split-screen)
    center_col = w // 2
    left_half = gray[:, :center_col]
    right_half = gray[:, center_col:]
    left_avg = left_half.mean()
    right_avg = right_half.mean()

    # Check if one side is black
    right_black = right_avg < 25 and right_half.var() < 100
    left_black = left_avg < 25 and left_half.var() < 100

    if right_black or left_black:
        return "split_guest_off"

    # Check vertical divider line (dark AND light divider)
    center_strip = gray[:, center_col - 2:center_col + 2]
    left_right_avg = (left_avg + right_avg) / 2
    center_deviation = abs(float(center_strip.mean()) - left_right_avg)
    if center_deviation > 25:
        return "split_both"

    # Also detect divider as sharp horizontal brightness edge
    center_edge = cv2.Canny(gray[:, center_col - 4:center_col + 4], 30, 100)
    if center_edge.mean() > 0.05:
        return "split_both"

    # Screen share: high edge density + contrast (higher threshold to avoid false-positives from dual layout)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.sum()) / edges.size
    if edge_density > 0.15 and gray.var() > 800:
        return "screen_share"

    return "solo"


# ─── Chat Overlay Detection ──────────────────────────────────────────────

def _detect_chat_region(frame_bgr: np.ndarray) -> Optional[Dict]:
    layout_cfg = cfg.get("layout", {})
    if not layout_cfg.get("has_chat_overlay", False):
        return None

    h, w = frame_bgr.shape[:2]
    chat_side = layout_cfg.get("chat", {}).get("side", "right")
    sample_w = min(w // 4, 350)

    if chat_side == "right":
        edge = frame_bgr[:, w - sample_w:]
        center = frame_bgr[:, w // 4:3 * w // 4]
    else:
        edge = frame_bgr[:, :sample_w]
        center = frame_bgr[:, w // 4:3 * w // 4]

    edge_gray = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
    center_gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)

    edge_avg = edge_gray.mean()
    center_avg = center_gray.mean()
    edge_var = edge_gray.var()

    brightness_diff = abs(edge_avg - center_avg)
    if brightness_diff > 30 and edge_var > 500:
        return {
            "chat_detected": True,
            "chat_side": chat_side,
            "chat_x": w - sample_w if chat_side == "right" else 0,
            "chat_width": sample_w,
        }
    return None


# ─── Main Analysis ────────────────────────────────────────────────────────

class PremiumAnalyzer:
    """Drop-in replacement for analyze_clip with Premium tracking + smoothing."""

    def __init__(self):
        self.face_detector = FaceDetector()
        self.tracker = ByteTrack()
        self.smooth_crop: Optional[SmoothCrop] = None
        self._frame_idx = 0
        self.host_detector: Optional[HostDetector] = None
        self._init_host_detector()

    def _init_host_detector(self):
        """Initialize host detector with reference photos from config."""
        host_photos_path = cfg.get("premium", {}).get("host_ref_photos", "")
        if not host_photos_path:
            self.host_detector = HostDetector([])
            return

        ref_photos = []
        photos_dir = Path(host_photos_path)
        if photos_dir.exists() and photos_dir.is_dir():
            for img_path in photos_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            ref_photos.append(img)
                    except Exception as e:
                        log.warning("Failed to load reference photo %s: %s", img_path, e)
        elif Path(host_photos_path).exists():
            try:
                img = cv2.imread(host_photos_path)
                if img is not None:
                    ref_photos.append(img)
            except Exception as e:
                log.warning("Failed to load reference photo: %s", e)

        self.host_detector = HostDetector(ref_photos)

    def reset(self):
        self.tracker = ByteTrack()
        self.smooth_crop = None
        self._frame_idx = 0

    def analyze_clip(
        self,
        video_path: str,
        start: float,
        end: float,
        transcript_segments: Optional[List[Dict]] = None,
        clip_id: str = "clip",
    ) -> Dict:
        """Full premium analysis. Returns same schema as frame_analyzer.analyze_clip."""
        self.reset()
        log.info("[%s] 🚀 Premium analysis: %.1fs-%.1fs", clip_id, start, end)

        info = _get_video_info(video_path)
        frame_w, frame_h = info["width"], info["height"]
        self.smooth_crop = SmoothCrop(frame_w, frame_h)

        # Sample frames
        duration = max(0.01, end - start)
        margin = min(0.5, duration * 0.1)
        sample_times = np.linspace(start + margin, end - margin, num=12)

        all_faces = []
        layouts = []
        chats = []

        for t in sample_times:
            frame = _extract_frame(video_path, float(t), frame_w, frame_h)
            if frame is None:
                continue

            xyxy, conf = self.face_detector.detect(frame)
            tracks = self.tracker.update(xyxy, conf)

            layout = _classify_layout(frame)
            layouts.append(layout)

            chat = _detect_chat_region(frame)
            if chat:
                chats.append(chat)

            for trk in tracks:
                if trk["hits"] >= 2:
                    b = trk["bbox"]
                    cx = (b[0] + b[2]) / 2
                    cy = (b[1] + b[3]) / 2
                    # Extract face region brightness + image for host identification
                    face_brightness = 128
                    face_img_roi = None
                    try:
                        x1, y1, x2, y2 = map(int, [b[0], b[1], b[2], b[3]])
                        if frame is not None and y2 > y1 and x2 > x1:
                            face_roi = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                            if face_roi.size > 0:
                                face_brightness = int(np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)))
                                face_img_roi = face_roi.copy()
                    except Exception:
                        pass
                    all_faces.append({
                        "x": cx, "y": cy,
                        "w": b[2] - b[0],
                        "h": b[3] - b[1],
                        "id": trk["id"],
                        "face_img": face_img_roi,
                        "brightness": face_brightness,
                    })
            self._frame_idx += 1

        # Determine dominant layout
        if layouts:
            from collections import Counter
            dominant_layout = Counter(layouts).most_common(1)[0][0]
        else:
            dominant_layout = "solo"

        # Determine best face (most tracked, highest confidence)
        best_face = None
        host_face = None
        if all_faces:
            from collections import Counter
            face_id_counts = Counter(f["id"] for f in all_faces)
            if face_id_counts:
                best_id = face_id_counts.most_common(1)[0][0]
                best_candidates = [f for f in all_faces if f["id"] == best_id]
                if best_candidates:
                    mid_idx = len(best_candidates) // 2
                    best_face = best_candidates[mid_idx]

            # ── Host Identification ───────────────────────────────────────────
            # Group faces by unique positions (different people)
            unique_faces = []
            for f in all_faces:
                is_new = True
                for uf in unique_faces:
                    if abs(f["x"] - uf["x"]) < 150 and abs(f["y"] - uf["y"]) < 100:
                        if f.get("id") == uf.get("id"):
                            is_new = False
                            break
                if is_new:
                    unique_faces.append(f)

            if unique_faces and self.host_detector:
                host_idx = self.host_detector.identify_host(unique_faces)
                if 0 <= host_idx < len(unique_faces):
                    host_face = unique_faces[host_idx]
                    host_face["is_host"] = True
                    log.info("[%s] Host identified at position (%.0f, %.0f)", clip_id, host_face["x"], host_face["y"])
                    # Use host as best face if identified
                    if best_face is None or host_face.get("w", 0) * host_face.get("h", 0) > 0:
                        best_face = host_face

        # Build export strategy (with host priority)
        strategy = self._build_strategy(best_face, dominant_layout, frame_w, frame_h, chats, host_face)

        # Layout metadata
        layout_info = {
            "layout_type": dominant_layout,
            "prefer_solo": dominant_layout == "solo",
            "has_black_panel": dominant_layout == "split_guest_off",
            "black_panel_side": "right" if dominant_layout == "split_guest_off" else None,
            "guest_cam_on": dominant_layout == "split_both",
            "guest_cam_off": dominant_layout == "split_guest_off",
            "is_screen_share": dominant_layout == "screen_share",
            "chat_overlay": chats[-1] if chats else None,
            "host_detected": host_face is not None,
            "host_position": {"x": host_face["x"], "y": host_face["y"]} if host_face else None,
        }

        # Audio / silence
        silence = _detect_dead_air(video_path, start, end)

        # Speed factor from transcript
        speed_factor = 1.0
        if transcript_segments:
            clip_segs = [
                s for s in transcript_segments
                if s["end"] > start and s["start"] < end
            ]
            total_text = " ".join(s.get("text", "") for s in clip_segs)
            words = len(total_text.split())
            clip_dur = max(0.01, end - start)
            wpm = (words / (clip_dur / 60)) if clip_dur > 0 else 0
            if wpm > 150:
                speed_factor = 1.15
            elif wpm < 80:
                speed_factor = 1.25
            else:
                silence_ratio = max(0.0, (clip_dur - words * 0.35) / clip_dur)
                if silence_ratio > 0.5:
                    speed_factor = 1.35
                elif silence_ratio > 0.3:
                    speed_factor = 1.25

        strategy["speed_factor"] = speed_factor
        strategy["skip_silence"] = silence.get("has_dead_air", False)

        # Lighting analysis + studio-grade color grading — set on strategy for export to use
        strategy["apply_lighting_fix"] = True
        lighting_filter = ""
        avg_brightness = 128
        if all_faces:
            face_brightness = []
            for f in all_faces:
                br = f.get("brightness", 128)
                if isinstance(br, (int, float)):
                    face_brightness.append(br)
            if face_brightness:
                avg_brightness = np.mean(face_brightness)
        # Studio grade: natural contrast, warm saturation, light denoise, no washout
        if avg_brightness < 80:
            lighting_filter = "eq=contrast=1.25:saturation=1.15:brightness=-0.02,curves=all='0/0 0.15/0.12 0.50/0.42 0.85/0.78 1/1',hqdn3d=1.5:1.5:1:1,unsharp=3:3:0.5:3:3:0.0"
        elif avg_brightness > 190:
            lighting_filter = "eq=contrast=1.3:saturation=1.1:brightness=-0.06,curves=all='0/0 0.15/0.10 0.50/0.40 0.85/0.75 1/0.95',hqdn3d=1.5:1.5:1:1,unsharp=3:3:0.5:3:3:0.0"
        else:
            lighting_filter = "eq=contrast=1.15:saturation=1.05:brightness=-0.03,hqdn3d=1:1:1:1,unsharp=3:3:0.3:3:3:0.0"
        strategy["lighting_filter"] = lighting_filter
        return {
            "black_frames": {"has_black_frames": dominant_layout in ("blank", "split_guest_off"), "is_mostly_black": dominant_layout == "blank"},
            "lighting": {"needs_correction": True, "lighting_filter": lighting_filter, "avg_brightness": avg_brightness},
            "layout": layout_info,
            "dead_air": silence,
            "export_strategy": strategy,
            "premium": {
                "face_tracks": len(all_faces),
                "tracked_ids": list(set(f["id"] for f in all_faces)) if all_faces else [],
                "dominant_layout": dominant_layout,
                "detection_backend": self.face_detector.backend,
            },
        }

    def _build_strategy(
        self,
        best_face: Optional[Dict],
        layout_type: str,
        frame_w: int,
        frame_h: int,
        chats: List[Dict],
        host_face: Optional[Dict] = None,
    ) -> Dict:
        strategy = {
            "use_solo_frame": False,
            "use_vertical_stack": False,
            "has_black_panel": layout_type == "split_guest_off",
            "black_panel_side": "right" if layout_type == "split_guest_off" else None,
            "guest_cam_on": layout_type == "split_both",
            "guest_cam_off": layout_type == "split_guest_off",
            "is_screen_share": layout_type == "screen_share",
            "should_drop": False,
            "active_crop": None,
            "speed_factor": 1.0,
            "apply_lighting_fix": False,
            "lighting_filter": "",
            "skip_silence": False,
            "export_aspect_ratio": "9:16",
            "chat_exclusion": None,
            "host_prioritized": host_face is not None,
        }

        if layout_type == "blank":
            strategy["should_drop"] = True
            return strategy

        if layout_type == "split_guest_off":
            strategy["should_drop"] = True
            return strategy

        # Host priority in split layout: crop to host instead of vertical stack
        if layout_type == "split_both" and host_face and self.smooth_crop:
            crop = self.smooth_crop.get_crop(host_face["x"], host_face["y"], self._frame_idx)
            strategy["use_solo_frame"] = True
            strategy["use_vertical_stack"] = False
            strategy["active_crop"] = crop
            log.info("Host prioritized in split layout - using solo crop")
            return strategy

        if layout_type == "split_both":
            strategy["use_vertical_stack"] = True
            return strategy

        if layout_type == "screen_share":
            return strategy

        # Use host face if available, otherwise best_face
        target_face = host_face if host_face else best_face
        if target_face and self.smooth_crop:
            crop = self.smooth_crop.get_crop(target_face["x"], target_face["y"], self._frame_idx)
            strategy["use_solo_frame"] = True
            strategy["active_crop"] = crop
        else:
            strategy["should_drop"] = True

        if chats:
            c = chats[-1]
            strategy["chat_exclusion"] = {
                "exclude_chat": True,
                "chat_side": c["chat_side"],
                "chat_exclude_width": c["chat_width"],
            }

        return strategy


# ─── Helpers ──────────────────────────────────────────────────────────────

def _get_video_info(path: str) -> Dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(result.stdout)
        s = data.get("streams", [{}])[0]
        return {"width": int(s.get("width", 1920)), "height": int(s.get("height", 1080))}
    except Exception:
        return {"width": 1920, "height": 1080}


def _extract_frame(video_path: str, timestamp: float, w: int, h: int) -> Optional[np.ndarray]:
    cmd = [
        "ffmpeg", "-y", "-ss", f"{timestamp:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-vf", f"scale={w}:{h}",
        "pipe:1",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=10)
    except Exception:
        return None
    if not res.stdout or len(res.stdout) != w * h * 3:
        return None
    return np.frombuffer(res.stdout, dtype=np.uint8).reshape((h, w, 3))


def _detect_dead_air(video_path: str, start: float, end: float) -> Dict:
    duration = max(0.01, end - start)
    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
        "-i", video_path,
        "-af", "silencedetect=noise=-35dB:d=1",
        "-f", "null", "-",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        silence_count = (res.stderr or "").count("silence_start")
        return {"has_dead_air": silence_count > 0}
    except Exception:
        return {"has_dead_air": False}


# ─── Host Identification via Reference Photos ────────────────────────────────

class HostDetector:
    """
    Identifies the host among multiple detected faces using reference photos.
    Uses simple template matching (histogram comparison) for host identification.
    Falls back to largest face + facecam region when no references provided.
    """

    def __init__(self, reference_photos: List[np.ndarray]):
        self.reference_embeddings: List[np.ndarray] = []
        self.has_host_reference = len(reference_photos) > 0

        for ref_img in reference_photos:
            if ref_img is not None and ref_img.size > 0:
                try:
                    gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                    if gray.size > 0:
                        resized = cv2.resize(gray, (64, 64))
                        hist = cv2.calcHist([resized], [0], None, [32], [0, 256])
                        hist = cv2.normalize(hist, hist).flatten()
                        self.reference_embeddings.append(hist)
                except Exception as e:
                    log.warning("Failed to process reference photo: %s", e)

        if self.has_host_reference:
            log.info("HostDetector initialized with %d reference photos", len(self.reference_embeddings))
        else:
            log.info("HostDetector initialized in fallback mode (largest face + facecam region)")

    def _compute_face_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Compute histogram embedding for a face region."""
        try:
            if face_img is None or face_img.size == 0:
                return None
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            if gray.size == 0:
                return None
            resized = cv2.resize(gray, (64, 64))
            hist = cv2.calcHist([resized], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        except Exception:
            return None

    def _histogram_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compute similarity between two histograms (0-1)."""
        if hist1 is None or hist2 is None:
            return 0.0
        if len(hist1) != len(hist2):
            return 0.0
        return float(np.clip(np.dot(hist1, hist2), 0, 1))

    def identify_host(self, faces: List[Dict]) -> int:
        """
        Identify which face is the host.
        Returns: index of host face in faces list.

        Strategy:
        1. If reference photos: match by histogram similarity
        2. Else: use facecam region as hint
        3. Fallback: largest face
        """
        if not faces:
            return -1

        if not self.has_host_reference:
            return self._identify_by_facecam_region(faces)

        best_idx = 0
        best_score = -1.0

        for idx, face in enumerate(faces):
            face_img = face.get("face_img")
            if face_img is None:
                continue

            embedding = self._compute_face_embedding(face_img)
            if embedding is None:
                continue

            max_sim = 0.0
            for ref_emb in self.reference_embeddings:
                sim = self._histogram_similarity(embedding, ref_emb)
                max_sim = max(max_sim, sim)

            if max_sim > best_score:
                best_score = max_sim
                best_idx = idx

        return best_idx if best_score > 0.1 else self._identify_by_facecam_region(faces)

    def _identify_by_facecam_region(self, faces: List[Dict]) -> int:
        """Fallback: use facecam region config to identify host."""
        layout_cfg = cfg.get("layout", {})
        if not layout_cfg.get("has_facecam", False):
            return self._identify_largest_face(faces)

        fc = layout_cfg.get("facecam", {})
        fc_x = fc.get("x", 0)
        fc_y = fc.get("y", 540)
        fc_w = fc.get("width", 320)
        fc_h = fc.get("height", 180)

        best_idx = 0
        best_dist = float('inf')

        for idx, face in enumerate(faces):
            # x,y in face dicts are already CENTER coordinates (set in analyze_clip)
            face_cx = face.get("x", 0)
            face_cy = face.get("y", 0)

            fc_center_x = fc_x + fc_w / 2
            fc_center_y = fc_y + fc_h / 2

            dist = ((face_cx - fc_center_x) ** 2 + (face_cy - fc_center_y) ** 2) ** 0.5

            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        return best_idx

    def _identify_largest_face(self, faces: List[Dict]) -> int:
        """Final fallback: largest face area."""
        if not faces:
            return 0
        return max(range(len(faces)), key=lambda i: faces[i].get("w", 0) * faces[i].get("h", 0))


def get_fallback_host_position(frame_w: int, frame_h: int) -> Dict:
    """Get fallback host position when no faces detected - uses facecam region."""
    layout_cfg = cfg.get("layout", {})
    if layout_cfg.get("has_facecam", False):
        fc = layout_cfg.get("facecam", {})
        return {
            "x": fc.get("x", 0) + fc.get("width", 320) // 2,
            "y": fc.get("y", 540) + fc.get("height", 180) // 2,
        }
    return {
        "x": frame_w // 4,
        "y": frame_h // 2,
    }


def analyze_clip_with_host_priority(analysis: Dict) -> Dict:
    """Analyze clip ensuring host is prioritized for framing."""
    if "faces" not in analysis or not analysis["faces"]:
        return analysis

    layout_type = analysis.get("layout", {}).get("layout_type", "solo")

    if layout_type == "split_both_active":
        faces = analysis["faces"]
        host_idx = -1

        for idx, face in enumerate(faces):
            if face.get("is_host", False):
                host_idx = idx
                break

        if host_idx >= 0:
            analysis["primary_face"] = faces[host_idx]
            analysis["use_solo_frame"] = True
            analysis["use_vertical_stack"] = False
            log.info("Host prioritized in split layout - using solo crop")
        else:
            analysis["use_vertical_stack"] = True

    return analysis
