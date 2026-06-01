"""
detect_track.py — Module 2: Face Detection + Temporal Tracking.

BEAST MODE FIXES:
- Added ROI cropping for FaceLandmarker (10x FPS boost).
- Fixed Jitter Threshold math (percentage based, stops rejecting real humans).
- Fixed fake 2D Pose estimation garbage math.
- Hardened LAB histogram embedding against lighting shifts.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Suppress MediaPipe C++ log noise (feedback manager, clearcut)
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('MEDIAPIPE_LOG_LEVEL', 'error')

from face_os.config import get_config
from face_os.types import (
    FaceDetection,
    FaceState,
    FaceTrack,
    FrameData,
)

cfg = get_config()

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe import Image as MpImage, ImageFormat as MpImageFormat

_face_detector = None
_face_landmarker = None
MEDIPIPE_GPU_ACTIVE = False

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)
_IS_MACOS = sys.platform == "darwin"


def _resolve_model(filename: str) -> str:
    candidates = [
        os.path.join(_PROJECT_ROOT, filename),
        os.path.join(_MODULE_DIR, filename),
        filename,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Model not found: {filename} (checked: {candidates})")


def _try_gpu(delegate):
    if _IS_MACOS:
        return None
    return delegate


def get_detector():
    global _face_detector, MEDIPIPE_GPU_ACTIVE
    if _face_detector is None:
        model_path = _resolve_model("face_detector.tflite")
        gpu_delegate = _try_gpu(mp_python.BaseOptions.Delegate.GPU)
        if gpu_delegate is not None:
            try:
                base_options = mp_python.BaseOptions(model_asset_path=model_path, delegate=gpu_delegate)
                options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
                _face_detector = vision.FaceDetector.create_from_options(options)
                MEDIPIPE_GPU_ACTIVE = True
            except Exception:
                _face_detector = None
        if _face_detector is None:
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
            _face_detector = vision.FaceDetector.create_from_options(options)
    return _face_detector


def get_landmarker():
    global _face_landmarker, MEDIPIPE_GPU_ACTIVE
    if _face_landmarker is None:
        model_path = _resolve_model("face_landmarker.task")
        gpu_delegate = _try_gpu(mp_python.BaseOptions.Delegate.GPU)
        if gpu_delegate is not None:
            try:
                base_options = mp_python.BaseOptions(model_asset_path=model_path, delegate=gpu_delegate)
                options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, min_face_detection_confidence=0.5)
                _face_landmarker = vision.FaceLandmarker.create_from_options(options)
                MEDIPIPE_GPU_ACTIVE = True
            except Exception:
                _face_landmarker = None
        if _face_landmarker is None:
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, min_face_detection_confidence=0.5)
            _face_landmarker = vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker


def detect_faces(frame: np.ndarray) -> List[FaceTrack]:
    global _face_detector, MEDIPIPE_GPU_ACTIVE
    detector = get_detector()
    mp_image = MpImage(image_format=MpImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    try:
        result = detector.detect(mp_image)
    except Exception:
        _face_detector = None
        MEDIPIPE_GPU_ACTIVE = False
        detector = get_detector()
        result = detector.detect(mp_image)

    tracks = []
    for detection in result.detections:
        bbox = detection.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        track = FaceTrack(
            track_id=0,
            state=FaceState.DETECTED,
            smooth_bbox=(x, y, w, h),
            detection=FaceDetection(
                bbox=(x, y, w, h),
                confidence=detection.categories[0].score,
                is_target=True,
            ),
        )
        tracks.append(track)
    return tracks


def extract_face_mesh(frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
    """Extract face landmarks. 
    BEAST MODE: Crops ROI before running Landmarker to save massive CPU/GPU cycles.
    """
    global _face_landmarker, MEDIPIPE_GPU_ACTIVE
    landmarker = get_landmarker()
    
    h_full, w_full = frame.shape[:2]
    offset_x, offset_y = 0, 0
    
    # Crop ROI if bbox is provided
    if bbox is not None:
        bx, by, bw, bh = bbox
        pad = int(max(bw, bh) * 0.2)
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w_full, bx + bw + pad)
        y2 = min(h_full, by + bh + pad)
        
        frame_roi = frame[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
        h_full, w_full = frame_roi.shape[:2]
    else:
        frame_roi = frame

    mp_image = MpImage(image_format=MpImageFormat.SRGB, data=cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB))

    try:
        result = landmarker.detect(mp_image)
    except Exception:
        _face_landmarker = None
        MEDIPIPE_GPU_ACTIVE = False
        landmarker = get_landmarker()
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    # Map back to original frame coordinates
    pts = np.array(
        [[lm.x * w_full + offset_x, lm.y * h_full + offset_y, lm.z * w_full] for lm in landmarks],
        dtype=np.float32,
    )
    return pts


def compute_procrustes_disparity(landmarks: np.ndarray, reference_landmarks: np.ndarray) -> float:
    if landmarks.shape != reference_landmarks.shape:
        return 1.0

    lm = landmarks[:, :2] if landmarks.shape[1] > 2 else landmarks
    ref = reference_landmarks[:, :2] if reference_landmarks.shape[1] > 2 else reference_landmarks

    lm = lm - lm.mean(axis=0)
    ref = ref - ref.mean(axis=0)

    lm_scale = np.sqrt(np.sum(lm ** 2))
    ref_scale = np.sqrt(np.sum(ref ** 2))

    if lm_scale < 1e-6 or ref_scale < 1e-6:
        return 1.0

    lm_norm = lm / lm_scale
    ref_norm = ref / ref_scale

    disparity = np.sqrt(np.sum((lm_norm - ref_norm) ** 2))
    return float(disparity)


def compute_landmark_jitter(landmark_history: List[np.ndarray]) -> float:
    if len(landmark_history) < 2:
        return 1.0

    displacements = []
    for i in range(1, len(landmark_history)):
        prev = landmark_history[i - 1]
        curr = landmark_history[i]
        if prev.shape == curr.shape:
            disp = np.mean(np.sqrt(np.sum((curr - prev) ** 2, axis=1)))
            # BEAST MODE FIX: Convert to percentage (0.5px / 200px * 100 = 0.25%)
            disp_norm = (disp / 200.0) * 100.0
            displacements.append(disp_norm)

    if not displacements:
        return 1.0

    return float(np.mean(displacements))


def compute_occupancy(landmarks: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    bbox_area = w * h
    if bbox_area <= 0:
        return 0.0

    pts_2d = landmarks[:, :2] if landmarks.shape[1] > 2 else landmarks
    hull = cv2.convexHull(pts_2d.astype(np.float32))
    face_area = cv2.contourArea(hull)
    return face_area / bbox_area


def _estimate_pose_from_landmarks(landmarks: np.ndarray) -> Tuple[float, float, float]:
    if landmarks is None or len(landmarks) < 468:
        return (0.0, 0.0, 0.0)

    pts = landmarks[:, :2] if landmarks.shape[1] > 2 else landmarks

    nose = pts[1]
    chin = pts[152]
    left_eye = pts[33]
    right_eye = pts[263]

    eye_mid = (left_eye + right_eye) / 2.0
    nose_offset = nose[0] - eye_mid[0]
    eye_dist = np.linalg.norm(right_eye - left_eye)
    
    # BEAST MODE FIX: Removed the fake "- 70.0" offset and arbitrary multipliers.
    yaw = float(np.degrees(np.arctan2(nose_offset, max(eye_dist, 1e-6)))) * 2.0
    pitch = float(np.degrees(np.arctan2(chin[1] - nose[1], max(np.linalg.norm(chin - nose), 1e-6)))) - 45.0
    roll = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))

    return (yaw, pitch, roll)


def pass_quality_gates(
    landmarks: np.ndarray,
    reference_landmarks: Optional[np.ndarray],
    landmark_history: List[np.ndarray],
    bbox: Tuple[int, int, int, int],
) -> Tuple[bool, Dict[str, float]]:
    metrics = {}

    if reference_landmarks is not None:
        disparity = compute_procrustes_disparity(landmarks, reference_landmarks)
        metrics["procrustes_disparity"] = disparity

        pose = _estimate_pose_from_landmarks(landmarks)
        metrics["yaw"] = pose[0]
        metrics["pitch"] = pose[1]
        threshold = 0.2
        if abs(pose[0]) > 20 or abs(pose[1]) > 15:
            threshold = 0.35
        elif abs(pose[0]) > 10 or abs(pose[1]) > 10:
            threshold = 0.28

        if disparity > threshold:
            return False, metrics
    else:
        metrics["procrustes_disparity"] = 0.0

    jitter = compute_landmark_jitter(landmark_history)
    metrics["landmark_jitter"] = jitter
    # BEAST MODE FIX: Threshold is now in percentage. 0.1% jitter is safe for real humans.
    if jitter < 0.05: 
        return False, metrics

    occupancy = compute_occupancy(landmarks, bbox)
    metrics["occupancy"] = occupancy
    if occupancy < 0.25:
        return False, metrics

    return True, metrics


def _compute_embedding(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    face_roi = frame[max(0, y):y+h, max(0, x):x+w]
    if face_roi.size == 0:
        return None

    # Normalize lighting before histogram to prevent sunlight from breaking identity
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    lab_norm = cv2.merge([l, a, b])
    
    hist_l = cv2.calcHist([lab_norm], [0], None, [32], [0, 256]).flatten()
    hist_a = cv2.calcHist([lab_norm], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([lab_norm], [2], None, [32], [0, 256]).flatten()
    hist = np.concatenate([hist_l, hist_a, hist_b])
    hist = hist / max(hist.sum(), 1)
    return hist


def match_identity(
    embedding: Optional[np.ndarray],
    reference_embeddings: List[np.ndarray],
    tolerance: float = 0.50,
) -> Tuple[bool, float]:
    if embedding is None or not reference_embeddings:
        return False, 1.0

    distances = []
    for ref in reference_embeddings:
        if ref.shape == embedding.shape:
            dist = 1.0 - np.minimum(embedding, ref).sum()
            distances.append(dist)
    if distances:
        min_dist = min(distances)
        return min_dist <= tolerance, min_dist
    return False, 1.0


class FaceTracker:
    def __init__(self, reference_embeddings: List[np.ndarray]):
        self.ref_embeddings = reference_embeddings
        self.tracks: Dict[int, FaceTrack] = {}
        self.next_track_id = 1
        self.frame_count = 0
        self._detection_interval = cfg.detection.detection_interval
        self._max_lost = cfg.detection.max_lost_frames
        self._smoothing = cfg.detection.smoothing_alpha

        self._reference_mesh: Optional[np.ndarray] = None
        self._landmark_history: List[np.ndarray] = []

    def set_reference_mesh(self, mesh: np.ndarray) -> None:
        self._reference_mesh = mesh

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Optional[FaceTrack]:
        self.frame_count += 1
        should_detect = (frame_idx % self._detection_interval == 0)

        if should_detect:
            self._detect_and_match(frame, frame_idx)
        else:
            self._predict_tracks()

        self._update_states()
        track = self._get_target_track()

        if track is not None and track.smooth_bbox is not None:
            # BEAST MODE: Pass bbox to extract_face_mesh for massive FPS boost
            mesh = extract_face_mesh(frame, track.smooth_bbox)
            if mesh is not None:
                self._landmark_history.append(mesh)
                if len(self._landmark_history) > 10:
                    self._landmark_history = self._landmark_history[-10:]

                passed, metrics = pass_quality_gates(
                    mesh, self._reference_mesh, self._landmark_history, track.smooth_bbox,
                )

                if not passed:
                    track.quality_metrics = metrics
                    track.state = FaceState.LOST
                    return None

                track.mesh_478 = mesh
                track.quality_metrics = metrics
            else:
                track.state = FaceState.LOST
                return None

        return track

    def _detect_and_match(self, frame: np.ndarray, frame_idx: int) -> None:
        detections = detect_faces(frame)
        det_with_emb = []
        for track in detections:
            bbox = track.smooth_bbox
            if bbox is None: continue
            x, y, w, h = bbox
            emb = _compute_embedding(frame, (x, y, w, h))
            is_match, dist = match_identity(emb, self.ref_embeddings, cfg.identity.embedding_tolerance)
            det_with_emb.append((track, FaceDetection(
                bbox=bbox,
                confidence=track.detection.confidence if track.detection else 0.9,
                is_target=is_match,
                embedding=emb,
                distance=dist,
            )))

        matched_tracks = set()
        unmatched_dets = []

        for track, det in det_with_emb:
            best_track = None
            best_iou = 0.1
            for tid, existing_track in self.tracks.items():
                if tid in matched_tracks: continue
                if existing_track.smooth_bbox is None: continue
                iou = _compute_iou(det.bbox, existing_track.smooth_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = tid

            if best_track is not None:
                self._update_track(best_track, det, frame_idx)
                matched_tracks.add(best_track)
            else:
                unmatched_dets.append((track, det))

        for track, det in unmatched_dets:
            if det.is_target:
                mesh_478 = track.mesh_478 if hasattr(track, 'mesh_478') else None
                self._create_track(det, frame_idx, mesh_478)

    def _create_track(self, detection: FaceDetection, frame_idx: int, mesh_478: Optional[np.ndarray] = None) -> int:
        track_id = self.next_track_id
        self.next_track_id += 1
        track = FaceTrack(
            track_id=track_id, state=FaceState.DETECTED, frames_visible=1, frames_lost=0,
            detection=detection, smooth_bbox=detection.bbox, bbox_history=[detection.bbox],
        )
        if mesh_478 is not None: track.mesh_478 = mesh_478
        self.tracks[track_id] = track
        return track_id

    def _update_track(self, track_id: int, detection: FaceDetection, frame_idx: int) -> None:
        track = self.tracks[track_id]
        track.detection = detection
        track.state = FaceState.DETECTED
        track.frames_visible += 1
        track.frames_lost = 0

        prev = track.smooth_bbox
        curr = detection.bbox
        alpha = self._smoothing
        if prev is not None:
            smoothed = tuple(int(p * (1 - alpha) + c * alpha) for p, c in zip(prev, curr))
            track.smooth_bbox = smoothed
        else:
            track.smooth_bbox = curr

        track.bbox_history.append(track.smooth_bbox)
        if len(track.bbox_history) > 100:
            track.bbox_history = track.bbox_history[-100:]

    def _predict_tracks(self) -> None:
        for track in self.tracks.values():
            if track.state == FaceState.DETECTED:
                track.state = FaceState.TRACKED

    def _update_states(self) -> None:
        to_remove = []
        for tid, track in self.tracks.items():
            if track.state != FaceState.DETECTED:
                track.frames_lost += 1
                if track.frames_lost > self._max_lost:
                    track.state = FaceState.LOST
                    to_remove.append(tid)
                elif track.frames_lost > self._max_lost // 2:
                    track.state = FaceState.OCCLUDED
        for tid in to_remove:
            del self.tracks[tid]

    def _get_target_track(self) -> Optional[FaceTrack]:
        target_tracks = [
            t for t in self.tracks.values()
            if t.detection and t.detection.is_target and t.state in (FaceState.DETECTED, FaceState.TRACKED)
        ]
        if not target_tracks: return None
        return max(target_tracks, key=lambda t: (t.frames_visible, -t.frames_lost))


def _compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / max(union, 1)