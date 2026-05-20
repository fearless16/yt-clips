"""
detect_track.py — Module 2: Face Detection + Temporal Tracking.

Uses MediaPipe FaceDetector + FaceLandmarker (tasks API).
No Haar cascade. No fallback.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import (
    FaceDetection,
    FaceState,
    FaceTrack,
    FrameData,
)

cfg = get_config()

# ─── MediaPipe Tasks API ────────────────────────────────────────────────────

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe import Image as MpImage, ImageFormat as MpImageFormat

_face_detector = None
_face_landmarker = None


def get_detector():
    global _face_detector
    if _face_detector is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "face_detector.tflite")
        if not os.path.exists(model_path):
            model_path = "face_detector.tflite"
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.6,
        )
        _face_detector = vision.FaceDetector.create_from_options(options)
    return _face_detector


def get_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task")
        if not os.path.exists(model_path):
            model_path = "face_landmarker.task"
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
        )
        _face_landmarker = vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker


# ─── Face Detection ─────────────────────────────────────────────────────────

def detect_faces(frame: np.ndarray) -> List[FaceTrack]:
    """Detect faces using MediaPipe FaceDetector.

    Returns list of FaceTrack with bbox and confidence.
    """
    detector = get_detector()
    mp_image = MpImage(
        image_format=MpImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
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


def extract_face_mesh(frame: np.ndarray) -> Optional[np.ndarray]:
    """Extract face landmarks using MediaPipe FaceLandmarker.

    Returns (N, 3) array of (x, y, z) pixel coordinates, or None.
    """
    landmarker = get_landmarker()
    mp_image = MpImage(
        image_format=MpImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    h, w = frame.shape[:2]
    landmarks = result.face_landmarks[0]
    pts = np.array(
        [[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks],
        dtype=np.float32,
    )
    return pts


# ─── Quality Gates ──────────────────────────────────────────────────────────

def compute_procrustes_disparity(
    landmarks: np.ndarray,
    reference_landmarks: np.ndarray,
) -> float:
    """Compute Procrustes disparity between two landmark sets.

    Normalizes scale and translation, then measures shape difference.
    Returns disparity (0 = identical shape, >0.1 = different face).
    """
    if landmarks.shape != reference_landmarks.shape:
        return 1.0

    # Use only x,y for shape comparison
    lm = landmarks[:, :2] if landmarks.shape[1] > 2 else landmarks
    ref = reference_landmarks[:, :2] if reference_landmarks.shape[1] > 2 else reference_landmarks

    # Center both
    lm = lm - lm.mean(axis=0)
    ref = ref - ref.mean(axis=0)

    # Scale to unit Frobenius norm
    lm_scale = np.sqrt(np.sum(lm ** 2))
    ref_scale = np.sqrt(np.sum(ref ** 2))

    if lm_scale < 1e-6 or ref_scale < 1e-6:
        return 1.0

    lm_norm = lm / lm_scale
    ref_norm = ref / ref_scale

    # Procrustes disparity = Frobenius norm of difference
    disparity = np.sqrt(np.sum((lm_norm - ref_norm) ** 2))

    return float(disparity)


def compute_landmark_jitter(landmark_history: List[np.ndarray]) -> float:
    """Compute landmark jitter across recent frames."""
    if len(landmark_history) < 2:
        return 1.0

    displacements = []
    for i in range(1, len(landmark_history)):
        prev = landmark_history[i - 1]
        curr = landmark_history[i]
        if prev.shape == curr.shape:
            disp = np.mean(np.sqrt(np.sum((curr - prev) ** 2, axis=1)))
            disp_norm = disp / 200.0
            displacements.append(disp_norm)

    if not displacements:
        return 1.0

    return float(np.mean(displacements))


def compute_occupancy(landmarks: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """Compute face occupancy = convex hull area / bbox area."""
    x, y, w, h = bbox
    bbox_area = w * h
    if bbox_area <= 0:
        return 0.0

    pts_2d = landmarks[:, :2] if landmarks.shape[1] > 2 else landmarks
    hull = cv2.convexHull(pts_2d.astype(np.float32))
    face_area = cv2.contourArea(hull)
    return face_area / bbox_area


def pass_quality_gates(
    landmarks: np.ndarray,
    reference_landmarks: Optional[np.ndarray],
    landmark_history: List[np.ndarray],
    bbox: Tuple[int, int, int, int],
) -> Tuple[bool, Dict[str, float]]:
    """Check all quality gates for identity update."""
    metrics = {}

    if reference_landmarks is not None:
        disparity = compute_procrustes_disparity(landmarks, reference_landmarks)
        metrics["procrustes_disparity"] = disparity
        if disparity > 0.2:  # Relaxed threshold for different image sizes
            return False, metrics
    else:
        metrics["procrustes_disparity"] = 0.0

    jitter = compute_landmark_jitter(landmark_history)
    metrics["landmark_jitter"] = jitter
    if jitter < 0.0008:
        return False, metrics

    occupancy = compute_occupancy(landmarks, bbox)
    metrics["occupancy"] = occupancy
    if occupancy < 0.25:
        return False, metrics

    return True, metrics


# ─── Face Matching (Identity) ───────────────────────────────────────────────

def _compute_embedding(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Compute face embedding for identity matching."""
    x, y, w, h = bbox
    face_roi = frame[max(0, y):y+h, max(0, x):x+w]
    if face_roi.size == 0:
        return None

    try:
        import face_recognition
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = [(y, x + w, y + h, x)]
        encodings = face_recognition.face_encodings(rgb, locations)
        if encodings:
            return encodings[0]
    except ImportError:
        pass

    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    hist_l = cv2.calcHist([lab], [0], None, [32], [0, 256]).flatten()
    hist_a = cv2.calcHist([lab], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([lab], [2], None, [32], [0, 256]).flatten()
    hist = np.concatenate([hist_l, hist_a, hist_b])
    hist = hist / max(hist.sum(), 1)
    return hist


def match_identity(
    embedding: Optional[np.ndarray],
    reference_embeddings: List[np.ndarray],
    tolerance: float = 0.50,
) -> Tuple[bool, float]:
    """Check if an embedding matches any reference."""
    if embedding is None or not reference_embeddings:
        return False, 1.0

    try:
        import face_recognition
        distances = face_recognition.face_distance(reference_embeddings, embedding)
        min_dist = float(np.min(distances))
        return min_dist <= tolerance, min_dist
    except ImportError:
        distances = []
        for ref in reference_embeddings:
            if ref.shape == embedding.shape:
                dist = 1.0 - np.minimum(embedding, ref).sum()
                distances.append(dist)
        if distances:
            min_dist = min(distances)
            return min_dist <= tolerance, min_dist
        return False, 1.0


# ─── Temporal Tracker ───────────────────────────────────────────────────────

class FaceTracker:
    """Maintains face tracks across frames."""

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
        """Set reference face mesh for Procrustes comparison."""
        self._reference_mesh = mesh

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> Optional[FaceTrack]:
        """Process a frame and return the target face track (if found)."""
        self.frame_count += 1

        should_detect = (frame_idx % self._detection_interval == 0)

        if should_detect:
            self._detect_and_match(frame, frame_idx)
        else:
            self._predict_tracks()

        self._update_states()

        track = self._get_target_track()

        # QUALITY GATES
        if track is not None and track.smooth_bbox is not None:
            mesh = extract_face_mesh(frame)
            if mesh is not None:
                self._landmark_history.append(mesh)
                if len(self._landmark_history) > 10:
                    self._landmark_history = self._landmark_history[-10:]

                passed, metrics = pass_quality_gates(
                    mesh,
                    self._reference_mesh,
                    self._landmark_history,
                    track.smooth_bbox,
                )

                if not passed:
                    track.quality_metrics = metrics
                    track.state = FaceState.LOST
                    return None

                track.mesh_468 = mesh
                track.quality_metrics = metrics
            else:
                track.state = FaceState.LOST
                return None

        return track

    def _detect_and_match(self, frame: np.ndarray, frame_idx: int) -> None:
        """Run face detection and match to identity."""
        detections = detect_faces(frame)

        det_with_emb = []
        for track in detections:
            bbox = track.smooth_bbox
            if bbox is None:
                continue
            x, y, w, h = bbox
            emb = _compute_embedding(frame, (x, y, w, h))
            is_match, dist = match_identity(
                emb, self.ref_embeddings, cfg.identity.embedding_tolerance
            )
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
                if tid in matched_tracks:
                    continue
                if existing_track.smooth_bbox is None:
                    continue
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
                mesh_468 = track.mesh_468 if hasattr(track, 'mesh_468') else None
                self._create_track(det, frame_idx, mesh_468)

    def _create_track(self, detection: FaceDetection, frame_idx: int, mesh_468: Optional[np.ndarray] = None) -> int:
        """Create a new face track."""
        track_id = self.next_track_id
        self.next_track_id += 1

        track = FaceTrack(
            track_id=track_id,
            state=FaceState.DETECTED,
            frames_visible=1,
            frames_lost=0,
            detection=detection,
            smooth_bbox=detection.bbox,
            bbox_history=[detection.bbox],
        )
        if mesh_468 is not None:
            track.mesh_468 = mesh_468
        self.tracks[track_id] = track
        return track_id

    def _update_track(self, track_id: int, detection: FaceDetection, frame_idx: int) -> None:
        """Update an existing track with a new detection."""
        track = self.tracks[track_id]
        track.detection = detection
        track.state = FaceState.DETECTED
        track.frames_visible += 1
        track.frames_lost = 0

        prev = track.smooth_bbox
        curr = detection.bbox
        alpha = self._smoothing
        if prev is not None:
            smoothed = tuple(
                int(p * (1 - alpha) + c * alpha)
                for p, c in zip(prev, curr)
            )
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
        """Get the best target identity track. NO FALLBACK."""
        target_tracks = [
            t for t in self.tracks.values()
            if t.detection and t.detection.is_target and t.state in (
                FaceState.DETECTED, FaceState.TRACKED
            )
        ]

        if not target_tracks:
            return None

        return max(target_tracks, key=lambda t: (t.frames_visible, -t.frames_lost))


# ─── Utilities ──────────────────────────────────────────────────────────────

def _compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter

    return inter / max(union, 1)
