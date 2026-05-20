"""
detect_track.py — Module 2: Face Detection + Temporal Tracking.

Core philosophy:
  - Detect sparsely (every N frames), track densely (every frame)
  - Match detected faces to target identity via embeddings
  - Maintain persistent face tracks across occlusions and missed detections
  - Smooth bounding boxes to prevent jitter

This module does NOT enhance or modify frames.
It only answers: "Where is the target face in this frame?"

QUALITY GATES (all must pass for identity update):
  1. Procrustes disparity < 0.1 (face shape matches reference)
  2. Landmark jitter > 0.0008 (real face, not poster)
  3. Occupancy > 0.25 (face fills enough of bbox)
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

# ─── MediaPipe Face Detection ───────────────────────────────────────────────

_MP_FACE_DETECTOR = None
_MP_FACE_MESH = None


def _get_mp_face_detector():
    global _MP_FACE_DETECTOR
    if _MP_FACE_DETECTOR is None:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import os

        model_path = os.path.join(os.path.dirname(__file__), "..", "face_detector.tflite")
        if not os.path.exists(model_path):
            model_path = "face_detector.tflite"

        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            min_detection_confidence=0.6,
            running_mode=vision.RunningMode.IMAGE,
        )
        _MP_FACE_DETECTOR = vision.FaceDetector.create_from_options(options)
    return _MP_FACE_DETECTOR


def _get_mp_face_mesh():
    """Face mesh not available — using 68-point landmarks instead."""
    return None


# ─── Face Detection ─────────────────────────────────────────────────────────

def detect_faces(frame: np.ndarray, min_size: int = 60) -> List[Tuple[int, int, int, int, float]]:
    """Detect faces using MediaPipe FaceDetection.

    Returns list of (x, y, w, h, confidence).
    """
    import mediapipe as mp

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detector = _get_mp_face_detector()
    result = detector.detect(mp_image)

    detections = []
    if result.detections:
        for det in result.detections:
            score = det.categories[0].score
            if score < 0.6:
                continue
            bbox = det.bounding_box
            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)
            x2 = min(w, bbox.origin_x + bbox.width)
            y2 = min(h, bbox.origin_y + bbox.height)
            bw = x2 - x1
            bh = y2 - y1
            if bw < min_size or bh < min_size:
                continue
            detections.append((x1, y1, bw, bh, float(score)))

    return detections


def extract_face_mesh(frame: np.ndarray) -> Optional[np.ndarray]:
    """Extract face landmarks using dlib 68-point landmarks.

    Returns (68, 2) array of (x, y) pixel coordinates, or None.
    """
    try:
        import dlib
        detector = dlib.get_frontal_face_detector()
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            return None
        predictor = dlib.shape_predictor(predictor_path)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        if not faces:
            return None

        # Use largest face
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = predictor(gray, face)

        pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)
        return pts
    except Exception:
        return None


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

    # Center and normalize both
    lm = landmarks - landmarks.mean(axis=0)
    ref = reference_landmarks - reference_landmarks.mean(axis=0)

    # Scale to unit Frobenius norm
    lm_scale = np.sqrt(np.sum(lm ** 2))
    ref_scale = np.sqrt(np.sum(ref ** 2))

    if lm_scale < 1e-6 or ref_scale < 1e-6:
        return 1.0

    lm_norm = lm / lm_scale
    ref_norm = ref / ref_scale

    # Procrustes disparity = 1 - |trace(L^T R)|
    # Simplified: just compute Frobenius norm of difference
    disparity = np.sqrt(np.sum((lm_norm - ref_norm) ** 2))

    return float(disparity)


def compute_landmark_jitter(
    landmark_history: List[np.ndarray],
) -> float:
    """Compute landmark jitter across recent frames.

    Low jitter (<0.0008) indicates poster/static image.
    Returns mean displacement per landmark between consecutive frames.
    """
    if len(landmark_history) < 2:
        return 1.0  # Assume real if not enough history

    displacements = []
    for i in range(1, len(landmark_history)):
        prev = landmark_history[i - 1]
        curr = landmark_history[i]
        if prev.shape == curr.shape:
            # Mean displacement per landmark, normalized by image size
            disp = np.mean(np.sqrt(np.sum((curr - prev) ** 2, axis=1)))
            # Normalize by bbox size (assume ~200px face)
            disp_norm = disp / 200.0
            displacements.append(disp_norm)

    if not displacements:
        return 1.0

    return float(np.mean(displacements))


def compute_occupancy(
    landmarks: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> float:
    """Compute face occupancy = convex hull area / bbox area.

    Returns occupancy ratio (0-1). <0.25 means face is too small in bbox.
    """
    x, y, w, h = bbox
    bbox_area = w * h
    if bbox_area <= 0:
        return 0.0

    hull = cv2.convexHull(landmarks.astype(np.float32))
    face_area = cv2.contourArea(hull)
    return face_area / bbox_area


def pass_quality_gates(
    landmarks: np.ndarray,
    reference_landmarks: Optional[np.ndarray],
    landmark_history: List[np.ndarray],
    bbox: Tuple[int, int, int, int],
) -> Tuple[bool, Dict[str, float]]:
    """Check all quality gates for identity update.

    Returns (passed, metrics_dict).
    """
    metrics = {}

    # Gate 1: Procrustes disparity (face shape matches reference)
    if reference_landmarks is not None:
        disparity = compute_procrustes_disparity(landmarks, reference_landmarks)
        metrics["procrustes_disparity"] = disparity
        if disparity > 0.1:
            return False, metrics
    else:
        metrics["procrustes_disparity"] = 0.0

    # Gate 2: Landmark jitter (real face, not poster)
    jitter = compute_landmark_jitter(landmark_history)
    metrics["landmark_jitter"] = jitter
    if jitter < 0.0008:
        return False, metrics

    # Gate 3: Occupancy (face fills enough of bbox)
    occupancy = compute_occupancy(landmarks, bbox)
    metrics["occupancy"] = occupancy
    if occupancy < 0.25:
        return False, metrics

    return True, metrics


# ─── Face Matching (Identity) ───────────────────────────────────────────────

def _compute_embedding(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Compute face embedding for identity matching.

    Uses face_recognition library (dlib) if available, else falls back
    to a simple histogram-based descriptor.
    """
    x, y, w, h = bbox
    face_roi = frame[max(0, y):y+h, max(0, x):x+w]
    if face_roi.size == 0:
        return None

    try:
        import face_recognition
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = [(y, x + w, y + h, x)]  # top, right, bottom, left
        encodings = face_recognition.face_encodings(rgb, locations)
        if encodings:
            return encodings[0]
    except ImportError:
        pass

    # Fallback: LAB histogram as rough embedding
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
    """Check if an embedding matches any reference.

    Returns (is_match, min_distance).
    """
    if embedding is None or not reference_embeddings:
        return False, 1.0

    try:
        import face_recognition
        distances = face_recognition.face_distance(reference_embeddings, embedding)
        min_dist = float(np.min(distances))
        return min_dist <= tolerance, min_dist
    except ImportError:
        # Fallback: histogram intersection
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
    """Maintains face tracks across frames.

    Detects every N frames, predicts position via EMA in between.
    Matches detections to target identity via embeddings.
    """

    def __init__(self, reference_embeddings: List[np.ndarray]):
        self.ref_embeddings = reference_embeddings
        self.tracks: Dict[int, FaceTrack] = {}
        self.next_track_id = 1
        self.frame_count = 0
        self._detection_interval = cfg.detection.detection_interval
        self._max_lost = cfg.detection.max_lost_frames
        self._smoothing = cfg.detection.smoothing_alpha

        # Quality gate state
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
        """Process a frame and return the target face track (if found).

        This is the main entry point — called for every frame.
        """
        self.frame_count += 1

        # Should we detect this frame?
        should_detect = (frame_idx % self._detection_interval == 0)

        if should_detect:
            self._detect_and_match(frame, frame_idx)
        else:
            # Predict positions for existing tracks
            self._predict_tracks()

        # Update track states
        self._update_states()

        # Return the target identity track (best match)
        track = self._get_target_track()

        # QUALITY GATES: check all gates before returning track
        if track is not None and track.smooth_bbox is not None:
            # Extract face mesh for quality checks
            mesh = extract_face_mesh(frame)
            if mesh is not None:
                # Update landmark history
                self._landmark_history.append(mesh)
                if len(self._landmark_history) > 10:
                    self._landmark_history = self._landmark_history[-10:]

                # Run quality gates
                passed, metrics = pass_quality_gates(
                    mesh,
                    self._reference_mesh,
                    self._landmark_history,
                    track.smooth_bbox,
                )

                if not passed:
                    # Store metrics on track for debugging
                    track.quality_metrics = metrics
                    track.state = FaceState.LOST
                    return None

                # Store mesh on track for downstream use
                track.face_mesh = mesh
                track.quality_metrics = metrics
            else:
                # No mesh detected — cannot verify quality
                track.state = FaceState.LOST
                return None

        return track

    def _detect_and_match(self, frame: np.ndarray, frame_idx: int) -> None:
        """Run face detection and match to identity."""
        detections = detect_faces(frame, cfg.detection.min_face_size)

        # Compute embeddings for all detections
        det_with_emb = []
        for (x, y, w, h, conf) in detections:
            emb = _compute_embedding(frame, (x, y, w, h))
            is_match, dist = match_identity(
                emb, self.ref_embeddings, cfg.identity.embedding_tolerance
            )
            det_with_emb.append(FaceDetection(
                bbox=(x, y, w, h),
                confidence=conf,
                is_target=is_match,
                embedding=emb,
                distance=dist,
            ))

        # Match detections to existing tracks (nearest neighbor)
        matched_tracks = set()
        unmatched_dets = []

        for det in det_with_emb:
            best_track = None
            best_iou = 0.1  # Min IoU to consider a match

            for tid, track in self.tracks.items():
                if tid in matched_tracks:
                    continue
                if track.smooth_bbox is None:
                    continue
                iou = _compute_iou(det.bbox, track.smooth_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = tid

            if best_track is not None:
                self._update_track(best_track, det, frame_idx)
                matched_tracks.add(best_track)
            else:
                unmatched_dets.append(det)

        # Create new tracks for unmatched target detections
        for det in unmatched_dets:
            if det.is_target:
                self._create_track(det, frame_idx)

    def _create_track(self, detection: FaceDetection, frame_idx: int) -> int:
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
        self.tracks[track_id] = track
        return track_id

    def _update_track(self, track_id: int, detection: FaceDetection, frame_idx: int) -> None:
        """Update an existing track with a new detection."""
        track = self.tracks[track_id]
        track.detection = detection
        track.state = FaceState.DETECTED
        track.frames_visible += 1
        track.frames_lost = 0

        # Smooth bbox
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
        # Keep history bounded
        if len(track.bbox_history) > 100:
            track.bbox_history = track.bbox_history[-100:]

    def _predict_tracks(self) -> None:
        """Predict track positions when not detecting (no-op for now — uses last known)."""
        for track in self.tracks.values():
            if track.state == FaceState.DETECTED:
                track.state = FaceState.TRACKED
            elif track.state == FaceState.TRACKED:
                # Keep last known position (could add Kalman prediction here)
                pass

    def _update_states(self) -> None:
        """Update track states based on visibility."""
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
        """Get the best target identity track.

        If no target is found, return None. NO FALLBACK to non-target tracks.
        """
        target_tracks = [
            t for t in self.tracks.values()
            if t.detection and t.detection.is_target and t.state in (
                FaceState.DETECTED, FaceState.TRACKED
            )
        ]

        if not target_tracks:
            return None

        # Prefer most recently detected, then most frames visible
        return max(target_tracks, key=lambda t: (t.frames_visible, -t.frames_lost))


# ─── Utilities ──────────────────────────────────────────────────────────────

def _compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """Compute Intersection over Union for two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to (x1, y1, x2, y2)
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter

    return inter / max(union, 1)
