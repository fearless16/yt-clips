"""
detect_track.py — Module 2: Face Detection + Temporal Tracking.

Core philosophy:
  - Detect sparsely (every N frames), track densely (every frame)
  - Match detected faces to target identity via embeddings
  - Maintain persistent face tracks across occlusions and missed detections
  - Smooth bounding boxes to prevent jitter

This module does NOT enhance or modify frames.
It only answers: "Where is the target face in this frame?"
"""

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

# ─── Cascade fallback (always available) ────────────────────────────────────

_CASCADE = None


def _get_cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _CASCADE


# �─── Face Detection ─────────────────────────────────────────────────────────

def detect_faces(frame: np.ndarray, min_size: int = 60) -> List[Tuple[int, int, int, int, float]]:
    """Detect faces in a frame using Haar Cascade.

    Returns list of (x, y, w, h, confidence).
    """
    cascade = _get_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(min_size, min_size),
    )

    results = []
    for (x, y, w, h) in faces:
        # Haar doesn't give confidence — estimate from face quality
        roi = gray[y:y+h, x:x+w]
        confidence = min(1.0, float(np.std(roi)) / 50.0)  # Higher std = sharper face
        results.append((int(x), int(y), int(w), int(h), confidence))

    return results


def detect_faces_dnn(frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int, float]]:
    """Detect faces using DNN (if available). Falls back to cascade."""
    try:
        # Try OpenCV DNN
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0),
        )
        net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel",
        )
        net.setInput(blob)
        detections = net.forward()

        results = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf >= confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                results.append((
                    max(0, x1), max(0, y1),
                    max(0, x2 - x1), max(0, y2 - y1),
                    float(conf),
                ))
        return results
    except Exception:
        return detect_faces(frame)


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
        return self._get_target_track()

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
        """Get the best target identity track."""
        target_tracks = [
            t for t in self.tracks.values()
            if t.detection and t.detection.is_target and t.state in (
                FaceState.DETECTED, FaceState.TRACKED
            )
        ]

        if not target_tracks:
            # Fall back to any active track with lowest distance
            active = [
                t for t in self.tracks.values()
                if t.state in (FaceState.DETECTED, FaceState.TRACKED)
            ]
            if active:
                return min(active, key=lambda t: t.detection.distance if t.detection else 1.0)
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
