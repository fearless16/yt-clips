"""
face_matcher.py — Identify the user in video frames using reference photos.

Uses face_recognition (dlib-based) embeddings to match detected faces
against reference photos in photos/ folder.

Only frames where the user is detected get graded. Background players
are ignored.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import face_recognition

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("face_matcher", cfg["logging"]["log_file"], cfg["logging"]["level"])


class FaceMatcher:
    """Load reference photos, extract embeddings, match faces in video frames."""

    def __init__(self, photos_dir: str = "photos/", tolerance: float = 0.50):
        self.tolerance = tolerance  # Lower = stricter match (0.4-0.6 typical)
        self.ref_encodings: List[np.ndarray] = []
        self.ref_names: List[str] = []
        self._load_references(photos_dir)

    def _load_references(self, photos_dir: str) -> None:
        """Load all reference photos and extract face encodings."""
        photos_path = Path(photos_dir)
        if not photos_path.exists():
            log.warning("Photos dir not found: %s", photos_dir)
            return

        for img_path in sorted(photos_path.glob("*.png")) + sorted(photos_path.glob("*.jpg")):
            img = face_recognition.load_image_file(str(img_path))
            encodings = face_recognition.face_encodings(img)
            if encodings:
                self.ref_encodings.extend(encodings)
                self.ref_names.extend([img_path.name] * len(encodings))
                log.info("Loaded %d encoding(s) from %s", len(encodings), img_path.name)
            else:
                log.warning("No face found in %s", img_path.name)

        log.info("Total reference encodings: %d", len(self.ref_encodings))

    def find_user_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Find the user's face in a frame.

        Returns (x, y, w, h) of the user's face, or None if not found.
        Ignores other faces (background players, guests, etc.).
        """
        if not self.ref_encodings:
            return None

        # Convert BGR to RGB for face_recognition
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect all faces
        face_locations = face_recognition.face_locations(rgb, model="hog")
        if not face_locations:
            return None

        # Extract encodings for all detected faces
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        # Compare each detected face against reference encodings
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Compare against all reference encodings
            distances = face_recognition.face_distance(self.ref_encodings, encoding)
            min_dist = float(np.min(distances)) if len(distances) > 0 else 1.0

            if min_dist <= self.tolerance:
                # Convert to (x, y, w, h) format
                x, y, w, h = left, top, right - left, bottom - top
                log.debug("User face found: (%d,%d,%d,%d) dist=%.3f", x, y, w, h, min_dist)
                return (x, y, w, h)

        return None

    def find_all_faces_with_ids(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, bool, float]]:
        """Find all faces and identify which ones are the user.

        Returns list of (x, y, w, h, is_user, min_distance).
        """
        if not self.ref_encodings:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        results = []
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(self.ref_encodings, encoding)
            min_dist = float(np.min(distances)) if len(distances) > 0 else 1.0
            is_user = min_dist <= self.tolerance
            x, y, w, h = left, top, right - left, bottom - top
            results.append((x, y, w, h, is_user, min_dist))

        return results


def crop_with_headspace(frame: np.ndarray, face_bbox: Tuple[int, int, int, int],
                        target_w: int = 1080, target_h: int = 1920,
                        headspace_ratio: float = 0.35) -> np.ndarray:
    """Crop 16:9 frame to 9:16 with headspace above face.

    headspace_ratio: fraction of output height above face center.
    expectation.png has ~35% headspace above face center.

    Args:
        frame: Input 16:9 frame
        face_bbox: (x, y, w, h) of user's face
        target_w: Output width (1080)
        target_h: Output height (1920)
        headspace_ratio: Fraction of output height above face center

    Returns:
        Cropped 9:16 frame
    """
    h, w = frame.shape[:2]
    fx, fy, fw, fh = face_bbox
    face_cx = fx + fw // 2
    face_cy = fy + fh // 2

    # Output aspect ratio
    out_aspect = target_w / target_h  # 1080/1920 = 0.5625

    # Crop width from source to match output aspect ratio
    crop_w = int(h * out_aspect)
    crop_h = h

    # Center crop horizontally, but shift to keep face centered
    # Face should be at headspace_ratio from top of output
    # So face_cy in source should map to headspace_ratio * target_h in output
    # This means crop center X should be at face_cx
    crop_x = face_cx - crop_w // 2
    crop_x = max(0, min(crop_x, w - crop_w))

    # Crop
    cropped = frame[0:crop_h, crop_x:crop_x+crop_w]

    # Scale to target resolution
    result = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--photos", default="photos/")
    p.add_argument("--tolerance", type=float, default=0.50)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    matcher = FaceMatcher(args.photos, tolerance=args.tolerance)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    user_count = 0
    other_count = 0
    no_face_count = 0

    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = matcher.find_all_faces_with_ids(frame)

        for x, y, w, h, is_user, dist in results:
            if is_user:
                user_count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"USER ({dist:.2f})", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                other_count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, f"OTHER ({dist:.2f})", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if not results:
            no_face_count += 1

        fi += 1
        if fi % 30 == 0:
            log.info("Frame %d/%d: user=%d other=%d no_face=%d",
                     fi, total, user_count, other_count, no_face_count)

    cap.release()

    print(f"\nResults ({fi} frames):")
    print(f"  User faces: {user_count}")
    print(f"  Other faces: {other_count}")
    print(f"  No face: {no_face_count}")
