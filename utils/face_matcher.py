import os
import glob
import logging
import numpy as np
from pathlib import Path

log = logging.getLogger("face_matcher")

_HOST_ENCODINGS = None

# Resolve photos directory relative to the project root (where config.yaml lives),
# NOT the current working directory — avoids broken globs when the worker CWD differs.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PHOTOS_DIR = _PROJECT_ROOT / "photos"

def get_host_encodings():
    global _HOST_ENCODINGS
    if _HOST_ENCODINGS is not None:
        return _HOST_ENCODINGS

    try:
        import face_recognition
    except ImportError:
        log.warning("face_recognition not installed. Dynamic matching unavailable.")
        _HOST_ENCODINGS = []
        return []

    encodings = []
    # Search for reference photos using absolute path relative to project root
    photo_paths = sorted(
        str(p) for p in _PHOTOS_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not photo_paths:
        log.warning("No reference photos found in '%s' directory.", _PHOTOS_DIR)
        _HOST_ENCODINGS = []
        return []

    log.info("Loading ML face recognition model and encoding %d reference photos from %s...",
             len(photo_paths), _PHOTOS_DIR)
    for path in photo_paths:
        try:
            image = face_recognition.load_image_file(path)
            image = np.ascontiguousarray(image)
            # Find all face encodings in the image
            face_encs = face_recognition.face_encodings(image, num_jitters=0)
            if face_encs:
                # Assume the largest face or first face is the host
                encodings.append(face_encs[0])
                log.info("  Encoded face from %s", os.path.basename(path))
            else:
                log.warning("No faces found in reference photo: %s", path)
        except Exception as e:
            log.error("Failed to load reference photo %s: %s", path, e)
            
    _HOST_ENCODINGS = encodings
    log.info("Successfully loaded %d host face encodings.", len(_HOST_ENCODINGS))
    return _HOST_ENCODINGS

def find_host_in_frame(frame_bgr: np.ndarray, facecam_bounds: dict = None) -> dict:
    """
    Detects faces in the frame and returns the bounding box of the host.
    Args:
        frame_bgr: BGR numpy array of the full frame
        facecam_bounds: Optional dict with x, y, width, height of facecam region.
                        When provided and full-frame detection fails, tries detection
                        on the cropped facecam region for better accuracy on small faces.
    Returns: {"x": x, "y": y, "width": w, "height": h} or None
    """
    host_encs = get_host_encodings()
    if not host_encs:
        return None

    try:
        import face_recognition
        # Convert BGR (OpenCV) to RGB (face_recognition) and ensure contiguous memory for dlib
        rgb_frame = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        
        # Use CNN model for speed/accuracy if GPU is available, else HOG
        use_gpu = False
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            pass
            
        model_type = "cnn" if use_gpu else "hog"
        face_locations = face_recognition.face_locations(rgb_frame, model=model_type)
        
        # If full-frame detection fails and facecam bounds provided, try cropped region
        if not face_locations and facecam_bounds:
            fc_x = facecam_bounds.get("x", 0)
            fc_y = facecam_bounds.get("y", 0)
            fc_w = facecam_bounds.get("width", 320)
            fc_h = facecam_bounds.get("height", 180)
            fh, fw = frame_bgr.shape[:2]
            # Clamp to frame bounds
            x1, y1 = max(0, fc_x), max(0, fc_y)
            x2, y2 = min(fw, fc_x + fc_w), min(fh, fc_y + fc_h)
            if x2 > x1 and y2 > y1:
                facecam_rgb = rgb_frame[y1:y2, x1:x2]
                face_locations = face_recognition.face_locations(facecam_rgb, model=model_type)
                if face_locations:
                    # Offset back to full-frame coordinates
                    face_locations = [(top + y1, right + x1, bottom + y1, left + x1)
                                      for (top, right, bottom, left) in face_locations]
                    log.debug("Face found via facecam-region crop (%d,%d %dx%d)",
                              fc_x, fc_y, fc_w, fc_h)
        
        if not face_locations:
            return None
            
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=0)
        
        best_match_idx = -1
        best_distance = 1.0 # Lower is better, 0.6 is typical strict threshold
        
        for i, face_encoding in enumerate(face_encodings):
            # Compare distance against all host reference encodings
            distances = face_recognition.face_distance(host_encs, face_encoding)
            min_dist = np.min(distances)
            
            if min_dist < best_distance:
                best_distance = min_dist
                best_match_idx = i
                
        # 0.6 is the default strictness threshold in face_recognition
        if best_match_idx != -1 and best_distance <= 0.65:
            # face_locations is (top, right, bottom, left)
            top, right, bottom, left = face_locations[best_match_idx]
            return {
                "x": left,
                "y": top,
                "width": right - left,
                "height": bottom - top,
                "confidence": 1.0 - best_distance
            }
            
    except Exception as e:
        log.error("Face recognition failed on frame: %s", e)
        
    return None
