#!/usr/bin/env python3
"""
Face Pipeline Diagnostic Tool
==============================
Validates the end-to-end face detection, matching, and 9:16 crop pipeline.

Usage:
    python tests/diagnose_face_pipeline.py <video_path> [--frames N] [--facecam X Y W H] [--outdir DIR]

What it does:
    1. Loads reference photos from photos/ (absolute path, not CWD-dependent)
    2. Extracts N random frames from the video via ffmpeg (no full download)
    3. Runs face_recognition (HOG) + Haar Cascade on each frame
    4. Compares each detected face against reference encodings
    5. Computes the 9:16 crop region and validates face is centered
    6. Saves debug images with annotations
    7. Prints detailed pass/fail report
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_reference_encodings(photos_dir: Path) -> Tuple[List[np.ndarray], List[str]]:
    """Load face encodings from reference photos."""
    try:
        import face_recognition
    except ImportError:
        print("ERROR: face_recognition not installed. pip install face_recognition")
        sys.exit(1)

    encodings = []
    names = []
    photo_paths = sorted(
        p for p in photos_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not photo_paths:
        print(f"ERROR: No reference photos in {photos_dir}")
        sys.exit(1)

    for path in photo_paths:
        img = face_recognition.load_image_file(str(path))
        img = np.ascontiguousarray(img)
        encs = face_recognition.face_encodings(img, num_jitters=0)
        if encs:
            encodings.append(encs[0])
            names.append(path.name)
            print(f"  [OK] Encoded: {path.name} (face found)")
        else:
            print(f"  [SKIP] No face in: {path.name}")

    print(f"  Loaded {len(encodings)} reference encodings from {len(photo_paths)} photos\n")
    return encodings, names


def get_video_info(video_path: str) -> Dict:
    """Get video dimensions, fps, duration via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", video_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(res.stdout)
    stream = info["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": eval(stream.get("r_frame_rate", "30/1")),
        "total_frames": int(stream.get("nb_frames", 0)),
        "duration": float(stream.get("duration", 0)),
    }


def extract_frame_at(video_path: str, timestamp: float, width: int, height: int) -> Optional[np.ndarray]:
    """Extract a single frame at the given timestamp via ffmpeg pipe."""
    cmd = [
        "ffmpeg", "-y", "-ss", f"{timestamp:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-vf", f"scale={width}:{height}",
        "pipe:1"
    ]
    res = subprocess.run(cmd, capture_output=True, timeout=15)
    if not res.stdout:
        return None
    frame = np.frombuffer(res.stdout, dtype=np.uint8)
    expected = width * height * 3
    if frame.size != expected:
        return None
    return frame.reshape((height, width, 3))


def detect_faces_hog(rgb_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces using face_recognition HOG model. Returns [(top, right, bottom, left), ...]."""
    import face_recognition
    return face_recognition.face_locations(rgb_frame, model="hog")


def detect_faces_haar(gray_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces using Haar Cascade. Returns [(x, y, w, h), ...]."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray_frame, scaleFactor=1.03, minNeighbors=3, minSize=(30, 30))
    return [(x, y, w, h) for (x, y, w, h) in faces]


def match_face_encoding(
    rgb_frame: np.ndarray,
    face_location: Tuple[int, int, int, int],
    ref_encodings: List[np.ndarray]
) -> Tuple[float, int]:
    """Match a detected face against reference encodings. Returns (distance, ref_index)."""
    import face_recognition
    encs = face_recognition.face_encodings(rgb_frame, [face_location], num_jitters=0)
    if not encs:
        return 1.0, -1
    distances = face_recognition.face_distance(ref_encodings, encs[0])
    best_idx = int(np.argmin(distances))
    return float(distances[best_idx]), best_idx


def compute_9x16_crop(face_x: int, face_y: int, face_w: int, face_h: int,
                      frame_w: int, frame_h: int) -> Dict:
    """Compute the 9:16 crop region centered on the face."""
    target_h = frame_h
    target_w = int(target_h * 9 / 16)
    face_center_x = face_x + face_w // 2
    crop_x = face_center_x - target_w // 2
    crop_x = max(0, min(crop_x, frame_w - target_w))
    return {
        "crop_x": crop_x,
        "crop_y": 0,
        "crop_w": target_w,
        "crop_h": target_h,
        "face_center_x": face_center_x,
        "face_in_crop": crop_x <= face_x and (face_x + face_w) <= (crop_x + target_w),
        "face_centered": abs(face_center_x - (crop_x + target_w // 2)) < target_w * 0.15,
    }


def validate_facecam_crop(facecam_bounds: Dict, frame_w: int, frame_h: int,
                          detected_faces: List[Tuple]) -> Dict:
    """Check if any detected face falls within the configured facecam bounds."""
    fc_x = facecam_bounds.get("x", 0)
    fc_y = facecam_bounds.get("y", 0)
    fc_w = facecam_bounds.get("width", 320)
    fc_h = facecam_bounds.get("height", 180)
    margin = 50

    for face in detected_faces:
        if len(face) == 4:
            # HOG format: (top, right, bottom, left)
            if all(isinstance(v, int) for v in face):
                top, right, bottom, left = face
                fx, fy = left, top
            else:
                continue
        else:
            continue

        if (fc_x - margin) <= fx <= (fc_x + fc_w + margin) and \
           (fc_y - margin) <= fy <= (fc_y + fc_h + margin):
            return {"in_facecam": True, "face_pos": (fx, fy), "facecam": (fc_x, fc_y, fc_w, fc_h)}

    return {"in_facecam": False, "facecam": (fc_x, fc_y, fc_w, fc_h)}


def draw_debug_frame(
    frame: np.ndarray,
    hog_faces: List[Tuple],
    haar_faces: List[Tuple],
    match_results: List[Dict],
    crop_info: Optional[Dict],
    frame_idx: int,
    timestamp: float,
    facecam_bounds: Optional[Dict] = None,
) -> np.ndarray:
    """Draw debug annotations on a copy of the frame."""
    canvas = frame.copy()

    # Draw facecam bounds if provided
    if facecam_bounds:
        fc_x = facecam_bounds["x"]
        fc_y = facecam_bounds["y"]
        fc_w = facecam_bounds["width"]
        fc_h = facecam_bounds["height"]
        cv2.rectangle(canvas, (fc_x, fc_y), (fc_x + fc_w, fc_y + fc_h), (0, 255, 255), 2)
        cv2.putText(canvas, "FACECAM", (fc_x, fc_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw HOG faces
    for top, right, bottom, left in hog_faces:
        cv2.rectangle(canvas, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(canvas, "HOG", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw Haar faces
    for x, y, w, h in haar_faces:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(canvas, "HAAR", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Draw match info
    for mr in match_results:
        top, right, bottom, left = mr["location"]
        color = (0, 255, 0) if mr["matched"] else (0, 0, 255)
        label = f"{'MATCH' if mr['matched'] else 'NO_MATCH'} d={mr['distance']:.3f}"
        cv2.putText(canvas, label, (left, bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw 9:16 crop region
    if crop_info:
        cx, cy, cw, ch = crop_info["crop_x"], crop_info["crop_y"], crop_info["crop_w"], crop_info["crop_h"]
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (0, 165, 255), 2)
        status = "CENTERED" if crop_info["face_centered"] else "OFF-CENTER"
        cv2.putText(canvas, f"9:16 CROP ({status})", (cx, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # Header
    cv2.putText(canvas, f"frame={frame_idx} t={timestamp:.1f}s {frame.shape[1]}x{frame.shape[0]}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Face Pipeline Diagnostic Tool")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--frames", type=int, default=10, help="Number of random frames to sample")
    parser.add_argument("--facecam", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="Facecam bounds: x y width height")
    parser.add_argument("--outdir", default="diagnostics",
                        help="Output directory for debug images")
    parser.add_argument("--match-threshold", type=float, default=0.65,
                        help="Face match distance threshold (default: 0.65)")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    photos_dir = Path(_PROJECT_ROOT) / "photos"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FACE PIPELINE DIAGNOSTIC")
    print("=" * 70)

    # 1. Load reference encodings
    print("\n[1/5] Loading reference encodings...")
    ref_encodings, ref_names = load_reference_encodings(photos_dir)
    if not ref_encodings:
        print("FATAL: No valid reference encodings. Check photos/ directory.")
        sys.exit(1)

    # 2. Get video info
    print("[2/5] Probing video...")
    info = get_video_info(video_path)
    print(f"  {info['width']}x{info['height']}, {info['fps']:.1f}fps, "
          f"{info['duration']:.1f}s, {info['total_frames']} frames")

    # Use facecam from CLI or config
    facecam_bounds = None
    if args.facecam:
        facecam_bounds = {"x": args.facecam[0], "y": args.facecam[1],
                          "width": args.facecam[2], "height": args.facecam[3]}
        print(f"  Facecam bounds: {facecam_bounds}")
    else:
        # Try loading from config
        try:
            from utils.config import load_config
            cfg = load_config()
            layout_cfg = cfg.get("layout", {})
            if layout_cfg.get("has_facecam"):
                facecam_bounds = layout_cfg.get("facecam")
                print(f"  Facecam bounds (from config): {facecam_bounds}")
        except Exception:
            pass

    # 3. Sample frames
    print(f"\n[3/5] Extracting {args.frames} random frames...")
    duration = info["duration"]
    timestamps = np.linspace(1.0, max(1.0, duration - 1.0), args.frames)

    results = []
    total_hog_found = 0
    total_haar_found = 0
    total_matched = 0
    total_facecam_match = 0
    total_crop_valid = 0

    for i, ts in enumerate(timestamps):
        frame = extract_frame_at(video_path, float(ts), info["width"], info["height"])
        if frame is None:
            print(f"  [SKIP] frame {i}: extraction failed at t={ts:.1f}s")
            continue

        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        hog_faces = detect_faces_hog(rgb)
        haar_faces = detect_faces_haar(gray)

        total_hog_found += len(hog_faces)
        total_haar_found += len(haar_faces)

        # Also try facecam-cropped detection if bounds available and full-frame fails
        facecam_hog_faces = []
        if facecam_bounds and not hog_faces:
            fc_x = facecam_bounds["x"]
            fc_y = facecam_bounds["y"]
            fc_w = facecam_bounds["width"]
            fc_h = facecam_bounds["height"]
            x1, y1 = max(0, fc_x), max(0, fc_y)
            x2, y2 = min(info["width"], fc_x + fc_w), min(info["height"], fc_y + fc_h)
            if x2 > x1 and y2 > y1:
                cropped_rgb = rgb[y1:y2, x1:x2]
                fc_faces = detect_faces_hog(cropped_rgb)
                for loc in fc_faces:
                    top, right, bottom, left = loc
                    # Offset back to full-frame coords
                    facecam_hog_faces.append((top + y1, right + x1, bottom + y1, left + x1))
                if fc_faces:
                    print(f"  [INFO] frame {i}: face found via facecam-region crop (full-frame missed it)")

        all_hog_faces = hog_faces + facecam_hog_faces

        # Match each HOG face against references
        match_results = []
        frame_matched = False
        for loc in all_hog_faces:
            dist, ref_idx = match_face_encoding(rgb, loc, ref_encodings)
            matched = dist <= args.match_threshold
            if matched:
                frame_matched = True
            match_results.append({
                "location": loc,
                "distance": dist,
                "ref_idx": ref_idx,
                "ref_name": ref_names[ref_idx] if ref_idx >= 0 else "none",
                "matched": matched,
            })

        if frame_matched:
            total_matched += 1

        # Facecam validation
        facecam_result = None
        if facecam_bounds and all_hog_faces:
            facecam_result = validate_facecam_crop(facecam_bounds, info["width"], info["height"], all_hog_faces)
            if facecam_result["in_facecam"]:
                total_facecam_match += 1

        # 9:16 crop validation (use best matched face)
        crop_info = None
        best_match = None
        for mr in match_results:
            if mr["matched"]:
                best_match = mr
                break
        if not best_match and match_results:
            best_match = min(match_results, key=lambda m: m["distance"])

        if best_match:
            top, right, bottom, left = best_match["location"]
            crop_info = compute_9x16_crop(left, top, right - left, bottom - top,
                                          info["width"], info["height"])
            if crop_info["face_in_crop"] and crop_info["face_centered"]:
                total_crop_valid += 1

        # Save debug image
        debug_img = draw_debug_frame(
            frame, all_hog_faces, haar_faces, match_results, crop_info,
            i, float(ts), facecam_bounds
        )
        debug_path = outdir / f"frame_{i:03d}_t{ts:.1f}s.jpg"
        cv2.imwrite(str(debug_path), debug_img)

        # Print per-frame summary
        status_parts = []
        status_parts.append(f"HOG={len(all_hog_faces)}")
        status_parts.append(f"HAAR={len(haar_faces)}")
        if match_results:
            best = min(match_results, key=lambda m: m["distance"])
            tag = "MATCH" if best["matched"] else "MISS"
            status_parts.append(f"{tag}(d={best['distance']:.3f})")
        if crop_info:
            status_parts.append("CROP:" + ("OK" if crop_info["face_centered"] else "OFF"))
        if facecam_result:
            status_parts.append("FC:" + ("IN" if facecam_result["in_facecam"] else "OUT"))

        print(f"  frame {i:3d} t={ts:6.1f}s: {' | '.join(status_parts)}")

        results.append({
            "frame": i,
            "timestamp": float(ts),
            "hog_faces": len(all_hog_faces),
            "haar_faces": len(haar_faces),
            "matches": match_results,
            "crop": crop_info,
            "facecam": facecam_result,
        })

    # 4. Save raw 9:16 crops for matched frames
    print(f"\n[4/5] Saving 9:16 crops...")
    for r in results:
        if r["crop"] and any(m["matched"] for m in r["matches"]):
            ts = r["timestamp"]
            frame = extract_frame_at(video_path, ts, info["width"], info["height"])
            if frame is None:
                continue
            c = r["crop"]
            cropped = frame[0:c["crop_h"], c["crop_x"]:c["crop_x"] + c["crop_w"]]
            if cropped.size > 0:
                crop_path = outdir / f"crop_9x16_frame{r['frame']:03d}_t{ts:.1f}s.jpg"
                cv2.imwrite(str(crop_path), cropped)

    # 5. Summary report
    n = len(results)
    print(f"\n{'=' * 70}")
    print("DIAGNOSTIC REPORT")
    print(f"{'=' * 70}")
    print(f"  Video:              {os.path.basename(video_path)}")
    print(f"  Dimensions:         {info['width']}x{info['height']}")
    print(f"  Reference photos:   {len(ref_encodings)}")
    print(f"  Frames sampled:     {n}")
    print(f"  HOG detections:     {total_hog_found}/{n} frames had faces")
    print(f"  Haar detections:    {total_haar_found}/{n} frames had faces")
    print(f"  Host matched:       {total_matched}/{n} frames ({100*total_matched/max(1,n):.0f}%)")
    if facecam_bounds:
        print(f"  In facecam region:  {total_facecam_match}/{n} frames ({100*total_facecam_match/max(1,n):.0f}%)")
    print(f"  Valid 9:16 crop:    {total_crop_valid}/{n} frames ({100*total_crop_valid/max(1,n):.0f}%)")

    # Verdict
    print()
    if total_matched == 0:
        print("  VERDICT: FAIL — No faces matched reference in ANY frame.")
        print("  Check: photos/ has clear face photos, video has visible facecam.")
    elif total_matched < n * 0.5:
        print(f"  VERDICT: POOR — Only {100*total_matched//n}% frames matched.")
        print("  Check: facecam bounds, video quality, reference photo clarity.")
    elif total_crop_valid < total_matched * 0.8:
        print("  VERDICT: CROP ISSUES — Faces matched but 9:16 crops are off-center.")
    else:
        print("  VERDICT: PASS — Face detection and crop pipeline working correctly.")

    print(f"\n  Debug images saved to: {outdir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
