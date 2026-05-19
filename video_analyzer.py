"""
video_analyzer.py — Pre-analysis: face/lighting map + reference photo matching.

Samples the full VOD, detects faces, analyzes lighting, matches against
reference photos, and produces a per-second quality map that feeds into
highlight detection for smarter clip selection.

Usage:
    python video_analyzer.py <video_path> --photos photos/ --reference expectation.png
    python video_analyzer.py input/video.mp4  # auto-detect photos/
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("video_analyzer", cfg["logging"]["log_file"], cfg["logging"]["level"])

# ─── Face detection (prefer face_recognition, fallback to OpenCV) ─────────

HAS_FACE_REC = False
try:
    import face_recognition
    HAS_FACE_REC = True
except ImportError:
    pass

HAAR_CASCADE = None


def _get_haar_cascade():
    global HAAR_CASCADE
    if HAAR_CASCADE is None:
        HAAR_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return HAAR_CASCADE


def _detect_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces, return list of (top, right, bottom, left)."""
    if HAS_FACE_REC:
        try:
            locations = face_recognition.face_locations(frame, model="hog")
            if locations:
                return locations
        except Exception:
            pass
    # Fallback: OpenCV Haar
    cascade = _get_haar_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    return [(y, x + w, y + h, x) for (x, y, w, h) in rects]


# ─── Lighting analysis ────────────────────────────────────────────────────

def _analyze_lighting(frame: np.ndarray) -> Dict[str, float]:
    """Analyze lighting properties of a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Overall brightness (mean luminance)
    brightness = float(np.mean(gray))

    # Contrast (std of luminance)
    contrast = float(np.std(gray))

    # Histogram spread (how evenly distributed tones are)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    # Entropy — higher = more tonal variety
    nonzero = hist[hist > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    # Face region brightness (if face detected)
    faces = _detect_faces(frame)
    face_brightness = 0.0
    face_contrast = 0.0
    face_area_ratio = 0.0

    if faces:
        top, right, bottom, left = max(
            faces, key=lambda r: (r[2] - r[0]) * (r[1] - r[3])
        )
        face_gray = gray[top:bottom, left:right]
        if face_gray.size > 0:
            face_brightness = float(np.mean(face_gray))
            face_contrast = float(np.std(face_gray))
            face_area_ratio = (face_gray.shape[0] * face_gray.shape[1]) / (h * w)

    # Exposure quality — penalize too dark or too blown-out
    overexposed = float(np.mean(gray > 240))  # fraction of near-white pixels
    underexposed = float(np.mean(gray < 15))   # fraction of near-black pixels

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "entropy": round(entropy, 2),
        "face_brightness": round(face_brightness, 2),
        "face_contrast": round(face_contrast, 2),
        "face_area_ratio": round(face_area_ratio, 4),
        "overexposed_pct": round(overexposed * 100, 2),
        "underexposed_pct": round(underexposed * 100, 2),
        "face_count": len(faces),
    }


# ─── Reference photo matching ─────────────────────────────────────────────

def _load_reference_embeddings(photos_dir: Path) -> List[np.ndarray]:
    """Load face encodings from reference photos."""
    if not HAS_FACE_REC:
        return []
    if not photos_dir.exists():
        return []

    embeddings = []
    for ext in ("jpg", "jpeg", "png", "webp"):
        for img_path in photos_dir.glob(f"*.{ext}"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb)
                embeddings.extend(encs)
            except Exception as e:
                log.debug("Failed to encode %s: %s", img_path.name, e)
    log.info("Loaded %d reference face embeddings from %s", len(embeddings), photos_dir)
    return embeddings


def _match_face_to_references(
    frame_rgb: np.ndarray,
    face_location: Tuple[int, int, int, int],
    ref_embeddings: List[np.ndarray],
) -> float:
    """Return best similarity score between detected face and reference photos."""
    if not ref_embeddings or not HAS_FACE_REC:
        return 0.0
    top, right, bottom, left = face_location
    # Add generous padding for better encoding
    h, w = frame_rgb.shape[:2]
    pad = int(max(bottom - top, right - left) * 0.4)
    y1, y2 = max(0, top - pad), min(h, bottom + pad)
    x1, x2 = max(0, left - pad), min(w, right + pad)

    face_crop = frame_rgb[y1:y2, x1:x2]
    if face_crop.size == 0:
        return 0.0

    encs = face_recognition.face_encodings(face_crop, num_jitters=1)
    if not encs:
        return 0.0

    face_enc = encs[0]
    best = 0.0
    for ref_enc in ref_embeddings:
        dist = face_recognition.face_distance([ref_enc], face_enc)[0]
        sim = 1.0 - dist  # cosine-like similarity
        if sim > best:
            best = sim
    return round(best, 4)


# ─── Frame sampler ─────────────────────────────────────────────────────────

def _sample_frames(
    video_path: str,
    interval_sec: float = 2.0,
):
    """Extract frames at regular intervals using ffmpeg GPU (fallback CPU).
    
    Yields (timestamp, frame_bgr) one at a time — no memory accumulation.
    """
    probe = _probe_video(video_path)
    fps = probe.get("fps", 30)
    width = probe.get("width", 0)
    height = probe.get("height", 0)
    duration = probe.get("duration", 0)
    step = max(1, int(fps * interval_sec))

    log.info("Sampling frames every %.1fs from %.0fs video (%.0f fps, %d frames, %dx%d)",
             interval_sec, duration, fps, int(duration * fps), width, height)

    if width and height and shutil.which("ffmpeg"):
        yield from _sample_frames_gpu(video_path, fps, width, height, step, interval_sec, duration)
    else:
        yield from _sample_frames_cpu(video_path, fps, step, interval_sec)


def _sample_frames_gpu(video_path, fps, width, height, step, interval_sec, duration=0):
    """GPU-accelerated frame sampling via ffmpeg (auto-detect hwaccel). Falls back to CPU."""
    frame_size = width * height * 3
    import subprocess
    # Detect if CUDA is available
    has_cuda = False
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=5,
        )
        has_cuda = "cuda" in r.stdout.lower()
    except Exception:
        pass

    hwaccel = "cuda" if has_cuda else "none"
    cmd = [
        "ffmpeg",
        "-hwaccel", hwaccel,
        "-i", video_path,
        "-vf", f"select=not(mod(n\\,{step}))",
        "-vsync", "0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
    )
    idx = 0
    log_every = max(1, 300 // interval_sec)  # log every ~5 min of video
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
            yield (idx * interval_sec, frame)
            idx += 1
            if idx % log_every == 0:
                log.info("GPU progress: %.1fs / %.0fs (%.0f%%)",
                         idx * interval_sec, duration,
                         idx * interval_sec / duration * 100)
    finally:
        proc.kill()
        proc.wait()
    log.info("GPU: sampled %d frames in %.1fs of video",
             idx, idx * interval_sec)


def _sample_frames_cpu(video_path, fps, step, interval_sec):
    """CPU fallback via OpenCV. Yields (ts, frame)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return
    idx = 0
    count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                yield (idx / fps if fps > 0 else 0, frame)
                count += 1
            idx += 1
    finally:
        cap.release()
    log.info("CPU: sampled %d frames", count)


# ─── Quality scoring ───────────────────────────────────────────────────────

# Ideal ranges for talking-head YouTube Shorts (learned from expectation.png)
IDEAL = {
    "brightness": (100, 180),      # Well-lit face, not washed out
    "contrast": (40, 80),          # Enough contrast for sharpness
    "face_area_ratio": (0.08, 0.5),# Face should fill reasonable portion
    "face_brightness": (100, 200), # Face well-lit
}


def _score_frame(
    lighting: Dict[str, float],
    ref_match: float,
    face_detected: bool,
) -> float:
    """
    Score a single frame 0.0 (bad) → 1.0 (perfect).
    Weights: face match (30%), lighting (30%), face presence (20%), exposure (20%).
    """
    score = 0.0

    # 1. Face presence (20 pts) — reduced penalty for sports/low-face content
    if not face_detected:
        score += 0.05  # Reduced penalty: allows sports clips without close-ups
    else:
        score += 0.20

    # 2. Reference face match (30 pts)
    score += ref_match * 0.30

    # 3. Lighting quality (30 pts)
    brightness = lighting["brightness"]
    b_lo, b_hi = IDEAL["brightness"]
    if b_lo <= brightness <= b_hi:
        lighting_score = 1.0
    else:
        dist = min(abs(brightness - b_lo), abs(brightness - b_hi))
        lighting_score = max(0, 1.0 - dist / 80.0)
    score += lighting_score * 0.15

    contrast = lighting["contrast"]
    c_lo, c_hi = IDEAL["contrast"]
    if c_lo <= contrast <= c_hi:
        contrast_score = 1.0
    else:
        dist = min(abs(contrast - c_lo), abs(contrast - c_hi))
        contrast_score = max(0, 1.0 - dist / 50.0)
    score += contrast_score * 0.15

    # 4. Exposure quality (20 pts) — penalize clipping
    over = lighting["overexposed_pct"]
    under = lighting["underexposed_pct"]
    exposure_penalty = (over + under) / 100.0
    score += max(0, 0.20 - exposure_penalty * 0.20)

    return round(min(1.0, max(0.0, score)), 4)


# ─── Main analysis ─────────────────────────────────────────────────────────

def analyze_video(
    video_path: str,
    photos_dir: str = "photos/",
    reference_image: str = "expectation.png",
    sample_interval: float = 2.0,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Analyze full video: face detection, lighting map, reference matching.

    Returns:
        Dict with per_second scores, summary stats, and segment recommendations.
    """
    t_start = time.perf_counter()
    video_path = str(Path(video_path).resolve())
    photos_path = Path(photos_dir)
    ref_path = Path(reference_image)

    log.info("=" * 60)
    log.info("VIDEO ANALYZER — Pre-clipping Quality Map")
    log.info("=" * 60)
    log.info("Video: %s", video_path)
    log.info("Photos: %s", photos_path)
    log.info("Reference: %s", ref_path)

    # Get video duration
    probe = _probe_video(video_path)
    duration = probe.get("duration", 0)

    # Load reference face embeddings
    ref_embeddings = _load_reference_embeddings(photos_path)

    # Also try expectation.png as a reference
    if ref_path.exists() and HAS_FACE_REC:
        try:
            img = cv2.imread(str(ref_path))
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb)
                ref_embeddings.extend(encs)
                log.info("Added reference from %s", ref_path)
        except Exception:
            pass

    # Sample frames (streaming, GPU-accelerated — no memory accumulation)
    per_frame = []
    for ts, frame in _sample_frames(video_path, interval_sec=sample_interval):
        lighting = _analyze_lighting(frame)
        face_detected = lighting["face_count"] > 0

        # Match against reference photos
        ref_match = 0.0
        if face_detected and ref_embeddings:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = _detect_faces(frame)
            if faces:
                best_face = max(faces, key=lambda r: (r[2]-r[0]) * (r[1]-r[3]))
                ref_match = _match_face_to_references(rgb, best_face, ref_embeddings)

        quality = _score_frame(lighting, ref_match, face_detected)

        per_frame.append({
            "timestamp": round(ts, 2),
            "lighting": lighting,
            "ref_match": ref_match,
            "quality": quality,
        })

    if not per_frame:
        log.error("No frames extracted — aborting")
        return {"summary": {"frames_sampled": 0, "duration_sec": 0, "error": "no frames"},
                "per_second": [], "best_segments": [], "per_frame": []}

    # Aggregate to per-second scores
    per_second = _aggregate_to_seconds(per_frame)

    # Find best segments
    segments = _find_best_segments(per_second, duration)

    # Summary stats
    all_scores = [f["quality"] for f in per_frame]
    with_face = [f for f in per_frame if f["lighting"]["face_count"] > 0]

    summary = {
        "video": video_path,
        "duration_sec": round(duration, 1),
        "frames_sampled": len(per_frame),
        "frames_with_face": len(with_face),
        "face_detection_rate": round(len(with_face) / max(1, len(per_frame)) * 100, 1),
        "avg_quality": round(np.mean(all_scores), 4) if all_scores else 0,
        "max_quality": round(max(all_scores), 4) if all_scores else 0,
        "min_quality": round(min(all_scores), 4) if all_scores else 0,
        "avg_brightness": round(np.mean([f["lighting"]["brightness"] for f in per_frame]), 1),
        "avg_ref_match": round(np.mean([f["ref_match"] for f in per_frame]), 4),
        "reference_photos_used": len(ref_embeddings),
        "analysis_time_sec": round(time.perf_counter() - t_start, 1),
    }

    # Build output
    result = {
        "summary": summary,
        "per_second": per_second,
        "best_segments": segments,
        "per_frame": per_frame,
    }

    # Save
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(cfg["paths"]["temp"]) / f"{stem}_analysis.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    _print_summary(summary, segments)
    log.info("Analysis saved: %s", output_path)

    return result


def _probe_video(video_path: str) -> Dict:
    """Get video metadata via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json", video_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(r.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format", {})
        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30
        else:
            fps = float(fps_str)
        return {
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "fps": round(fps, 2),
            "duration": float(fmt.get("duration", 0)),
        }
    except Exception:
        return {"width": 0, "height": 0, "fps": 30, "duration": 0}


def _aggregate_to_seconds(per_frame: List[Dict]) -> List[Dict]:
    """Aggregate per-frame scores to per-second averages."""
    buckets = defaultdict(list)
    for f in per_frame:
        sec = int(f["timestamp"])
        buckets[sec].append(f)

    per_second = []
    for sec in sorted(buckets.keys()):
        frames_in_sec = buckets[sec]
        avg_quality = np.mean([f["quality"] for f in frames_in_sec])
        avg_brightness = np.mean([f["lighting"]["brightness"] for f in frames_in_sec])
        max_ref = max(f["ref_match"] for f in frames_in_sec)
        any_face = any(f["lighting"]["face_count"] > 0 for f in frames_in_sec)

        per_second.append({
            "second": sec,
            "quality": round(float(avg_quality), 4),
            "brightness": round(float(avg_brightness), 1),
            "ref_match": round(float(max_ref), 4),
            "face_present": any_face,
        })

    return per_second


def _find_best_segments(
    per_second: List[Dict],
    total_duration: float,
    window: int = 25,
    top_n: int = 10,
) -> List[Dict]:
    """Find the best contiguous segments for clipping."""
    if not per_second:
        return []

    scores = [ps["quality"] for ps in per_second]
    n = len(scores)

    # Sliding window average
    candidates = []
    for i in range(max(1, n - window + 1)):
        end = min(i + window, n)
        window_scores = scores[i:end]
        avg = np.mean(window_scores)
        # Bonus for face presence throughout
        face_ratio = sum(1 for ps in per_second[i:end] if ps["face_present"]) / len(window_scores)
        # Bonus for reference match
        avg_ref = np.mean([ps["ref_match"] for ps in per_second[i:end]])

        composite = avg * 0.6 + face_ratio * 0.25 + avg_ref * 0.15

        start_time = per_second[i]["second"]
        end_time = per_second[min(end - 1, n - 1)]["second"] + 1

        candidates.append({
            "start_sec": start_time,
            "end_sec": min(end_time, total_duration),
            "duration_sec": end_time - start_time,
            "avg_quality": round(float(avg), 4),
            "face_ratio": round(float(face_ratio), 2),
            "avg_ref_match": round(float(avg_ref), 4),
            "composite_score": round(float(composite), 4),
        })

    # Deduplicate overlapping segments, keep best
    candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    selected = []
    for c in candidates:
        overlaps = False
        for s in selected:
            if c["start_sec"] < s["end_sec"] and c["end_sec"] > s["start_sec"]:
                overlaps = True
                break
        if not overlaps:
            selected.append(c)
        if len(selected) >= top_n:
            break

    return selected


def _print_summary(summary: Dict, segments: List[Dict]):
    """Print analysis summary to terminal."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from rich import box

        console = Console()

        console.print()
        console.print(Panel(
            Text("VIDEO ANALYSIS — Quality Map", style="bold cyan", justify="center"),
            border_style="cyan", width=70,
        ))

        t = Table(title="Summary", box=box.ROUNDED, header_style="bold cyan")
        t.add_column("Metric", style="cyan")
        t.add_column("Value", justify="right")
        t.add_row("Duration", f"{summary['duration_sec']:.0f}s")
        t.add_row("Frames Sampled", str(summary['frames_sampled']))
        t.add_row("Face Detection Rate", f"{summary['face_detection_rate']}%")
        t.add_row("Avg Quality", f"{summary['avg_quality']:.3f}")
        t.add_row("Max Quality", f"{summary['max_quality']:.3f}")
        t.add_row("Avg Brightness", f"{summary['avg_brightness']:.0f}")
        t.add_row("Avg Ref Match", f"{summary['avg_ref_match']:.3f}")
        t.add_row("Reference Photos", str(summary['reference_photos_used']))
        t.add_row("Analysis Time", f"{summary['analysis_time_sec']:.1f}s")
        console.print(t)

        if segments:
            st = Table(title="Best Segments for Clipping", box=box.SIMPLE, header_style="bold green")
            st.add_column("#", justify="right", style="dim")
            st.add_column("Start", justify="right")
            st.add_column("End", justify="right")
            st.add_column("Duration", justify="right")
            st.add_column("Quality", justify="right")
            st.add_column("Face %", justify="right")
            st.add_column("Ref Match", justify="right")
            st.add_column("Score", justify="right", style="bold green")

            for i, seg in enumerate(segments, 1):
                st.add_row(
                    str(i),
                    f"{seg['start_sec']:.0f}s",
                    f"{seg['end_sec']:.0f}s",
                    f"{seg['duration_sec']:.0f}s",
                    f"{seg['avg_quality']:.3f}",
                    f"{seg['face_ratio']*100:.0f}%",
                    f"{seg['avg_ref_match']:.3f}",
                    f"{seg['composite_score']:.3f}",
                )
            console.print()
            console.print(st)
        console.print()
    except ImportError:
        # Fallback: plain text
        print(f"\nVideo Analysis: {summary['duration_sec']:.0f}s")
        print(f"  Face detection: {summary['face_detection_rate']}%")
        print(f"  Avg quality: {summary['avg_quality']:.3f}")
        print(f"  Best segments: {len(segments)}")
        for i, seg in enumerate(segments[:5], 1):
            print(f"    {i}. {seg['start_sec']:.0f}s-{seg['end_sec']:.0f}s (score={seg['composite_score']:.3f})")


# ─── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze video for face/lighting quality")
    parser.add_argument("video", help="Path to source video")
    parser.add_argument("--photos", default="photos/", help="Reference photos directory")
    parser.add_argument("--reference", default="expectation.png", help="Reference image")
    parser.add_argument("--interval", type=float, default=2.0, help="Frame sampling interval (seconds)")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = analyze_video(
        video_path=args.video,
        photos_dir=args.photos,
        reference_image=args.reference,
        sample_interval=args.interval,
        output_path=args.output,
    )
