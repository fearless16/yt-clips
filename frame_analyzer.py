import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("analyzer", cfg["logging"]["log_file"], cfg["logging"]["level"])


# ─── Face Detection Cascade (OpenCV) ────────────────────────────────────────
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_crop(frame_bgr: np.ndarray, frame_width: int, frame_height: int) -> Optional[Dict]:
    """
    Detects face and returns crop coordinates to keep eyes at 30% from top.
    Returns None if no face detected.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    if len(faces) == 0:
        return None
    
    # Get largest face (closest to camera)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    # Calculate target crop: 9:16 aspect ratio, centered on face
    # Position eyes at ~30% from top of final frame
    target_w = frame_width
    target_h = int(frame_width * 16 / 9)
    
    # Center X on face
    crop_x = max(0, x + w//2 - target_w//2)
    crop_x = min(crop_x, frame_width - target_w)
    
    # Position Y so eyes are at 30% from top
    # Assume eyes are at 40% down from top of face box
    eye_y = y + int(h * 0.4)
    desired_eye_y = int(target_h * 0.3)
    crop_y = eye_y - desired_eye_y
    crop_y = max(0, min(crop_y, frame_height - target_h))
    
    return {
        "x": int(crop_x),
        "y": int(crop_y),
        "width": int(target_w),
        "height": int(target_h),
        "confidence": 0.9  # High confidence face detected
    }


# ─── Safe subprocess wrapper ────────────────────────────────────────────────

def _run_cmd(cmd, timeout=15):
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning("Command timeout: %s", " ".join(cmd))
    except Exception as e:
        log.warning("Command failed: %s", e)
    return None


def save_preview_snapshot(video_path: str, timestamp: float, output_path: str) -> bool:
    """Save a preview snapshot (JPEG) at the given timestamp."""
    cmd = [
        "ffmpeg", "-y", "-ss", f"{timestamp:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path
    ]
    res = _run_cmd(cmd, timeout=10)
    return res is not None and res.returncode == 0


# ─── Brightness Sampling (Improved) ─────────────────────────────────────────

def _frame_stats(frame: bytes) -> Dict:
    """Compute avg + variance (better than avg only)."""
    if not frame:
        return {"avg": 128.0, "var": 0.0}

    n = len(frame)
    avg = sum(frame) / n
    var = sum((x - avg) ** 2 for x in frame) / n
    return {"avg": avg, "var": var}


def _sample_brightness_series(video_path: str, start: float, end: float, sample_count: int = 5) -> List[Dict]:
    duration = max(0.01, end - start)
    sample_count = max(1, min(sample_count, 10))  # limit to avoid RAM issues

    frame_size = 160 * 90
    fps = sample_count / duration

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", video_path,
        "-vf", f"fps={fps:.2f},scale=160:90,format=gray",
        "-frames:v", str(sample_count),
        "-f", "rawvideo", "-pix_fmt", "gray",
        "pipe:1",
    ]

    res = _run_cmd(cmd, timeout=20)
    if not res or res.returncode != 0 or not res.stdout:
        return []

    data = res.stdout
    frames = min(sample_count, len(data) // frame_size)

    samples = []
    for i in range(frames):
        frame = data[i * frame_size:(i + 1) * frame_size]
        stats = _frame_stats(frame)
        samples.append(stats)

    return samples


# ─── Black Frame Detection (Improved) ───────────────────────────────────────

def detect_black_frames(samples: List[Dict]) -> Dict:
    threshold = cfg.get("quality", {}).get("black_threshold", 20)
    
    # CRITICAL FIX: Require BOTH low brightness AND low variance for true black detection
    # This prevents false positives from dark scenes that still have content
    black = [s for s in samples if s["avg"] < threshold and s["var"] < 50]
    
    return {
        "has_black_frames": len(black) > 0,
        "black_ratio": len(black) / len(samples) if samples else 0,
        "is_mostly_black": len(black) / len(samples) > 0.6 if samples else False,  # For segment skipping
    }


# ─── Lighting Analysis (Improved) ───────────────────────────────────────────

def analyze_lighting(samples: List[Dict]) -> Dict:
    if not samples:
        return {"needs_correction": False}

    avg = sum(s["avg"] for s in samples) / len(samples)

    if avg < 70:
        return {
            "needs_correction": True,
            "lighting_filter": f"eq=gamma=1.3:contrast=1.1"
        }
    elif avg > 200:
        return {
            "needs_correction": True,
            "lighting_filter": f"eq=gamma=0.8:contrast=0.9"
        }

    return {"needs_correction": False}


# ─── Layout Detection (Multi-frame voting) ──────────────────────────────────

def detect_layout(video_path: str, start: float, end: float) -> Dict:
    """
    Detects if video has split-screen layout (e.g., host + guest side-by-side).
    Also detects if one panel is black (guest camera off).
    """
    timestamps = [
        start + (end - start) * x for x in [0.2, 0.4, 0.5, 0.6, 0.8]
    ]

    votes = []
    black_panel_votes = []
    black_panel_side_votes = []

    for t in timestamps:
        cmd = [
            "ffmpeg", "-y", "-ss", str(t),
            "-i", video_path,
            "-frames:v", "1",
            "-vf", "scale=320:180,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "pipe:1",
        ]

        res = _run_cmd(cmd)
        if not res or not res.stdout:
            continue

        data = res.stdout
        mid = 160
        
        # Calculate left and right panel brightness
        left_pixels = [data[i] for i in range(0, len(data), 320)]
        right_pixels = [data[i] for i in range(mid, len(data), 320)]
        
        left = sum(left_pixels)
        right = sum(right_pixels)
        
        # Check for split-screen: significant difference OR both panels active
        # For split-screen, left and right should have similar variance
        left_var = sum((x - left/len(left_pixels))**2 for x in left_pixels) / len(left_pixels)
        right_var = sum((x - right/len(right_pixels))**2 for x in right_pixels) / len(right_pixels)
        
        # Split-screen detection: both panels have content (variance > threshold)
        # If BOTH panels are active, we should DROP this segment (user preference)
        is_split = left_var > 1000 and right_var > 1000
        
        # Check if right panel is black (guest camera off)
        # Right panel average brightness < 25 AND low variance
        right_avg = right / len(right_pixels) if right_pixels else 0
        right_is_black = right_avg < 25 and right_var < 100
        
        # Check if LEFT panel is black (rare case)
        left_avg = left / len(left_pixels) if left_pixels else 0
        left_is_black = left_avg < 25 and left_var < 100
        
        if is_split:
            votes.append("split")
        if right_is_black:
            black_panel_votes.append(True)
            black_panel_side_votes.append("right")
        if left_is_black:
            black_panel_votes.append(True)
            black_panel_side_votes.append("left")

    is_split_screen = votes.count("split") >= 2
    has_black_panel = len(black_panel_votes) >= 2
    
    # Determine which side is black (majority vote)
    black_panel_side = None
    if has_black_panel:
        right_count = black_panel_side_votes.count("right")
        left_count = black_panel_side_votes.count("left")
        black_panel_side = "right" if right_count >= left_count else "left"
    
    return {
        "layout_type": "split" if is_split_screen else "solo",
        "prefer_solo": not is_split_screen,
        "has_black_panel": has_black_panel,
        "black_panel_side": black_panel_side,
        "is_multi_active_frame": is_split_screen and not has_black_panel  # NEW: Both panels active
    }


# ─── Safe ffprobe ───────────────────────────────────────────────────────────

def _get_video_dimensions(video_path: str) -> Dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path,
    ]

    res = _run_cmd(cmd)
    if not res or not res.stdout:
        return {"width": 1920, "height": 1080}

    try:
        data = json.loads(res.stdout)
        streams = data.get("streams", [])
        if not streams:
            return {"width": 1920, "height": 1080}

        s = streams[0]
        return {
            "width": int(s.get("width", 1920)),
            "height": int(s.get("height", 1080))
        }
    except Exception:
        return {"width": 1920, "height": 1080}


# ─── Silence Detection (safer) ──────────────────────────────────────────────

def detect_dead_air(video_path: str, start: float, end: float) -> Dict:
    duration = max(0.01, end - start)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", video_path,
        "-af", "silencedetect=noise=-35dB:d=1",
        "-f", "null", "-"
    ]

    res = _run_cmd(cmd, timeout=30)
    if not res:
        return {"has_dead_air": False}

    silence_count = res.stderr.decode(errors="replace").count("silence_start")
    return {"has_dead_air": silence_count > 0}


# ─── Main Analyzer ──────────────────────────────────────────────────────────

def analyze_clip(
    video_path: str,
    start: float,
    end: float,
    transcript_segments: Optional[List[Dict]] = None,
    clip_id: str = "clip"
) -> Dict:
    log.info("[%s] Analyzing clip...", clip_id)

    samples = _sample_brightness_series(video_path, start, end)

    black = detect_black_frames(samples)
    lighting = analyze_lighting(samples)
    layout = detect_layout(video_path, start, end)
    silence = detect_dead_air(video_path, start, end)

    # Basic Tempo Analysis if transcript available
    is_fast_paced = False
    if transcript_segments:
        # Filter segments within our range
        clip_segments = [s for s in transcript_segments if s["start"] < end and s["end"] > start]
        total_text = " ".join([s["text"] for s in clip_segments])
        words = len(total_text.split())
        duration = end - start
        wpm = (words / (duration / 60)) if duration > 0 else 0
        is_fast_paced = wpm > 150 # Simple heuristic
        log.debug("[%s] Tempo: %.1f WPM (fast=%s)", clip_id, wpm, is_fast_paced)

    # CRITICAL FIX: Only use solo frame if there's NO split screen OR if one panel is black
    # If split screen with both panels active, DROP this segment (user preference)
    # If split screen with black panel, use solo mode to show only the active panel
    should_use_solo = (
        layout["prefer_solo"] or  # No split screen detected
        layout.get("has_black_panel", False) or  # One panel is black (guest camera off)
        black["black_ratio"] > 0.6  # >60% black frames (was 30%, too aggressive)
    )
    
    # NEW: Mark segments with multiple active frames for DROPPING
    should_drop = layout.get("is_multi_active_frame", False)
    
    strategy = {
        "use_solo_frame": should_use_solo,
        "has_black_panel": layout.get("has_black_panel", False),
        "black_panel_side": layout.get("black_panel_side"),
        "is_multi_active_frame": layout.get("is_multi_active_frame", False),
        "should_drop": should_drop,  # NEW: Drop if both panels active
        "speed_factor": 1.25 if (silence["has_dead_air"] and not is_fast_paced) else 1.0,
        "apply_lighting_fix": lighting["needs_correction"],
        "lighting_filter": lighting.get("lighting_filter", ""),
        "skip_silence": silence["has_dead_air"],
    }

    return {
        "black_frames": black,
        "lighting": lighting,
        "layout": layout,
        "dead_air": silence,
        "export_strategy": strategy
    }