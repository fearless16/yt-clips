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
    arr = np.frombuffer(frame, dtype=np.uint8).astype(np.float32)
    return {"avg": float(arr.mean()), "var": float(arr.var())}


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
        "-vf", f"fps={sample_count}/{duration:.3f},scale=160:90,format=gray",
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

def _has_vertical_divider(frame_array: np.ndarray, width: int) -> bool:
    """
    Check for a clear vertical divider line near the center of the frame.
    A true split-screen (OBS/Zoom) has a sharp brightness discontinuity at the center.
    Background posters or screen shares do NOT have this.
    """
    center = width // 2
    # Sample a narrow band (±3 pixels) around the center
    band_width = 3
    left_band = frame_array[:, max(0, center - band_width - 5):max(0, center - band_width)]
    right_band = frame_array[:, center + band_width:center + band_width + 5]
    center_band = frame_array[:, center - band_width:center + band_width]

    if left_band.size == 0 or right_band.size == 0 or center_band.size == 0:
        return False

    # Divider = center band is significantly darker/different from both sides
    center_avg = float(center_band.mean())
    left_avg = float(left_band.mean())
    right_avg = float(right_band.mean())

    # The center strip should be notably darker (black divider line)
    # OR there should be a sharp gradient across the center
    has_dark_divider = center_avg < min(left_avg, right_avg) - 30
    has_sharp_edge = abs(left_avg - right_avg) > 40

    return has_dark_divider or has_sharp_edge


def _detect_faces_per_half(frame_array: np.ndarray, width: int) -> dict:
    """
    Detect faces in left and right halves separately.
    Returns count of faces on each side.
    """
    left_half = frame_array[:, :width // 2]
    right_half = frame_array[:, width // 2:]

    left_faces = FACE_CASCADE.detectMultiScale(left_half, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    right_faces = FACE_CASCADE.detectMultiScale(right_half, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    return {
        "left": len(left_faces),
        "right": len(right_faces),
    }


def _is_screen_share_frame(frame_array: np.ndarray) -> bool:
    """
    Heuristic: screen share frames tend to have high edge density
    (text, UI elements) and relatively uniform brightness bands.
    """
    # High variance overall = lots of content
    overall_var = float(frame_array.var())
    # Compute horizontal gradient (edges)
    grad = np.abs(np.diff(frame_array.astype(np.int16), axis=1))
    edge_density = float(grad.mean())

    # Screen shares: moderate-high variance + high edge density + no strong center divider
    return overall_var > 800 and edge_density > 8


def detect_layout(video_path: str, start: float, end: float) -> Dict:
    """
    Detects if video has split-screen layout (e.g., host + guest side-by-side).
    Also detects if one panel is black (guest camera off) or screen sharing.

    CRITICAL: Requires BOTH a vertical divider AND faces on both halves to
    flag as multi-active. Background posters / screen shares do NOT count.
    """
    duration = max(0.01, end - start)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", video_path,
        "-vf", f"fps=5/{duration:.3f},scale=320:180,format=gray",
        "-frames:v", "5",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "pipe:1"
    ]

    divider_votes = []
    dual_face_votes = []
    black_panel_votes = []
    black_panel_side_votes = []
    screen_share_votes = []

    res = _run_cmd(cmd)
    if res and res.stdout:
        data = res.stdout
        frame_size = 320 * 180
        frames = len(data) // frame_size

        for i in range(frames):
            frame_data = data[i * frame_size : (i + 1) * frame_size]
            frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(180, 320)
            left_half  = frame_array[:, :160]
            right_half = frame_array[:, 160:]

            left_avg  = float(left_half.mean())
            right_avg = float(right_half.mean())
            left_var  = float(left_half.var())
            right_var = float(right_half.var())

            right_is_black = right_avg < 25 and right_var < 100
            left_is_black = left_avg < 25 and left_var < 100

            if right_is_black:
                black_panel_votes.append(True)
                black_panel_side_votes.append("right")
            if left_is_black:
                black_panel_votes.append(True)
                black_panel_side_votes.append("left")

            # Check for vertical divider (true split-screen signature)
            if _has_vertical_divider(frame_array, 320):
                divider_votes.append(True)

            # Check for faces on both sides (requires real people, not posters)
            face_counts = _detect_faces_per_half(frame_array, 320)
            if face_counts["left"] > 0 and face_counts["right"] > 0:
                dual_face_votes.append(True)

            # Check for screen share
            if _is_screen_share_frame(frame_array):
                screen_share_votes.append(True)

    has_black_panel = len(black_panel_votes) >= 2

    # TRUE split-screen requires BOTH: vertical divider AND faces on both halves
    # This prevents false positives from background posters or screen sharing
    has_divider = len(divider_votes) >= 2
    has_dual_faces = len(dual_face_votes) >= 2
    is_split_screen = has_divider and has_dual_faces

    is_screen_share = len(screen_share_votes) >= 3 and not is_split_screen

    # Determine which side is black (majority vote)
    black_panel_side = None
    if has_black_panel:
        right_count = black_panel_side_votes.count("right")
        left_count = black_panel_side_votes.count("left")
        black_panel_side = "right" if right_count >= left_count else "left"

    # Determine layout type
    if is_screen_share:
        layout_type = "screen_share"
    elif is_split_screen:
        layout_type = "split"
    else:
        layout_type = "solo"

    log.debug(
        "Layout: type=%s divider=%d/%d dual_face=%d/%d screen_share=%d/%d black_panel=%s",
        layout_type, len(divider_votes), 5, len(dual_face_votes), 5,
        len(screen_share_votes), 5, black_panel_side,
    )

    return {
        "layout_type": layout_type,
        "prefer_solo": layout_type != "split",
        "has_black_panel": has_black_panel,
        "black_panel_side": black_panel_side,
        "is_multi_active_frame": is_split_screen and not has_black_panel,
        "is_screen_share": is_screen_share,
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

    # Sample a representative frame for face detection
    mid_time = (start + end) / 2.0
    face_crop = None
    info = _get_video_dimensions(video_path)
    
    cmd = [
        "ffmpeg", "-y", "-ss", f"{mid_time:.3f}",
        "-i", video_path, "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-vf", f"scale={info['width']}:{info['height']}",
        "pipe:1"
    ]
    res = _run_cmd(cmd, timeout=10)
    if res and res.stdout:
        frame = np.frombuffer(res.stdout, dtype=np.uint8)
        if frame.size == info["width"] * info["height"] * 3:
            frame_bgr = frame.reshape((info["height"], info["width"], 3))
            face_crop = detect_face_crop(frame_bgr, info["width"], info["height"])

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

    # Layout decision matrix:
    # - solo / screen_share → use_solo_frame (center crop or vertical stack)
    # - split + black panel → crop to active panel
    # - split + both active (true dual-cam) → DROP
    # Screen share segments are ALWAYS kept (vertical stack mode)
    is_screen_share = layout.get("is_screen_share", False)
    
    should_use_solo = (
        layout["prefer_solo"] or
        layout.get("has_black_panel", False) or
        black["black_ratio"] > 0.6
    )
    
    # Only drop if CONFIRMED multi-active (divider + dual faces) or mostly black
    # NEVER drop screen share segments
    should_drop = (
        (layout.get("is_multi_active_frame", False) and not is_screen_share)
        or black.get("is_mostly_black", False)
    )
    
    # Screen share → force vertical stack (blurred bg + sharp center)
    use_vertical_stack = is_screen_share
    
    strategy = {
        "use_solo_frame": should_use_solo and not use_vertical_stack,
        "use_vertical_stack": use_vertical_stack,
        "has_black_panel": layout.get("has_black_panel", False),
        "black_panel_side": layout.get("black_panel_side"),
        "is_multi_active_frame": layout.get("is_multi_active_frame", False),
        "is_screen_share": is_screen_share,
        "should_drop": should_drop,
        "active_crop": face_crop,
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