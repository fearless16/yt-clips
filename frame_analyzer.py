import json
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from utils.config import load_config
from utils.logger import get_logger
from utils.face_matcher import find_host_in_frame

cfg = load_config()
log = get_logger("analyzer", cfg["logging"]["log_file"], cfg["logging"]["level"])

# Per-clip crop smoothing state — thread-local to prevent bleed across parallel exports
_crop_local = threading.local()

def _get_prev_crop_x() -> Optional[int]:
    """Get the previous crop X position for the current thread."""
    return getattr(_crop_local, 'prev_crop_x', None)

def _set_prev_crop_x(value: Optional[int]) -> None:
    """Set the previous crop X position for the current thread."""
    _crop_local.prev_crop_x = value

def _get_last_face_bbox() -> Optional[Tuple[int, int, int, int]]:
    """Get the last detected face bounding box for the current thread."""
    return getattr(_crop_local, 'last_face_bbox', None)

def _set_last_face_bbox(value: Optional[Tuple[int, int, int, int]]) -> None:
    """Set the last detected face bounding box for the current thread."""
    _crop_local.last_face_bbox = value

# Backward-compatible module-level accessor (read-only, for tests)
_PREV_CROP_X = None  # Only used by tests checking reset; actual state is in _crop_local

def reset_crop_state() -> None:
    """Call once at the start of each new clip to prevent crop history bleeding across clips."""
    global _PREV_CROP_X
    _crop_local.prev_crop_x = None
    _crop_local.last_face_bbox = None
    _PREV_CROP_X = None  # Keep module-level in sync for backward compat

def _smooth_int(prev: Optional[int], current: int, alpha: float = 0.25) -> int:
    if prev is None:
        return current
    return int(prev * (1.0 - alpha) + current * alpha)

def _detect_face_at_timestamp(video_path: str, timestamp: float, frame_w: int, frame_h: int) -> Optional[Dict]:
    cmd = [
        "ffmpeg", "-y", "-ss", f"{timestamp:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-vf", f"scale={frame_w}:{frame_h}",
        "pipe:1"
    ]
    res = _run_cmd(cmd, timeout=10)
    if not res or not res.stdout:
        return None
    frame = np.frombuffer(res.stdout, dtype=np.uint8)
    if frame.size != frame_w * frame_h * 3:
        return None
    frame_bgr = frame.reshape((frame_h, frame_w, 3))
    return detect_face_crop(frame_bgr, frame_w, frame_h)

def detect_face_crop(frame_bgr: np.ndarray, frame_width: int, frame_height: int) -> Optional[Dict]:
    # ── 0. Face-centered crop for YouTube Shorts style ───────────────────────
    # Reference (expectation.png): face at 30% from top, 30% headroom.
    # Crop region = face + 70% headroom above + body below.
    # This positions face at ~30% from top of crop, which after fill-crop
    # puts face at ~30% from top of final 1080x1920 frame.

    def _apply_top_padding(face_top_y: int, face_height: int, face_width: int = 0):
        """Return (crop_y, crop_h, crop_w) with face-centered positioning.
        Target: face at ~30% of output height from top (matching expectation.png).
        Crop is calculated from FACE SIZE, not height, to prevent oversized faces."""
        # Target: face should be ~25% of output width (1080px) = 270px
        target_face_w = 270
        if face_width > 0:
            # Calculate scale: how much to zoom in
            scale = target_face_w / face_width
            # Crop dimensions from source
            crop_w = int(1080 / scale)  # Source region width
            crop_h = int(1920 / scale)  # Source region height (9:16)
        else:
            # Fallback: use height-based calculation
            headroom = int(face_height * 0.80)
            body_below = int(face_height * 1.20)
            crop_h = face_height + headroom + body_below
            crop_w = int(crop_h * 9 / 16)
        
        # Don't exceed frame bounds
        crop_h = min(crop_h, frame_height)
        crop_w = min(crop_w, frame_width)
        
        # Position face at ~30% from top (matching expectation.png composition)
        # face_top_y - crop_h * 0.70 puts face at 30% from top
        crop_y = max(0, min(frame_height - crop_h, face_top_y - int(crop_h * 0.70)))
        
        return crop_y, crop_h, crop_w

    # ── 1. Dynamic Host Matching (Priority) ──────────────────────────────────
    # Try to find the host specifically using reference photos
    # Pass facecam bounds so detection can crop to the facecam region if full-frame fails
    layout_cfg = cfg.get("layout", {})
    facecam_bounds = layout_cfg.get("facecam") if layout_cfg.get("has_facecam") else None
    host_box = find_host_in_frame(frame_bgr, facecam_bounds=facecam_bounds)
    if host_box:
        hx, hy = host_box["x"], host_box["y"]
        hw, hh = host_box["width"], host_box["height"]
        
        crop_y, crop_h, crop_w = _apply_top_padding(hy, hh, hw)
        crop_x = hx + hw // 2 - crop_w // 2
        crop_x = max(0, min(crop_x, frame_width - crop_w))
        
        # Smooth and return
        crop_x = _smooth_int(_get_prev_crop_x(), crop_x, alpha=0.25)
        _set_prev_crop_x(crop_x)
        
        return {
            "x": int(crop_x), "y": int(crop_y),
            "width": int(crop_w), "height": int(crop_h),
            "face_w": int(hw), "face_h": int(hh),
            "face_x": int(hx), "face_y": int(hy),
            "confidence": host_box.get("confidence", 0.9),
            "is_dynamic_match": True
        }

    # ── 2. Fallback to DNN face detection ───────────────────
    from utils.face_detect import detect_faces
    faces = detect_faces(frame_bgr, score_threshold=0.5)
    if len(faces) == 0:
        return None

    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0

    # Load host encodings if available (M3.2 identity matching)
    host_encs = []
    try:
        from utils.face_matcher import get_host_encodings
        host_encs = get_host_encodings()
    except Exception:
        pass

    try:
        import face_recognition
    except ImportError:
        log.warning("face_recognition not installed — skipping face matching")
        return None

    best_face = None
    best_score = -1.0
    last_bbox = _get_last_face_bbox()

    for (x, y, w, h) in faces:
        is_host = False
        if host_encs:
            try:
                face_crop = frame_bgr[y:y+h, x:x+w]
                if face_crop.size > 0:
                    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    encs = face_recognition.face_encodings(rgb_crop)
                    if encs:
                        matches = face_recognition.compare_faces(host_encs, encs[0], tolerance=0.6)
                        if any(matches):
                            is_host = True
            except Exception:
                pass

        track_score = 0.0
        if last_bbox:
            track_score = _iou((x, y, w, h), last_bbox)

        area = float(w * h)
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        norm_x = center_x / float(frame_width)
        norm_y = center_y / float(frame_height)
        size_ratio = h / float(frame_height)
        score = area
        if size_ratio > 0.28:
            score *= 0.001
        if norm_y < 0.55 and 0.25 < norm_x < 0.75 and h < frame_height * 0.5:
            continue
        if 0.05 < size_ratio < 0.22:
            score *= 8.0
        if norm_y > 0.55:
            score *= 3.0
        if norm_x < 0.35 or norm_x > 0.65:
            score *= 3.0

        if is_host:
            score *= 20.0
        if track_score > 0.5:
            score *= 5.0

        if score > best_score:
            best_score = score
            best_face = (x, y, w, h)

    if best_face is None or best_score < 10:
        return None

    # Track this face in cheap path
    _set_last_face_bbox(best_face)

    x, y, w, h = best_face
    crop_y, crop_h, crop_w = _apply_top_padding(y, h, w)
    crop_x = x + w // 2 - crop_w // 2
    crop_x = max(0, min(crop_x, frame_width - crop_w))
    crop_x = _smooth_int(_get_prev_crop_x(), crop_x, alpha=0.25)
    _set_prev_crop_x(crop_x)

    return {
        "x": int(crop_x), "y": int(crop_y),
        "width": int(crop_w), "height": int(crop_h),
        "face_w": int(w), "face_h": int(h),
        "face_x": int(x), "face_y": int(y),
        "confidence": min(1.0, best_score / 5000.0),
    }

def _run_cmd(cmd, timeout=15):
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning("Command timeout: %s", " ".join(cmd))
    except Exception as e:
        log.warning("Command failed: %s", e)
    return None

def save_preview_snapshot(video_path: str, timestamp: float, output_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-ss", f"{timestamp:.3f}",
        "-i", video_path, "-frames:v", "1", "-q:v", "2", output_path
    ]
    res = _run_cmd(cmd, timeout=10)
    return res is not None and res.returncode == 0

def _frame_stats(frame: bytes) -> Dict:
    if not frame:
        return {"avg": 128.0, "var": 0.0}
    arr = np.frombuffer(frame, dtype=np.uint8).astype(np.float32)
    return {"avg": float(arr.mean()), "var": float(arr.var())}

def _sample_brightness_series(video_path: str, start: float, end: float, sample_count: int = 5) -> List[Dict]:
    duration = max(0.01, end - start)
    sample_count = max(1, min(sample_count, 10))
    frame_size = 160 * 90

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", video_path,
        "-vf", f"fps={sample_count}/{duration:.3f},scale=160:90,format=gray",
        "-frames:v", str(sample_count),
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    res = _run_cmd(cmd, timeout=20)
    if not res or res.returncode != 0 or not res.stdout:
        return []

    data = res.stdout
    frames = min(sample_count, len(data) // frame_size)
    return [_frame_stats(data[i * frame_size:(i + 1) * frame_size]) for i in range(frames)]

def detect_black_frames(samples: List[Dict]) -> Dict:
    threshold = cfg.get("quality", {}).get("black_threshold", 20)
    black = [s for s in samples if isinstance(s, dict) and s.get("avg", 255) < threshold and s.get("var", 9999) < 50]
    return {
        "has_black_frames": len(black) > 0,
        "black_ratio": len(black) / len(samples) if samples else 0,
        "is_mostly_black": len(black) / len(samples) > 0.6 if samples else False,
    }

def analyze_lighting(samples: List[Dict]) -> Dict:
    if not samples:
        return {"needs_correction": False}
    avgs = [s.get("avg", 128) for s in samples if isinstance(s, dict)]
    if not avgs:
        return {"needs_correction": False}
    avg = sum(avgs) / len(avgs)
    quality_cfg = cfg.get("quality", {})
    backlit_thresh = quality_cfg.get("backlit_brightness_threshold", 80)
    overexposed_thresh = quality_cfg.get("overexposed_brightness_threshold", 210)
    if avg < backlit_thresh:
        return {"needs_correction": True, "lighting_filter": "eq=gamma=1.3:contrast=1.1"}
    elif avg > overexposed_thresh:
        return {"needs_correction": True, "lighting_filter": "eq=gamma=0.8:contrast=0.9"}
    return {"needs_correction": False}

def _has_vertical_divider(frame_array: np.ndarray, width: int) -> bool:
    center = width // 2
    band_width = 3
    left_band = frame_array[:, max(0, center - band_width - 5):max(0, center - band_width)]
    right_band = frame_array[:, center + band_width:center + band_width + 5]
    center_band = frame_array[:, center - band_width:center + band_width]
    if left_band.size == 0 or right_band.size == 0 or center_band.size == 0:
        return False
    center_avg = float(center_band.mean())
    left_avg = float(left_band.mean())
    right_avg = float(right_band.mean())
    left_right_avg = (left_avg + right_avg) / 2
    has_bright_divider = center_avg > left_right_avg + 25  # white divider
    has_dark_divider = center_avg < left_right_avg - 25     # dark divider
    has_sharp_edge = abs(left_avg - right_avg) > 40
    return has_bright_divider or has_dark_divider or has_sharp_edge


def _detect_chat_overlay(frame_array: np.ndarray, width: int, height: int, cfg: dict) -> Optional[Dict]:
    """
    Detect chat overlay in the frame.
    Chat typically appears on the right or left side of streams.
    Returns dict with chat_x, chat_width if detected.
    """
    layout_cfg = cfg.get("layout", {})
    if not layout_cfg.get("has_chat_overlay", False):
        return None

    chat_side = layout_cfg.get("chat", {}).get("side", "right")
    est_width = layout_cfg.get("chat", {}).get("estimated_width", 350)
    bright_thresh = layout_cfg.get("chat", {}).get("brightness_threshold", 30)

    # Sample the edge region (right or left 25% of frame)
    sample_width = min(width // 4, est_width)
    if chat_side == "right":
        edge_region = frame_array[:, width - sample_width:]
    else:
        edge_region = frame_array[:, :sample_width]

    if edge_region.size == 0:
        return None

    edge_avg = float(edge_region.mean())
    edge_var = float(edge_region.var())

    # Center region for comparison (middle 50%)
    center_region = frame_array[:, width // 4:3 * width // 4]
    center_avg = float(center_region.mean()) if center_region.size > 0 else edge_avg

    # Chat detection heuristics:
    # 1. Edge region is significantly different in brightness from center
    # 2. High variance (text/details) in chat area vs smooth gameplay
    brightness_diff = abs(edge_avg - center_avg)

    is_chat_like = (
        brightness_diff > bright_thresh
        and edge_var > 500
        and (edge_var < center_region.var() * 2 if center_region.size > 0 else True)
    )

    if is_chat_like:
        chat_x = width - sample_width if chat_side == "right" else 0
        return {
            "chat_detected": True,
            "chat_side": chat_side,
            "chat_x": chat_x,
            "chat_width": sample_width,
            "brightness_diff": brightness_diff
        }

    return None

def _detect_faces_per_half(frame_array: np.ndarray, width: int) -> dict:
    # frame_array is 320x180 grayscale from ffmpeg pipe
    # Convert to BGR 3-channel for DNN, resize to 300x300 for better detection
    gray_3ch = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2BGR)
    left_half = gray_3ch[:, :width // 2]
    right_half = gray_3ch[:, width // 2:]
    from utils.face_detect import detect_faces
    left_faces = detect_faces(left_half, score_threshold=0.3)
    right_faces = detect_faces(right_half, score_threshold=0.3)
    return {"left": len(left_faces), "right": len(right_faces)}

def _is_screen_share_frame(frame_array: np.ndarray) -> bool:
    overall_var = float(frame_array.var())
    grad = np.abs(np.diff(frame_array.astype(np.int16), axis=1))
    edge_density = float(grad.mean())
    bright_pixels = np.sum(frame_array > 210)
    dark_pixels = np.sum(frame_array < 40)
    contrast_ratio = (bright_pixels + dark_pixels) / frame_array.size
    return bool(overall_var > 500 and edge_density > 6 and contrast_ratio > 0.18)

def detect_layout(video_path: str, start: float, end: float) -> Dict:
    """
    Layout decision table:
      solo                → crop + scale to 9:16 (you solo in frame)
      split + both cams   → vertical stack (guest has video)
      split + one black   → crop to active half only (guest cam off → DROP)
      screen share        → vertical stack (blurred bg + sharp content)
    """
    duration = max(0.01, end - start)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", video_path,
        "-vf", f"fps=12/{duration:.3f},scale=320:180,format=gray",
        "-frames:v", "12",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"
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
            frame_data = data[i * frame_size:(i + 1) * frame_size]
            frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(180, 320)
            left_half = frame_array[:, :160]
            right_half = frame_array[:, 160:]
            left_avg = float(left_half.mean())
            right_avg = float(right_half.mean())
            left_var = float(left_half.var())
            right_var = float(right_half.var())

            right_is_black = right_avg < 25 and right_var < 100
            left_is_black = left_avg < 25 and left_var < 100

            if right_is_black:
                black_panel_votes.append(True)
                black_panel_side_votes.append("right")
            if left_is_black:
                black_panel_votes.append(True)
                black_panel_side_votes.append("left")

            if _has_vertical_divider(frame_array, 320):
                divider_votes.append(True)

            face_counts = _detect_faces_per_half(frame_array, 320)
            if face_counts["left"] > 0 and face_counts["right"] > 0:
                dual_face_votes.append(True)

            if _is_screen_share_frame(frame_array):
                screen_share_votes.append(True)

    has_black_panel = len(black_panel_votes) >= 2
    has_divider = len(divider_votes) >= 2
    has_dual_faces = len(dual_face_votes) >= 2

    # True split-screen: divider present AND faces on both halves
    is_split_screen = has_divider and has_dual_faces

    # Screen share: high edge density but no split-screen
    is_screen_share = len(screen_share_votes) >= 3 and not is_split_screen

    black_panel_side = None
    if has_black_panel:
        right_count = black_panel_side_votes.count("right")
        left_count = black_panel_side_votes.count("left")
        black_panel_side = "right" if right_count >= left_count else "left"

    # ── Layout classification ─────────────────────────────────────────────────
    # split + both active  → guest has video  → vertical stack (KEEP)
    # split + black panel  → guest cam off    → crop active side only (DROP)
    # screen share         → presentation     → vertical stack (KEEP)
    # solo                 → just you         → center crop 9:16 (KEEP)
    if is_screen_share:
        layout_type = "screen_share"
    elif is_split_screen and not has_black_panel:
        layout_type = "split_both_active"   # guest has video → vertical stack
    elif is_split_screen and has_black_panel:
        layout_type = "split_guest_off"     # guest cam off   → drop
    else:
        layout_type = "solo"

    log.debug(
        "Layout: %s | divider=%d dual_face=%d screen_share=%d black_panel=%s",
        layout_type, len(divider_votes), len(dual_face_votes),
        len(screen_share_votes), black_panel_side,
    )

    # ── Chat overlay detection ─────────────────────────────────────────────────
    chat_overlay = None
    chat_cfg = cfg.get("layout", {}).get("chat", {})
    if chat_cfg and chat_cfg.get("side"):
        # Re-run analysis with chat detection
        chat_side = chat_cfg.get("side", "right")
        est_width = chat_cfg.get("estimated_width", 350)
        bright_thresh = chat_cfg.get("brightness_threshold", 30)

        # Use the first frame for chat detection
        if res and res.stdout:
            sample_width = min(320 // 4, int(est_width * (320 / 1920)))
            frame_data = res.stdout[:320 * 180]
            if len(frame_data) >= 320 * 180:
                frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(180, 320)
                if chat_side == "right":
                    edge_region = frame_array[:, 320 - sample_width:]
                else:
                    edge_region = frame_array[:, :sample_width]

                if edge_region.size > 0:
                    edge_avg = float(edge_region.mean())
                    center_region = frame_array[:, 80:240]
                    center_avg = float(center_region.mean()) if center_region.size > 0 else edge_avg
                    brightness_diff = abs(edge_avg - center_avg)

                    if brightness_diff > bright_thresh and float(edge_region.var()) > 500:
                        chat_overlay = {
                            "chat_detected": True,
                            "chat_side": chat_side,
                            "chat_x": 320 - sample_width if chat_side == "right" else 0,
                            "chat_width": sample_width,
                        }

    return {
        "layout_type": layout_type,
        "prefer_solo": layout_type == "solo",
        "has_black_panel": has_black_panel,
        "black_panel_side": black_panel_side,
        "guest_cam_off": layout_type == "split_guest_off",
        "guest_cam_on": layout_type == "split_both_active",
        "is_screen_share": is_screen_share,
        "chat_overlay": chat_overlay,
    }

def _get_video_dimensions(video_path: str) -> Dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", video_path,
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
        return {"width": int(s.get("width", 1920)), "height": int(s.get("height", 1080))}
    except Exception:
        return {"width": 1920, "height": 1080}

def detect_dead_air(video_path: str, start: float, end: float) -> Dict:
    duration = max(0.01, end - start)
    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
        "-i", video_path,
        "-af", "silencedetect=noise=-35dB:d=1",
        "-f", "null", "-"
    ]
    res = _run_cmd(cmd, timeout=30)
    if not res:
        return {"has_dead_air": False}
    silence_count = res.stderr.decode(errors="replace").count("silence_start")
    return {"has_dead_air": silence_count > 0}

def analyze_clip(
    video_path: str,
    start: float,
    end: float,
    transcript_segments: Optional[List[Dict]] = None,
    clip_id: str = "clip"
) -> Dict:
    log.info("[%s] Analyzing clip...", clip_id)

    # Reset temporal crop smoothing — prevents history bleeding from previous clips
    reset_crop_state()

    samples = _sample_brightness_series(video_path, start, end)
    black = detect_black_frames(samples)
    lighting = analyze_lighting(samples)
    layout = detect_layout(video_path, start, end)
    silence = detect_dead_air(video_path, start, end)

    info = _get_video_dimensions(video_path)
    
    # ── Facecam / Host Location ────────────────────────────────────────────────
    layout_cfg = cfg.get("layout", {})
    has_facecam = layout_cfg.get("has_facecam", False)
    
    margin = min(0.5, (end - start) * 0.1)
    sample_times = np.linspace(start + margin, end - margin, num=5)
    face_candidates = []
    
    for t in sample_times:
        face = _detect_face_at_timestamp(video_path, float(t), info["width"], info["height"])
        if face:
            face_candidates.append(face)

    face_crop = None
    host_absent = False

    # 1. Prioritize any candidate that is a dynamic match (matched reference photo)
    dynamic_matches = [f for f in face_candidates if f.get("is_dynamic_match")]
    if dynamic_matches:
        log.info("[%s] Found dynamic face match using reference photos", clip_id)
        face_crop = max(dynamic_matches, key=lambda f: f.get("confidence", 0.0))
    
    # 2. If no dynamic match, but we have a facecam config, check for presence there
    elif has_facecam:
        fc = layout_cfg.get("facecam", {})
        log.info("[%s] No dynamic match; validating config facecam bounds", clip_id)
        fc_x, fc_y = fc.get("x", 0), fc.get("y", 0)
        fc_w, fc_h = fc.get("width", 320), fc.get("height", 180)
        
        for f in face_candidates:
            fx, fy = f.get("face_x", f.get("x", 0)), f.get("face_y", f.get("y", 0))
            if (fc_x - 50) <= fx <= (fc_x + fc_w + 50) and (fc_y - 50) <= fy <= (fc_y + fc_h + 50):
                # Valid face found in config area
                face_crop = f
                break
        
        if not face_crop:
            log.warning("[%s] Host absent from configured facecam bounds", clip_id)
            host_absent = True
    
    # 3. Last fallback: just pick the best detected face overall
    elif face_candidates:
        face_crop = max(face_candidates, key=lambda f: (f.get("face_w", 0) * f.get("face_h", 0)) * f.get("confidence", 0.0))


    is_fast_paced = False
    silence_ratio = 0.0
    if transcript_segments:
        clip_segments = [s for s in transcript_segments if s["start"] < end and s["end"] > start]
        total_text = " ".join([s["text"] for s in clip_segments])
        words = len(total_text.split())
        duration = end - start
        wpm = (words / (duration / 60)) if duration > 0 else 0
        is_fast_paced = wpm > 150

        # Analyze silence for variable speed
        # Estimate silence based on words: avg 0.35s per word, rest is silence
        speech_duration = words * 0.35
        silence_ratio = max(0.0, (duration - speech_duration) / duration) if duration > 0 else 0.0

        log.debug("[%s] Tempo: %.1f WPM (fast=%s), Silence: %.1f%%", clip_id, wpm, is_fast_paced, silence_ratio * 100)

    layout_type = layout.get("layout_type", "solo")
    is_screen_share = layout.get("is_screen_share", False)
    guest_cam_on = layout.get("guest_cam_on", False)
    guest_cam_off = layout.get("guest_cam_off", False)

    # ── Screen-share priority rule ────────────────────────────────────────────
    # Screen-share detection takes priority OVER face detection.
    # Only override screen-share back to solo if the face is dominant (>25% of
    # frame height) — meaning it's clearly a facecam-only shot misclassified as
    # screen-share, NOT a genuine presentation with a small facecam overlay.
    if is_screen_share and face_crop:
        face_dominance = face_crop.get("face_h", 0) / max(info["height"], 1)
        if face_dominance > 0.25:
            # Face is so large it occupies >25% height — definitely not screen-share
            log.debug("[%s] Screen-share overridden: face dominance=%.2f", clip_id, face_dominance)
            is_screen_share = False
            layout_type = "solo"
            layout["prefer_solo"] = True
        else:
            # Keep screen-share classification regardless of face presence
            log.debug("[%s] Screen-share kept (face dominance=%.2f < 0.25)", clip_id, face_dominance)

    should_drop = False
    use_vertical_stack = False
    use_solo_frame = False

    if guest_cam_on:
        # Guest has live video → vertical stack (side-by-side → stacked 9:16)
        use_vertical_stack = True
        should_drop = False
        log.info("[%s] Guest cam ON → vertical stack", clip_id)

    elif guest_cam_off:
        # Guest cam off (black panel) → drop this clip entirely
        should_drop = True
        log.info("[%s] Guest cam OFF (black panel) → DROP", clip_id)

    elif is_screen_share:
        # Screen share → preserve the content inside a vertical Shorts canvas.
        # Export keeps is_screen_share=true so export.py can use the readable
        # blurred-background layout instead of a tight center crop.
        use_vertical_stack = False
        use_solo_frame = False
        should_drop = False
        log.info("[%s] Screen share → 9:16 shorts canvas", clip_id)

    else:
        # Solo frame (just you)
        use_solo_frame = face_crop is not None
        should_drop = (
            black.get("is_mostly_black", False)
            or face_crop is None
        )
        if should_drop:
            log.info("[%s] Solo but no face detected → DROP", clip_id)

    # ── Chat overlay handling for 9:16 crop ────────────────────────────────────
    chat_overlay = layout.get("chat_overlay")
    exclude_chat_from_crop = False
    crop_adjustment = None

    if chat_overlay and chat_overlay.get("chat_detected"):
        exclude_chat_from_crop = True
        # Calculate chat exclusion zone
        chat_side = chat_overlay.get("chat_side", "right")
        chat_width = chat_overlay.get("chat_width", 0)
        # Scale to full resolution
        scale_factor = info["width"] / 320
        crop_adjustment = {
            "exclude_chat": True,
            "chat_side": chat_side,
            "chat_exclude_width": int(chat_width * scale_factor)
        }
        log.info("[%s] Chat detected on %s — will exclude from 9:16 crop", clip_id, chat_side)

    # ── Variable Speed Logic ────────────────────────────────────────────────────
    # Analyze silence ratio to determine optimal speed
    # High silence (dead air) → speed up more aggressively
    # Fast speech → slight speed up to increase retention
    # Normal speech → use global speed factor
    if silence_ratio > 0.5:
        speed_factor = 1.35  # Aggressive speed up for dead air
    elif silence_ratio > 0.3:
        speed_factor = 1.25
    elif is_fast_paced:
        speed_factor = 1.15  # Slight speed up for fast speech
    else:
        speed_factor = 1.0  # Normal speed (global config will multiply later)

    log.info("[%s] Variable speed: %.2fx (silence=%.1f%%, fast_paced=%s)",
            clip_id, speed_factor, silence_ratio * 100, is_fast_paced)

    strategy = {
        "use_solo_frame": use_solo_frame,
        "use_vertical_stack": use_vertical_stack,
        "has_black_panel": layout.get("has_black_panel", False),
        "black_panel_side": layout.get("black_panel_side"),
        "guest_cam_on": guest_cam_on,
        "guest_cam_off": guest_cam_off,
        "is_screen_share": is_screen_share,
        "should_drop": should_drop,
        "active_crop": face_crop,
        "speed_factor": speed_factor,
        "apply_lighting_fix": lighting["needs_correction"],
        "lighting_filter": lighting.get("lighting_filter", ""),
        "skip_silence": silence["has_dead_air"],
        "export_aspect_ratio": "9:16",
        "chat_exclusion": crop_adjustment,
    }

    return {
        "black_frames": black,
        "lighting": lighting,
        "layout": layout,
        "dead_air": silence,
        "export_strategy": strategy,
    }
