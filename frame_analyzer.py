"""
frame_analyzer.py — Intelligent Frame Analysis Engine.

Analyses source video frames to make smart export decisions:
  1. Black screen detection (skip dead frames)
  2. Multi-frame / solo-frame detection
  3. Backlight / lighting issue detection
  4. Dead air (silence) detection
  5. Tempo / pacing analysis for speed adjustment
  6. Preview snapshot generation for QA

Usage:
    from frame_analyzer import analyze_clip
    result = analyze_clip("input/video.mp4", start_sec=95.0, end_sec=125.0)
"""

import json
import math
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("analyzer", cfg["logging"]["log_file"], cfg["logging"]["level"])


# ─── Frame Sampling via FFmpeg ────────────────────────────────────────────────

def _extract_frame_stats(video_path: str, timestamp: float) -> Dict:
    """
    Extract frame statistics at a given timestamp using FFmpeg signalstats.
    Returns brightness (YAVG), darkness ratio, and saturation info.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-select_streams", "v:0",
        "-show_entries", "frame_tags=lavfi.signalstats.YAVG,lavfi.signalstats.YLOW,lavfi.signalstats.YHIGH",
        "-of", "json",
    ]
    # signalstats needs a filter — use ffmpeg instead
    cmd2 = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", "signalstats=stat=tout+vrep+brng,metadata=mode=print",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd2, capture_output=True, text=True, timeout=10)
        output = result.stderr
        stats = {}
        for line in output.split("\n"):
            if "lavfi.signalstats" in line:
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip().split(".")[-1]
                    try:
                        stats[key] = float(parts[1].strip())
                    except ValueError:
                        pass
        return stats
    except Exception as e:
        log.debug("signalstats failed at %.1f: %s", timestamp, e)
        return {}


def _get_frame_brightness(video_path: str, timestamp: float) -> float:
    """
    Get average brightness of a frame using FFmpeg's blackdetect-style analysis.
    Returns value 0.0 (black) to 255.0 (white).
    """
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", "format=gray,stats",
        "-f", "null", "-"
    ]
    # More reliable: use crop to sample center + compute mean
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", "format=yuv420p,cropdetect=round=2:reset=1",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        # Parse cropdetect output for brightness hints
        # Fallback: use thumbnail + raw pixel sampling
        return _sample_brightness_raw(video_path, timestamp)
    except Exception:
        return 128.0  # Assume mid-brightness on failure


def _sample_brightness_raw(video_path: str, timestamp: float) -> float:
    """
    Extract a single frame as raw grayscale pixels and compute mean brightness.
    Most reliable method — no parsing FFmpeg text output.
    """
    temp_dir = Path(cfg["paths"]["temp"])
    temp_dir.mkdir(parents=True, exist_ok=True)
    raw_path = str(temp_dir / "frame_sample.gray")

    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", "scale=160:90,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray",
        raw_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        raw = Path(raw_path)
        if raw.exists() and raw.stat().st_size > 0:
            data = raw.read_bytes()
            avg = sum(data) / len(data)
            raw.unlink(missing_ok=True)
            return avg
    except Exception as e:
        log.debug("Raw brightness sampling failed: %s", e)

    return 128.0


def _save_preview_snapshot(video_path: str, timestamp: float, output_path: str) -> Optional[str]:
    """Save a preview snapshot (JPEG) at the given timestamp."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if Path(output_path).exists():
            return output_path
    except Exception:
        pass
    return None


# ─── Black Screen Detection ──────────────────────────────────────────────────

def detect_black_frames(
    video_path: str, start: float, end: float, sample_count: int = 5
) -> Dict:
    """
    Sample frames across the clip and detect black/near-black frames.

    Returns:
        {
            "has_black_frames": bool,
            "black_ratio": float (0-1),
            "samples": [{"time": float, "brightness": float, "is_black": bool}, ...]
        }
    """
    quality_cfg = cfg.get("quality", {})
    black_threshold = quality_cfg.get("black_threshold", 20)

    duration = end - start
    step = duration / (sample_count + 1)
    samples = []

    for i in range(1, sample_count + 1):
        t = start + step * i
        brightness = _sample_brightness_raw(video_path, t)
        is_black = brightness < black_threshold
        samples.append({"time": t, "brightness": round(brightness, 1), "is_black": is_black})

    black_count = sum(1 for s in samples if s["is_black"])

    return {
        "has_black_frames": black_count > 0,
        "black_ratio": black_count / len(samples) if samples else 0,
        "samples": samples,
    }


# ─── Lighting Analysis ───────────────────────────────────────────────────────

def analyze_lighting(
    video_path: str, start: float, end: float, sample_count: int = 5
) -> Dict:
    """
    Detect backlit / underexposed scenes.

    Strategy: Sample multiple frames, check if average brightness is low
    (backlit = subject dark, background bright → overall dark with high contrast).

    Returns:
        {
            "avg_brightness": float,
            "needs_correction": bool,
            "correction_type": str ("backlit" | "underexposed" | "overexposed" | "normal"),
            "gamma_boost": float,  # suggested gamma correction value
            "exposure_boost": float,  # suggested exposure EV adjustment
        }
    """
    quality_cfg = cfg.get("quality", {})
    backlit_threshold = quality_cfg.get("backlit_brightness_threshold", 80)
    overexposed_threshold = quality_cfg.get("overexposed_brightness_threshold", 210)

    duration = end - start
    step = duration / (sample_count + 1)
    brightnesses = []

    for i in range(1, sample_count + 1):
        t = start + step * i
        b = _sample_brightness_raw(video_path, t)
        brightnesses.append(b)

    avg_b = sum(brightnesses) / len(brightnesses) if brightnesses else 128

    if avg_b < backlit_threshold:
        # Backlit or underexposed — boost gamma and exposure
        # The darker it is, the more we boost
        gamma_boost = min(2.0, max(1.1, 128.0 / max(avg_b, 1)))
        exposure_boost = min(1.5, max(0.2, (128 - avg_b) / 100.0))
        correction_type = "backlit" if avg_b < 60 else "underexposed"
    elif avg_b > overexposed_threshold:
        gamma_boost = max(0.6, min(0.95, avg_b / 255.0))
        exposure_boost = -0.3
        correction_type = "overexposed"
    else:
        gamma_boost = 1.0
        exposure_boost = 0.0
        correction_type = "normal"

    return {
        "avg_brightness": round(avg_b, 1),
        "needs_correction": correction_type != "normal",
        "correction_type": correction_type,
        "gamma_boost": round(gamma_boost, 2),
        "exposure_boost": round(exposure_boost, 2),
    }


# ─── Dead Air / Silence Detection ────────────────────────────────────────────

def detect_dead_air(
    video_path: str, start: float, end: float
) -> Dict:
    """
    Detect silence segments within a clip using FFmpeg silencedetect.

    Returns:
        {
            "has_dead_air": bool,
            "silence_ratio": float (0-1),
            "silence_segments": [{"start": float, "end": float, "duration": float}, ...],
            "active_segments": [{"start": float, "end": float}, ...],
        }
    """
    quality_cfg = cfg.get("quality", {})
    silence_threshold_db = quality_cfg.get("silence_threshold_db", -35)
    min_silence_duration = quality_cfg.get("min_silence_duration", 1.0)

    duration = end - start

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-t", str(duration),
        "-i", video_path,
        "-af", f"silencedetect=noise={silence_threshold_db}dB:d={min_silence_duration}",
        "-f", "null", "-"
    ]

    silence_segments = []
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stderr = result.stderr

        # Parse silencedetect output
        current_start = None
        for line in stderr.split("\n"):
            if "silence_start:" in line:
                try:
                    val = line.split("silence_start:")[1].strip().split()[0]
                    current_start = float(val) + start  # offset to absolute time
                except (ValueError, IndexError):
                    pass
            elif "silence_end:" in line and current_start is not None:
                try:
                    parts = line.split("silence_end:")[1].strip().split()
                    silence_end = float(parts[0]) + start
                    silence_dur = float(parts[-1]) if len(parts) > 1 else silence_end - current_start
                    silence_segments.append({
                        "start": round(current_start, 2),
                        "end": round(silence_end, 2),
                        "duration": round(silence_dur, 2),
                    })
                    current_start = None
                except (ValueError, IndexError):
                    current_start = None

    except Exception as e:
        log.warning("Silence detection failed: %s", e)

    # Build active (non-silent) segments
    total_silence = sum(s["duration"] for s in silence_segments)
    silence_ratio = total_silence / duration if duration > 0 else 0

    active_segments = _compute_active_segments(start, end, silence_segments)

    return {
        "has_dead_air": silence_ratio > 0.15,
        "silence_ratio": round(silence_ratio, 3),
        "silence_segments": silence_segments,
        "active_segments": active_segments,
    }


def _compute_active_segments(
    start: float, end: float, silence_segments: List[Dict]
) -> List[Dict]:
    """Compute non-silent segments by subtracting silence from [start, end]."""
    if not silence_segments:
        return [{"start": start, "end": end}]

    active = []
    cursor = start
    for s in sorted(silence_segments, key=lambda x: x["start"]):
        if s["start"] > cursor:
            active.append({"start": round(cursor, 2), "end": round(s["start"], 2)})
        cursor = max(cursor, s["end"])

    if cursor < end:
        active.append({"start": round(cursor, 2), "end": round(end, 2)})

    # Filter out tiny active segments (< 0.5s)
    return [a for a in active if a["end"] - a["start"] >= 0.5]


# ─── Speech Tempo Analysis ───────────────────────────────────────────────────

def analyze_tempo(
    transcript_segments: List[Dict], start: float, end: float
) -> Dict:
    """
    Analyze speech tempo within a clip window.
    Determine if clip feels slow and recommend speed adjustment.

    Returns:
        {
            "wpm": float,
            "feels_slow": bool,
            "recommended_speed": float (1.0 or 1.25),
            "speech_density": float (0-1, ratio of time with speech),
        }
    """
    quality_cfg = cfg.get("quality", {})
    slow_wpm_threshold = quality_cfg.get("slow_wpm_threshold", 100)

    # Get transcript segments overlapping this window
    overlapping = []
    for seg in transcript_segments:
        if seg["end"] > start and seg["start"] < end:
            overlapping.append(seg)

    if not overlapping:
        return {
            "wpm": 0, "feels_slow": True,
            "recommended_speed": 1.25, "speech_density": 0.0,
        }

    # Calculate total words and speech duration
    total_words = 0
    total_speech_time = 0.0
    for seg in overlapping:
        seg_start = max(seg["start"], start)
        seg_end = min(seg["end"], end)
        seg_dur = seg_end - seg_start
        text = seg.get("text", "")
        words = len(text.split())
        total_words += words
        total_speech_time += seg_dur

    clip_duration = end - start
    speech_density = total_speech_time / clip_duration if clip_duration > 0 else 0

    # Words per minute
    wpm = (total_words / total_speech_time * 60) if total_speech_time > 0 else 0

    # Determine if it feels slow
    # Very low WPM (<60) = always speed up regardless of density
    # Low WPM (<100) + low density (<60%) = speed up
    feels_slow = (wpm < slow_wpm_threshold * 0.6) or \
                 (wpm < slow_wpm_threshold and speech_density < 0.6)

    return {
        "wpm": round(wpm, 1),
        "feels_slow": feels_slow,
        "recommended_speed": 1.25 if feels_slow else 1.0,
        "speech_density": round(speech_density, 3),
    }


# ─── Multi-Frame / Layout Detection ──────────────────────────────────────────

def detect_layout(
    video_path: str, start: float, end: float
) -> Dict:
    """
    Detect if the stream has multiple frames/panels (e.g., gameplay + facecam,
    split screen, picture-in-picture).

    Strategy:
    1. Sample frames and check for distinct regions (brightness boundaries)
    2. Detect horizontal splits (side-by-side frames)
    3. Check if one panel is black/dead

    Returns:
        {
            "layout_type": str ("solo" | "split_horizontal" | "split_vertical" | "pip"),
            "has_dead_panel": bool,
            "active_region": dict or None,  # crop coords for the active panel
            "prefer_solo": bool,
        }
    """
    # Sample a frame from the middle of the clip
    mid_time = (start + end) / 2
    return _analyze_frame_layout(video_path, mid_time)


def _analyze_frame_layout(video_path: str, timestamp: float) -> Dict:
    """
    Analyze a single frame to detect multi-panel layout.

    Uses horizontal and vertical strip brightness analysis to find
    panel boundaries.
    """
    temp_dir = Path(cfg["paths"]["temp"])
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Get source dimensions
    info = _get_video_dimensions(video_path)
    sw, sh = info["width"], info["height"]

    # Extract a low-res frame for analysis (fast)
    analysis_w, analysis_h = 320, 180
    raw_path = str(temp_dir / "layout_frame.gray")

    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", f"scale={analysis_w}:{analysis_h},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray",
        raw_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        raw = Path(raw_path)
        if not raw.exists() or raw.stat().st_size == 0:
            return _default_layout()

        data = raw.read_bytes()
        raw.unlink(missing_ok=True)

        # Analyze vertical strips (for horizontal split detection)
        left_half = []
        right_half = []
        mid_col = analysis_w // 2

        for row in range(analysis_h):
            for col in range(analysis_w):
                pixel = data[row * analysis_w + col]
                if col < mid_col:
                    left_half.append(pixel)
                else:
                    right_half.append(pixel)

        left_avg = sum(left_half) / len(left_half) if left_half else 128
        right_avg = sum(right_half) / len(right_half) if right_half else 128

        # Analyze horizontal strips (for vertical split detection)
        top_half = []
        bottom_half = []
        mid_row = analysis_h // 2

        for row in range(analysis_h):
            for col in range(analysis_w):
                pixel = data[row * analysis_w + col]
                if row < mid_row:
                    top_half.append(pixel)
                else:
                    bottom_half.append(pixel)

        top_avg = sum(top_half) / len(top_half) if top_half else 128
        bottom_avg = sum(bottom_half) / len(bottom_half) if bottom_half else 128

        black_thresh = cfg.get("quality", {}).get("black_threshold", 20)

        # Check for horizontal split with one dead panel
        has_horiz_split = False
        dead_side = None
        if left_avg < black_thresh and right_avg > black_thresh + 30:
            has_horiz_split = True
            dead_side = "left"
        elif right_avg < black_thresh and left_avg > black_thresh + 30:
            has_horiz_split = True
            dead_side = "right"
        elif abs(left_avg - right_avg) > 60:
            # Significant brightness difference — likely split screen
            has_horiz_split = True

        # Check for vertical split (less common in gaming streams)
        has_vert_split = False
        if abs(top_avg - bottom_avg) > 60:
            has_vert_split = True

        # Determine layout
        if has_horiz_split and dead_side:
            # One panel is black — use solo mode with the active panel
            active_x = sw // 2 if dead_side == "left" else 0
            return {
                "layout_type": "split_horizontal_dead",
                "has_dead_panel": True,
                "dead_side": dead_side,
                "active_region": {
                    "x": active_x, "y": 0,
                    "width": sw // 2, "height": sh,
                },
                "prefer_solo": True,
                "left_brightness": round(left_avg, 1),
                "right_brightness": round(right_avg, 1),
            }
        elif has_horiz_split:
            return {
                "layout_type": "split_horizontal",
                "has_dead_panel": False,
                "active_region": None,
                "prefer_solo": False,
                "left_brightness": round(left_avg, 1),
                "right_brightness": round(right_avg, 1),
            }

        return _default_layout()

    except Exception as e:
        log.warning("Layout detection failed: %s", e)
        return _default_layout()


def _default_layout() -> Dict:
    return {
        "layout_type": "solo",
        "has_dead_panel": False,
        "active_region": None,
        "prefer_solo": True,
    }


def _get_video_dimensions(video_path: str) -> Dict:
    """Get source video width and height."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        return {"width": stream["width"], "height": stream["height"]}
    except Exception:
        return {"width": 1920, "height": 1080}


# ─── Comprehensive Clip Analysis ─────────────────────────────────────────────

def analyze_clip(
    video_path: str,
    start: float,
    end: float,
    transcript_segments: Optional[List[Dict]] = None,
    save_preview: bool = True,
    clip_id: str = "clip",
) -> Dict:
    """
    Run full analysis on a clip segment. Returns all decisions needed for export.

    Returns:
        {
            "clip_id": str,
            "layout": {...},
            "lighting": {...},
            "dead_air": {...},
            "tempo": {...},
            "black_frames": {...},
            "preview_path": str or None,
            "export_strategy": {
                "use_solo_frame": bool,
                "active_crop": dict or None,
                "speed_factor": float,
                "apply_lighting_fix": bool,
                "lighting_filter": str,
                "skip_silence": bool,
                "active_segments": list,
            }
        }
    """
    log.info("[%s] ═══ Frame Analysis Engine ═══", clip_id)

    # 1. Layout detection
    log.info("[%s] Detecting frame layout...", clip_id)
    layout = detect_layout(video_path, start, end)
    log.info("[%s]   Layout: %s | Dead panel: %s",
             clip_id, layout["layout_type"], layout["has_dead_panel"])

    # 2. Black frame detection
    log.info("[%s] Scanning for black frames...", clip_id)
    black_frames = detect_black_frames(video_path, start, end)
    log.info("[%s]   Black frames: %s (ratio=%.1f%%)",
             clip_id, black_frames["has_black_frames"],
             black_frames["black_ratio"] * 100)

    # 3. Lighting analysis
    log.info("[%s] Analyzing lighting conditions...", clip_id)
    lighting = analyze_lighting(video_path, start, end)
    log.info("[%s]   Brightness: %.1f | Correction: %s (gamma=%.2f)",
             clip_id, lighting["avg_brightness"],
             lighting["correction_type"], lighting["gamma_boost"])

    # 4. Dead air detection
    log.info("[%s] Detecting dead air / silence...", clip_id)
    dead_air = detect_dead_air(video_path, start, end)
    log.info("[%s]   Silence ratio: %.1f%% | Dead air: %s",
             clip_id, dead_air["silence_ratio"] * 100, dead_air["has_dead_air"])

    # 5. Tempo analysis
    tempo = {"wpm": 0, "feels_slow": False, "recommended_speed": 1.0, "speech_density": 0}
    if transcript_segments:
        log.info("[%s] Analyzing speech tempo...", clip_id)
        tempo = analyze_tempo(transcript_segments, start, end)
        log.info("[%s]   WPM: %.1f | Slow: %s | Speed: %.2fx",
                 clip_id, tempo["wpm"], tempo["feels_slow"],
                 tempo["recommended_speed"])

    # 6. Save preview snapshot
    preview_path = None
    if save_preview:
        preview_dir = Path(cfg["paths"]["temp"]) / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_file = str(preview_dir / f"{clip_id}_preview.jpg")
        mid_time = (start + end) / 2
        preview_path = _save_preview_snapshot(video_path, mid_time, preview_file)
        if preview_path:
            log.info("[%s]   📸 Preview saved: %s", clip_id, preview_path)

    # ─── Build Export Strategy ────────────────────────────────────────────────
    strategy = _build_export_strategy(layout, lighting, dead_air, tempo, black_frames)
    log.info("[%s] ═══ Strategy: solo=%s | speed=%.2fx | light_fix=%s | skip_silence=%s ═══",
             clip_id, strategy["use_solo_frame"], strategy["speed_factor"],
             strategy["apply_lighting_fix"], strategy["skip_silence"])

    return {
        "clip_id": clip_id,
        "layout": layout,
        "lighting": lighting,
        "dead_air": dead_air,
        "tempo": tempo,
        "black_frames": black_frames,
        "preview_path": preview_path,
        "export_strategy": strategy,
    }


def _build_export_strategy(
    layout: Dict, lighting: Dict, dead_air: Dict,
    tempo: Dict, black_frames: Dict,
) -> Dict:
    """
    Synthesize all analysis results into concrete export decisions.
    """
    # Solo frame decision: prefer solo if dead panel detected or layout suggests it
    use_solo = layout.get("prefer_solo", False) or layout.get("has_dead_panel", False)
    active_crop = layout.get("active_region")

    # If black frame ratio is high, force solo
    if black_frames["black_ratio"] > 0.3:
        use_solo = True

    # Speed
    speed_factor = tempo.get("recommended_speed", 1.0)

    # Lighting
    apply_lighting = lighting.get("needs_correction", False)
    lighting_filter = ""
    if apply_lighting:
        gamma = lighting["gamma_boost"]
        correction_type = lighting["correction_type"]
        if correction_type in ("backlit", "underexposed"):
            # Boost gamma + slight contrast + brightness via eq filter
            lighting_filter = f"eq=gamma={gamma}:contrast=1.15:brightness=0.05"
        elif correction_type == "overexposed":
            lighting_filter = f"eq=gamma={gamma}:contrast=0.95:brightness=-0.05"

    # Silence skipping
    skip_silence = dead_air.get("has_dead_air", False)
    active_segments = dead_air.get("active_segments", [])

    return {
        "use_solo_frame": use_solo,
        "active_crop": active_crop,
        "speed_factor": speed_factor,
        "apply_lighting_fix": apply_lighting,
        "lighting_filter": lighting_filter,
        "skip_silence": skip_silence,
        "active_segments": active_segments,
    }
