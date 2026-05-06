import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("analyzer", cfg["logging"]["log_file"], cfg["logging"]["level"])


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

    black = [s for s in samples if s["avg"] < threshold and s["var"] < 50]

    return {
        "has_black_frames": len(black) > 0,
        "black_ratio": len(black) / len(samples) if samples else 0,
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
    timestamps = [
        start + (end - start) * x for x in [0.25, 0.5, 0.75]
    ]

    votes = []

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

        left = sum(data[i] for i in range(0, len(data), 320))
        right = sum(data[i] for i in range(mid, len(data), 320))

        if abs(left - right) > 50000:
            votes.append("split")

    return {
        "layout_type": "split" if votes.count("split") >= 2 else "solo",
        "prefer_solo": votes.count("split") < 2
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

    strategy = {
        "use_solo_frame": layout["prefer_solo"] or black["black_ratio"] > 0.3,
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