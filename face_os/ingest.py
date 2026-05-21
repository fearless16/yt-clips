"""
ingest.py — Module 1: Video Ingestion + Audio Sync.

Handles:
  - Video file loading + metadata extraction
  - Audio stream detection and sync validation
  - Frame-by-frame reading with seeking support
  - Reference image/video loading for identity enrollment

No processing — pure I/O and metadata.
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import VideoMeta


cfg = get_config()


def load_video_meta(video_path: str) -> VideoMeta:
    """Extract video metadata using ffprobe.

    Returns VideoMeta with dimensions, fps, duration, codec info.
    Falls back to OpenCV if ffprobe fails.
    """
    meta = VideoMeta(path=video_path)

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,codec_name",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(result.stdout)

        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        meta.width = int(stream.get("width", 0))
        meta.height = int(stream.get("height", 0))
        meta.codec = stream.get("codec_name", "")
        meta.duration = float(fmt.get("duration", 0))

        # Parse frame rate
        rate = stream.get("r_frame_rate", "30/1")
        if "/" in rate:
            num, den = rate.split("/", 1)
            meta.fps = float(num) / float(den)
        else:
            meta.fps = float(rate)

        if meta.fps > 0:
            meta.total_frames = int(meta.duration * meta.fps)

    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, IndexError):
        # Fallback to OpenCV
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            meta.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            meta.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            meta.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            meta.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            meta.duration = meta.total_frames / meta.fps if meta.fps > 0 else 0
            cap.release()

    # Check audio stream
    meta.has_audio = _has_audio(video_path)

    return meta


def _has_audio(video_path: str) -> bool:
    """Check if video has an audio stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        return bool(data.get("streams"))
    except Exception:
        return False


def frame_reader(
    video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Yield (frame_idx, timestamp_sec, bgr_frame) from video.

    Efficient seeking — skips frames via OpenCV rather than decoding all.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    while True:
        if end_frame and frame_idx >= end_frame:
            break
        if frame_idx >= total:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % step == 0:
            timestamp = frame_idx / fps
            yield frame_idx, timestamp, frame

        frame_idx += 1

    cap.release()


def load_reference_images(
    ref_dir: str = "photos/",
    primary_ref: str = "expectation.png",
) -> Tuple[Optional[np.ndarray], list]:
    """Load reference face images for identity enrollment.

    Returns:
        (primary_image, [all_reference_images])
    """
    ref_path = Path(primary_ref)
    primary = None
    if ref_path.exists():
        primary = cv2.imread(str(ref_path))

    all_refs = []
    photos_dir = Path(ref_dir)
    if photos_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for img_path in sorted(photos_dir.glob(ext)):
                img = cv2.imread(str(img_path))
                if img is not None:
                    all_refs.append(img)

    return primary, all_refs


def validate_av_sync(video_path: str, tolerance: float = 0.5) -> bool:
    """Check if audio and video durations match within tolerance."""
    if not _has_audio(video_path):
        return True  # No audio to desync

    for stream_type in ("v:0", "a:0"):
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", stream_type,
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            dur = float(result.stdout.strip()) if result.stdout.strip() else 0.0
        except (ValueError, subprocess.TimeoutExpired):
            dur = 0.0

        if stream_type == "v:0":
            v_dur = dur
        else:
            a_dur = dur

    if v_dur > 0 and a_dur > 0:
        return abs(v_dur - a_dur) <= tolerance
    return True
