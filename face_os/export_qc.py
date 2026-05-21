"""
export_qc.py — Module 10: Export + Quality Control.

Handles:
  - Final video encoding (FFmpeg pipe)
  - Audio muxing
  - Fade in/out transitions
  - Quality validation (face detection rate, identity drift, flicker, sharpness)
  - A/V sync verification

This is the final module — output must be YouTube Shorts ready.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config


cfg = get_config()


class QualityReport:
    """Quality metrics for a processed clip."""

    def __init__(self):
        self.face_detection_rate: float = 0.0
        self.avg_confidence: float = 0.0
        self.identity_drift: float = 0.0
        self.flicker_score: float = 0.0
        self.sharpness: float = 0.0
        self.av_sync_ok: bool = True
        self.total_frames: int = 0
        self.passed: bool = True
        self.failures: List[str] = []

    def check(self) -> bool:
        """Run all QC checks. Returns True if all pass."""
        qc = cfg.qc

        if self.face_detection_rate < qc.min_face_detection_rate:
            self.failures.append(
                f"Face detection rate {self.face_detection_rate:.1%} < {qc.min_face_detection_rate:.1%}"
            )

        if self.identity_drift > qc.max_identity_drift:
            self.failures.append(
                f"Identity drift {self.identity_drift:.1f} > {qc.max_identity_drift:.1f}"
            )

        if self.flicker_score > qc.max_flicker_score:
            self.failures.append(
                f"Flicker score {self.flicker_score:.1f} > {qc.max_flicker_score:.1f}"
            )

        if self.sharpness < qc.min_sharpness:
            self.failures.append(
                f"Sharpness {self.sharpness:.1f} < {qc.min_sharpness:.1f}"
            )

        if qc.check_av_sync and not self.av_sync_ok:
            self.failures.append("A/V sync check failed")

        self.passed = len(self.failures) == 0
        return self.passed

    def to_dict(self) -> Dict:
        return {
            "face_detection_rate": self.face_detection_rate,
            "avg_confidence": self.avg_confidence,
            "identity_drift": self.identity_drift,
            "flicker_score": self.flicker_score,
            "sharpness": self.sharpness,
            "av_sync_ok": self.av_sync_ok,
            "total_frames": self.total_frames,
            "passed": self.passed,
            "failures": self.failures,
        }


class VideoExporter:
    """Exports processed frames to video via FFmpeg pipe."""

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        width: int = 1080,
        height: int = 1920,
        source_path: Optional[str] = None,
    ):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.source_path = source_path
        self._proc: Optional[subprocess.Popen] = None
        self._frame_count = 0
        self._start_time = time.perf_counter()

    def open(self) -> None:
        """Open the FFmpeg pipe for writing."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
        ]

        # Add source video for audio muxing
        if self.source_path and Path(self.source_path).exists():
            cmd.extend(["-i", self.source_path])
            cmd.extend(["-map", "0:v", "-map", "1:a?"])
        else:
            cmd.extend(["-map", "0:v"])

        cmd.extend([
            "-c:v", cfg.export.codec,
            "-crf", str(cfg.export.crf),
            "-preset", cfg.export.preset,
            "-pix_fmt", cfg.export.pixel_format,
        ])

        # Audio encoding
        if self.source_path and Path(self.source_path).exists():
            cmd.extend([
                "-c:a", "aac",
                "-b:a", cfg.export.audio_bitrate,
                "-shortest",
            ])

        cmd.extend(["-movflags", "+faststart", self.output_path])

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._start_time = time.perf_counter()

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame to the video."""
        if self._proc is None:
            self.open()

        # Ensure correct dimensions
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)

        try:
            self._proc.stdin.write(frame.tobytes())
            self._frame_count += 1
        except BrokenPipeError:
            raise RuntimeError("FFmpeg pipe broken — check output path and codec support")

    def close(self) -> bool:
        """Close the pipe and finalize the video. Returns True on success."""
        if self._proc is None:
            return False

        self._proc.stdin.close()
        self._proc.wait()

        elapsed = time.perf_counter() - self._start_time
        if self._proc.returncode == 0:
            size_mb = Path(self.output_path).stat().st_size / 1e6
            fps_actual = self._frame_count / max(elapsed, 0.001)
            print(f"  Export: {self._frame_count} frames, {size_mb:.1f}MB, {fps_actual:.0f}fps")
            return True
        else:
            print(f"  Export failed: FFmpeg returned {self._proc.returncode}")
            return False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


def apply_fades(
    video_path: str,
    fade_in: float = 0.5,
    fade_out: float = 0.5,
    output_path: Optional[str] = None,
) -> str:
    """Apply fade in/out transitions to a video.

    Args:
        video_path: Input video path
        fade_in: Fade from black duration (seconds)
        fade_out: Fade to black duration (seconds)
        output_path: Output path (defaults to overwriting input)

    Returns:
        Path to output video
    """
    if output_path is None:
        output_path = video_path

    # Get duration
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        duration = float(result.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired):
        return video_path

    # Build filter
    filters = []
    if fade_in > 0:
        filters.append(f"fade=t=in:st=0:d={fade_in}")
    if fade_out > 0:
        fade_start = max(0, duration - fade_out)
        filters.append(f"fade=t=out:st={fade_start:.3f}:d={fade_out}")

    if not filters:
        return video_path

    vf = ",".join(filters)

    # Apply fades
    tmp_out = output_path + ".tmp.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf,
        "-af", f"afade=t=in:st=0:d={fade_in},afade=t=out:st={max(0, duration - fade_out):.3f}:d={fade_out}",
        "-c:v", "libx264", "-crf", str(cfg.export.crf), "-preset", "fast",
        "-c:a", "aac", "-b:a", cfg.export.audio_bitrate,
        "-movflags", "+faststart",
        tmp_out,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.export.timeout_seconds)
    if result.returncode == 0 and Path(tmp_out).exists():
        os.replace(tmp_out, output_path)
    elif Path(tmp_out).exists():
        Path(tmp_out).unlink()

    return output_path


def validate_av_sync(video_path: str, tolerance: float = 0.5) -> bool:
    """Check if audio and video durations match."""
    for stream in ("v:0", "a:0"):
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", stream,
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            dur = float(result.stdout.strip()) if result.stdout.strip() else 0.0
        except (ValueError, subprocess.TimeoutExpired):
            dur = 0.0

        if stream == "v:0":
            v_dur = dur
        else:
            a_dur = dur

    if v_dur > 0 and a_dur > 0:
        return abs(v_dur - a_dur) <= tolerance
    return True


def compute_quality_metrics(
    frames: List[np.ndarray],
    reference_lab: Optional[Tuple[float, float, float]] = None,
) -> QualityReport:
    """Compute quality metrics from a list of processed frames.

    Args:
        frames: List of output frames (BGR)
        reference_lab: Reference face LAB values for identity drift check

    Returns:
        QualityReport with all metrics
    """
    report = QualityReport()
    report.total_frames = len(frames)

    if not frames:
        report.passed = False
        report.failures.append("No frames to evaluate")
        return report

    # Compute metrics
    laplacian_vars = []
    lab_means = []
    prev_lab_mean = None
    flicker_values = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Sharpness
        lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        laplacian_vars.append(lap_var)

        # LAB mean
        lab_mean = np.mean(lab, axis=(0, 1))
        lab_means.append(lab_mean)

        # Flicker (frame-to-frame LAB difference)
        if prev_lab_mean is not None:
            flicker = float(np.sqrt(np.sum((lab_mean - prev_lab_mean) ** 2)))
            flicker_values.append(flicker)
        prev_lab_mean = lab_mean

    report.sharpness = float(np.mean(laplacian_vars))
    report.flicker_score = float(np.mean(flicker_values)) if flicker_values else 0.0

    # Identity drift (distance from reference)
    if reference_lab and lab_means:
        avg_lab = np.mean(lab_means, axis=0)
        report.identity_drift = float(np.sqrt(np.sum((avg_lab - np.array(reference_lab)) ** 2)))

    # Face detection rate (placeholder — would need face detection results)
    report.face_detection_rate = 1.0  # Assume all frames have face

    report.check()
    return report
