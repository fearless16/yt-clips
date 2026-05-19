"""
temporal_consistency.py — Pass 3: Temporal smoothing and drift correction.

After selective enhancement (Pass 2), this pass ensures:
- No flicker between consecutive frames
- Smooth transitions at segment boundaries
- Identity drift correction (face stays consistent over time)
- Lighting consistency (no sudden brightness jumps)

Key techniques:
- Temporal averaging of face region (IIR filter)
- Histogram matching between consecutive frames
- Drift detection and re-sync to reference identity
- Boundary blending at segment edges

Usage:
    python temporal_consistency.py <video_path> --analysis analysis.json
    python temporal_consistency.py <video_path> --analysis analysis.json --output final.mp4
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("temporal_consistency", cfg["logging"]["log_file"], cfg["logging"]["level"])


# ─── Face detection (shared) ──────────────────────────────────────────────

from utils.face_detect import detect_face as _detect_face


# ─── Temporal face smoothing (IIR filter) ─────────────────────────────────

class FaceTemporalFilter:
    """
    Applies temporal smoothing to face region across frames.
    
    Uses IIR filter: output = alpha * current + (1 - alpha) * previous
    This eliminates flicker while preserving motion.
    """
    
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.prev_face = None
        self.prev_bbox = None
        self.prev_frame_size = None
    
    def smooth_face(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """Apply temporal smoothing to face region."""
        if face_bbox is None:
            self.prev_face = None
            self.prev_bbox = None
            return frame
        
        fx, fy, fw, fh = face_bbox
        
        # Validate bbox
        if fy + fh > frame.shape[0] or fx + fw > frame.shape[1] or fw <= 0 or fh <= 0:
            return frame
        
        current_face = frame[fy:fy+fh, fx:fx+fw].copy()
        
        if self.prev_face is not None and self.prev_bbox is not None:
            # Resize previous face to current size if needed
            prev_px, prev_py, prev_pw, prev_ph = self.prev_bbox
            if prev_pw != fw or prev_ph != fh:
                prev_resized = cv2.resize(self.prev_face, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
            else:
                prev_resized = self.prev_face
            
            # IIR filter
            smoothed = cv2.addWeighted(
                current_face, self.alpha,
                prev_resized, 1 - self.alpha,
                0
            )
            
            # Blend with soft edges
            mask = self._create_blend_mask(fw, fh)
            blended = current_face * (1 - mask[:, :, None]) + smoothed * mask[:, :, None]
            
            frame[fy:fy+fh, fx:fx+fw] = blended.clip(0, 255).astype(np.uint8)
            
            # Update state
            self.prev_face = frame[fy:fy+fh, fx:fx+fw].copy()
            self.prev_bbox = face_bbox
        else:
            # First frame, just store
            self.prev_face = current_face
            self.prev_bbox = face_bbox
        
        return frame
    
    def _create_blend_mask(self, w: int, h: int) -> np.ndarray:
        """Create soft edge mask for blending."""
        mask = np.ones((h, w), dtype=np.float32)
        border = max(2, min(w, h) // 10)
        
        if border > 0:
            mask[:border, :] = np.linspace(0, 1, border)[:, None]
            mask[-border:, :] = np.linspace(1, 0, border)[:, None]
            mask[:, :border] = np.linspace(0, 1, border)[None, :]
            mask[:, -border:] = np.linspace(1, 0, border)[None, :]
        
        return mask
    
    def reset(self):
        """Reset filter state (e.g., at segment boundary)."""
        self.prev_face = None
        self.prev_bbox = None


# ─── Histogram matching ───────────────────────────────────────────────────

def _match_histograms(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match histogram of source frame to reference frame.
    
    Eliminates sudden brightness/color jumps between frames.
    """
    matched = np.empty_like(source)
    
    for i in range(3):
        src_hist, bins = np.histogram(source[:, :, i].flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference[:, :, i].flatten(), 256, [0, 256])
        
        src_cdf = src_hist.cumsum()
        src_cdf = src_cdf * ref_hist.max() / src_cdf[-1]
        
        ref_cdf = ref_hist.cumsum()
        ref_cdf = ref_cdf * src_hist.max() / ref_cdf[-1]
        
        interp_table = np.interp(src_cdf, ref_cdf, bins[:-1]).astype(np.uint8)
        matched[:, :, i] = interp_table[source[:, :, i]]
    
    return matched


class GlobalTemporalFilter:
    """
    Applies temporal smoothing to entire frame (not just face).
    
    Prevents background flicker and lighting jumps.
    """
    
    def __init__(self, alpha: float = 0.85):
        self.alpha = alpha
        self.prev_frame = None
    
    def smooth_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply global temporal smoothing."""
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return frame
        
        # IIR filter on entire frame
        smoothed = cv2.addWeighted(
            frame, self.alpha,
            self.prev_frame, 1 - self.alpha,
            0
        )
        
        self.prev_frame = smoothed.copy()
        return smoothed.astype(np.uint8)
    
    def reset(self):
        self.prev_frame = None


# ─── Drift detection ──────────────────────────────────────────────────────

def _detect_identity_drift(
    current_face: np.ndarray,
    reference_face: np.ndarray,
    threshold: float = 0.15,
) -> bool:
    """
    Detect if face has drifted significantly from reference identity.
    
    Only catches MAJOR drift (face lost, wrong person, extreme angle).
    Normal expressions, mouth movement, slight pose changes are NOT drift.
    
    Threshold tuned for talking-head content:
    - Normal frame variation: 5-20
    - Mouth open/close: 15-30
    - Head turn: 20-40
    - Face lost / wrong person: >60
    """
    if current_face.size == 0 or reference_face.size == 0:
        return False
    
    # Resize to same size for comparison
    if current_face.shape != reference_face.shape:
        current_face = cv2.resize(current_face, (reference_face.shape[1], reference_face.shape[0]))
    
    # Use simple mean color difference as drift indicator
    diff = np.mean(np.abs(current_face.astype(np.float64) - reference_face.astype(np.float64)))
    
    # Only trigger on extreme changes (face lost, completely different person)
    return diff > 65


# ─── Segment boundary handling ────────────────────────────────────────────

def _blend_segment_boundary(
    frame: np.ndarray,
    prev_frame: np.ndarray,
    blend_strength: float = 0.5,
) -> np.ndarray:
    """
    Smooth transition at segment boundaries.
    
    Blends current frame with previous frame to avoid hard cuts.
    """
    if prev_frame is None:
        return frame
    
    return cv2.addWeighted(
        frame, 1 - blend_strength,
        prev_frame, blend_strength,
        0
    ).astype(np.uint8)


# ─── Main consistency pipeline ────────────────────────────────────────────

def apply_temporal_consistency(
    video_path: str,
    analysis_path: str,
    output_path: Optional[str] = None,
    face_alpha: float = 0.7,
    global_alpha: float = 0.85,
    drift_threshold: float = 0.15,
) -> str:
    """
    Pass 3: Apply temporal consistency to enhanced video.
    
    Args:
        video_path: Input enhanced video (from Pass 2)
        analysis_path: JSON from state_analyzer.py
        output_path: Output video path
        face_alpha: Temporal smoothing strength for face (0-1, lower = smoother)
        global_alpha: Temporal smoothing strength for entire frame
        drift_threshold: Identity drift detection threshold
    
    Returns:
        Path to consistent video
    """
    t_start = time.perf_counter()
    video_path = str(Path(video_path).resolve())
    analysis_path = str(Path(analysis_path).resolve())
    
    log.info("=" * 60)
    log.info("TEMPORAL CONSISTENCY — Pass 3: Flicker Removal + Drift Correction")
    log.info("=" * 60)
    
    # Load analysis
    with open(analysis_path) as f:
        analysis = json.load(f)
    
    segments = analysis.get("segments", [])
    per_frame = analysis.get("per_frame", [])
    
    # Build segment boundary timestamps
    segment_boundaries = set()
    for seg in segments:
        segment_boundaries.add(seg["start_sec"])
    
    log.info("Loaded analysis: %d segments, %d frames", len(segments), len(per_frame))
    log.info("Segment boundaries at: %s", sorted(segment_boundaries))
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return video_path
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    log.info("Input: %dx%d @ %.1ffps (%d frames)", src_w, src_h, fps, total_frames)
    
    # Initialize filters
    face_filter = FaceTemporalFilter(alpha=face_alpha)
    global_filter = GlobalTemporalFilter(alpha=global_alpha)
    
    # Reference face for drift detection (from first detected face)
    reference_face = None
    
    # Temp output - use /tmp for fast local I/O
    temp_dir = Path("/tmp/yt_clips_consistency")
    temp_dir.mkdir(exist_ok=True)
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Clean old frames
    for f in frames_dir.glob("*.jpg"):
        f.unlink()
    
    # Process frames
    frame_idx = 0
    flicker_corrections = 0
    drift_resets = 0
    boundary_blends = 0
    
    t_proc = time.perf_counter()
    prev_frame = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps if fps > 0 else 0.0
            
            # Check if we're at a segment boundary
            at_boundary = any(abs(timestamp - b) < 1.0 / fps for b in segment_boundaries)
            
            if at_boundary:
                # Reset filters at segment boundary for clean transition
                face_filter.reset()
                boundary_blends += 1
                
                # Blend with previous frame
                if prev_frame is not None:
                    frame = _blend_segment_boundary(frame, prev_frame, blend_strength=0.3)
            
            # Detect face
            face = _detect_face(frame)
            face_bbox = None
            
            if face is not None:
                fx, fy, fw, fh = face
                face_bbox = (fx, fy, fw, fh)
                
                # Store reference face from first detection
                if reference_face is None:
                    reference_face = frame[fy:fy+fh, fx:fx+fw].copy()
                
                # Check for identity drift only every 2 seconds (not every frame)
                # and update reference face every 5 seconds to account for gradual changes
                if reference_face is not None and frame_idx > fps and frame_idx % int(fps * 2) == 0:
                    current_face = frame[fy:fy+fh, fx:fx+fw]
                    if _detect_identity_drift(current_face, reference_face, drift_threshold):
                        # Drift detected: reset filter and re-sync
                        face_filter.reset()
                        drift_resets += 1
                        log.debug("Drift detected at t=%.2fs, resetting filter", timestamp)
                
                # Update reference face every 5 seconds to account for gradual lighting changes
                if frame_idx > 0 and frame_idx % int(fps * 5) == 0:
                    reference_face = frame[fy:fy+fh, fx:fx+fw].copy()
            
            # Apply face temporal smoothing
            if face_bbox is not None:
                frame = face_filter.smooth_face(frame, face_bbox)
            
            # Apply global temporal smoothing
            frame = global_filter.smooth_frame(frame)
            
            # Write frame as jpg
            cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            prev_frame = frame.copy()
            
            frame_idx += 1
            
            if frame_idx % 50 == 0:
                elapsed = time.perf_counter() - t_proc
                rate = frame_idx / elapsed if elapsed > 0 else 0
                log.info("  Processed %d/%d frames (%.0f fps)", frame_idx, total_frames, rate)
    
    finally:
        cap.release()
    
    t_process = time.perf_counter() - t_proc
    log.info("Processing: %d frames in %.1fs (%.0f fps)", frame_idx, t_process, frame_idx / t_process if t_process > 0 else 0)
    log.info("  Flicker corrections: %d | Drift resets: %d | Boundary blends: %d",
             flicker_corrections, drift_resets, boundary_blends)
    
    # Encode with FFmpeg from frames
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(cfg["paths"]["temp"]) / f"{stem}_consistent.mp4")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Extract audio
    audio_path = str(temp_dir / "audio.aac")
    audio_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-c:a", "copy",
        audio_path,
    ]
    subprocess.run(audio_cmd, capture_output=True, text=True)
    
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.jpg"),
        "-i", audio_path,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("FFmpeg encode failed: %s", result.stderr[-300:])
        cmd_no_audio = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "%06d.jpg"),
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(cmd_no_audio, capture_output=True, text=True)
    
    t_total = time.perf_counter() - t_start
    out_size = Path(output_path).stat().st_size / 1e6
    log.info("Consistent video: %s (%.1f MB, %.1fs total)", output_path, out_size, t_total)
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return output_path


# ─── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 3: Temporal consistency")
    parser.add_argument("video", help="Path to enhanced video (from Pass 2)")
    parser.add_argument("--analysis", "-a", required=True, help="Path to state analysis JSON")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--face-alpha", type=float, default=0.7, help="Face smoothing strength (0-1)")
    parser.add_argument("--global-alpha", type=float, default=0.85, help="Global smoothing strength (0-1)")
    parser.add_argument("--drift-threshold", type=float, default=0.15, help="Drift detection threshold")
    args = parser.parse_args()
    
    apply_temporal_consistency(
        video_path=args.video,
        analysis_path=args.analysis,
        output_path=args.output,
        face_alpha=args.face_alpha,
        global_alpha=args.global_alpha,
        drift_threshold=args.drift_threshold,
    )
