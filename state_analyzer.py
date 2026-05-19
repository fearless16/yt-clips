"""
state_analyzer.py — Pass 1: State-aware clip analysis for selective enhancement.

Analyzes talking-head clips to classify frames by enhancement safety:
- Mouth state (open/closed ratio) → decides sharpening aggression
- Eye state (open/blink) → avoids fake eye reconstruction
- Pose stability per segment → decides if GFPGAN is safe
- Lighting stability per segment → decides relight strength
- Motion blur score → avoids ghosting on fast movement
- Output: enhancement map (heavy/light/skip per frame)

Usage:
    python state_analyzer.py <video_path> --output analysis.json
    python state_analyzer.py <video_path> --sample-rate 2 --segment-size 30
"""

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("state_analyzer", cfg["logging"]["log_file"], cfg["logging"]["level"])


# ─── Face detection (shared) ──────────────────────────────────────────────

from utils.face_detect import detect_face as _detect_face


# ─── Mouth state estimation ───────────────────────────────────────────────

def _estimate_mouth_open(face_gray: np.ndarray, face_w: int, face_h: int) -> float:
    """
    Estimate mouth openness ratio from a cropped face.
    
    Uses lower-third region analysis: mouth area has higher variance when open
    (dark interior + teeth contrast) vs closed (smooth lip line).
    
    Returns: 0.0 (closed) → 1.0 (wide open)
    """
    h, w = face_gray.shape
    
    # Mouth region: lower third, center half
    y_start = int(h * 0.55)
    y_end = int(h * 0.85)
    x_start = int(w * 0.25)
    x_end = int(w * 0.75)
    
    mouth_roi = face_gray[y_start:y_end, x_start:x_end]
    if mouth_roi.size == 0:
        return 0.0
    
    # Method 1: Vertical gradient in mouth region
    # Open mouth = strong vertical edges (lip boundary)
    sobel_y = cv2.Sobel(mouth_roi, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.mean(np.abs(sobel_y))
    
    # Method 2: Intensity variance in mouth region
    # Open mouth = high variance (dark interior + bright teeth)
    variance = np.var(mouth_roi.astype(np.float64))
    
    # Method 3: Dark pixel ratio (open mouth has dark interior)
    dark_ratio = np.mean(mouth_roi < 40)
    
    # Normalize and combine
    # Typical ranges (empirical): edge_strength 5-30, variance 200-2000, dark_ratio 0-0.3
    mouth_score = (
        min(edge_strength / 30.0, 1.0) * 0.4 +
        min(variance / 2000.0, 1.0) * 0.35 +
        min(dark_ratio / 0.30, 1.0) * 0.25
    )
    
    return float(np.clip(mouth_score, 0.0, 1.0))


# ─── Eye state estimation ─────────────────────────────────────────────────

def _estimate_eye_open(face_gray: np.ndarray, face_w: int, face_h: int) -> float:
    """
    Estimate eye openness from upper face region.
    
    Uses horizontal projection in eye region: open eyes create dark bands
    (pupils) separated by lighter regions. Closed eyes = uniform gradient.
    
    Returns: 0.0 (closed/blink) → 1.0 (wide open)
    """
    h, w = face_gray.shape
    
    # Eye region: upper third
    y_start = int(h * 0.20)
    y_end = int(h * 0.45)
    eye_roi = face_gray[y_start:y_end, :]
    
    if eye_roi.size == 0:
        return 0.0
    
    # Horizontal projection: sum pixel values per row
    row_sums = np.mean(eye_roi.astype(np.float64), axis=1)
    
    # Open eyes = bimodal distribution (dark pupil bands + lighter sclera/skin)
    # Closed eyes = unimodal (smooth gradient)
    if len(row_sums) < 4:
        return 0.0
    
    # Variance of row sums indicates eye openness
    row_variance = np.var(row_sums)
    
    # Also check for dark horizontal bands (pupils)
    dark_threshold = np.mean(eye_roi) - np.std(eye_roi) * 0.5
    dark_bands = np.mean(eye_roi < dark_threshold)
    
    # Combine
    eye_score = (
        min(row_variance / 300.0, 1.0) * 0.6 +
        min(dark_bands / 0.10, 1.0) * 0.4
    )
    
    return float(np.clip(eye_score, 0.0, 1.0))


# ─── Pose estimation (lightweight) ────────────────────────────────────────

def _estimate_pose(face_gray: np.ndarray, face_x: int, face_y: int, 
                   face_w: int, face_h: int, frame_w: int, frame_h: int) -> Dict[str, float]:
    """
    Estimate head pose from face position and symmetry.
    
    Returns:
        yaw: -1.0 (left) → 1.0 (right)
        pitch: -1.0 (down) → 1.0 (up)
        off_center: 0.0 (centered) → 1.0 (edge of frame)
    """
    # Face center relative to frame center
    face_center_x = face_x + face_w / 2
    face_center_y = face_y + face_h / 2
    
    frame_center_x = frame_w / 2
    frame_center_y = frame_h / 2
    
    # Normalized offset
    yaw = (face_center_x - frame_center_x) / (frame_w / 2)
    pitch = (face_center_y - frame_center_y) / (frame_h / 2)
    
    # Off-center score
    off_center = np.sqrt(yaw**2 + pitch**2)
    
    # Symmetry check: compare left/right halves of face
    # Asymmetric lighting = turned head
    left_half = face_gray[:, :face_w // 2]
    right_half = face_gray[:, face_w // 2:]
    
    left_mean = np.mean(left_half.astype(np.float64))
    right_mean = np.mean(right_half.astype(np.float64))
    
    # Brightness asymmetry indicates yaw
    brightness_diff = abs(left_mean - right_mean) / max(left_mean, right_mean, 1)
    yaw_confidence = min(brightness_diff / 0.3, 1.0)
    
    return {
        "yaw": float(np.clip(yaw, -1.0, 1.0)),
        "pitch": float(np.clip(pitch, -1.0, 1.0)),
        "off_center": float(np.clip(off_center, 0.0, 1.0)),
        "yaw_confidence": float(yaw_confidence),
    }


# ─── Lighting analysis ────────────────────────────────────────────────────

def _analyze_lighting(frame: np.ndarray, face_gray: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Analyze lighting conditions.
    
    Returns brightness, contrast, color temperature, face lighting quality.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    
    # Color temperature estimate (warm vs cool)
    bgr = frame.astype(np.float64)
    blue_mean = np.mean(bgr[:, :, 0])
    red_mean = np.mean(bgr[:, :, 2])
    color_temp = float(red_mean / max(blue_mean, 1))  # >1 = warm, <1 = cool
    
    # Face lighting quality
    face_brightness = 0.0
    face_contrast = 0.0
    face_lighting_score = 0.0
    
    if face_gray is not None and face_gray.size > 0:
        face_brightness = float(np.mean(face_gray))
        face_contrast = float(np.std(face_gray))
        
        # Ideal face lighting: 100-180 brightness, 30-70 contrast
        b_score = 1.0 - min(abs(face_brightness - 140) / 100, 1.0)
        c_score = 1.0 - min(abs(face_contrast - 50) / 40, 1.0)
        face_lighting_score = float(b_score * 0.6 + c_score * 0.4)
    
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "color_temp": round(color_temp, 3),
        "face_brightness": round(face_brightness, 2),
        "face_contrast": round(face_contrast, 2),
        "face_lighting_score": round(face_lighting_score, 3),
    }


# ─── Motion blur estimation ───────────────────────────────────────────────

def _estimate_motion_blur(frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> float:
    """
    Estimate motion blur in frame.
    
    Uses Laplacian variance: sharp images have high variance,
    blurry images have low variance.
    
    Returns: 0.0 (very blurry) → 1.0 (sharp)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(laplacian))
    
    # Normalize: typical sharp video = 200-800, blurry = 20-100
    blur_score = min(sharpness / 300.0, 1.0)
    
    # Also check inter-frame difference for motion detection
    motion_score = 0.0
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        motion_magnitude = np.mean(diff)
        # High motion + low sharpness = motion blur
        if motion_magnitude > 15 and sharpness < 150:
            motion_score = 0.3  # Penalty for motion blur
    
    return float(np.clip(blur_score - motion_score, 0.0, 1.0))


# ─── Compression artifact detection ───────────────────────────────────────

def _detect_compression_artifacts(face_gray: np.ndarray) -> float:
    """
    Detect compression artifacts in face region.
    
    Uses blockiness detection (8x8 DCT artifacts common in H.264).
    
    Returns: 0.0 (clean) → 1.0 (heavy artifacts)
    """
    if face_gray is None or face_gray.size == 0:
        return 0.0
    
    h, w = face_gray.shape
    
    # Check for 8x8 block boundaries
    # Horizontal block edges
    h_diffs = []
    for y in range(0, h - 8, 8):
        row = face_gray[y:y+2, :]
        if row.size > 0:
            diff = np.mean(np.abs(np.diff(row.astype(np.float64), axis=1)))
            h_diffs.append(diff)
    
    # Vertical block edges
    v_diffs = []
    for x in range(0, w - 8, 8):
        col = face_gray[:, x:x+2]
        if col.size > 0:
            diff = np.mean(np.abs(np.diff(col.astype(np.float64), axis=0)))
            v_diffs.append(diff)
    
    if not h_diffs or not v_diffs:
        return 0.0
    
    # Blockiness = high gradient at block boundaries relative to interior
    avg_h = np.mean(h_diffs)
    avg_v = np.mean(v_diffs)
    blockiness = (avg_h + avg_v) / 2
    
    # Normalize
    artifact_score = min(blockiness / 15.0, 1.0)
    
    return float(artifact_score)


# ─── Enhancement decision logic ───────────────────────────────────────────

def _classify_frame_state(frame_data: Dict) -> Dict:
    """
    Classify a frame's enhancement safety level.
    
    Returns:
        enhancement_level: "heavy" | "light" | "skip"
        reasons: list of factors
        confidence: 0.0-1.0
    """
    mouth_open = frame_data["mouth_open"]
    eye_open = frame_data["eye_open"]
    pose = frame_data["pose"]
    lighting = frame_data["lighting"]
    sharpness = frame_data["sharpness"]
    artifacts = frame_data["compression_artifacts"]
    face_detected = frame_data["face_detected"]
    
    reasons = []
    penalty = 0.0
    confidence = 1.0
    
    # No face = skip enhancement entirely
    if not face_detected:
        return {
            "enhancement_level": "skip",
            "reasons": ["no_face_detected"],
            "confidence": 0.9,
        }
    
    # Mouth open = risky for aggressive sharpening (ghosting)
    if mouth_open > 0.75:
        penalty += 0.3
        reasons.append("mouth_wide_open")
    elif mouth_open > 0.5:
        penalty += 0.1
        reasons.append("mouth_partially_open")
    
    # Eyes closed = risky for eye reconstruction (uncanny)
    if eye_open < 0.15:
        penalty += 0.25
        reasons.append("eyes_closed_blink")
    
    # Extreme pose = GFPGAN may distort
    if abs(pose["yaw"]) > 0.5:
        penalty += 0.2
        reasons.append("extreme_yaw")
    if abs(pose["pitch"]) > 0.4:
        penalty += 0.15
        reasons.append("extreme_pitch")
    
    # Off-center face = less reliable enhancement
    if pose["off_center"] > 0.6:
        penalty += 0.1
        reasons.append("face_off_center")
    
    # Poor lighting = conservative enhancement
    if lighting["face_lighting_score"] < 0.3:
        penalty += 0.2
        reasons.append("poor_face_lighting")
    
    # Motion blur = skip heavy enhancement (ghosting risk)
    if sharpness < 0.3:
        penalty += 0.3
        reasons.append("motion_blur")
    elif sharpness < 0.5:
        penalty += 0.15
        reasons.append("slight_blur")
    
    # Heavy compression artifacts = conservative (can't restore what isn't there)
    if artifacts > 0.6:
        penalty += 0.15
        reasons.append("heavy_compression")
    
    # Decision
    if penalty < 0.2:
        level = "heavy"
        confidence = max(0.5, 1.0 - penalty)
    elif penalty < 0.5:
        level = "light"
        confidence = max(0.3, 1.0 - penalty)
    else:
        level = "skip"
        confidence = max(0.2, 1.0 - penalty)
    
    return {
        "enhancement_level": level,
        "reasons": reasons if reasons else ["stable_frame"],
        "confidence": round(confidence, 3),
        "penalty_score": round(penalty, 3),
    }


# ─── Segment analysis ─────────────────────────────────────────────────────

def _analyze_segments(per_frame: List[Dict], segment_size_sec: float = 2.0, fps: float = 30.0) -> List[Dict]:
    """
    Group frames into segments and analyze stability.
    
    Returns per-segment: lighting stability, pose stability, 
    dominant enhancement level, recommended strategy.
    """
    frames_per_segment = int(segment_size_sec * fps)
    segments = []
    
    for i in range(0, len(per_frame), frames_per_segment):
        seg_frames = per_frame[i:i + frames_per_segment]
        if len(seg_frames) < 3:
            continue
        
        # Lighting stability (lower variance = more stable)
        brightness_values = [f["lighting"]["brightness"] for f in seg_frames]
        brightness_std = float(np.std(brightness_values))
        lighting_stable = brightness_std < 15.0
        
        # Pose stability
        yaw_values = [f["pose"]["yaw"] for f in seg_frames]
        pitch_values = [f["pose"]["pitch"] for f in seg_frames]
        pose_std = float(np.sqrt(np.var(yaw_values) + np.var(pitch_values)))
        pose_stable = pose_std < 0.1
        
        # Mouth state consistency
        mouth_values = [f["mouth_open"] for f in seg_frames]
        mouth_avg = float(np.mean(mouth_values))
        mouth_stable = float(np.std(mouth_values)) < 0.15
        
        # Enhancement level distribution
        levels = [f["enhancement"]["enhancement_level"] for f in seg_frames]
        heavy_count = levels.count("heavy")
        light_count = levels.count("light")
        skip_count = levels.count("skip")
        total = len(levels)
        
        # Dominant level
        if heavy_count > total * 0.6:
            dominant = "heavy"
        elif skip_count > total * 0.5:
            dominant = "skip"
        else:
            dominant = "light"
        
        # Recommended strategy
        if lighting_stable and pose_stable and dominant == "heavy":
            strategy = "full_enhancement"
        elif lighting_stable and pose_stable:
            strategy = "selective_enhancement"
        elif not lighting_stable:
            strategy = "conservative_relight"
        elif not pose_stable:
            strategy = "temporal_propagation"
        else:
            strategy = "light_enhancement"
        
        seg_start = seg_frames[0]["timestamp"]
        seg_end = seg_frames[-1]["timestamp"]
        
        segments.append({
            "segment_id": len(segments),
            "start_sec": round(seg_start, 2),
            "end_sec": round(seg_end, 2),
            "duration_sec": round(seg_end - seg_start, 2),
            "frame_count": len(seg_frames),
            "lighting_stable": lighting_stable,
            "brightness_std": round(brightness_std, 2),
            "pose_stable": pose_stable,
            "pose_std": round(pose_std, 3),
            "mouth_stable": mouth_stable,
            "mouth_avg": round(mouth_avg, 3),
            "heavy_pct": round(heavy_count / total * 100, 1),
            "light_pct": round(light_count / total * 100, 1),
            "skip_pct": round(skip_count / total * 100, 1),
            "dominant_level": dominant,
            "recommended_strategy": strategy,
        })
    
    return segments


# ─── Frame extraction ─────────────────────────────────────────────────────

def _extract_frames(video_path: str, sample_rate: int = 1) -> Tuple[List[np.ndarray], float, int]:
    """
    Extract frames from video.
    
    sample_rate: 1 = every frame, 2 = every 2nd frame, etc.
    Returns: (frames, fps, total_frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return [], 0, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_idx = 0
    
    log.info("Extracting frames from %s (%d frames @ %.1f fps, sample_rate=%d)",
             video_path, total_frames, fps, sample_rate)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            frames.append(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            log.info("  Extracted %d/%d frames", len(frames), total_frames // sample_rate)
    
    cap.release()
    log.info("Extracted %d frames", len(frames))
    
    return frames, fps, total_frames


# ─── Main analysis pipeline ───────────────────────────────────────────────

def analyze_clip(
    video_path: str,
    sample_rate: int = 2,
    segment_size_sec: float = 2.0,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Full clip analysis: per-frame states + per-segment strategies.
    
    Args:
        video_path: Path to input video
        sample_rate: Frame sampling rate (1=all, 2=every 2nd, etc.)
        segment_size_sec: Segment duration for stability analysis
        output_path: Where to save analysis JSON
    
    Returns:
        Analysis dict with per_frame, segments, and summary
    """
    t_start = time.perf_counter()
    video_path = str(Path(video_path).resolve())
    
    log.info("=" * 60)
    log.info("STATE ANALYZER — Pass 1: Clip Analysis")
    log.info("=" * 60)
    
    # Extract frames
    frames, fps, total_frames = _extract_frames(video_path, sample_rate)
    if not frames:
        return {"error": "no_frames_extracted"}
    
    # Per-frame analysis
    per_frame = []
    prev_frame = None
    
    for i, frame in enumerate(frames):
        timestamp = i * sample_rate / fps if fps > 0 else 0.0
        
        # Face detection
        face = _detect_face(frame)
        face_detected = face is not None
        face_gray = None
        face_x, face_y, face_w, face_h = 0, 0, 0, 0
        
        if face_detected:
            face_x, face_y, face_w, face_h = face
            # Crop face with padding
            h, w = frame.shape[:2]
            pad = int(max(face_w, face_h) * 0.2)
            x1 = max(0, face_x - pad)
            y1 = max(0, face_y - pad)
            x2 = min(w, face_x + face_w + pad)
            y2 = min(h, face_y + face_h + pad)
            face_crop = frame[y1:y2, x1:x2]
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Mouth state
        mouth_open = _estimate_mouth_open(face_gray, face_w, face_h) if face_detected else 0.0
        
        # Eye state
        eye_open = _estimate_eye_open(face_gray, face_w, face_h) if face_detected else 0.0
        
        # Pose
        pose = _estimate_pose(face_gray, face_x, face_y, face_w, face_h, 
                             frame.shape[1], frame.shape[0]) if face_detected else {
            "yaw": 0.0, "pitch": 0.0, "off_center": 1.0, "yaw_confidence": 0.0
        }
        
        # Lighting
        lighting = _analyze_lighting(frame, face_gray)
        
        # Motion blur
        sharpness = _estimate_motion_blur(frame, prev_frame)
        
        # Compression artifacts
        artifacts = _detect_compression_artifacts(face_gray) if face_detected else 0.0
        
        frame_data = {
            "timestamp": round(timestamp, 3),
            "frame_index": i * sample_rate,
            "face_detected": face_detected,
            "mouth_open": round(mouth_open, 3),
            "eye_open": round(eye_open, 3),
            "pose": pose,
            "lighting": lighting,
            "sharpness": round(sharpness, 3),
            "compression_artifacts": round(artifacts, 3),
        }
        
        # Enhancement classification
        enhancement = _classify_frame_state(frame_data)
        frame_data["enhancement"] = enhancement
        
        per_frame.append(frame_data)
        prev_frame = frame
        
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            log.info("  Analyzed %d/%d frames (%.0f fps)", len(per_frame), len(frames), rate)
    
    # Segment analysis
    segments = _analyze_segments(per_frame, segment_size_sec, fps / sample_rate)
    
    # Summary statistics
    faces_detected = sum(1 for f in per_frame if f["face_detected"])
    heavy_count = sum(1 for f in per_frame if f["enhancement"]["enhancement_level"] == "heavy")
    light_count = sum(1 for f in per_frame if f["enhancement"]["enhancement_level"] == "light")
    skip_count = sum(1 for f in per_frame if f["enhancement"]["enhancement_level"] == "skip")
    
    avg_mouth_open = float(np.mean([f["mouth_open"] for f in per_frame]))
    avg_eye_open = float(np.mean([f["eye_open"] for f in per_frame]))
    avg_sharpness = float(np.mean([f["sharpness"] for f in per_frame]))
    
    summary = {
        "video": video_path,
        "total_frames_analyzed": len(per_frame),
        "sample_rate": sample_rate,
        "fps": fps,
        "face_detection_rate": round(faces_detected / max(1, len(per_frame)) * 100, 1),
        "enhancement_distribution": {
            "heavy": heavy_count,
            "light": light_count,
            "skip": skip_count,
            "heavy_pct": round(heavy_count / max(1, len(per_frame)) * 100, 1),
            "light_pct": round(light_count / max(1, len(per_frame)) * 100, 1),
            "skip_pct": round(skip_count / max(1, len(per_frame)) * 100, 1),
        },
        "avg_mouth_open": round(avg_mouth_open, 3),
        "avg_eye_open": round(avg_eye_open, 3),
        "avg_sharpness": round(avg_sharpness, 3),
        "segment_count": len(segments),
        "analysis_time_sec": round(time.perf_counter() - t_start, 1),
    }
    
    result = {
        "summary": summary,
        "per_frame": per_frame,
        "segments": segments,
    }
    
    # Save
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(cfg["paths"]["temp"]) / f"{stem}_state_analysis.json")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    _print_summary(summary, segments)
    log.info("Analysis saved: %s", output_path)
    
    return result


def _print_summary(summary: Dict, segments: List[Dict]):
    """Print analysis summary."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from rich import box
        
        console = Console()
        
        console.print()
        console.print(Panel(
            Text("STATE ANALYSIS — Enhancement Map", style="bold cyan", justify="center"),
            border_style="cyan", width=70,
        ))
        
        t = Table(title="Summary", box=box.ROUNDED, header_style="bold cyan")
        t.add_column("Metric", style="cyan")
        t.add_column("Value", justify="right")
        t.add_row("Frames Analyzed", str(summary["total_frames_analyzed"]))
        t.add_row("Face Detection Rate", f"{summary['face_detection_rate']}%")
        t.add_row("Avg Mouth Open", f"{summary['avg_mouth_open']:.3f}")
        t.add_row("Avg Eye Open", f"{summary['avg_eye_open']:.3f}")
        t.add_row("Avg Sharpness", f"{summary['avg_sharpness']:.3f}")
        t.add_row("Heavy Frames", f"{summary['enhancement_distribution']['heavy']} ({summary['enhancement_distribution']['heavy_pct']}%)")
        t.add_row("Light Frames", f"{summary['enhancement_distribution']['light']} ({summary['enhancement_distribution']['light_pct']}%)")
        t.add_row("Skip Frames", f"{summary['enhancement_distribution']['skip']} ({summary['enhancement_distribution']['skip_pct']}%)")
        t.add_row("Segments", str(summary["segment_count"]))
        t.add_row("Analysis Time", f"{summary['analysis_time_sec']:.1f}s")
        console.print(t)
        
        if segments:
            st = Table(title="Segments", box=box.SIMPLE, header_style="bold green")
            st.add_column("#", justify="right", style="dim")
            st.add_column("Start", justify="right")
            st.add_column("End", justify="right")
            st.add_column("Lighting", justify="center")
            st.add_column("Pose", justify="center")
            st.add_column("Dominant", justify="center")
            st.add_column("Strategy", justify="left")
            
            for seg in segments:
                st.add_row(
                    str(seg["segment_id"]),
                    f"{seg['start_sec']:.1f}s",
                    f"{seg['end_sec']:.1f}s",
                    "stable" if seg["lighting_stable"] else "unstable",
                    "stable" if seg["pose_stable"] else "unstable",
                    seg["dominant_level"],
                    seg["recommended_strategy"],
                )
            console.print()
            console.print(st)
        console.print()
    except ImportError:
        print(f"\nState Analysis: {summary['total_frames_analyzed']} frames")
        print(f"  Face detection: {summary['face_detection_rate']}%")
        print(f"  Heavy: {summary['enhancement_distribution']['heavy']} frames")
        print(f"  Light: {summary['enhancement_distribution']['light']} frames")
        print(f"  Skip: {summary['enhancement_distribution']['skip']} frames")
        for seg in segments[:5]:
            print(f"  Segment {seg['segment_id']}: {seg['start_sec']:.1f}s-{seg['end_sec']:.1f}s → {seg['recommended_strategy']}")


# ─── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State-aware clip analysis for selective enhancement")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--sample-rate", type=int, default=2, help="Frame sampling rate (1=all, 2=every 2nd)")
    parser.add_argument("--segment-size", type=float, default=2.0, help="Segment size in seconds")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    args = parser.parse_args()
    
    result = analyze_clip(
        video_path=args.video,
        sample_rate=args.sample_rate,
        segment_size_sec=args.segment_size,
        output_path=args.output,
    )
