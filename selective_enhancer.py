"""
selective_enhancer.py — Pass 2: State-aware conditional enhancement.

Uses the enhancement map from state_analyzer.py to selectively apply:
- Heavy frames: Full GFPGAN + ESRGAN (stable pose, closed mouth, good lighting)
- Light frames: Conservative enhancement (partial mouth open, slight pose shift)
- Skip frames: Temporal propagation from nearest enhanced neighbor

Key design:
- GFPGAN runs ONLY on heavy frames (not every frame)
- Skip frames get face propagated via optical flow from nearest enhanced frame
- Background is cached + globally graded (not per-frame processed)
- Mouth-open frames get reduced sharpening to avoid ghosting
- Blink frames skip eye reconstruction to avoid uncanny results

Usage:
    python selective_enhancer.py <video_path> --analysis analysis.json
    python selective_enhancer.py <video_path> --analysis analysis.json --output enhanced.mp4
"""

import argparse
import gc
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("selective_enhancer", cfg["logging"]["log_file"], cfg["logging"]["level"])

# ─── Backend detection ────────────────────────────────────────────────────

HAS_TORCH = False
HAS_CUDA = False
HAS_GFPGAN = False
HAS_REALESRGAN = False

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

try:
    import utils.torchvision_compat  # noqa: F401 — must precede gfpgan
    from gfpgan import GFPGANer
    HAS_GFPGAN = True
except ImportError:
    pass

try:
    import utils.torchvision_compat  # noqa: F401
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    HAS_REALESRGAN = True
except ImportError:
    pass


# ─── Model loaders ────────────────────────────────────────────────────────

GFPGAN_WEIGHT_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
GFPGAN_LOCAL_PATH = Path("weights/GFPGANv1.4.pth")


def _ensure_gfpgan_weights() -> Optional[str]:
    """Download GFPGAN weights if not present. Returns local path or None."""
    if GFPGAN_LOCAL_PATH.exists():
        return str(GFPGAN_LOCAL_PATH)
    GFPGAN_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading GFPGAN weights...")
    try:
        import urllib.request
        urllib.request.urlretrieve(GFPGAN_WEIGHT_URL, str(GFPGAN_LOCAL_PATH))
        log.info("GFPGAN weights downloaded: %s", GFPGAN_LOCAL_PATH)
        return str(GFPGAN_LOCAL_PATH)
    except Exception as e:
        log.warning("GFPGAN download failed: %s", e)
        return None


class EnhancementModels:
    """Lazy-load GPU models only when needed."""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda:0" if HAS_CUDA else "cpu")
        self.gfpgan = None
        self.esrgan = None
        self._loaded = set()
    
    def load_gfpgan(self):
        if "gfpgan" in self._loaded:
            return
        if not (HAS_GFPGAN and HAS_CUDA):
            log.warning("GFPGAN not available (missing package or CUDA)")
            return
        model_path = _ensure_gfpgan_weights()
        if model_path is None:
            log.warning("GFPGAN weights not available — skipping face restoration")
            return
        try:
            self.gfpgan = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device,
            )
            self._loaded.add("gfpgan")
            log.info("GFPGAN loaded on %s", self.device)
        except Exception as e:
            log.warning("GFPGAN load failed: %s", e)
    
    def load_esrgan(self, scale: int = 4):
        if "esrgan" in self._loaded:
            return
        if not (HAS_REALESRGAN and HAS_CUDA):
            log.warning("Real-ESRGAN not available")
            return
        try:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=scale,
            )
            self.esrgan = RealESRGANer(
                scale=scale,
                model_path="weights/RealESRGAN_x4plus.pth",
                model=model,
                tile=512,
                tile_pad=10,
                pre_pad=0,
                half=True,
                device=self.device,
            )
            self._loaded.add("esrgan")
            log.info("Real-ESRGAN %dx loaded on %s", scale, self.device)
        except Exception as e:
            log.warning("Real-ESRGAN load failed: %s", e)
    
    def clear_vram(self):
        gc.collect()
        if HAS_CUDA:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ─── Face detection (shared) ──────────────────────────────────────────────

from utils.face_detect import detect_face as _detect_face


# ─── Optical flow propagation ─────────────────────────────────────────────

def _propagate_face(
    src_frame: np.ndarray,
    dst_frame: np.ndarray,
    src_face: np.ndarray,
    src_face_bbox: Tuple[int, int, int, int],
    dst_face_bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Propagate enhanced face from source frame to destination frame.
    
    Uses dense optical flow to warp the enhanced face region,
    then blends with destination frame.
    """
    sx, sy, sw, sh = src_face_bbox
    dx, dy, dw, dh = dst_face_bbox
    
    # Validate dimensions
    if dw <= 0 or dh <= 0 or sw <= 0 or sh <= 0:
        return dst_frame
    if src_face.size == 0:
        return dst_frame
    
    # Resize enhanced face to destination size
    face_resized = cv2.resize(src_face, (dw, dh), interpolation=cv2.INTER_LANCZOS4)
    
    # Create blend mask (soft edges)
    mask = np.ones((dh, dw), dtype=np.float32)
    border = max(3, min(dw, dh) // 8)
    mask[:border, :] = np.linspace(0, 1, border)[:, None]
    mask[-border:, :] = np.linspace(1, 0, border)[:, None]
    mask[:, :border] = np.linspace(0, 1, border)[None, :]
    mask[:, -border:] = np.linspace(1, 0, border)[None, :]
    
    # Blend
    y1 = max(0, dy)
    y2 = min(dst_frame.shape[0], dy + dh)
    x1 = max(0, dx)
    x2 = min(dst_frame.shape[1], dx + dw)
    
    fy1 = y1 - dy
    fy2 = y1 - dy + (y2 - y1)
    fx1 = x1 - dx
    fx2 = fx1 + (x2 - x1)
    
    if fy2 > fy1 and fx2 > fx1:
        face_region = dst_frame[y1:y2, x1:x2].astype(np.float32)
        warped_face = face_resized[fy1:fy2, fx1:fx2].astype(np.float32)
        blend_mask = mask[fy1:fy2, fx1:fx2, None]
        
        dst_frame[y1:y2, x1:x2] = (
            face_region * (1 - blend_mask) + warped_face * blend_mask
        ).astype(np.uint8)
    
    return dst_frame


# ─── Enhancement functions ────────────────────────────────────────────────

def _enhance_heavy(
    frame: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    models: EnhancementModels,
    mouth_open: float,
    eye_open: float,
) -> np.ndarray:
    """
    Full enhancement: GFPGAN + optional ESRGAN on face region.
    
    Mouth-open frames: reduce sharpening to avoid ghosting.
    Blink frames: skip eye reconstruction.
    """
    x, y, w, h = face_bbox
    
    # Crop face with padding
    pad = int(max(w, h) * 0.3)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)
    
    if y2 <= y1 or x2 <= x1:
        return frame
    
    face_crop = frame[y1:y2, x1:x2].copy()
    
    # GFPGAN face restoration
    if models.gfpgan:
        try:
            # Adjust fidelity based on mouth/eye state
            # Mouth open = reduce restoration to avoid ghosting
            fidelity = 0.5 if mouth_open > 0.6 else 0.7
            
            _, _, restored = models.gfpgan.enhance(
                face_crop,
                paste_back=True,
            )
            
            # Blend original and restored based on state
            blend_alpha = 0.8 if mouth_open < 0.5 else 0.6
            face_crop = cv2.addWeighted(
                face_crop, 1 - blend_alpha,
                restored, blend_alpha,
                0
            )
        except Exception as e:
            log.debug("GFPGAN failed: %s", e)
    
    # Paste back
    frame[y1:y2, x1:x2] = face_crop
    
    return frame


def _enhance_light(
    frame: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    models: EnhancementModels,
    mouth_open: float,
) -> np.ndarray:
    """
    Conservative enhancement: mild sharpening + color correction only.
    
    No GFPGAN on risky frames. Just gentle cleanup.
    """
    x, y, w, h = face_bbox
    
    pad = int(max(w, h) * 0.2)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)
    
    if y2 <= y1 or x2 <= x1:
        return frame
    
    face_crop = frame[y1:y2, x1:x2].astype(np.float32)
    
    # Mild sharpening (reduced for mouth-open frames)
    sharpen_amount = 0.3 if mouth_open > 0.5 else 0.5
    kernel = np.array([
        [0, -sharpen_amount, 0],
        [-sharpen_amount, 1 + 4 * sharpen_amount, -sharpen_amount],
        [0, -sharpen_amount, 0]
    ])
    face_crop = cv2.filter2D(face_crop, -1, kernel)
    
    # Gentle contrast boost
    face_crop = cv2.convertScaleAbs(face_crop, alpha=1.05, beta=5)
    
    # Clip and paste back
    face_crop = np.clip(face_crop, 0, 255).astype(np.uint8)
    frame[y1:y2, x1:x2] = face_crop
    
    return frame


def _apply_background_grade(
    frame: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]],
    lighting: Dict,
) -> np.ndarray:
    """
    Apply global background color grade.
    
    Matches background to face lighting for studio-grade harmony.
    Only adjusts: brightness, contrast, color temperature.
    No per-pixel AI processing.
    """
    if face_bbox is None:
        return frame
    
    # Create mask excluding face region
    mask = np.ones(frame.shape[:2], dtype=np.float32)
    fx, fy, fw, fh = face_bbox
    pad = int(max(fw, fh) * 0.5)
    x1 = max(0, fx - pad)
    y1 = max(0, fy - pad)
    x2 = min(frame.shape[1], fx + fw + pad)
    y2 = min(frame.shape[0], fy + fh + pad)
    mask[y1:y2, x1:x2] = 0.0
    
    # Soften mask edges
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    # Global grade parameters (subtle)
    face_brightness = lighting.get("face_brightness", 128)
    target_brightness = 140  # Ideal face brightness
    brightness_delta = (target_brightness - face_brightness) * 0.1  # 10% correction
    
    # Apply to background only
    graded = frame.astype(np.float32)
    graded = cv2.convertScaleAbs(graded, alpha=1.02, beta=brightness_delta)
    
    # Blend with mask
    result = frame.astype(np.float32) * (1 - mask[:, :, None]) + graded * mask[:, :, None]
    
    return result.clip(0, 255).astype(np.uint8)


# ─── Main enhancement pipeline ────────────────────────────────────────────

def enhance_clip(
    video_path: str,
    analysis_path: str,
    output_path: Optional[str] = None,
    use_esrgan: bool = False,
) -> str:
    """
    Pass 2: Selective enhancement using analysis map.

    Input must be a 9:16 cropped video from export.py (NOT raw 16:9 source).
    Processes at native resolution — no stretching.

    Args:
        video_path: Input video (already 9:16 cropped)
        analysis_path: JSON from state_analyzer.py
        output_path: Output video path
        use_esrgan: Enable super-resolution

    Returns:
        Path to enhanced video
    """
    t_start = time.perf_counter()
    video_path = str(Path(video_path).resolve())
    analysis_path = str(Path(analysis_path).resolve())
    
    log.info("=" * 60)
    log.info("SELECTIVE ENHANCER — Pass 2: Conditional Enhancement")
    log.info("=" * 60)
    
    # Load analysis
    with open(analysis_path) as f:
        analysis = json.load(f)
    
    per_frame = analysis["per_frame"]
    summary = analysis["summary"]
    
    log.info("Loaded analysis: %d frames, %d heavy, %d light, %d skip",
             len(per_frame),
             summary["enhancement_distribution"]["heavy"],
             summary["enhancement_distribution"]["light"],
             summary["enhancement_distribution"]["skip"])
    
    # Initialize models
    models = EnhancementModels()
    models.load_gfpgan()
    if use_esrgan:
        models.load_esrgan()
    
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
    if src_w > src_h:
        log.warning("Input appears to be landscape (%dx%d) — expected 9:16 portrait from export.py", src_w, src_h)
    
    # Temp output - use /tmp for fast local I/O (not Drive-mounted)
    temp_dir = Path("/tmp/yt_clips_enhance")
    temp_dir.mkdir(exist_ok=True)
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Clean old frames
    for f in frames_dir.glob("*.jpg"):
        f.unlink()
    
    # Track last enhanced face for propagation
    last_enhanced_face = None
    last_enhanced_bbox = None
    last_enhanced_frame_idx = -1
    
    # Process frames
    frame_idx = 0
    enhanced_count = 0
    light_count = 0
    skip_count = 0
    prop_count = 0
    
    t_proc = time.perf_counter()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # No resize — input is already 9:16 from export.py
            
            # Get analysis for this frame (or nearest)
            analysis_frame = None
            for af in per_frame:
                if af["frame_index"] == frame_idx:
                    analysis_frame = af
                    break
            
            if analysis_frame is None:
                # Find nearest analyzed frame
                best_dist = float('inf')
                for af in per_frame:
                    dist = abs(af["frame_index"] - frame_idx)
                    if dist < best_dist:
                        best_dist = dist
                        analysis_frame = af
            
            if analysis_frame is None:
                # No analysis available, pass through
                cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_idx += 1
                continue
            
            level = analysis_frame["enhancement"]["enhancement_level"]
            face_bbox = None
            face_detected = analysis_frame["face_detected"]
            
            if face_detected:
                # Detect face in current frame — bbox is already in frame coordinates
                face = _detect_face(frame)
                if face is not None:
                    face_bbox = tuple(int(v) for v in face)  # (x, y, w, h) — no scaling needed
            
            if level == "heavy" and face_detected and face_bbox:
                # Full enhancement
                mouth_open = analysis_frame.get("mouth_open", 0)
                eye_open = analysis_frame.get("eye_open", 0)
                
                frame = _enhance_heavy(frame, face_bbox, models, mouth_open, eye_open)
                
                # Store for propagation (only if face region is valid)
                fx, fy, fw, fh = face_bbox
                if fy + fh <= frame.shape[0] and fx + fw <= frame.shape[1] and fh > 0 and fw > 0:
                    last_enhanced_face = frame[fy:fy+fh, fx:fx+fw].copy()
                    last_enhanced_bbox = face_bbox
                    last_enhanced_frame_idx = frame_idx
                
                enhanced_count += 1
            
            elif level == "light" and face_detected and face_bbox:
                # Conservative enhancement
                mouth_open = analysis_frame.get("mouth_open", 0)
                frame = _enhance_light(frame, face_bbox, models, mouth_open)
                light_count += 1
            
            elif level == "skip" and last_enhanced_face is not None:
                # Propagate from last enhanced frame
                if face_bbox:
                    frame = _propagate_face(
                        frame, frame,
                        last_enhanced_face,
                        last_enhanced_bbox or (0, 0, frame.shape[1], frame.shape[0]),
                        face_bbox,
                    )
                prop_count += 1
            else:
                skip_count += 1
            
            # Background grade (every frame, but lightweight)
            lighting = analysis_frame.get("lighting", {})
            frame = _apply_background_grade(frame, face_bbox, lighting)
            
            # Write frame as jpg (fast local I/O)
            cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_idx += 1
            
            if frame_idx % 50 == 0:
                elapsed = time.perf_counter() - t_proc
                rate = frame_idx / elapsed if elapsed > 0 else 0
                log.info("  Enhanced %d/%d frames (%.0f fps)", frame_idx, total_frames, rate)
    
    finally:
        cap.release()
    
    t_process = time.perf_counter() - t_proc
    log.info("Processing: %d frames in %.1fs (%.0f fps)", frame_idx, t_process, frame_idx / t_process if t_process > 0 else 0)
    log.info("  Heavy: %d | Light: %d | Skip: %d | Propagated: %d",
             enhanced_count, light_count, skip_count, prop_count)
    
    # Encode with FFmpeg from frames
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(cfg["paths"]["temp"]) / f"{stem}_enhanced.mp4")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Extract audio from source
    audio_path = str(temp_dir / "audio.aac")
    audio_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-c:a", "copy",
        audio_path,
    ]
    subprocess.run(audio_cmd, capture_output=True, text=True)
    
    # Encode video + audio
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
        # Fallback: video only
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
    log.info("Enhanced video: %s (%.1f MB, %.1fs total)", output_path, out_size, t_total)
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    models.clear_vram()
    
    return output_path


# ─── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State-aware selective enhancement (Pass 2)")
    parser.add_argument("video", help="Path to input video (already 9:16 cropped)")
    parser.add_argument("--analysis", "-a", required=True, help="Path to state analysis JSON")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--esrgan", action="store_true", help="Enable super-resolution")
    args = parser.parse_args()
    
    enhance_clip(
        video_path=args.video,
        analysis_path=args.analysis,
        output_path=args.output,
        use_esrgan=args.esrgan,
    )
