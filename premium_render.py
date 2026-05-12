"""
premium_render.py — FILM/RIFE frame interpolation + GFPGAN face enhancement
+ Gaussian-smoothed speed profile + two-pass VBR encoding.

Usage (Colab T4):
    from premium_render import PremiumRender
    pr = PremiumRender()
    pr.render_clip(input_path, output_path, start, end, speed_map)
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from utils.config import load_config
from utils.logger import get_logger, ProgressManager

cfg = load_config()
log = get_logger("premium_render", cfg["logging"]["log_file"], cfg["logging"]["level"])

# ─── Backend detection ────────────────────────────────────────────────────

HAS_TORCH = False
HAS_GFPGAN = False
try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    pass

try:
    from gfpgan import GFPGANer
    HAS_GFPGAN = True
except ImportError:
    pass


# ─── Gaussian Speed Profile ───────────────────────────────────────────────

def generate_speed_profile(
    duration: float,
    fps: float = 60,
    base_speed: float = 1.0,
    max_speed: float = 1.25,
    silence_regions: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Gaussian-smoothed speed profile for a clip.
    
    Returns:
        time_stamps: array of time points (seconds)
        speed_factors: array of speed multipliers (1.0-1.25)
    """
    n_frames = min(max(2, int(duration * fps)), 60000)
    time_stamps = np.linspace(0, duration, n_frames)
    speed = np.ones(n_frames) * base_speed

    if silence_regions:
        for s_start, s_end in silence_regions:
            mask = (time_stamps >= s_start) & (time_stamps <= s_end)
            speed[mask] = max_speed

    # Gaussian smoothing — no abrupt changes
    from scipy.ndimage import gaussian_filter1d
    sigma = max(1.0, duration * fps * 0.05)  # 5% of clip as transition window
    speed = gaussian_filter1d(speed, sigma=sigma, mode="nearest")
    speed = np.clip(speed, base_speed, max_speed)

    return time_stamps, speed


# ─── Frame Interpolation (RIFE / FILM stub) ──────────────────────────────

class FrameInterpolator:
    """30→60fps via RIFE or FILM. Falls back to FFmpeg framerate filter."""

    def __init__(self):
        self.model = None
        self.backend = "ffmpeg"
        if HAS_TORCH:
            try:
                self._load_rife()
            except Exception as e:
                log.warning("RIFE load failed: %s — using FFmpeg fallback", e)

    def _load_rife(self):
        try:
            from torch.nn import functional as F
            import torch
            import sys
            sys.path.insert(0, str(Path.cwd() / "RIFE"))
            from model.RIFE_HDv3 import Model
            model = Model()
            model.load_model(Path.cwd() / "RIFE" / "train_log")
            model.eval()
            model.device()
            self.model = model
            self.backend = "rife"
            log.info("RIFE frame interpolation loaded (GPU)")
        except Exception as e:
            log.warning("RIFE model init failed: %s", e)
            raise

    def interpolate(
        self,
        video_path: str,
        start: float,
        end: float,
        output_path: str,
        speed_profile: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Render interpolated video at 60fps.
        Uses RIFE if available, otherwise FFmpeg framerate filter.
        """
        clip_duration = end - start
        target_fps = 60

        if self.backend == "rife" and self.model:
            return self._interpolate_rife(video_path, start, end, output_path, speed_profile)
        return self._interpolate_ffmpeg(video_path, start, end, output_path, clip_duration, target_fps, speed_profile)

    def _interpolate_rife(self, video_path, start, end, output_path, speed_profile):
        import torch
        import cv2
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w, out_h = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(output_path, fourcc, 60.0, (out_w, out_h))

        frames = []
        while cap.get(cv2.CAP_PROP_POS_MSEC) < end * 1000:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if len(frames) == 2:
                from torch.nn import functional as F
                I0 = torch.from_numpy(frames[0]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                I1 = torch.from_numpy(frames[1]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                if torch.cuda.is_available():
                    I0 = I0.cuda()
                    I1 = I1.cuda()
                with torch.no_grad():
                    mid = self.model.inference(I0, I1)
                mid_np = (mid.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                writer.write(frames[0])
                writer.write(mid_np)
                frames = [frames[1]]

        if frames:
            writer.write(frames[0])
        cap.release()
        writer.release()
        return True

    def _interpolate_ffmpeg(self, video_path, start, end, output_path, clip_duration, target_fps, speed_profile):
        avg_speed = float(np.mean(speed_profile)) if speed_profile is not None else 1.0
        # Build atempo chain (FFmpeg limits atempo to 0.5-2.0 range)
        tempo_filters = []
        remaining = avg_speed
        while remaining > 2.0:
            tempo_filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            tempo_filters.append("atempo=0.5")
            remaining *= 2.0
        tempo_filters.append(f"atempo={remaining:.6f}")
        af = ",".join(tempo_filters)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{clip_duration:.3f}",
            "-i", video_path,
            "-vf", f"setpts={1/avg_speed:.3f}*PTS,framerate=fps={target_fps}:interp_start=0:interp_end=1:scene=0.3",
            "-af", af,
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("FFmpeg interpolation failed: %s", result.stderr[:500])
            return False
        return True


# ─── Face Enhancement (GFPGAN) ────────────────────────────────────────────

class FaceEnhancer:
    """GFPGAN face restoration. No-op fallback on CPU."""

    def __init__(self):
        self.enhancer = None
        self.backend = "none"
        if HAS_GFPGAN and HAS_TORCH:
            try:
                self.enhancer = GFPGANer(
                    model_path="experiments/pretrained_models/GFPGANv1.4.pth",
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                )
                self.backend = "gfpgan"
                log.info("GFPGAN face enhancer loaded (GPU)")
            except Exception as e:
                log.warning("GFPGAN load failed: %s", e)

    def enhance_clip(self, video_path: str, output_path: str) -> bool:
        """Apply GFPGAN to every Nth frame for consistency. No-op if no GPU."""
        if self.backend == "none":
            Path(output_path).write_bytes(Path(video_path).read_bytes())
            return True
        return self._enhance_frames(video_path, output_path)

    def _enhance_frames(self, video_path: str, output_path: str) -> bool:
        import cv2
        import shutil
        temp_dir = Path(tempfile.mkdtemp())
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                log.error("Cannot open video for face enhancement")
                return False
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            w = int(cap.get(3))
            h = int(cap.get(4))
            if w <= 0 or h <= 0:
                cap.release()
                return False
            temp_out = temp_dir / "enhanced.mp4"
            writer = cv2.VideoWriter(str(temp_out), fourcc, fps, (w, h))
            if not writer.isOpened():
                cap.release()
                return False

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % 10 == 0 and self.enhancer:
                    _, _, frame = self.enhancer.enhance(frame, paste_back=True)
                writer.write(frame)
                frame_idx += 1

            cap.release()
            writer.release()
            if temp_out.exists() and temp_out.stat().st_size > 0:
                shutil.copy2(str(temp_out), output_path)
                return True
            return False
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ─── Two-Pass VBR Encoding ───────────────────────────────────────────────

def encode_two_pass(input_path: str, output_path: str, bitrate: str = "15M") -> bool:
    """Two-pass VBR for optimal bit allocation."""
    import shutil, glob as gb
    logbase = tempfile.mktemp(suffix="")

    def _parse_bitrate(br: str) -> tuple:
        """Return (value, suffix) from '15M' or '1500K'. Fallback to 8M on junk."""
        try:
            if br and br[-1].upper() in ("M", "K"):
                val, suffix = br[:-1], br[-1].upper()
                float(val)
                return val, suffix
            if br:
                float(br)
                return br, "M"
        except (ValueError, TypeError, IndexError):
            pass
        return "8", "M"

    bval, bsuffix = _parse_bitrate(bitrate)
    maxrate = f"{float(bval) * 1.5:.0f}{bsuffix}"
    bufsize = f"{float(bval) * 2:.0f}{bsuffix}"

    pass1 = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "veryfast",
        "-b:v", bitrate, "-maxrate", maxrate,
        "-bufsize", bufsize,
        "-pass", "1", "-passlogfile", logbase,
        "-f", "null", "-",
    ]
    r1 = subprocess.run(pass1, capture_output=True, text=True)
    if r1.returncode != 0:
        log.error("VBR pass 1 failed: %s", r1.stderr[:500])
        for f in gb.glob(f"{logbase}*"):
            Path(f).unlink(missing_ok=True)
        return False

    pass2 = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "veryfast",
        "-b:v", bitrate, "-maxrate", maxrate,
        "-bufsize", bufsize,
        "-pass", "2", "-passlogfile", logbase,
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]
    r2 = subprocess.run(pass2, capture_output=True, text=True)
    for f in gb.glob(f"{logbase}*"):
        Path(f).unlink(missing_ok=True)
    if r2.returncode != 0:
        log.error("VBR pass 2 failed: %s", r2.stderr[:500])
        return False
    return True


# ─── Master Render Pipeline ──────────────────────────────────────────────

class PremiumRender:
    def __init__(self):
        self.interpolator = FrameInterpolator()
        self.enhancer = FaceEnhancer()

    def render_clip(
        self,
        video_path: str,
        start: float,
        end: float,
        output_path: str,
        clip_id: str = "clip",
        face_enhance: bool = True,
        two_pass: bool = True,
        silence_regions: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[str]:
        """Full premium render: interpolation → enhancement → encode."""
        import shutil
        temps = []
        try:
            with ProgressManager() as pm:
                pm.add(f"{clip_id}: render", total=100)
                temp1 = Path(tempfile.mktemp(suffix=".mp4"))
                temps.append(temp1)
                duration = end - start
                speed_profile = generate_speed_profile(
                    duration,
                    silence_regions=silence_regions,
                )

                pm.update(f"{clip_id}: render", description=f"{clip_id}: interpolate 30→60fps")
                ok = self.interpolator.interpolate(str(temp1.parent / "interp.mp4"), start, end, str(temp1), speed_profile[1])
                if not ok:
                    log.error("[%s] Interpolation failed", clip_id)
                    return None
                pm.update(f"{clip_id}: render", advance=50)

                if face_enhance and self.enhancer.backend != "none":
                    pm.update(f"{clip_id}: render", description=f"{clip_id}: face enhancement")
                    temp2 = Path(tempfile.mktemp(suffix=".mp4"))
                    temps.append(temp2)
                    ok = self.enhancer.enhance_clip(str(temp1), str(temp2))
                    if not ok:
                        log.warning("[%s] Face enhancement failed, using interpolated", clip_id)
                    else:
                        temp1 = temp2
                    pm.update(f"{clip_id}: render", advance=25)

                if two_pass:
                    pm.update(f"{clip_id}: render", description=f"{clip_id}: two-pass VBR")
                    ok = encode_two_pass(str(temp1), output_path, bitrate=cfg["export"].get("video_bitrate", "15M"))
                else:
                    shutil.copy2(str(temp1), output_path)
                    ok = True

                pm.update(f"{clip_id}: render", advance=25)
                if ok:
                    log.info("[%s] Premium render done → %s", clip_id, output_path)
                return output_path if ok else None
        finally:
            for t in temps:
                try:
                    t.unlink(missing_ok=True)
                except OSError:
                    pass
