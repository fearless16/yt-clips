"""
super_res.py — Real-ESRGAN 4x super-resolution for frame upscaling.

Replaces lanczos interpolation with AI hallucinated detail.
405x720 (720p crop) → 1620x2880 (4x) → 1080x1920 (lanczos downscale)
Looks significantly sharper than direct 405x720 → 1080x1920 lanczos.

Colab T4: ~0.5-1s per frame (x4plus model)
Mac CPU: too slow, auto-skips with warning.

Usage:
    from utils.super_res import SuperResEnhancer
    enhancer = SuperResEnhancer(scale=4)
    if enhancer.available:
        enhancer.upscale_video(input_path, output_path)
"""
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("super_res", cfg["logging"]["log_file"], cfg["logging"]["level"])

# ─── Backend detection ────────────────────────────────────────────────────

HAS_TORCH = False
HAS_REALESRGAN = False
HAS_CUDA = False

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

try:
    import utils.torchvision_compat  # noqa: F401 — must precede realesrgan
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    HAS_REALESRGAN = True
except ImportError:
    pass


# ─── Super Resolution Enhancer ────────────────────────────────────────────

class SuperResEnhancer:
    """Real-ESRGAN 4x upscaler. No-op fallback when no GPU."""

    def __init__(self, scale: int = 4, model_name: str = "RealESRGAN_x4plus_anime_6B"):
        self.upsampler = None
        self.scale = scale
        self.model_name = model_name
        self.available = False

        if not (HAS_REALESRGAN and HAS_TORCH and HAS_CUDA):
            missing = []
            if not HAS_REALESRGAN:
                missing.append("realesrgan")
            if not HAS_TORCH:
                missing.append("torch")
            if not HAS_CUDA:
                missing.append("CUDA GPU")
            log.warning("Super-res unavailable: missing %s", ", ".join(missing))
            return

        WEIGHT_URLS = {
            "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        }
        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(exist_ok=True)
        local_path = weights_dir / f"{model_name}.pth"
        model_path = str(local_path) if local_path.exists() else WEIGHT_URLS.get(model_name, "")

        try:
            self.upsampler = RealESRGANer(
                scale=scale,
                model_path=model_path,
                model=self._build_model(),
                tile=512,
                tile_pad=10,
                pre_pad=0,
                half=True,
                device=None,
            )
            self.available = True
            log.info("Real-ESRGAN %dx loaded (%s, GPU)", scale, model_name)
        except Exception as e:
            log.warning("Real-ESRGAN load failed: %s", e)

    def _build_model(self):
        """Build RRDBNet architecture matching the model checkpoint."""
        if self.model_name == "RealESRGAN_x4plus":
            return RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4,
            )
        elif self.model_name == "RealESRGAN_x4plus_anime_6B":
            return RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=6, num_grow_ch=32, scale=4,
            )
        # Default
        return RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )

    def upscale_frame(self, img: "np.ndarray") -> "np.ndarray":
        """Upscale a single BGR numpy frame. Returns original on failure."""
        if not self.available or self.upsampler is None:
            return img
        try:
            output, _ = self.upsampler.enhance(img, outscale=self.scale)
            return output
        except Exception as e:
            log.debug("Frame upscale failed: %s", e)
            return img

    def upscale_video(
        self,
        input_path: str,
        output_path: str,
        target_w: int = 1080,
        target_h: int = 1920,
    ) -> bool:
        """
        Upscale every frame of a video using Real-ESRGAN.
        Extracts frames → upscales → re-encodes with ffmpeg.
        """
        if not self.available:
            log.warning("Super-res not available; copying input unchanged")
            Path(output_path).write_bytes(Path(input_path).read_bytes())
            return True

        import cv2
        import numpy as np

        input_p = Path(input_path)
        output_p = Path(output_path)
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # 1. Extract frames
            frames_dir = temp_dir / "frames"
            frames_dir.mkdir()
            cmd_extract = (
                f'ffmpeg -y -i "{input_path}" '
                f'"{frames_dir}/%06d.png"'
            )
            import subprocess
            r = subprocess.run(cmd_extract, shell=True, capture_output=True, text=True)
            if r.returncode != 0:
                log.error("Frame extraction failed: %s", r.stderr[-200:])
                return False

            frame_files = sorted(frames_dir.glob("*.png"))
            if not frame_files:
                log.error("No frames extracted")
                return False

            log.info("Super-res: %d frames to upscale (%dx)", len(frame_files), self.scale)

            # 2. Upscale each frame
            upscaled_dir = temp_dir / "upscaled"
            upscaled_dir.mkdir()
            import time
            t_sr = time.perf_counter()

            for i, fp in enumerate(frame_files):
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                t_frame = time.perf_counter()
                sr_img = self.upscale_frame(img)
                frame_ms = (time.perf_counter() - t_frame) * 1000
                # Downscale to target resolution
                if sr_img.shape[0] != target_h or sr_img.shape[1] != target_w:
                    sr_img = cv2.resize(sr_img, (target_w, target_h),
                                        interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(str(upscaled_dir / fp.name), sr_img)

                if (i + 1) % 5 == 0 or i == 0:
                    elapsed = time.perf_counter() - t_sr
                    eta = (elapsed / (i + 1)) * (len(frame_files) - i - 1)
                    log.info("  Super-res: %d/%d frames (%.0fms/frame, ETA %.0fs)",
                             i + 1, len(frame_files), frame_ms, eta)

            log.info("Super-res upscale done in %.1fs", time.perf_counter() - t_sr)

            # 3. Get audio from original
            audio_tmp = temp_dir / "audio.aac"
            cmd_audio = (
                f'ffmpeg -y -i "{input_path}" '
                f'-vn -acodec copy "{audio_tmp}"'
            )
            subprocess.run(cmd_audio, shell=True, capture_output=True)

            # 4. Re-encode with audio
            has_audio = audio_tmp.exists() and audio_tmp.stat().st_size > 0

            cmd_encode = (
                f'ffmpeg -y -framerate 30 '
                f'-i "{upscaled_dir}/%06d.png" '
            )
            if has_audio:
                cmd_encode += f'-i "{audio_tmp}" -map 0:v -map 1:a '
            cmd_encode += (
                f'-c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p '
                f'-c:a aac -b:a 192k -shortest '
                f'"{output_path}"'
            )

            log.info("Super-res: re-encoding %d frames → %s", len(frame_files), Path(output_path).name)
            t_enc = time.perf_counter()
            r = subprocess.run(cmd_encode, shell=True, capture_output=True, text=True)
            if r.returncode != 0:
                log.error("Super-res encode failed: %s", r.stderr[-200:])
                return False

            t_total = time.perf_counter() - t_sr
            t_encode = time.perf_counter() - t_enc
            log.info("✅ Super-res done in %.1fs (upscale=%.1fs, encode=%.1fs): %s",
                     t_total, t_total - t_encode, t_encode, output_path)
            return True

        except Exception as e:
            log.error("Super-res crash: %s", e)
            return False
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def upscale_frames_in_dir(
    frames_dir: str,
    scale: int = 4,
    target_w: int = 1080,
    target_h: int = 1920,
) -> str:
    """
    Upscale all PNG frames in a directory in-place.
    Returns the directory path (same as input).
    """
    import cv2
    import numpy as np

    enhancer = SuperResEnhancer(scale=scale)
    if not enhancer.available:
        log.warning("Super-res not available; leaving frames unchanged")
        return frames_dir

    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob("*.png"))

    if not frame_files:
        return frames_dir

    log.info("Super-res: %d frames in %s", len(frame_files), frames_dir)

    for i, fp in enumerate(frame_files):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        sr_img = enhancer.upscale_frame(img)
        if sr_img.shape[0] != target_h or sr_img.shape[1] != target_w:
            sr_img = cv2.resize(sr_img, (target_w, target_h),
                                interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(fp), sr_img)

        if (i + 1) % 10 == 0 or i == 0:
            log.info("  Super-res: %d/%d", i + 1, len(frame_files))

    return frames_dir
