"""
replicate_enhance.py — Cloud video enhancement via Replicate API.

Provides face-aware reframe (16:9 → 9:16) and video upscaling without local GPU.
Uses Replicate's pay-per-second GPU pricing (~$0.003-0.13/sec).

Models:
  - luma/reframe-video: AI aspect ratio conversion, keeps subject centered
  - topazlabs/video-upscale: Professional upscaling to 4K
  - lucataco/real-esrgan-video: Budget upscaling

Usage:
    from utils.replicate_enhance import ReplicateEnhancer
    enhancer = ReplicateEnhancer()
    if enhancer.available:
        enhancer.reframe(input_path, output_path, aspect_ratio="9:16")
        enhancer.upscale(input_path, output_path, scale=2)
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("replicate_enhance", cfg["logging"]["log_file"], cfg["logging"]["level"])

HAS_REPLICATE = False
try:
    import replicate
    HAS_REPLICATE = True
except ImportError:
    pass


class ReplicateEnhancer:
    """Cloud video enhancement via Replicate API."""

    def __init__(self):
        self.available = HAS_REPLICATE and os.environ.get("REPLICATE_API_TOKEN", "")
        if not self.available and HAS_REPLICATE and not os.environ.get("REPLICATE_API_TOKEN"):
            log.warning("REPLICATE_API_TOKEN not set — cloud enhancement disabled")
        elif not HAS_REPLICATE:
            log.info("replicate package not installed — cloud enhancement disabled (pip install replicate)")

    def reframe(
        self,
        input_path: str,
        output_path: str,
        aspect_ratio: str = "9:16",
        model: str = "luma/reframe-video",
    ) -> Optional[str]:
        """AI-powered aspect ratio conversion. Keeps subject centered.

        Args:
            input_path: Source video path.
            output_path: Destination path for reframed video.
            aspect_ratio: Target ratio (9:16, 1:1, 16:9, etc.).
            model: Replicate model to use.

        Returns:
            Output path on success, None on failure.
        """
        if not self.available:
            log.debug("Replicate not available — skipping reframe")
            return None

        log.info("Replicate reframe: %s → %s (%s)", input_path, output_path, aspect_ratio)
        try:
            with open(input_path, "rb") as f:
                output = replicate.run(
                    model,
                    input={
                        "video": f,
                        "aspect_ratio": aspect_ratio,
                    },
                )

            # Replicate returns a URL or file object
            output_url = str(output) if not isinstance(output, str) else output

            # Download the result
            import urllib.request
            urllib.request.urlretrieve(output_url, output_path)

            out_size = Path(output_path).stat().st_size
            log.info("Reframe complete: %s (%.1f MB)", output_path, out_size / 1e6)
            return output_path

        except Exception as e:
            log.error("Reframe failed: %s", e)
            return None

    def upscale(
        self,
        input_path: str,
        output_path: str,
        scale: int = 2,
        model: str = "lucataco/real-esrgan-video",
    ) -> Optional[str]:
        """Video upscaling via cloud GPU.

        Args:
            input_path: Source video path.
            output_path: Destination path for upscaled video.
            scale: Upscale factor (2, 4).
            model: Replicate model to use.

        Returns:
            Output path on success, None on failure.
        """
        if not self.available:
            log.debug("Replicate not available — skipping upscale")
            return None

        log.info("Replicate upscale: %s → %s (%dx)", input_path, output_path, scale)
        try:
            with open(input_path, "rb") as f:
                output = replicate.run(
                    model,
                    input={
                        "video": f,
                        "scale": scale,
                    },
                )

            output_url = str(output) if not isinstance(output, str) else output

            import urllib.request
            urllib.request.urlretrieve(output_url, output_path)

            out_size = Path(output_path).stat().st_size
            log.info("Upscale complete: %s (%.1f MB)", output_path, out_size / 1e6)
            return output_path

        except Exception as e:
            log.error("Upscale failed: %s", e)
            return None

    def enhance_clip(
        self,
        input_path: str,
        output_path: str,
        do_reframe: bool = True,
        do_upscale: bool = True,
        reframe_model: str = "luma/reframe-video",
        upscale_model: str = "lucataco/real-esrgan-video",
        upscale_scale: int = 2,
    ) -> Optional[str]:
        """Full enhancement pipeline: reframe → upscale.

        Args:
            input_path: Source clip path.
            output_path: Final enhanced clip path.
            do_reframe: Whether to reframe aspect ratio.
            do_upscale: Whether to upscale resolution.
            reframe_model: Model for reframe step.
            upscale_model: Model for upscale step.
            upscale_scale: Scale factor for upscale.

        Returns:
            Output path on success, None on failure.
        """
        if not self.available:
            return None

        temp_dir = tempfile.mkdtemp(prefix="replicate_")
        current = input_path
        result = None

        try:
            if do_reframe:
                reframe_out = os.path.join(temp_dir, "reframed.mp4")
                log.info("Step 1/2: Reframe → 9:16")
                current = self.reframe(current, reframe_out, model=reframe_model) or current

            if do_upscale:
                upscale_out = output_path if not do_reframe else os.path.join(temp_dir, "upscaled.mp4")
                log.info("Step 2/2: Upscale %dx", upscale_scale)
                result = self.upscale(current, upscale_out, scale=upscale_scale, model=upscale_model)

                if do_reframe and result:
                    shutil.copy2(result, output_path)
                    result = output_path
            else:
                if current != input_path:
                    shutil.copy2(current, output_path)
                    result = output_path

            return result

        except Exception as e:
            log.error("Enhancement pipeline failed: %s", e)
            return None

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
