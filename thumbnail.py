import os
import json
import random
import subprocess
from pathlib import Path
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("thumbnail", cfg["logging"]["log_file"], cfg["logging"]["level"])

def _find_best_font() -> str:
    """Find a professional font available on the current OS."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Impact.ttf",          # macOS
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", # Linux/Colab
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",   # Linux
        "C:\\Windows\\Fonts\\impact.ttf",                         # Windows
        "impact.ttf",                                             # Local fallback
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""  # Fallback to default in _add_text

class ThumbnailGenerator:
    def __init__(self):
        self.config = cfg.get("thumbnail", {})
        self.font_path = self.config.get("font_path") or _find_best_font()
        self.font_size = self.config.get("font_size", 120)
        self.text_color = self.config.get("text_color", "#FFFFFF")
        self.stroke_color = self.config.get("stroke_color", "#000000")
        self.stroke_width = self.config.get("stroke_width", 8)

    def generate_for_clip(self, video_path: str, metadata_path: str, output_path: str):
        """Generates a thumbnail for a given clip using AI if available."""
        log.info(f"Generating thumbnail for {video_path}...")
        
        # Load metadata for text and context
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        
        title = meta.get("title", "Cricket Highlights")
        
        # ─── Option 1: AI Generation (Nano Banana Pro) ─────────────────────────
        from utils.ai_client import AIClient
        ai = AIClient()
        
        if self.config.get("use_ai", False):
            prompt = f"Cricket action shot: {title}. High contrast, 4K, professional YouTube Short style."
            success = ai.generate_image(prompt, output_path)
            if success:
                log.info(f"✅ AI Thumbnail (Nano Banana Pro) saved to {output_path}")
                return True
            log.warning("AI Thumbnail generation failed, falling back to frame extraction.")

        # ─── Option 2: Frame Extraction + Pillow (Fallback) ────────────────────
        # 1. Extract a frame (using ffmpeg)
        frame_path = Path(video_path).with_suffix(".jpg")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-ss", "00:00:01", "-i", video_path,
                "-vframes", "1", str(frame_path), "-loglevel", "error"
            ], check=True)
        except subprocess.CalledProcessError as e:
            log.error(f"FFmpeg frame extraction failed: {e}")
            return False
        
        if not frame_path.exists():
            log.error(f"Failed to extract frame from {video_path}")
            return False

        # 2. Open frame and apply effects
        img = Image.open(frame_path).convert("RGB")
        
        # Split title into 2-3 words for the thumbnail text
        words = title.split()
        short_text = " ".join(words[:3]).upper() if words else "CRICKET LIVE"

        # Apply enhancements
        img = self._apply_enhancements(img)
        
        # Add text
        img = self._add_text(img, short_text)
        
        # Add logo/watermark if exists
        logo_path = self.config.get("template_path")
        if logo_path and os.path.exists(logo_path):
            img = self._add_logo(img, logo_path)

        # Save final thumbnail
        img.save(output_path, quality=95)
        log.info(f"✅ Thumbnail saved to {output_path}")
        
        # Cleanup temp frame
        if frame_path.exists():
            os.remove(frame_path)
            
        return True

    def _apply_enhancements(self, img: Image.Image) -> Image.Image:
        """Apply subtle blur to edges or contrast boost."""
        # Optional: Add a slight vignette or contrast boost here
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        return img

    def _add_text(self, img: Image.Image, text: str) -> Image.Image:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(self.font_path, self.font_size)
        except:
            log.warning(f"Font not found at {self.font_path}, using default.")
            font = ImageFont.load_default()

        w, h = img.size
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        x = (w - tw) // 2
        y = h - th - 150  # 150px from bottom

        # ─── Add Semi-Transparent Overlay for Contrast ────────────────────────
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        # Gradient or solid block at the bottom
        overlay_draw.rectangle([0, y - 20, w, h], fill=(0, 0, 0, 100))
        img.paste(overlay, (0, 0), overlay)

        # ─── Draw Drop Shadow (Blurred) ───────────────────────────────────────
        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.text((x + 5, y + 5), text, font=font, fill=(0, 0, 0, 200))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=3))
        img.paste(shadow, (0, 0), shadow)
        
        # ─── Draw Stroke/Outline (Traditional) ────────────────────────────────
        for adj in range(-self.stroke_width, self.stroke_width + 1):
            for adj2 in range(-self.stroke_width, self.stroke_width + 1):
                draw.text((x + adj, y + adj2), text, font=font, fill=self.stroke_color)

        # ─── Draw Main Text ───────────────────────────────────────────────────
        draw.text((x, y), text, font=font, fill=self.text_color)
        return img

    def _add_logo(self, img: Image.Image, logo_path: str) -> Image.Image:
        logo = Image.open(logo_path).convert("RGBA")
        # Resize logo to ~15% of width
        w, h = img.size
        logo_w = int(w * 0.15)
        logo_h = int(logo_w * (logo.height / logo.width))
        logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
        
        # Place logo in top-right
        img.paste(logo, (w - logo_w - 40, 40), logo)
        return img

def process_all_thumbnails(shorts_dir: str):
    generator = ThumbnailGenerator()
    shorts_path = Path(shorts_dir)
    
    # Find all .mp4 files that have a matching _metadata.json
    videos = list(shorts_path.glob("*.mp4"))
    for video in videos:
        meta_path = video.with_name(f"{video.stem}_metadata.json")
        if meta_path.exists():
            thumb_path = video.with_name(f"{video.stem}_thumb.jpg")
            generator.generate_for_clip(str(video), str(meta_path), str(thumb_path))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_all_thumbnails(sys.argv[1])
    else:
        log.error("Usage: python thumbnail.py <shorts_directory>")


def generate_thumbnail_variants(video_path: str, metadata: dict, count: int = 3) -> List[str]:
    """
    Generate multiple thumbnail variants for A/B testing.
    
    Args:
        video_path: Path to the video file
        metadata: Metadata dict with title, clip_id, etc.
        count: Number of variants to generate (default: 3)
        
    Returns:
        List of paths to generated thumbnail variants
    """
    import shutil
    generator = ThumbnailGenerator()
    video_file = Path(video_path)
    output_dir = video_file.parent
    
    variants = []
    
    # First generate base thumbnail if it doesn't exist
    base_thumb = output_dir / "thumbnail.jpg"
    if not base_thumb.exists():
        meta_path = output_dir / "metadata.json"
        if meta_path.exists():
            generator.generate_for_clip(str(video_file), str(meta_path), str(base_thumb))
    
    # Create variants by copying and modifying
    base_title = metadata.get("title", "Cricket Highlights")
    
    for i in range(count):
        variant_output = output_dir / f"thumbnail_v{i+1}.jpg"
        
        try:
            # Copy base thumbnail
            if base_thumb.exists():
                shutil.copy(base_thumb, variant_output)
                
                # Modify with variant text
                from PIL import Image, ImageDraw, ImageFont
                
                img = Image.open(variant_output)
                draw = ImageDraw.Draw(img)
                w, h = img.size
                
                # Variant titles
                if i == 0:
                    text = base_title[:45]
                elif i == 1:
                    text = f"{base_title[:40]} 🔥"
                else:
                    text = f"🏏 {base_title[:40]}"
                
                font_size = int(h * 0.07)
                try:
                    font = ImageFont.truetype(generator.font_path, font_size)
                except:
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                x = (w - text_w) // 2
                y = h - text_h - 100
                
                # Clear area and redraw
                draw.rectangle([x-5, y-5, x+text_w+5, y+text_h+30], fill="#00000080")
                draw.text((x+2, y+2), text, font=font, fill="#000000")
                draw.text((x, y), text, font=font, fill="#FFFFFF")
                
                img.save(variant_output, quality=95)
                variants.append(str(variant_output))
            else:
                # Fallback: create placeholder
                variants.append(str(base_thumb) if base_thumb.exists() else str(video_file.with_suffix(".txt")))
                
        except Exception as e:
            log.error(f"Failed to generate variant {i+1}: {e}")
            if base_thumb.exists():
                variants.append(str(base_thumb))
    
    return variants
