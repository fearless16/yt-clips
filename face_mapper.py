"""
face_mapper.py — Reference-guided face enhancement (expectation.png specific).

Extracted reference profile from expectation.png deep analysis:
- Lighting: Dramatic side light from RIGHT (left_cheek=69.6, right_cheek=147.7)
- Skin: Warm yellow-orange (NOT red). LAB avg=(108.5, 139.6, 146.7)
- Shadows: Cool/blue (B=22), Highlights: Warm/red (R=85) — split-tone grading
- Background: Dark top (20-32) → blue bottom (177-181)
- Contrast: 39.5% shadows, 43.6% midtones, 16.8% highlights
- Vignette: 21% center-brighter-than-edges
- Sharpness: Laplacian std=15.2, detail layer std=8.6
- Saturation: mean=125, range=60-241
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

from utils.config import load_config
from utils.logger import get_logger
from utils.face_detect import detect_face

cfg = load_config()
log = get_logger("face_mapper", cfg["logging"]["log_file"], cfg["logging"]["level"])


# ─── Reference parameters (extracted from expectation.png) ───────────────

REF = {
    # Skin tone per face region (LAB)
    "skin_forehead":    (160.5, 143.4, 152.6),
    "skin_right_cheek": (147.7, 144.8, 153.0),  # highlight side
    "skin_left_cheek":  (69.6, 135.3, 139.1),    # shadow side
    "skin_nose":        (136.7, 147.2, 151.1),
    "skin_chin":        (107.3, 142.1, 144.1),
    "skin_lips":        (167.6, 149.8, 155.0),
    # Average skin (weighted by area)
    "skin_avg_lab":     (108.5, 139.6, 146.7),

    # Lighting
    "face_brightness":  103,
    "face_contrast":    59,
    "lr_ratio":         2.12,     # right/left brightness ratio
    "light_direction":  "right",  # light comes from right

    # Shadow/highlight distribution
    "shadow_pct":       39.5,     # <85 brightness
    "midtone_pct":      43.6,     # 85-170
    "highlight_pct":    16.8,     # >170
    "peak_brightness":  175,

    # Color grading (split-tone)
    "shadow_color_bgr": (22, 17, 12),    # cool blue shadows
    "highlight_color_bgr": (0, 10, 85),  # warm red highlights

    # Sharpness
    "laplacian_std":    15.2,
    "detail_std":       8.6,
    "face_sharpness":   232,     # Laplacian variance

    # Saturation
    "face_saturation":  125,
    "sat_range":        (60, 241),

    # Vignette
    "vignette_ratio":   1.21,    # center/edge brightness

    # Background
    "bg_top_bgr":       (35, 27, 20),     # dark warm
    "bg_bottom_bgr":    (132, 174, 207),  # blue

    # Color temperature
    "color_temp":       1.89,    # R/B ratio
}


class ReferenceProfile:
    """Reference profile from expectation.png deep analysis."""

    def __init__(self):
        self.valid = False
        # Use hardcoded values from deep analysis (more precise than runtime extraction)
        self.skin_lab = REF["skin_avg_lab"]
        self.face_brightness = REF["face_brightness"]
        self.face_contrast = REF["face_contrast"]
        self.face_sharpness = REF["face_sharpness"]
        self.color_temp = REF["color_temp"]
        self.lr_ratio = REF["lr_ratio"]
        self.shadow_color = np.array(REF["shadow_color_bgr"], dtype=np.float64)
        self.highlight_color = np.array(REF["highlight_color_bgr"], dtype=np.float64)
        self.vignette_ratio = REF["vignette_ratio"]
        self.face_saturation = REF["face_saturation"]
        self.valid = True

    @classmethod
    def from_image(cls, image_path: str) -> "ReferenceProfile":
        """Extract profile from reference image (for custom references)."""
        profile = cls()  # Uses hardcoded defaults from expectation.png

        img = cv2.imread(image_path)
        if img is None:
            log.warning("Cannot read reference: %s — using defaults", image_path)
            return profile

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

        if len(faces) == 0:
            log.warning("No face in reference: %s — using defaults", image_path)
            return profile

        x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])
        face_roi = img[y:y+fh, x:x+fw]
        face_lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)

        # Override with extracted values
        profile.skin_lab = (
            float(np.mean(face_lab[:,:,0])),
            float(np.mean(face_lab[:,:,1])),
            float(np.mean(face_lab[:,:,2])),
        )
        profile.face_brightness = float(np.mean(gray[y:y+fh, x:x+fw]))
        profile.face_contrast = float(np.std(gray[y:y+fh, x:x+fw]))
        profile.face_sharpness = float(np.var(cv2.Laplacian(gray[y:y+fh, x:x+fw], cv2.CV_64F)))

        bgr_means = [float(np.mean(face_roi[:,:,i])) for i in range(3)]
        profile.color_temp = bgr_means[2] / max(bgr_means[0], 1)

        left = float(np.mean(cv2.cvtColor(face_roi[:, :fw//2], cv2.COLOR_BGR2GRAY)))
        right = float(np.mean(cv2.cvtColor(face_roi[:, fw//2:], cv2.COLOR_BGR2GRAY)))
        profile.lr_ratio = right / max(left, 1)

        face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        profile.face_saturation = float(np.mean(face_hsv[:,:,1]))

        log.info("Reference extracted: LAB=(%.0f,%.0f,%.0f) bright=%.0f lr_ratio=%.2f",
                 *profile.skin_lab, profile.face_brightness, profile.lr_ratio)

        return profile


# ─── Enhancement functions ────────────────────────────────────────────────

def apply_split_tone(frame: np.ndarray, shadow_color: np.ndarray, highlight_color: np.ndarray) -> np.ndarray:
    """
    Apply split-tone color grading: cool shadows + warm highlights.
    
    Reference: shadows are blue (BGR 22,17,12), highlights are warm (BGR 0,10,85).
    This creates the cinematic studio look.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    # Shadow/highlight masks (soft blend)
    shadow_mask = np.clip((128 - gray) / 128, 0, 1)  # 1 at black, 0 at mid
    highlight_mask = np.clip((gray - 128) / 128, 0, 1)  # 0 at mid, 1 at white
    
    result = frame.astype(np.float64)
    
    # Add shadow color (cool blue tint) — very subtle
    for c in range(3):
        result[:,:,c] += shadow_color[c] * shadow_mask * 0.15  # 15% strength
    
    # Add highlight color (warm red tint) — very subtle
    for c in range(3):
        result[:,:,c] += highlight_color[c] * highlight_mask * 0.08  # 8% strength
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_vignette(frame: np.ndarray, ratio: float = 1.21) -> np.ndarray:
    """
    Apply vignette: center brighter than edges.
    
    Reference: center is 1.21x brighter than edges.
    """
    h, w = frame.shape[:2]
    
    # Create radial gradient
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w / 2, h / 2
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Normalize: 1 at center, 0 at edges — very subtle
    vignette = 1 - (dist / max_dist) * (1 - 1/ratio) * 0.5  # 50% of full vignette
    vignette = vignette[:,:,None]  # Add channel dim
    
    result = frame.astype(np.float64) * vignette
    return np.clip(result, 0, 255).astype(np.uint8)


def match_skin_tone(source: np.ndarray, target_lab: Tuple[float, float, float]) -> np.ndarray:
    """
    Match skin tone to reference LAB values.
    
    Reference skin is warm yellow-orange (NOT red).
    Source is too red (a=149 vs ref a=139.6).
    
    Global correction to prevent flicker.
    """
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float64)
    
    for i in range(3):
        src_mean = np.mean(src_lab[:,:,i])
        tgt_mean = target_lab[i]
        diff = tgt_mean - src_mean
        
        if i == 0:
            # L channel: very gentle darkening (15% of gap)
            # Don't chase exact value — preserve source dynamics
            correction = diff * 0.15
        elif i == 1:
            # a channel: reduce red aggressively (65% of gap)
            # This is the key fix for the "too red" problem
            correction = diff * 0.65
        else:
            # b channel: match yellow (30% of gap)
            correction = diff * 0.30
        
        src_lab[:,:,i] += correction
    
    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


def match_contrast_curve(source: np.ndarray, target_contrast: float) -> np.ndarray:
    """
    Match contrast to reference.
    
    Reference contrast (std): 59.
    Uses gentle linear stretch — no peak shift (prevents flicker).
    """
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    current_contrast = float(np.std(gray))
    if current_contrast <= 0:
        return source
    
    ratio = target_contrast / current_contrast
    # Only increase contrast, never decrease (preserve source dynamics)
    ratio = max(1.0, min(ratio, 1.3))
    
    mean = np.mean(source.astype(np.float64))
    result = (source.astype(np.float64) - mean) * ratio + mean
    
    return np.clip(result, 0, 255).astype(np.uint8)


def sharpen_reference(source: np.ndarray) -> np.ndarray:
    """
    Apply sharpening to match reference detail level.
    
    Reference: Laplacian std=15.2, detail std=8.6.
    Source: typically much lower.
    
    Two-pass gentle sharpening (no halos).
    """
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    current_std = float(np.std(cv2.Laplacian(gray, cv2.CV_64F)))
    
    if current_std >= 12:  # Already close to reference (15.2)
        return source
    
    # Cap: don't sharpen more than 2x
    if current_std >= 8:
        amount = 0.2  # Very gentle
    else:
        amount = 0.3  # Gentle
    kernel = np.array([
        [0, -amount, 0],
        [-amount, 1 + 4*amount, -amount],
        [0, -amount, 0]
    ])
    result = cv2.filter2D(source, -1, kernel)
    
    # Pass 2: micro-detail enhancement (high-pass blend)
    blur = cv2.GaussianBlur(result, (3, 3), 0)
    detail = result.astype(np.float64) - blur.astype(np.float64)
    result = result.astype(np.float64) + detail * 0.5  # Boost detail by 50%
    
    return np.clip(result, 0, 255).astype(np.uint8)


def match_saturation(source: np.ndarray, target_sat: float = 125) -> np.ndarray:
    """
    Match face saturation to reference.
    
    Reference: mean=125, range=60-241.
    Source: typically lower (compressed video).
    """
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV).astype(np.float64)
    current_sat = np.mean(hsv[:,:,1])
    
    if current_sat <= 0:
        return source
    
    ratio = target_sat / current_sat
    ratio = max(0.8, min(ratio, 1.4))  # Gentle range
    
    hsv[:,:,1] *= ratio
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def enhance_frame(
    frame: np.ndarray,
    profile: ReferenceProfile,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Apply reference-guided enhancement to a single frame.
    
    Pipeline:
    1. Skin tone match (LAB transfer — reduce red, match yellow)
    2. Contrast curve (match shadow/highlight distribution)
    3. Split-tone grading (cool shadows + warm highlights)
    4. Saturation match
    5. Sharpening (detail enhancement)
    6. Vignette (center brighter)
    """
    if not profile.valid:
        return frame
    
    # Step 1: Skin tone match
    result = match_skin_tone(frame, profile.skin_lab)
    
    # Step 2: Contrast curve
    result = match_contrast_curve(result, profile.face_contrast)
    
    # Step 3: Split-tone grading (very subtle)
    result = apply_split_tone(result, profile.shadow_color, profile.highlight_color)
    
    # Step 4: Saturation
    result = match_saturation(result, profile.face_saturation)
    
    # Step 5: Sharpening
    result = sharpen_reference(result)
    
    # Step 6: Vignette (very subtle)
    result = apply_vignette(result, profile.vignette_ratio)
    
    return result


def enhance_video(
    source_path: str,
    reference_path: str,
    output_path: str,
) -> str:
    """
    Full reference-guided enhancement pipeline.
    
    1. Extract reference profile
    2. Pre-scan face positions (bidirectional fill)
    3. Process each frame with all 6 enhancement steps
    4. Encode output
    """
    t_start = time.perf_counter()
    
    # Load reference
    profile = ReferenceProfile.from_image(reference_path)
    if not profile.valid:
        log.error("Failed to extract reference profile")
        return source_path
    
    # Open source
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        log.error("Cannot open: %s", source_path)
        return source_path
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    log.info("Enhancing: %dx%d @ %.1ffps (%d frames)", src_w, src_h, fps, total)
    log.info("Reference: LAB=(%.0f,%.0f,%.0f) bright=%.0f contrast=%.0f sharp=%.0f",
             *profile.skin_lab, profile.face_brightness, profile.face_contrast, profile.face_sharpness)
    
    # Temp dir for frames
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir()
    
    # Pre-scan: detect face positions (bidirectional fill for stability)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_bboxes = {}
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) > 0:
            face_bboxes[idx] = tuple(int(v) for v in max(faces, key=lambda f: f[2]*f[3]))
        idx += 1
    
    # Bidirectional fill
    last_valid = None
    for i in range(total):
        if face_bboxes.get(i) is not None:
            last_valid = face_bboxes[i]
        elif last_valid is not None:
            face_bboxes[i] = last_valid
    
    next_valid = None
    for i in range(total - 1, -1, -1):
        if face_bboxes.get(i) is not None:
            next_valid = face_bboxes[i]
        elif next_valid is not None:
            face_bboxes[i] = next_valid
    
    # Process frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    t_proc = time.perf_counter()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            face_bbox = face_bboxes.get(frame_idx)
            enhanced = enhance_frame(frame, profile, face_bbox)
            cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.jpg"), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_idx += 1
            
            if frame_idx % 50 == 0:
                elapsed = time.perf_counter() - t_proc
                rate = frame_idx / elapsed if elapsed > 0 else 0
                log.info("  %d/%d frames (%.0f fps)", frame_idx, total, rate)
    finally:
        cap.release()
    
    # Encode
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.jpg"),
        "-i", source_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    
    # Verify
    if Path(output_path).exists():
        size = Path(output_path).stat().st_size / 1e6
        elapsed = time.perf_counter() - t_start
        log.info("Done: %s (%.1f MB, %.1fs)", output_path, size, elapsed)
    else:
        log.error("Output not created")
        return source_path
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reference-guided face enhancement")
    parser.add_argument("--source", required=True, help="Source video")
    parser.add_argument("--reference", default="expectation.png", help="Reference image")
    parser.add_argument("--output", "-o", default=None, help="Output path")
    args = parser.parse_args()
    
    output = args.output or "temp/face_mapped.mp4"
    enhance_video(args.source, args.reference, output)
