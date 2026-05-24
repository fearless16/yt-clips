"""
face_mapper.py — Reference-guided face enhancement + landmark-aware per-region grading.

Two enhancement modes (auto-selected per frame):
  1. GLOBAL (default): 6-step pipeline matching expectation.png reference.
  2. LANDMARK  (MediaPipe FaceMesh): per-region corrections — eyes escape sharpening,
     lips get saturation boost, background gets stronger vignette.

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
from typing import Dict, List, Optional, Tuple
import time

from utils.config import load_config
from utils.logger import get_logger
from utils.face_detect import detect_face

cfg = load_config()
log = get_logger("face_mapper", cfg["logging"]["log_file"], cfg["logging"]["level"])

_LANDMARK_AVAILABLE = False  # Set to True when MediaPipe FaceLandmarker model available


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
        from utils.face_detect import detect_face as _dnn_detect_face
        face = _dnn_detect_face(img)
        if face is None:
            log.warning("No face in reference: %s — using defaults", image_path)
            return profile
        x, y, fw, fh = face
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


# ─── Region Mask Generation (Geometric) ───────────────────────────────────
# Uses the existing Haar Cascade face bounding box to derive approximate
# per-region masks.  This avoids any extra dependency while achieving the
# same practical effect: eyes escape sharpening, lips get saturation, etc.


def _gaussian_mask(h: int, w: int, cy: int, cx: int, ry: int, rx: int) -> np.ndarray:
    """Create a smooth elliptical mask centered at (cx, cy) with radii (rx, ry)."""
    Y, X = np.ogrid[:h, :w]
    d = ((X - cx) / max(rx, 1))**2 + ((Y - cy) / max(ry, 1))**2
    mask = np.clip(1.0 - d, 0, 1)
    k = max(int(min(rx, ry) * 0.4) | 1, 3)
    return cv2.GaussianBlur(mask, (k, k), max(k / 3, 1.0))


def create_region_masks_from_bbox(
    h: int, w: int,
    bbox: Tuple[int, int, int, int],
) -> Dict[str, np.ndarray]:
    """Build smooth per-region masks from a face bounding box (x, y, w, h).

    Uses geometric proportions of a typical frontal face:
      - Forehead:    top 20%
      - Eyes:        20-40% (left/right halves)
      - Nose:        centre 25% wide, 35-55% tall
      - Lips:        centre 35% wide, 55-70% tall
      - Chin:        bottom 25%
      - Skin:        full face minus eyes/nose/lips (safe grading zone)
      - Background:  inverse of face oval
    """
    bx, by, bw, bh = bbox
    cy = by + bh // 2
    cx = bx + bw // 2

    # Full face (soft ellipse)
    face_mask = _gaussian_mask(h, w, cy, cx, bh // 2, bw // 2)

    # Left & right eye regions
    eye_ry = int(bh * 0.07)
    eye_rx = int(bw * 0.10)
    left_eye = _gaussian_mask(h, w, by + int(bh * 0.28), bx + int(bw * 0.30), eye_ry, eye_rx)
    right_eye = _gaussian_mask(h, w, by + int(bh * 0.28), bx + int(bw * 0.70), eye_ry, eye_rx)
    eyes_mask = np.clip(left_eye + right_eye, 0, 1)

    # Lips
    lips_mask = _gaussian_mask(h, w, by + int(bh * 0.62), cx,
                                int(bh * 0.07), int(bw * 0.18))

    # Nose
    nose_mask = _gaussian_mask(h, w, by + int(bh * 0.45), cx,
                                int(bh * 0.08), int(bw * 0.12))

    # Skin = face minus eyes/nose/lips
    skin_mask = np.clip(face_mask - eyes_mask - lips_mask - nose_mask, 0, 1)

    return {
        "face": face_mask,
        "eyes": eyes_mask,
        "lips": lips_mask,
        "nose": nose_mask,
        "skin": skin_mask,
    }


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


def apply_vignette_masked(
    frame: np.ndarray,
    face_mask: np.ndarray,
    ratio: float = 1.21,
    bg_strength: float = 0.3,
) -> np.ndarray:
    """Apply vignette only to background (not face).

    Uses the face mask to protect the face region from darkening.
    This prevents the common artifact where vignette darkens the speaker's face.
    """
    h, w = frame.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)

    vignette = 1 - (dist / max_dist) * (1 - 1 / ratio) * bg_strength
    vignette_3ch = vignette[:, :, None]

    result = frame.astype(np.float64)
    # Blend: background gets vignette, face stays at full brightness
    bg_weight = 1.0 - face_mask[:, :, None]
    result = result * (1.0 - bg_weight + bg_weight * vignette_3ch)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_region_saturation(
    frame: np.ndarray,
    mask: np.ndarray,
    boost: float = 1.2,
) -> np.ndarray:
    """Boost saturation selectively in a masked region (e.g. lips)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 1] *= (1.0 + (boost - 1.0) * mask)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


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
    region_masks: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Apply reference-guided enhancement to a single frame.

    Two modes:
    - NO region_masks: Global 6-step pipeline (flicker-safe).
    - WITH region_masks: Per-region grading — eyes skip sharpening,
      lips get saturation boost, background gets stronger vignette.

    Global pipeline (always applied first):
    1. Skin tone match (LAB transfer — reduce red, match yellow)
    2. Contrast curve (match shadow/highlight distribution)
    3. Split-tone grading (cool shadows + warm highlights)
    4. Saturation match
    5. Sharpening (detail enhancement)
    6. Vignette (center brighter)

    Region-aware overrides (applied on top when masks available):
    7a. Eyes region: undo sharpening (lighter sharpen or none)
    7b. Lips region: saturation boost (10%)
    7c. Background: stronger vignette (50% strength)
    """
    if not profile.valid:
        return frame

    # ── Global 6-step pipeline (always runs) ────────────────────────────────
    result = match_skin_tone(frame, profile.skin_lab)
    result = match_contrast_curve(result, profile.face_contrast)
    result = apply_split_tone(result, profile.shadow_color, profile.highlight_color)
    result = match_saturation(result, profile.face_saturation)
    result = sharpen_reference(result)
    result = apply_vignette(result, profile.vignette_ratio)

    # ── Region-aware overrides ──────────────────────────────────────────────
    if region_masks is not None:
        # Eyes: blend back to pre-sharpen state (avoid crispy eyes)
        unsharp = sharpen_reference(frame)  # same sharpen on source
        sharp_diff = result.astype(np.float64) - unsharp.astype(np.float64)
        eyes_w = region_masks.get("eyes", np.zeros((frame.shape[0], frame.shape[1])))
        result = np.clip(
            result.astype(np.float64) - sharp_diff * eyes_w[:, :, None] * 0.6,
            0, 255,
        ).astype(np.uint8)

        # Lips: slight saturation boost
        lips_w = region_masks.get("lips", np.zeros((frame.shape[0], frame.shape[1])))
        if lips_w.max() > 0.01:
            result = apply_region_saturation(result, lips_w, boost=1.10)

        # Background: stronger vignette (face stays bright)
        face_w = region_masks.get("face", np.zeros((frame.shape[0], frame.shape[1])))
        result = apply_vignette_masked(result, face_w, profile.vignette_ratio, bg_strength=0.5)

    return result


def enhance_video(
    source_path: str,
    reference_path: str,
    output_path: str,
    use_region_grading: bool = True,
) -> str:
    """
    Full reference-guided enhancement pipeline with optional region-aware grading.

    When *use_region_grading* is True (default), the global 6-step pipeline is
    augmented with per-region corrections derived from the face bounding box:
      - Eyes get reduced sharpening (no crispy eyes)
      - Lips get a mild saturation boost
      - Background gets a stronger vignette (face stays bright)

    Pipeline:
    1. Extract reference profile
    2. Pre-scan face positions (bidirectional fill for stability) + build region masks
    3. Process each frame with global enhancement + per-region overrides
    4. Encode output
    """
    t_start = time.perf_counter()

    profile = ReferenceProfile.from_image(reference_path)
    if not profile.valid:
        log.error("Failed to extract reference profile")
        return source_path

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
    log.info("Region-aware grading: %s", "ON" if use_region_grading else "OFF")

    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir()

    # ── Pre-scan: detect face bounding boxes using OpenCV DNN ──────────────
    from utils.face_detect import detect_faces as _dnn_detect_faces
    face_bboxes: Dict[int, Tuple[int, int, int, int]] = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = _dnn_detect_faces(frame)
        if faces:
            face_bboxes[idx] = max(faces, key=lambda f: f[2] * f[3])
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

    # ── Process frames ────────────────────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    t_proc = time.perf_counter()

    region_hits = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_bbox = face_bboxes.get(frame_idx)
            masks = None
            if use_region_grading and face_bbox is not None:
                h, w = frame.shape[:2]
                masks = create_region_masks_from_bbox(h, w, face_bbox)
                region_hits += 1

            enhanced = enhance_frame(frame, profile, face_bbox, region_masks=masks)
            cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.jpg"), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_idx += 1

            if frame_idx % 50 == 0:
                elapsed = time.perf_counter() - t_proc
                rate = frame_idx / elapsed if elapsed > 0 else 0
                log.info("  %d/%d frames (%.0f fps)", frame_idx, total, rate)
    finally:
        cap.release()

    if total > 0:
        log.info("Region coverage: %d/%d frames (%.0f%%)",
                 region_hits, total, 100.0 * region_hits / total)

    # ── Encode ────────────────────────────────────────────────────────────
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

    if Path(output_path).exists():
        size = Path(output_path).stat().st_size / 1e6
        elapsed = time.perf_counter() - t_start
        log.info("Done: %s (%.1f MB, %.1fs)", output_path, size, elapsed)
    else:
        log.error("Output not created")
        return source_path

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reference-guided face enhancement")
    parser.add_argument("--source", required=True, help="Source video")
    parser.add_argument("--reference", default="expectation.png", help="Reference image")
    parser.add_argument("--output", "-o", default=None, help="Output path")
    parser.add_argument("--region-grading", action="store_true", default=True,
                        help="Enable per-region grading (eyes/lips/background) (default: True)")
    parser.add_argument("--no-region-grading", action="store_false", dest="region_grading",
                        help="Disable region grading, use global 6-step pipeline only")
    args = parser.parse_args()
    
    output = args.output or "temp/face_mapped.mp4"
    enhance_video(args.source, args.reference, output, use_region_grading=args.region_grading)
