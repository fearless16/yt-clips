"""
face_enhance.py — Module 7: Face Region Enhancer.

Implements "Eye Dominance Rendering" — the principle that human perception
is obsessed with eyes. So:
  - Eyes always get highest quality processing
  - Brows, beard, face contour get boosted processing
  - Cheeks, forehead get standard processing
  - Background gets minimal processing

Also adds "Intentional Cinematic Noise" — subtle sensor grain that makes
the output look like real camera footage rather than a clean CG render.

This module operates on cropped 9:16 frames and uses region masks
from the landmarks module to apply per-region enhancement.
"""

from typing import Optional

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import EnhancementLevel, EnhancementMask


cfg = get_config()


# ─── Enhancement masks from landmarks ───────────────────────────────────────

def create_enhancement_mask(
    region_masks: dict,
    frame_shape: tuple,
) -> EnhancementMask:
    """Create enhancement intensity map from region masks.

    Maps each region to an enhancement level:
      - Eyes, brows → FULL (1.5x)
      - Beard, contour → STANDARD (1.2x)
      - Skin → STANDARD (1.0x)
      - Background → SKIP (0.0x)
    """
    h, w = frame_shape[:2]

    eye_mask = np.zeros((h, w), dtype=np.float32)
    for name in ("left_eye", "right_eye"):
        if name in region_masks:
            eye_mask = np.maximum(eye_mask, region_masks[name])

    brow_mask = np.zeros((h, w), dtype=np.float32)
    for name in ("left_brow", "right_brow"):
        if name in region_masks:
            brow_mask = np.maximum(brow_mask, region_masks[name])

    face_mask = region_masks.get("face", np.zeros((h, w), dtype=np.float32))
    skin_mask = region_masks.get("skin", np.zeros((h, w), dtype=np.float32))
    nose_mask = region_masks.get("nose", np.zeros((h, w), dtype=np.float32))
    mouth_mask = region_masks.get("mouth", np.zeros((h, w), dtype=np.float32))

    # Beard region: approximate as lower face (below nose, excluding mouth)
    beard_mask = np.zeros((h, w), dtype=np.float32)
    if face_mask.max() > 0 and nose_mask.max() > 0:
        # Create a mask for the lower face region
        nose_bottom = int(np.argmax(np.cumsum(nose_mask, axis=0).sum(axis=1)) + nose_mask.shape[0] * 0.05)
        lower_face = face_mask.copy()
        lower_face[:nose_bottom, :] = 0
        lower_face = np.clip(lower_face - mouth_mask, 0, 1)
        beard_mask = lower_face

    # Contour: face minus inner features
    contour_mask = face_mask.copy()
    for name in ("left_eye", "right_eye", "nose", "mouth"):
        if name in region_masks:
            contour_mask = np.clip(contour_mask - region_masks[name], 0, 1)

    # Background: inverse of face
    background_mask = np.clip(1.0 - face_mask, 0, 1)

    return EnhancementMask(
        face_mask=face_mask,
        eye_mask=eye_mask,
        brow_mask=brow_mask,
        beard_mask=beard_mask,
        contour_mask=contour_mask,
        skin_mask=skin_mask,
        background_mask=background_mask,
    )


# ─── Per-region enhancement ─────────────────────────────────────────────────

def enhance_eyes(
    frame: np.ndarray,
    eye_mask: np.ndarray,
    boost: float = 1.5,
) -> np.ndarray:
    """Enhance eyes: sharpen + brighten sclera + boost contrast.

    Eyes are the most important feature for perceived quality.
    """
    if eye_mask.max() < 0.01:
        return frame

    # Sharpen
    sharpened = _sharpen(frame, amount=0.4, radius=1.0)

    # Brighten slightly (sclera appears whiter)
    brightened = frame.astype(np.float32) + 8
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)

    # Combine: sharpen + brighten in eye region
    mask_3ch = eye_mask[:, :, np.newaxis] * boost
    result = frame.astype(np.float32) * (1 - mask_3ch) + sharpened.astype(np.float32) * mask_3ch * 0.6 + brightened.astype(np.float32) * mask_3ch * 0.4
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_brows(
    frame: np.ndarray,
    brow_mask: np.ndarray,
    boost: float = 1.3,
) -> np.ndarray:
    """Enhance eyebrows: increase contrast and definition."""
    if brow_mask.max() < 0.01:
        return frame

    # Increase contrast locally
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_l = float(np.mean(lab[:, :, 0]))
    lab[:, :, 0] = (lab[:, :, 0] - mean_l) * 1.2 + mean_l
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    contrasted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    mask_3ch = brow_mask[:, :, np.newaxis] * boost
    result = frame.astype(np.float32) * (1 - mask_3ch) + contrasted.astype(np.float32) * mask_3ch
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_beard(
    frame: np.ndarray,
    beard_mask: np.ndarray,
    boost: float = 1.2,
) -> np.ndarray:
    """Enhance beard: increase texture detail."""
    if beard_mask.max() < 0.01:
        return frame

    # Detail enhancement (high-pass blend)
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    detail = frame.astype(np.float32) - blurred.astype(np.float32)
    enhanced = frame.astype(np.float32) + detail * 0.5
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    mask_3ch = beard_mask[:, :, np.newaxis] * boost
    result = frame.astype(np.float32) * (1 - mask_3ch) + enhanced.astype(np.float32) * mask_3ch
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_skin(
    frame: np.ndarray,
    skin_mask: np.ndarray,
    smoothing: float = 0.3,
) -> np.ndarray:
    """Enhance skin: gentle smoothing while preserving texture.

    Uses bilateral filter for edge-preserving smoothing.
    """
    if skin_mask.max() < 0.01 or smoothing <= 0:
        return frame

    # Bilateral filter (preserves edges)
    smoothed = cv2.bilateralFilter(frame, 9, 75, 75)

    mask_3ch = skin_mask[:, :, np.newaxis] * smoothing
    result = frame.astype(np.float32) * (1 - mask_3ch) + smoothed.astype(np.float32) * mask_3ch
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_background(
    frame: np.ndarray,
    bg_mask: np.ndarray,
    vignette_strength: float = 0.3,
) -> np.ndarray:
    """Enhance background: apply vignette to darken edges."""
    if bg_mask.max() < 0.01 or vignette_strength <= 0:
        return frame

    h, w = frame.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)

    vignette = 1 - (dist / max_dist) * vignette_strength
    vignette_3ch = vignette[:, :, np.newaxis]

    # Apply vignette to background only
    bg_weight = bg_mask[:, :, np.newaxis]
    result = frame.astype(np.float32) * (1 - bg_weight + bg_weight * vignette_3ch)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─── Cinematic noise ────────────────────────────────────────────────────────

def add_cinematic_noise(
    frame: np.ndarray,
    strength: float = 0.02,
) -> np.ndarray:
    """Add subtle sensor grain for cinematic realism.

    Perfect clean renders look fake. Tiny amounts of noise make
    the brain register the output as "real camera footage."

    The noise is:
    - Gaussian (matches sensor noise distribution)
    - Slightly correlated (mimics real sensor patterns)
    - Weaker in highlights (like real sensors)
    - Applied in LAB to avoid color shifts
    """
    if strength <= 0:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]

    # Generate noise
    noise = np.random.randn(h, w).astype(np.float32) * strength * 255

    # Correlate slightly (box blur)
    noise = cv2.GaussianBlur(noise, (3, 3), 0.5)

    # Reduce noise in highlights (brighter pixels get less noise)
    l_channel = lab[:, :, 0] / 255.0
    highlight_factor = 1.0 - l_channel * 0.5  # 50% reduction at white
    noise *= highlight_factor

    # Apply to L channel only (no color shift)
    lab[:, :, 0] += noise
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result


# ─── Main enhancement pipeline ──────────────────────────────────────────────

def enhance_frame(
    frame: np.ndarray,
    enhancement_mask: Optional[EnhancementMask] = None,
    region_masks: Optional[dict] = None,
) -> np.ndarray:
    """Apply full enhancement pipeline to a frame.

    Pipeline order:
    1. Eyes (sharpen + brighten)
    2. Brows (contrast boost)
    3. Beard (texture detail)
    4. Skin (gentle smoothing)
    5. Background (vignette)
    6. Cinematic noise (global)

    Each step is masked — only affects its designated region.
    """
    if enhancement_mask is None and region_masks is not None:
        enhancement_mask = create_enhancement_mask(region_masks, frame.shape)

    if enhancement_mask is None:
        # No masks — apply global enhancement only
        result = _sharpen(frame, cfg.enhance.sharpen_amount, cfg.enhance.sharpen_radius)
        if cfg.enhance.use_cinematic_noise:
            result = add_cinematic_noise(result, cfg.enhance.noise_strength)
        return result

    result = frame.copy()

    # 1. Eyes
    result = enhance_eyes(result, enhancement_mask.eye_mask, cfg.enhance.eye_boost)

    # 2. Brows
    result = enhance_brows(result, enhancement_mask.brow_mask, cfg.enhance.brow_boost)

    # 3. Beard
    result = enhance_beard(result, enhancement_mask.beard_mask, cfg.enhance.beard_boost)

    # 4. Skin
    result = enhance_skin(result, enhancement_mask.skin_mask, cfg.enhance.skin_smoothing)

    # 5. Background vignette
    result = enhance_background(result, enhancement_mask.background_mask, 0.3)

    # 6. Global sharpen (gentle)
    result = _sharpen(result, cfg.enhance.sharpen_amount * 0.5, cfg.enhance.sharpen_radius)

    # 7. Cinematic noise
    if cfg.enhance.use_cinematic_noise:
        result = add_cinematic_noise(result, cfg.enhance.noise_strength)

    return result


# ─── Utilities ──────────────────────────────────────────────────────────────

def _sharpen(
    frame: np.ndarray,
    amount: float = 0.3,
    radius: float = 1.0,
) -> np.ndarray:
    """Apply unsharp mask sharpening."""
    if amount <= 0:
        return frame

    ksize = max(3, int(radius * 4) | 1)
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), radius)
    sharpened = frame.astype(np.float32) + (frame.astype(np.float32) - blurred.astype(np.float32)) * amount
    return np.clip(sharpened, 0, 255).astype(np.uint8)
