"""
face_enhance.py — Structure-Preserving Face Rendering.

OLD APPROACH: "enhance this frame" → sharpen, brighten, boost contrast
NEW APPROACH: "preserve identity structure" → keep what's real, reject what's hallucinated

EYE STRATEGY (from architecture-appearence-field.md):
  PRESERVE eye structure aggressively.
  NO hallucinated eyelashes.
  Human eyes detect tiny inconsistencies instantly.
  Even tiny eye artifact = uncanny valley.

PRIORITY MAP (perceptual importance, NOT area):
  CRITICAL: eyes, eyelids, eyebrows, beard contour, lips
  MEDIUM:   nose, forehead
  LOW:      cheeks, neck

RULE: Allocate quality based on perceptual importance, NOT area size.

CINEMATIC NOISE:
  Perfect clean output = FAKE.
  Need subtle grain, sensor noise, micro shimmer.
  Noise must vary spatially, stay statistically consistent.
"""

from typing import Optional

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import EnhancementMask

cfg = get_config()


# ─── Structure-preserving eye rendering ──────────────────────────────────────

def preserve_eyes(
    frame: np.ndarray,
    eye_mask: np.ndarray,
    identity_eyes: Optional[np.ndarray] = None,
    confidence: float = 0.5,
) -> np.ndarray:
    """Preserve eye structure — NOT enhance.

    Rules:
    1. NEVER hallucinate eyelashes, iris detail, or sclera brightness
    2. Preserve temporal stability (eyes must not flicker)
    3. If identity memory has good eye data, prefer it over source
    4. Only sharpen if confidence is very high (don't amplify noise)

    Args:
        frame: Source frame (BGR)
        eye_mask: Eye region mask (H, W) float [0, 1]
        identity_eyes: Identity memory's eye region (if available)
        confidence: How much we trust identity memory

    Returns:
        Frame with preserved eyes
    """
    if eye_mask.max() < 0.01:
        return frame

    result = frame.copy()

    # Strategy 1: If identity has high-confidence eye data, use it
    # This prevents temporal instability (blinks, gaze shifts)
    if identity_eyes is not None and confidence > 0.6:
        # Blend: identity eyes for stability, source for current expression
        # Low blend factor = more identity (stable)
        # High blend factor = more source (current expression)
        blend = 0.3  # 70% identity, 30% source for eyes
        mask_3ch = eye_mask[:, :, np.newaxis] * blend
        result = result.astype(np.float32) * (1 - mask_3ch) + identity_eyes.astype(np.float32) * mask_3ch
        result = np.clip(result, 0, 255).astype(np.uint8)

    # Strategy 2: Minimal sharpening only (preserve structure)
    # Only sharpen edges, don't boost overall contrast
    elif confidence > 0.4:
        # Edge-preserving: only sharpen where there are real edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0

        # Dilate edges slightly to cover edge pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Sharpen only at edges
        sharpened = _sharpen(frame, amount=0.3, radius=1.0)
        edge_mask = eye_mask * edges * 0.5  # Conservative
        mask_3ch = edge_mask[:, :, np.newaxis]
        result = result.astype(np.float32) * (1 - mask_3ch) + sharpened.astype(np.float32) * mask_3ch
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def preserve_brows(
    frame: np.ndarray,
    brow_mask: np.ndarray,
) -> np.ndarray:
    """Preserve eyebrow structure.

    Brows are critical for identity recognition.
    Don't boost contrast — preserve natural brow shape and density.
    """
    if brow_mask.max() < 0.01:
        return frame

    # Only slight edge enhancement (preserve natural density)
    sharpened = _sharpen(frame, amount=0.2, radius=0.8)
    mask_3ch = brow_mask[:, :, np.newaxis] * 0.3
    result = frame.astype(np.float32) * (1 - mask_3ch) + sharpened.astype(np.float32) * mask_3ch
    return np.clip(result, 0, 255).astype(np.uint8)


def preserve_beard(
    frame: np.ndarray,
    beard_mask: np.ndarray,
) -> np.ndarray:
    """Preserve beard texture.

    Beard texture is one of the most stable features.
    Don't smooth it. Don't over-sharpen it.
    Just preserve what's there.
    """
    if beard_mask.max() < 0.01:
        return frame

    # Minimal processing — preserve texture
    # Only slight sharpening for definition
    sharpened = _sharpen(frame, amount=0.15, radius=0.8)
    mask_3ch = beard_mask[:, :, np.newaxis] * 0.2
    result = frame.astype(np.float32) * (1 - mask_3ch) + sharpened.astype(np.float32) * mask_3ch
    return np.clip(result, 0, 255).astype(np.uint8)


def smooth_skin(
    frame: np.ndarray,
    skin_mask: np.ndarray,
    amount: float = 0.15,
) -> np.ndarray:
    """Gentle skin smoothing.

    Don't over-smooth — preserve natural skin texture.
    The goal is to reduce compression artifacts, not remove pores.
    """
    if skin_mask.max() < 0.01 or amount <= 0:
        return frame

    # Very gentle bilateral (preserve texture)
    smoothed = cv2.bilateralFilter(frame, 5, 40, 40)

    mask_3ch = skin_mask[:, :, np.newaxis] * amount
    result = frame.astype(np.float32) * (1 - mask_3ch) + smoothed.astype(np.float32) * mask_3ch
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_vignette(
    frame: np.ndarray,
    bg_mask: np.ndarray,
    strength: float = 0.3,
) -> np.ndarray:
    """Apply vignette to background to draw attention to face."""
    if bg_mask.max() < 0.01 or strength <= 0:
        return frame

    h, w = frame.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)

    vignette = 1 - (dist / max_dist) * strength
    vignette_3ch = vignette[:, :, np.newaxis]

    bg_weight = bg_mask[:, :, np.newaxis]
    result = frame.astype(np.float32) * (1 - bg_weight + bg_weight * vignette_3ch)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─── Cinematic noise ────────────────────────────────────────────────────────

def add_cinematic_noise(
    frame: np.ndarray,
    strength: float = 0.015,
) -> np.ndarray:
    """Add subtle sensor grain for cinematic realism.

    Rules from architecture doc:
    - Noise MUST vary spatially
    - Noise MUST stay statistically consistent
    - Perfect clean output = FAKE
    - Brain should register as "real camera footage"

    The noise:
    - Gaussian (matches sensor distribution)
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

    # Correlate slightly (mimics sensor pattern)
    noise = cv2.GaussianBlur(noise, (3, 3), 0.5)

    # Reduce in highlights (real sensor behavior)
    l_channel = lab[:, :, 0] / 255.0
    highlight_factor = 1.0 - l_channel * 0.5
    noise *= highlight_factor

    # Apply to L channel only (no color shift)
    lab[:, :, 0] += noise
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ─── Main rendering pipeline ────────────────────────────────────────────────

def render_frame(
    frame: np.ndarray,
    enhancement_mask: Optional[EnhancementMask] = None,
    region_masks: Optional[dict] = None,
    identity_eyes: Optional[np.ndarray] = None,
    eye_confidence: float = 0.5,
) -> np.ndarray:
    """Render a frame using structure-preserving approach.

    OLD: enhance_frame → sharpen, brighten, boost
    NEW: render_frame → preserve structure, minimal processing

    Pipeline:
    1. Eyes (preserve structure, identity-first)
    2. Brows (minimal edge enhancement)
    3. Beard (preserve texture)
    4. Skin (gentle smoothing, compression artifact removal)
    5. Background vignette
    6. Cinematic noise (subtle grain)
    """
    if enhancement_mask is None and region_masks is not None:
        enhancement_mask = _create_enhancement_mask(region_masks, frame.shape)

    if enhancement_mask is None:
        result = frame.copy()
        if cfg.enhance.use_cinematic_noise:
            result = add_cinematic_noise(result, cfg.enhance.noise_strength)
        return result

    result = frame.copy()

    # 1. Eyes (structure-preserving, identity-first)
    result = preserve_eyes(
        result, enhancement_mask.eye_mask,
        identity_eyes=identity_eyes,
        confidence=eye_confidence,
    )

    # 2. Brows (minimal)
    result = preserve_brows(result, enhancement_mask.brow_mask)

    # 3. Beard (preserve texture)
    result = preserve_beard(result, enhancement_mask.beard_mask)

    # 4. Skin (gentle smoothing)
    result = smooth_skin(result, enhancement_mask.skin_mask, cfg.enhance.skin_smoothing)

    # 5. Background vignette
    result = apply_vignette(result, enhancement_mask.background_mask, 0.3)

    # 6. Cinematic noise (subtle grain)
    if cfg.enhance.use_cinematic_noise:
        result = add_cinematic_noise(result, cfg.enhance.noise_strength)

    return result


# ─── Utilities ──────────────────────────────────────────────────────────────

def _create_enhancement_mask(region_masks: dict, frame_shape: tuple) -> EnhancementMask:
    """Create enhancement intensity map from region masks."""
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

    # Beard region
    beard_mask = np.zeros((h, w), dtype=np.float32)
    if face_mask.max() > 0 and nose_mask.max() > 0:
        nose_bottom = int(np.argmax(np.cumsum(nose_mask, axis=0).sum(axis=1)) + nose_mask.shape[0] * 0.05)
        lower_face = face_mask.copy()
        lower_face[:nose_bottom, :] = 0
        lower_face = np.clip(lower_face - mouth_mask, 0, 1)
        beard_mask = lower_face

    # Background
    background_mask = np.clip(1.0 - face_mask, 0, 1)

    return EnhancementMask(
        face_mask=face_mask,
        eye_mask=eye_mask,
        brow_mask=brow_mask,
        beard_mask=beard_mask,
        contour_mask=face_mask,
        skin_mask=skin_mask,
        background_mask=background_mask,
    )


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
