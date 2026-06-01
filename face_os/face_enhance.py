"""
face_enhance.py — Structure-Preserving Face Rendering.

BEAST MODE FIXES:
- Killed the 5x float32/uint8 conversion loop (FPS Chakravyuh).
- Replaced O(N^2) Bilateral Filter with Downscale->Blur->Upscale (CPU Saver).
- Nuked LAB color space conversion for noise (Fast Luminance Approximation).
- Consolidated all region blending into a single high-speed float32 pass.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import EnhancementMask

cfg = get_config()


# ─── Blink Detection ────────────────────────────────────────────────────────

class BlinkDetector:
    """Detect eye blinks using Eye Aspect Ratio (EAR)."""

    def __init__(self, ear_threshold: float = 0.20, consecutive_frames: int = 2):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self._blink_counter: int = 0
        self._is_blinking: bool = False
        self._blink_history: list = []
        self._last_good_left_eye: Optional[np.ndarray] = None
        self._last_good_right_eye: Optional[np.ndarray] = None
        self._last_good_frame_idx: int = -1

    def detect(self, landmarks: np.ndarray, frame_idx: int = 0) -> Tuple[bool, float]:
        left_eye = landmarks[[33, 159, 158, 133, 153, 145]]
        right_eye = landmarks[[362, 386, 385, 263, 380, 374]]

        left_ear = self._compute_ear(left_eye)
        right_ear = self._compute_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < self.ear_threshold:
            self._blink_counter += 1
        else:
            self._blink_counter = 0

        was_blinking = self._is_blinking
        self._is_blinking = self._blink_counter >= self.consecutive_frames

        self._blink_history.append({'frame_idx': frame_idx, 'ear': ear, 'is_blinking': self._is_blinking})
        if len(self._blink_history) > 100:
            self._blink_history = self._blink_history[-100:]

        return self._is_blinking, ear

    def _compute_ear(self, eye_points: np.ndarray) -> float:
        if len(eye_points) < 6: return 0.3
        p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
        p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
        p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
        if p1_p4 < 1e-6: return 0.3
        return float((p2_p6 + p3_p5) / (2.0 * p1_p4))

    def freeze_eyes(self, frame: np.ndarray, landmarks: np.ndarray, frame_idx: int) -> np.ndarray:
        is_blinking, ear = self.detect(landmarks, frame_idx)
        if not is_blinking:
            self._update_last_good(frame, landmarks, frame_idx)
            return frame

        if self._last_good_left_eye is None or self._last_good_right_eye is None:
            return frame

        result = frame.copy()
        result = self._freeze_eye_region(result, landmarks, 'left', self._last_good_left_eye)
        result = self._freeze_eye_region(result, landmarks, 'right', self._last_good_right_eye)
        return result

    def _update_last_good(self, frame: np.ndarray, landmarks: np.ndarray, frame_idx: int) -> None:
        left_eye = landmarks[[33, 159, 158, 133, 153, 145]]
        right_eye = landmarks[[362, 386, 385, 263, 380, 374]]

        left_bbox = self._eye_bbox(left_eye, frame.shape)
        right_bbox = self._eye_bbox(right_eye, frame.shape)

        if left_bbox:
            x1, y1, x2, y2 = left_bbox
            self._last_good_left_eye = frame[y1:y2, x1:x2].copy()
        if right_bbox:
            x1, y1, x2, y2 = right_bbox
            self._last_good_right_eye = frame[y1:y2, x1:x2].copy()
        self._last_good_frame_idx = frame_idx

    def _freeze_eye_region(self, frame: np.ndarray, landmarks: np.ndarray, side: str, frozen_eye: np.ndarray) -> np.ndarray:
        eye_points = landmarks[[33, 159, 158, 133, 153, 145]] if side == 'left' else landmarks[[362, 386, 385, 263, 380, 374]]
        bbox = self._eye_bbox(eye_points, frame.shape)
        if not bbox: return frame

        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0: return frame

        if frozen_eye.shape[0] != h or frozen_eye.shape[1] != w:
            frozen_eye = cv2.resize(frozen_eye, (w, h))

        mask = np.ones((h, w), dtype=np.float32)
        feather = min(5, h // 4, w // 4)
        if feather > 0:
            mask[:feather, :] *= np.linspace(0, 1, feather)[:, np.newaxis]
            mask[-feather:, :] *= np.linspace(1, 0, feather)[:, np.newaxis]
            mask[:, :feather] *= np.linspace(0, 1, feather)[np.newaxis, :]
            mask[:, -feather:] *= np.linspace(1, 0, feather)[np.newaxis, :]

        mask_3ch = mask[:, :, np.newaxis]
        frame[y1:y2, x1:x2] = (
            frame[y1:y2, x1:x2].astype(np.float32) * (1 - mask_3ch) +
            frozen_eye.astype(np.float32) * mask_3ch
        ).astype(np.uint8)
        return frame

    def _eye_bbox(self, eye_points: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        if len(eye_points) < 6: return None
        h, w = frame_shape[:2]
        x_min, x_max = int(np.min(eye_points[:, 0])), int(np.max(eye_points[:, 0]))
        y_min, y_max = int(np.min(eye_points[:, 1])), int(np.max(eye_points[:, 1]))
        pad_x, pad_y = max(5, (x_max - x_min) // 2), max(5, (y_max - y_min) // 2)
        
        x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
        x2, y2 = min(w, x_max + pad_x), min(h, y_max + pad_y)
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    @property
    def is_blinking(self) -> bool: return self._is_blinking
    
    @property
    def last_ear(self) -> float: return self._blink_history[-1]['ear'] if self._blink_history else 0.3

    def get_stats(self) -> dict:
        if not self._blink_history: return {'total_frames': 0, 'blink_frames': 0, 'blink_rate': 0.0}
        total = len(self._blink_history)
        blinks = sum(1 for h in self._blink_history if h['is_blinking'])
        return {'total_frames': total, 'blink_frames': blinks, 'blink_rate': blinks / max(1, total), 'last_ear': self.last_ear}


# ─── High-Speed Region Blending (BEAST MODE) ────────────────────────────────

def _blend_region(frame_f32: np.ndarray, mask: np.ndarray, identity_patch: Optional[np.ndarray], confidence: float, is_eye: bool = False, sharpen_amt: float = 0.0) -> np.ndarray:
    """Blends regions in a single float32 pass without repeated typecasting."""
    if mask.max() < 0.01:
        return frame_f32
        
    if is_eye and identity_patch is not None and confidence > 0.6:
        blend = 0.3
        m3 = mask[:, :, np.newaxis] * blend
        return frame_f32 * (1 - m3) + identity_patch.astype(np.float32) * m3
        
    if sharpen_amt > 0 and confidence > 0.4:
        blurred = cv2.GaussianBlur(frame_f32, (3, 3), 0.8)
        sharp_detail = (frame_f32 - blurred) * sharpen_amt
        m3 = mask[:, :, np.newaxis]
        return frame_f32 + sharp_detail * m3
        
    return frame_f32

def _smooth_skin_fast(frame_f32: np.ndarray, skin_mask: np.ndarray, amount: float = 0.15) -> np.ndarray:
    """Downscale -> Blur -> Upscale. 100x faster than Bilateral Filter."""
    if skin_mask.max() < 0.01 or amount <= 0:
        return frame_f32
        
    h, w = frame_f32.shape[:2]
    small = cv2.resize(frame_f32, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    smoothed_small = cv2.GaussianBlur(small, (5, 5), 0)
    smoothed = cv2.resize(smoothed_small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    m3 = skin_mask[:, :, np.newaxis] * amount
    return frame_f32 * (1 - m3) + smoothed * m3

def _apply_vignette_fast(frame_f32: np.ndarray, bg_mask: np.ndarray, strength: float = 0.3) -> np.ndarray:
    if bg_mask.max() < 0.01 or strength <= 0:
        return frame_f32
        
    h, w = frame_f32.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vignette = 1.0 - (dist / max_dist) * strength
    
    bg_weight = bg_mask * (1.0 - vignette)
    bg_weight_3ch = bg_weight[:, :, np.newaxis]
    return frame_f32 * (1.0 - bg_weight_3ch)


# ─── Cinematic Noise ─────────────────────────────────────────────────────────

class TemporalNoiseField:
    def __init__(self, h: int, w: int, alpha: float = 0.05):
        self.h, self.w, self.alpha = h, w, alpha
        self._base_noise = np.random.randn(h, w).astype(np.float32)
        self._sensor_pattern = self._generate_sensor_pattern()
        self._drift_noise = np.random.randn(h, w).astype(np.float32) * 0.1
        self._frame_count = 0

    def _generate_sensor_pattern(self) -> np.ndarray:
        pattern = np.zeros((self.h, self.w), dtype=np.float32)
        num_hot = max(1, (self.h * self.w) // 10000)
        hot_y = np.random.randint(0, self.h, num_hot)
        hot_x = np.random.randint(0, self.w, num_hot)
        pattern[hot_y, hot_x] = np.random.randn(num_hot) * 0.5
        pattern += np.random.randn(self.w) * 0.1
        return cv2.GaussianBlur(pattern, (5, 5), 1.0)

    def get_noise(self, strength: float = 0.015) -> np.ndarray:
        self._frame_count += 1
        new_noise = np.random.randn(self.h, self.w).astype(np.float32)
        self._base_noise = self._base_noise * (1 - self.alpha) + new_noise * self.alpha
        noise = self._base_noise * 0.7 + self._sensor_pattern * 0.2 + self._drift_noise * 0.1
        noise = cv2.GaussianBlur(noise, (3, 3), 0.5)
        return noise * strength * 255

    def reset(self) -> None:
        self._base_noise = np.random.randn(self.h, self.w).astype(np.float32)
        self._sensor_pattern = self._generate_sensor_pattern()
        self._drift_noise = np.random.randn(self.h, self.w).astype(np.float32) * 0.1
        self._frame_count = 0

_noise_field: Optional[TemporalNoiseField] = None

def _get_noise_field(h: int, w: int) -> TemporalNoiseField:
    global _noise_field
    if _noise_field is None or _noise_field.h != h or _noise_field.w != w:
        _noise_field = TemporalNoiseField(h, w)
    return _noise_field

def _add_cinematic_noise_fast(frame_f32: np.ndarray, strength: float = 0.015) -> np.ndarray:
    """Fast Luminance Approximation. Nuked the slow LAB conversion."""
    if strength <= 0: return frame_f32
    h, w = frame_f32.shape[:2]
    noise_field = _get_noise_field(h, w)
    noise = noise_field.get_noise(strength)
    
    # BGR Luminance weights: 0.114*B + 0.587*G + 0.299*R
    l_channel = (frame_f32[:, :, 0] * 0.114 + frame_f32[:, :, 1] * 0.587 + frame_f32[:, :, 2] * 0.299) / 255.0
    highlight_factor = 1.0 - l_channel * 0.5
    
    noise_3ch = (noise * highlight_factor)[:, :, np.newaxis]
    return frame_f32 + noise_3ch

def reset_noise_field() -> None:
    global _noise_field
    if _noise_field is not None: _noise_field.reset()


# ─── Main Rendering Pipeline ─────────────────────────────────────────────────

def render_frame(
    frame: np.ndarray,
    enhancement_mask: Optional[EnhancementMask] = None,
    region_masks: Optional[dict] = None,
    identity_eyes: Optional[np.ndarray] = None,
    eye_confidence: float = 0.5,
) -> np.ndarray:
    if enhancement_mask is None and region_masks is not None:
        enhancement_mask = _create_enhancement_mask(region_masks, frame.shape)

    if enhancement_mask is None:
        result = frame.copy()
        if getattr(cfg.enhance, 'use_cinematic_noise', True):
            result = add_cinematic_noise(result, getattr(cfg.enhance, 'noise_strength', 0.015))
        return result

    # BEAST MODE: Single float32 pass for ALL regions. No more 5x typecasting!
    result = frame.astype(np.float32)

    result = _blend_region(result, enhancement_mask.eye_mask, identity_eyes, eye_confidence, is_eye=True)
    result = _blend_region(result, enhancement_mask.brow_mask, None, 0.5, sharpen_amt=0.2)
    result = _blend_region(result, enhancement_mask.beard_mask, None, 0.5, sharpen_amt=0.15)
    result = _smooth_skin_fast(result, enhancement_mask.skin_mask, getattr(cfg.enhance, 'skin_smoothing', 0.15))
    result = _apply_vignette_fast(result, enhancement_mask.background_mask, 0.3)

    if getattr(cfg.enhance, 'use_cinematic_noise', True):
        result = _add_cinematic_noise_fast(result, getattr(cfg.enhance, 'noise_strength', 0.015))

    return np.clip(result, 0, 255).astype(np.uint8)

def add_cinematic_noise(frame: np.ndarray, strength: float = 0.015, use_temporal: bool = True) -> np.ndarray:
    """Wrapper for uint8 frames."""
    if strength <= 0: return frame
    f32 = frame.astype(np.float32)
    res = _add_cinematic_noise_fast(f32, strength)
    return np.clip(res, 0, 255).astype(np.uint8)


# ─── Utilities ───────────────────────────────────────────────────────────────

def _create_enhancement_mask(region_masks: dict, frame_shape: tuple) -> EnhancementMask:
    h, w = frame_shape[:2]
    eye_mask = np.maximum(region_masks.get("left_eye", np.zeros((h, w), dtype=np.float32)), 
                          region_masks.get("right_eye", np.zeros((h, w), dtype=np.float32)))
    brow_mask = np.maximum(region_masks.get("left_brow", np.zeros((h, w), dtype=np.float32)), 
                           region_masks.get("right_brow", np.zeros((h, w), dtype=np.float32)))
    face_mask = region_masks.get("face", np.zeros((h, w), dtype=np.float32))
    skin_mask = region_masks.get("skin", np.zeros((h, w), dtype=np.float32))
    nose_mask = region_masks.get("nose", np.zeros((h, w), dtype=np.float32))
    mouth_mask = region_masks.get("mouth", np.zeros((h, w), dtype=np.float32))

    beard_mask = np.zeros((h, w), dtype=np.float32)
    if face_mask.max() > 0 and nose_mask.max() > 0:
        nose_bottom = int(np.argmax(np.cumsum(nose_mask, axis=0).sum(axis=1)) + nose_mask.shape[0] * 0.05)
        lower_face = face_mask.copy()
        lower_face[:nose_bottom, :] = 0
        beard_mask = np.clip(lower_face - mouth_mask, 0, 1)

    return EnhancementMask(
        face_mask=face_mask, eye_mask=eye_mask, brow_mask=brow_mask,
        beard_mask=beard_mask, contour_mask=face_mask, skin_mask=skin_mask,
        background_mask=np.clip(1.0 - face_mask, 0, 1),
    )

def _sharpen(frame: np.ndarray, amount: float = 0.3, radius: float = 1.0) -> np.ndarray:
    """Fast Unsharp Mask using cv2.addWeighted (No manual float32 casting)."""
    if amount <= 0: return frame
    ksize = max(3, int(radius * 4) | 1)
    if ksize % 2 == 0: ksize += 1
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), radius)
    # sharpened = frame * (1 + amount) - blurred * amount
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase A: Signal Processing Repair
# ═══════════════════════════════════════════════════════════════════════════════

def _measure_sharpness(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Laplacian variance sharpness metric. Higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    if mask is not None and mask.max() > 0.01:
        m = (mask > 0.5).astype(np.float32)
        if m.shape[:2] != lap.shape[:2]:
            m = cv2.resize(m, (lap.shape[1], lap.shape[0]), interpolation=cv2.INTER_NEAREST)
        lap = lap * m
    return float(np.var(lap))


def adaptive_sharpen(
    frame: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
    fine_amount: float = 1.4,
    fine_radius: float = 0.6,
    coarse_amount: float = 0.8,
    coarse_radius: float = 1.2,
    target_sharpness: float = 300.0,
    deficit_gain: float = 1.5,
    min_amount_mult: float = 0.5,
    max_amount_mult: float = 2.5,
) -> np.ndarray:
    """Adaptive dual-radius USM — scales amount based on sharpness deficit.

    Sharp frames receive reduced amounts (avoid over-sharpening); blurry
    frames receive increased amounts up to max_amount_mult.

    Args:
        frame: uint8 BGR input image
        face_mask: optional float32 mask in [0,1] (outside pixels unchanged)
        fine_amount: base amount for fine-scale (0.6px) USM
        fine_radius: radius for fine-scale USM
        coarse_amount: base amount for coarse-scale (1.2px) USM
        coarse_radius: radius for coarse-scale USM
        target_sharpness: Laplacian variance target
        deficit_gain: how aggressively to scale amounts per unit deficit
        min_amount_mult: minimum multiplier (for already-sharp frames)
        max_amount_mult: maximum multiplier (for heavily blurred frames)
    """
    sharpness = _measure_sharpness(frame, face_mask)
    target = max(target_sharpness, 1.0)
    deficit = (target - sharpness) / target
    scale = 1.0 + deficit * deficit_gain
    scale = np.clip(scale, min_amount_mult, max_amount_mult)

    eff_fine = float(np.clip(fine_amount * scale, 0.0, fine_amount * max_amount_mult))
    eff_coarse = float(np.clip(coarse_amount * scale, 0.0, coarse_amount * max_amount_mult))

    if face_mask is not None and face_mask.max() > 0.01:
        sharpened = _sharpen(frame, amount=eff_fine, radius=fine_radius)
        sharpened = _sharpen(sharpened, amount=eff_coarse, radius=coarse_radius)
        m3 = np.clip(face_mask.astype(np.float32), 0.0, 1.0)
        if m3.shape[:2] != frame.shape[:2]:
            m3 = cv2.resize(m3, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        m3 = m3[:, :, np.newaxis]
        result = (frame.astype(np.float32) * (1.0 - m3) + sharpened.astype(np.float32) * m3)
        return np.clip(result, 0, 255).astype(np.uint8)

    frame = _sharpen(frame, amount=eff_fine, radius=fine_radius)
    return _sharpen(frame, amount=eff_coarse, radius=coarse_radius)


def enhance_contrast(
    frame: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """Adaptive contrast enhancement via CLAHE on luminance channel.

    Args:
        frame: uint8 BGR input image
        face_mask: optional float32 mask in [0,1] (outside pixels unchanged)
        clip_limit: CLAHE clip limit for local contrast limiting
        tile_grid_size: CLAHE tile grid (w, h)
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l_channel)

    if face_mask is not None and face_mask.max() > 0.01:
        m = np.clip(face_mask.astype(np.float32), 0.0, 1.0)
        if m.shape[:2] != frame.shape[:2]:
            m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        l_blended = (l_channel.astype(np.float32) * (1.0 - m) + l_eq.astype(np.float32) * m)
        l_eq = np.clip(l_blended, 0, 255).astype(np.uint8)

    lab[:, :, 0] = l_eq
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_energy_conservation(
    composite: np.ndarray,
    source: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
    energy_limit: float = 0.95,
) -> np.ndarray:
    """Normalize the energy of a composite to match the source.

    Matches the physical renderer's _apply_energy_conservation contract:
    - If output/target > energy_limit → scale down
    - If output/target < 0.5 → scale up (capped at 4x)

    Args:
        composite: uint8 BGR composite output
        source: uint8 BGR source reference for target energy
        face_mask: optional float32 mask (only interior pixels scaled)
        energy_limit: max energy ratio before scaling down
    """
    if face_mask is not None and face_mask.max() > 0.05:
        m = np.clip(face_mask.astype(np.float32), 0.0, 1.0)
        if m.shape[:2] != composite.shape[:2]:
            m = cv2.resize(m, (composite.shape[1], composite.shape[0]), interpolation=cv2.INTER_NEAREST)
        interior = m > 0.5
        src_energy = float(np.mean(source[interior])) if interior.any() else float(np.mean(source))
        out_energy = float(np.mean(composite[interior])) if interior.any() else float(np.mean(composite))
    else:
        src_energy = float(np.mean(source))
        out_energy = float(np.mean(composite))

    if src_energy <= 1e-8 or out_energy <= 1e-8:
        return composite

    ratio = out_energy / src_energy
    f32 = composite.astype(np.float32)
    if ratio > energy_limit:
        scale = energy_limit / ratio
        f32 *= scale
    elif ratio < 0.5:
        scale = min(0.5 / ratio, 4.0)
        f32 *= scale

    if face_mask is not None and face_mask.max() > 0.05:
        m = np.clip(face_mask.astype(np.float32), 0.0, 1.0)
        if m.shape[:2] != composite.shape[:2]:
            m = cv2.resize(m, (composite.shape[1], composite.shape[0]), interpolation=cv2.INTER_LINEAR)
        m3 = m[:, :, np.newaxis]
        result = composite.astype(np.float32) * (1.0 - m3) + f32 * m3
        return np.clip(result, 0, 255).astype(np.uint8)

    return np.clip(f32, 0, 255).astype(np.uint8)