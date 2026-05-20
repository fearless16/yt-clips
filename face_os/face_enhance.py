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

BLINK DETECTION (Module H):
  During blinks, eye patches must FREEZE.
  Don't update eye memory during blink.
  Use last known good eye state.

CINEMATIC NOISE:
  Perfect clean output = FAKE.
  Need subtle grain, sensor noise, micro shimmer.
  Noise must vary spatially, stay statistically consistent.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import EnhancementMask

cfg = get_config()


# ─── Blink Detection ────────────────────────────────────────────────────────

class BlinkDetector:
    """Detect eye blinks using Eye Aspect Ratio (EAR).

    Module H: Eye Dominance System
    - During blinks, eye patches must FREEZE
    - Don't update eye memory during blink
    - Use last known good eye state

    EAR Formula:
      EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Where p1-p6 are the 6 eye landmark points.
    EAR drops below threshold during blink.
    """

    def __init__(
        self,
        ear_threshold: float = 0.20,
        consecutive_frames: int = 2,
    ):
        """Initialize blink detector.

        Args:
            ear_threshold: EAR below this = blink
            consecutive_frames: Number of consecutive frames below threshold to confirm blink
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames

        # State tracking
        self._blink_counter: int = 0
        self._is_blinking: bool = False
        self._blink_history: list = []

        # Last known good eye state (frozen during blink)
        self._last_good_left_eye: Optional[np.ndarray] = None
        self._last_good_right_eye: Optional[np.ndarray] = None
        self._last_good_frame_idx: int = -1

    def detect(
        self,
        landmarks: np.ndarray,
        frame_idx: int = 0,
    ) -> Tuple[bool, float]:
        """Detect if eyes are blinking.

        Args:
            landmarks: 68-point facial landmarks (68, 2)
            frame_idx: Current frame index

        Returns:
            (is_blinking, ear_value)
        """
        # Extract eye landmarks
        # Left eye: points 36-41
        # Right eye: points 42-47
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Compute EAR for both eyes
        left_ear = self._compute_ear(left_eye)
        right_ear = self._compute_ear(right_eye)

        # Average EAR
        ear = (left_ear + right_ear) / 2.0

        # Detect blink
        if ear < self.ear_threshold:
            self._blink_counter += 1
        else:
            self._blink_counter = 0

        # Update blink state
        was_blinking = self._is_blinking
        self._is_blinking = self._blink_counter >= self.consecutive_frames

        # Track history
        self._blink_history.append({
            'frame_idx': frame_idx,
            'ear': ear,
            'is_blinking': self._is_blinking,
        })

        # Keep history bounded
        if len(self._blink_history) > 100:
            self._blink_history = self._blink_history[-100:]

        return self._is_blinking, ear

    def _compute_ear(self, eye_points: np.ndarray) -> float:
        """Compute Eye Aspect Ratio for one eye.

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Args:
            eye_points: (6, 2) array of eye landmark points

        Returns:
            EAR value (float)
        """
        if len(eye_points) < 6:
            return 0.3  # Default to open eye

        # Vertical distances
        p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
        p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])

        # Horizontal distance
        p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])

        # Avoid division by zero
        if p1_p4 < 1e-6:
            return 0.3

        # EAR
        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)

        return float(ear)

    def freeze_eyes(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:
        """Freeze eye regions during blink.

        During blink, use last known good eye state to prevent:
        - Eye artifacts from closed eyes
        - Temporal instability during blink

        Args:
            frame: Current frame (BGR)
            landmarks: 68-point facial landmarks
            frame_idx: Current frame index

        Returns:
            Frame with frozen eyes (if blinking)
        """
        is_blinking, ear = self.detect(landmarks, frame_idx)

        if not is_blinking:
            # Not blinking - update last good state
            self._update_last_good(frame, landmarks, frame_idx)
            return frame

        # Blinking - freeze eyes using last good state
        if self._last_good_left_eye is None or self._last_good_right_eye is None:
            return frame  # No good state available

        result = frame.copy()

        # Freeze left eye
        result = self._freeze_eye_region(
            result, landmarks, 'left',
            self._last_good_left_eye,
        )

        # Freeze right eye
        result = self._freeze_eye_region(
            result, landmarks, 'right',
            self._last_good_right_eye,
        )

        return result

    def _update_last_good(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        frame_idx: int,
    ) -> None:
        """Update last known good eye state."""
        # Extract eye regions
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Get bounding boxes
        left_bbox = self._eye_bbox(left_eye, frame.shape)
        right_bbox = self._eye_bbox(right_eye, frame.shape)

        if left_bbox is not None:
            x1, y1, x2, y2 = left_bbox
            self._last_good_left_eye = frame[y1:y2, x1:x2].copy()

        if right_bbox is not None:
            x1, y1, x2, y2 = right_bbox
            self._last_good_right_eye = frame[y1:y2, x1:x2].copy()

        self._last_good_frame_idx = frame_idx

    def _freeze_eye_region(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        side: str,
        frozen_eye: np.ndarray,
    ) -> np.ndarray:
        """Freeze one eye region using frozen state."""
        if side == 'left':
            eye_points = landmarks[36:42]
        else:
            eye_points = landmarks[42:48]

        bbox = self._eye_bbox(eye_points, frame.shape)
        if bbox is None:
            return frame

        x1, y1, x2, y2 = bbox

        # Resize frozen eye to fit
        if frozen_eye.shape[0] != (y2 - y1) or frozen_eye.shape[1] != (x2 - x1):
            frozen_eye = cv2.resize(frozen_eye, (x2 - x1, y2 - y1))

        # Blend frozen eye with current frame (feathered edges)
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
            return frame

        # Create feathered mask
        mask = np.ones((h, w), dtype=np.float32)
        feather = min(5, h // 4, w // 4)
        if feather > 0:
            mask[:feather, :] *= np.linspace(0, 1, feather)[:, np.newaxis]
            mask[-feather:, :] *= np.linspace(1, 0, feather)[:, np.newaxis]
            mask[:, :feather] *= np.linspace(0, 1, feather)[np.newaxis, :]
            mask[:, -feather:] *= np.linspace(1, 0, feather)[np.newaxis, :]

        # Apply frozen eye
        mask_3ch = mask[:, :, np.newaxis]
        frame[y1:y2, x1:x2] = (
            frame[y1:y2, x1:x2].astype(np.float32) * (1 - mask_3ch) +
            frozen_eye.astype(np.float32) * mask_3ch
        ).astype(np.uint8)

        return frame

    def _eye_bbox(
        self,
        eye_points: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box for eye region with padding."""
        if len(eye_points) < 6:
            return None

        h, w = frame_shape[:2]

        # Get bounds
        x_min = int(np.min(eye_points[:, 0]))
        x_max = int(np.max(eye_points[:, 0]))
        y_min = int(np.min(eye_points[:, 1]))
        y_max = int(np.max(eye_points[:, 1]))

        # Add padding
        pad_x = max(5, (x_max - x_min) // 2)
        pad_y = max(5, (y_max - y_min) // 2)

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    @property
    def is_blinking(self) -> bool:
        """Check if currently blinking."""
        return self._is_blinking

    @property
    def last_ear(self) -> float:
        """Get last computed EAR value."""
        if self._blink_history:
            return self._blink_history[-1]['ear']
        return 0.3

    def get_stats(self) -> dict:
        """Get blink detection statistics."""
        if not self._blink_history:
            return {'total_frames': 0, 'blink_frames': 0, 'blink_rate': 0.0}

        total = len(self._blink_history)
        blinks = sum(1 for h in self._blink_history if h['is_blinking'])

        return {
            'total_frames': total,
            'blink_frames': blinks,
            'blink_rate': blinks / max(1, total),
            'last_ear': self.last_ear,
        }


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

class TemporalNoiseField:
    """Temporally coherent sensor grain field.

    Architecture Module L: 'Noise MUST vary spatially, stay statistically consistent'

    The problem with independent random noise per frame:
      frame 1: noise_1
      frame 2: noise_2  (completely different)
      → micro shimmer flicker (brain detects as fake)

    The solution: temporally coherent noise field
      - Base noise field persists across frames
      - Slowly evolves over time (low-frequency temporal drift)
      - Sensor-pattern persistence (same hot pixels, same grain structure)
      - Like real camera sensor noise: consistent pattern, slow drift

    Math:
      noise_t = base_noise * (1 - α) + new_noise * α
      where α = 0.05 (very slow evolution)

    This creates:
      - Spatial variation (noise varies across frame)
      - Temporal consistency (noise pattern persists across frames)
      - Slow evolution (noise drifts slowly, like real sensor)
    """

    def __init__(self, h: int, w: int, alpha: float = 0.05):
        """Initialize noise field.

        Args:
            h: Frame height
            w: Frame width
            alpha: Evolution rate (0=static, 1=independent per frame)
        """
        self.h = h
        self.w = w
        self.alpha = alpha

        # Base noise field (persists across frames)
        self._base_noise = np.random.randn(h, w).astype(np.float32)

        # Correlated noise (sensor pattern)
        self._sensor_pattern = self._generate_sensor_pattern()

        # Temporal drift (slow low-frequency evolution)
        self._drift_noise = np.random.randn(h, w).astype(np.float32) * 0.1

        # Frame counter
        self._frame_count = 0

    def _generate_sensor_pattern(self) -> np.ndarray:
        """Generate persistent sensor pattern (like real camera).

        Real sensors have:
        - Hot pixels (always bright)
        - Column/row noise (readout pattern)
        - Fixed pattern noise (manufacturing defects)
        """
        pattern = np.zeros((self.h, self.w), dtype=np.float32)

        # Hot pixels (sparse, persistent)
        num_hot = max(1, (self.h * self.w) // 10000)
        hot_y = np.random.randint(0, self.h, num_hot)
        hot_x = np.random.randint(0, self.w, num_hot)
        pattern[hot_y, hot_x] = np.random.randn(num_hot) * 0.5

        # Column noise (readout pattern)
        col_noise = np.random.randn(self.w) * 0.1
        pattern += col_noise[np.newaxis, :]

        # Smooth to avoid sharp edges
        pattern = cv2.GaussianBlur(pattern, (5, 5), 1.0)

        return pattern

    def get_noise(self, strength: float = 0.015) -> np.ndarray:
        """Get temporally coherent noise for current frame.

        Args:
            strength: Noise intensity (0=none, 0.02=subtle, 0.05=visible)

        Returns:
            Noise field (H, W) float32, centered at 0
        """
        self._frame_count += 1

        # Generate new random component
        new_noise = np.random.randn(self.h, self.w).astype(np.float32)

        # Evolve base noise slowly (temporal coherence)
        # noise_t = base * (1-α) + new * α
        self._base_noise = self._base_noise * (1 - self.alpha) + new_noise * self.alpha

        # Combine: base noise + sensor pattern + temporal drift
        noise = (
            self._base_noise * 0.7 +           # Main noise (temporally coherent)
            self._sensor_pattern * 0.2 +         # Persistent sensor pattern
            self._drift_noise * 0.1              # Slow temporal drift
        )

        # Correlate slightly (mimics sensor readout)
        noise = cv2.GaussianBlur(noise, (3, 3), 0.5)

        # Scale by strength
        noise *= strength * 255

        return noise

    def reset(self) -> None:
        """Reset noise field (for new clip)."""
        self._base_noise = np.random.randn(self.h, self.w).astype(np.float32)
        self._sensor_pattern = self._generate_sensor_pattern()
        self._drift_noise = np.random.randn(self.h, self.w).astype(np.float32) * 0.1
        self._frame_count = 0


# Global noise field (persists across frames in a clip)
_noise_field: Optional[TemporalNoiseField] = None


def _get_noise_field(h: int, w: int) -> TemporalNoiseField:
    """Get or create global noise field."""
    global _noise_field
    if _noise_field is None or _noise_field.h != h or _noise_field.w != w:
        _noise_field = TemporalNoiseField(h, w)
    return _noise_field


def add_cinematic_noise(
    frame: np.ndarray,
    strength: float = 0.015,
    use_temporal: bool = True,
) -> np.ndarray:
    """Add subtle sensor grain for cinematic realism.

    Rules from architecture doc:
    - Noise MUST vary spatially
    - Noise MUST stay statistically consistent
    - Perfect clean output = FAKE
    - Brain should register as 'real camera footage'

    TEMPORALLY COHERENT (Module L):
    - Uses persistent noise field across frames
    - Slowly evolving noise (not independent per frame)
    - Sensor-pattern persistence (hot pixels, column noise)
    - Prevents micro-shimmer flicker
    """
    if strength <= 0:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]

    if use_temporal:
        # Temporally coherent noise
        noise_field = _get_noise_field(h, w)
        noise = noise_field.get_noise(strength)
    else:
        # Fallback: independent noise (for testing)
        noise = np.random.randn(h, w).astype(np.float32) * strength * 255
        noise = cv2.GaussianBlur(noise, (3, 3), 0.5)

    # Reduce in highlights (real sensor behavior)
    l_channel = lab[:, :, 0] / 255.0
    highlight_factor = 1.0 - l_channel * 0.5
    noise *= highlight_factor

    # Apply to L channel only (no color shift)
    lab[:, :, 0] += noise
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def reset_noise_field() -> None:
    """Reset global noise field (for new clip)."""
    global _noise_field
    if _noise_field is not None:
        _noise_field.reset()


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
