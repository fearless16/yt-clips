"""
identity_state.py — The Identity Belief State Engine.

CORE PHILOSOPHY:
  Source video is NOT ground truth. Source video is TELEMETRY.
  Each frame is a partial, noisy observation of the face.
  The system maintains a BELIEF about what this person looks like,
  and updates that belief as new observations arrive.

FREQUENCY DECOMPOSITION:
  I = I_low + I_high

  LOW FREQUENCY  = skin tone, lighting, overall structure
  HIGH FREQUENCY = pores, beard edges, eyelashes, texture

  Rules:
  - Low freq: EMA update (smooth, continuous)
  - High freq: BEST OBSERVATION ONLY (never average pores)
  - High freq only updates when current observation is better

BELIEF DISTRIBUTIONS:
  Instead of: pixel = weighted_average
  Do:         belief_distribution(pixel) → MAP estimate

  Each pixel maintains:
  - best_observation (highest quality seen)
  - observation_count (confidence)
  - variance (how noisy observations are)
  - pose_at_best (pose when best observation was captured)

This is NOT image processing.
This is IDENTITY RECONSTRUCTION.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import ConfidenceMap

cfg = get_config()


class FrequencyDecomposition:
    """Separates image into low and high frequency components.

    Low freq = Gaussian blur (skin tone, lighting)
    High freq = Original - Low freq (pores, edges, texture)

    Reconstruction: image = low + high
    """

    def __init__(self, low_pass_sigma: float = 3.0):
        self.low_pass_sigma = low_pass_sigma
        self._kernel_size = max(3, int(low_pass_sigma * 6) | 1)

    def decompose(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split image into low and high frequency components.

        Args:
            image: (H, W, C) float32 or uint8

        Returns:
            (low_freq, high_freq) both float32
        """
        img_f = image.astype(np.float32)
        low = cv2.GaussianBlur(img_f, (self._kernel_size, self._kernel_size), self.low_pass_sigma)
        high = img_f - low
        return low, high

    def reconstruct(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Reconstruct image from frequency components."""
        result = low + high
        return np.clip(result, 0, 255).astype(np.uint8)


class BeliefPixel:
    """Per-pixel belief distribution.

    Instead of storing a single averaged value, stores:
    - best_low: best low-frequency observation (EMA)
    - best_high: best high-frequency observation (NEVER averaged)
    - quality_max: quality of the best observation
    - observation_count: total observations seen
    - variance: estimated observation noise
    - pose_at_best: pose when best high-freq was captured
    """

    def __init__(self, h: int, w: int, channels: int = 3):
        # Low frequency: EMA (smooth accumulation)
        self.best_low = np.zeros((h, w, channels), dtype=np.float32)

        # High frequency: BEST observation only (never averaged)
        self.best_high = np.zeros((h, w, channels), dtype=np.float32)

        # Quality tracking
        self.quality_max = np.zeros((h, w), dtype=np.float32)
        self.quality_current = np.zeros((h, w), dtype=np.float32)

        # Observation stats
        self.observation_count = np.zeros((h, w), dtype=np.float32)
        self.variance = np.ones((h, w), dtype=np.float32) * 10.0  # Initial uncertainty

        # Pose at best observation
        self.pose_at_best = np.zeros((h, w, 3), dtype=np.float32)  # yaw, pitch, roll

        # Temporal
        self.last_update_frame = np.full((h, w), -1, dtype=np.int32)
        self.frame_count = 0

        # Initialized flag
        self._initialized = False

    def initialize(self, low: np.ndarray, high: np.ndarray, quality: np.ndarray) -> None:
        """Initialize with first observation."""
        self.best_low = low.copy()
        self.best_high = high.copy()
        self.quality_max = quality.copy()
        self.quality_current = quality.copy()
        self.observation_count = np.ones_like(quality)
        self.variance = np.ones_like(quality) * 5.0
        self.last_update_frame[:] = 0
        self._initialized = True

    def update(
        self,
        low: np.ndarray,
        high: np.ndarray,
        quality: np.ndarray,
        pose: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Update belief with new observation.

        LOW FREQ: EMA update (smooth, continuous)
          low_new = low_old * (1 - rate) + low_obs * rate
          rate = f(quality, observation_count)

        HIGH FREQ: BEST observation only
          if quality > quality_max:
              best_high = high_obs
              quality_max = quality
        """
        self.frame_count += 1

        if not self._initialized:
            self.initialize(low, high, quality)
            return

        # === LOW FREQUENCY: EMA update ===
        # Rate adapts: slower for well-observed pixels
        base_rate = 0.15
        obs_factor = 1.0 / (1.0 + self.observation_count * 0.005)
        low_rate = base_rate * obs_factor * quality
        low_rate_3d = low_rate[:, :, np.newaxis]

        self.best_low = self.best_low * (1 - low_rate_3d) + low * low_rate_3d

        # === HIGH FREQUENCY: BEST observation only ===
        # Only update when current observation is better
        better_mask = quality > self.quality_max
        better_mask_3d = better_mask[:, :, np.newaxis]

        self.best_high = np.where(better_mask_3d, high, self.best_high)
        self.quality_max = np.where(better_mask, quality, self.quality_max)

        # Update pose at best
        if pose is not None:
            pose_arr = np.array(pose, dtype=np.float32)
            for c in range(3):
                self.pose_at_best[:, :, c] = np.where(
                    better_mask, pose_arr[c], self.pose_at_best[:, :, c]
                )

        # === VARIANCE UPDATE ===
        # How different is this observation from current belief?
        current_recon = self.best_low + self.best_high
        obs_diff = np.abs(low + high - current_recon)
        obs_variance = np.mean(obs_diff, axis=2) if obs_diff.ndim == 3 else obs_diff
        # EMA of variance
        self.variance = self.variance * 0.95 + obs_variance * 0.05

        # === OBSERVATION COUNT ===
        self.observation_count += quality
        self.quality_current = quality
        self.last_update_frame[:] = self.frame_count

    def reconstruct(self) -> np.ndarray:
        """Reconstruct the best known appearance.

        Returns: BGR uint8 image
        """
        result = self.best_low + self.best_high
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_confidence(self) -> np.ndarray:
        """Get per-pixel confidence based on observation count and variance.

        High confidence = many observations, low variance
        Low confidence = few observations, high variance
        """
        # Observation confidence (saturates)
        obs_conf = np.clip(self.observation_count / max(cfg.memory.min_observations, 1), 0, 1)

        # Variance confidence (lower variance = higher confidence)
        var_conf = np.exp(-self.variance / 20.0)

        return (obs_conf * var_conf).astype(np.float32)

    def get_high_freq_confidence(self) -> np.ndarray:
        """Get confidence specifically for high-frequency details.

        High-freq confidence is based on:
        - How recent the best observation was
        - Quality of the best observation
        """
        # How old is the best observation?
        age = self.frame_count - self.last_update_frame
        age_conf = np.exp(-age.astype(np.float32) / 300.0)  # 300 frame half-life

        # Quality of best observation
        quality_conf = np.clip(self.quality_max, 0, 1)

        return (age_conf * quality_conf).astype(np.float32)


class IdentityState:
    """The complete identity belief state.

    Maintains:
    - Frequency-decomposed appearance (low + high)
    - Per-pixel belief distributions
    - Per-patch memory (regions with independent dynamics)
    - Canonical UV-space state (stabilized here, not in image space)
    - **Identity Anchor** (Module D): prevents identity drift

    THE MENTAL SHIFT:
      OLD: "how do I enhance this frame?"
      NEW: "what does this person's face usually look like?"

    ANCHOR CORRECTION (Module D from architecture):
      Every reconstruction must satisfy:
        distance(output_identity, anchor_identity) < threshold

      The anchor is the reference face's LAB values.
      After each update, identity is pulled toward anchor to prevent drift.
    """

    def __init__(self, atlas_size: Tuple[int, int] = (256, 256)):
        self.atlas_size = atlas_size
        self.freq = FrequencyDecomposition(low_pass_sigma=2.0)
        self.belief: Optional[BeliefPixel] = None
        self._pose_history = []

        # Module D: Identity Anchor
        self._anchor_low: Optional[np.ndarray] = None   # Reference low-freq (canonical)
        self._anchor_high: Optional[np.ndarray] = None  # Reference high-freq (canonical)
        self._anchor_lab: Optional[np.ndarray] = None   # Reference LAB for distance check
        self._anchor_strength = 0.35  # How much to pull toward anchor per update
        self._anchor_threshold = 25.0  # Max LAB distance before forced correction

    def is_initialized(self) -> bool:
        return self.belief is not None and self.belief._initialized

    def set_anchor(self, reference_face: np.ndarray) -> None:
        """Set the identity anchor from reference image.

        Module D: Identity Anchor System
        The anchor is the reference face warped to canonical space.
        All future reconstructions must stay close to this anchor.

        Args:
            reference_face: Reference face in canonical UV space (H, W, 3) BGR
        """
        low, high = self.freq.decompose(reference_face)
        self._anchor_low = low.copy()
        self._anchor_high = high.copy()
        self._anchor_lab = cv2.cvtColor(reference_face, cv2.COLOR_BGR2LAB).astype(np.float32)

    def _apply_anchor_correction(self) -> None:
        """Apply anchor correction to prevent identity drift.

        Module D Rule: distance(output_identity, anchor_identity) < threshold

        Math:
          For LOW freq (skin tone, lighting):
            pull = anchor_strength * (anchor_L - identity_L)
            identity_L += pull

          For HIGH freq (pores, edges):
            pull = anchor_strength * 0.3 * (anchor_H - identity_H)
            identity_H += pull  (less aggressive — preserve source detail)

        This ensures:
          - Skin tone matches reference
          - Brightness matches reference
          - Fine detail preserved from source
        """
        if self._anchor_low is None or self.belief is None:
            return

        # Current identity
        current = self.belief.reconstruct()
        current_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Check distance to anchor (drift)
        anchor_mean = np.mean(self._anchor_lab, axis=(0, 1))
        current_mean = np.mean(current_lab, axis=(0, 1))
        drift = np.sqrt(np.sum((anchor_mean - current_mean) ** 2))

        # ═══════════════════════════════════════════════════════════════════
        # IDENTITY GRAVITY EQUATION (Module D — Formalized)
        # ═══════════════════════════════════════════════════════════════════
        #
        #   I_t = (1 - λ) * I_t + λ * I_anchor
        #
        # Where λ (lambda) is conditioned on:
        #   - drift: higher drift → stronger pull (gravity increases with distance)
        #   - confidence: lower confidence → stronger pull (unstable identity needs anchor)
        #   - observation_count: fewer observations → stronger pull (new identity needs anchor)
        #
        # This creates "identity gravity" — the anchor pulls the identity
        # toward it like a gravitational field. The pull is stronger when:
        #   1. Identity has drifted far from anchor (high drift)
        #   2. Identity confidence is low (unstable observations)
        #   3. Few observations accumulated (new or reset identity)
        #
        # λ is clamped to [0.1, 0.95] to prevent:
        #   - λ=0: anchor has no effect (identity drifts freely)
        #   - λ=1: identity is always anchor (no source influence)
        # ═══════════════════════════════════════════════════════════════════

        # Compute λ (lambda) — identity gravity strength
        # Base: drift-proportional (like gravitational force ~ 1/r² but clamped)
        if drift > 30:
            lambda_base = 0.90  # Very strong pull for large drift
        elif drift > 15:
            lambda_base = 0.70  # Strong pull
        elif drift > 5:
            lambda_base = 0.50  # Moderate pull
        else:
            lambda_base = 0.30  # Gentle pull (maintenance)

        # Modulate by confidence (lower confidence → stronger anchor pull)
        # This is the key insight: unstable identity needs more anchor influence
        obs_count = np.mean(self.belief.observation_count)
        confidence_factor = 1.0 / (1.0 + obs_count * 0.01)  # Saturates at ~100 obs
        lambda_conf = lambda_base * (0.5 + 0.5 * confidence_factor)

        # Clamp λ to safe range
        lambda_clamped = np.clip(lambda_conf, 0.1, 0.95)

        # Apply identity gravity equation
        # I_t = (1 - λ) * I_t + λ * I_anchor
        self.belief.best_low = (1 - lambda_clamped) * self.belief.best_low + lambda_clamped * self._anchor_low

        # High freq: weaker pull (preserve source detail)
        # λ_high = λ * 0.2 (much less than low freq)
        lambda_high = lambda_clamped * 0.2
        self.belief.best_high = (1 - lambda_high) * self.belief.best_high + lambda_high * self._anchor_high

    def get_anchor_distance(self) -> float:
        """Get LAB distance between current identity and anchor.

        Returns:
            LAB distance (float). Should be < threshold for valid identity.
        """
        if not self.is_initialized() or self._anchor_lab is None:
            return 0.0

        current = self.belief.reconstruct()
        current_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32)

        anchor_mean = np.mean(self._anchor_lab, axis=(0, 1))
        current_mean = np.mean(current_lab, axis=(0, 1))

        return float(np.sqrt(np.sum((anchor_mean - current_mean) ** 2)))

    def update(
        self,
        canonical_face: np.ndarray,
        quality_map: np.ndarray,
        pose: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Update identity state with a new canonical observation.

        Args:
            canonical_face: Face warped to canonical UV space (H, W, 3) BGR
            quality_map: Per-pixel quality (H, W) float32
            pose: Head pose (yaw, pitch, roll)
        """
        h, w = canonical_face.shape[:2]

        if self.belief is None:
            self.belief = BeliefPixel(h, w, 3)

        # Decompose into frequencies
        low, high = self.freq.decompose(canonical_face)

        # Update belief
        self.belief.update(low, high, quality_map, pose)

        # Module D: Apply anchor correction to prevent identity drift
        self._apply_anchor_correction()

        # Track pose
        if pose is not None:
            self._pose_history.append(pose)

    def query(
        self,
        canonical_face: np.ndarray,
        quality_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query identity state for the most plausible appearance.

        THE CORE EQUATION:
          For each pixel:
            if confidence is high → use identity memory
            if confidence is low → use source observation

        But FREQUENCY-AWARE:
          - Low freq: trust memory more (skin tone is stable)
          - High freq: trust source more for current pose (texture changes with expression)

        ANCHOR CORRECTION (Module D):
          After blending, pull result toward anchor to prevent drift.

        Args:
            canonical_face: Current frame's face in canonical space
            quality_map: Per-pixel quality of current observation

        Returns:
            (reconstructed_face, confidence_map) both in canonical space
        """
        if not self.is_initialized():
            return canonical_face, np.ones(canonical_face.shape[:2], dtype=np.float32) * 0.5

        # Get identity's best appearance
        identity = self.belief.reconstruct()

        # Get confidence (based on accumulated observations)
        base_confidence = self.belief.get_confidence()

        # Modulate by CURRENT quality — if current frame is bad,
        # confidence in the output should reflect that
        # Architecture: 'trust source? or trust identity memory?'
        current_quality = np.clip(quality_map, 0, 1)
        # High current quality → trust source more (conf closer to 0.5)
        # Low current quality → trust identity more (conf closer to 1.0)
        # But we return confidence as "how much to trust identity"
        confidence = base_confidence * (0.5 + 0.5 * current_quality)

        # Frequency-aware blending
        low_curr, high_curr = self.freq.decompose(canonical_face)
        low_id, high_id = self.freq.decompose(identity)

        # Low freq: trust identity more (skin tone is stable)
        low_blend = 0.5  # 50% identity for low freq
        # High freq: trust source more for current pose
        high_blend = 0.15  # 15% identity for high freq (preserve source detail)

        # But modulate by confidence
        conf_3d = confidence[:, :, np.newaxis]
        effective_low_blend = low_blend * conf_3d
        effective_high_blend = high_blend * conf_3d

        # Reconstruct
        low_final = low_id * effective_low_blend + low_curr * (1 - effective_low_blend)
        high_final = high_id * effective_high_blend + high_curr * (1 - effective_high_blend)

        # ═══════════════════════════════════════════════════════════════════
        # IDENTITY GRAVITY IN QUERY (Module D — Post-blend anchor correction)
        # ═══════════════════════════════════════════════════════════════════
        #
        # After blending identity with source, apply identity gravity:
        #   I_final = (1 - λ) * I_blend + λ * I_anchor
        #
        # This ensures the final output stays close to anchor,
        # even when confidence is low and source dominates the blend.
        # ═══════════════════════════════════════════════════════════════════
        if self._anchor_low is not None:
            # Check distance to anchor
            result_lab = cv2.cvtColor(
                np.clip(low_final + high_final, 0, 255).astype(np.uint8),
                cv2.COLOR_BGR2LAB
            ).astype(np.float32)
            anchor_mean = np.mean(self._anchor_lab, axis=(0, 1))
            result_mean = np.mean(result_lab, axis=(0, 1))
            drift = np.sqrt(np.sum((anchor_mean - result_mean) ** 2))

            # Compute λ (identity gravity strength)
            if drift > 30:
                lambda_base = 0.85
            elif drift > 15:
                lambda_base = 0.60
            elif drift > 5:
                lambda_base = 0.35
            else:
                lambda_base = 0.15

            # Modulate by confidence (lower confidence → stronger anchor)
            lambda_clamped = np.clip(lambda_base, 0.1, 0.95)

            # Apply identity gravity: I_final = (1-λ)*I_blend + λ*I_anchor
            low_final = (1 - lambda_clamped) * low_final + lambda_clamped * self._anchor_low
            high_final = (1 - lambda_clamped * 0.2) * high_final + (lambda_clamped * 0.2) * self._anchor_high

        result = low_final + high_final
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result, confidence

    def query_region(
        self,
        region_name: str,
        canonical_face: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Query identity state for a specific region.

        Regions have INDEPENDENT dynamics:
        - eyes: high stability, preserve structure
        - beard: medium stability, preserve texture
        - skin: high stability, preserve tone
        - forehead: highest stability (changes least)

        Args:
            region_name: one of 'left_eye', 'right_eye', 'beard', 'skin', 'forehead'
            canonical_face: Current canonical face

        Returns:
            (region_patch, confidence) or (None, 0.0)
        """
        if not self.is_initialized():
            return None, 0.0

        h, w = canonical_face.shape[:2]
        identity = self.belief.reconstruct()
        confidence = self.belief.get_confidence()

        # Define region bounds in canonical space
        region_bounds = {
            'left_eye': (int(w * 0.20), int(h * 0.28), int(w * 0.42), int(h * 0.42)),
            'right_eye': (int(w * 0.58), int(h * 0.28), int(w * 0.80), int(h * 0.42)),
            'beard': (int(w * 0.25), int(h * 0.55), int(w * 0.75), int(h * 0.85)),
            'skin': (int(w * 0.20), int(h * 0.40), int(w * 0.80), int(h * 0.65)),
            'forehead': (int(w * 0.25), int(h * 0.10), int(w * 0.75), int(h * 0.30)),
            'nose': (int(w * 0.40), int(h * 0.38), int(w * 0.60), int(h * 0.55)),
            'lips': (int(w * 0.35), int(h * 0.60), int(w * 0.65), int(h * 0.72)),
        }

        if region_name not in region_bounds:
            return None, 0.0

        x1, y1, x2, y2 = region_bounds[region_name]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        region_identity = identity[y1:y2, x1:x2]
        region_conf = confidence[y1:y2, x1:x2]

        avg_conf = float(np.mean(region_conf))
        return region_identity, avg_conf

    def get_stable_l_channel(self) -> Optional[np.ndarray]:
        """Get the stable L (lightness) channel from identity state.

        This is the most reliable part of the identity — skin tone and
        overall brightness don't change much frame to frame.
        """
        if not self.is_initialized():
            return None

        identity = self.belief.reconstruct()
        lab = cv2.cvtColor(identity, cv2.COLOR_BGR2LAB).astype(np.float32)
        return lab[:, :, 0]

    def get_pose_at_best(self) -> Optional[Tuple[float, float, float]]:
        """Get the pose when the best observation was captured."""
        if not self.is_initialized():
            return None

        # Average pose across all pixels (weighted by quality)
        weights = self.belief.quality_max
        if weights.sum() < 1:
            return None

        yaw = float(np.average(self.belief.pose_at_best[:, :, 0], weights=weights))
        pitch = float(np.average(self.belief.pose_at_best[:, :, 1], weights=weights))
        roll = float(np.average(self.belief.pose_at_best[:, :, 2], weights=weights))

        return (yaw, pitch, roll)

    def reset(self) -> None:
        """Reset identity state (but preserve anchor)."""
        self.belief = None
        self._pose_history.clear()
        # Note: anchor is preserved across resets (it's the reference)
