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

from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np

from face_os.config import get_config

cfg = get_config()

# Region definitions for region-specific confidence
# These match the regions in patch_memory.py
REGION_DEFS = {
    "left_eye": {"bounds": (0.20, 0.28, 0.42, 0.42)},
    "right_eye": {"bounds": (0.58, 0.28, 0.80, 0.42)},
    "beard": {"bounds": (0.25, 0.55, 0.75, 0.85)},
    "skin": {"bounds": (0.20, 0.40, 0.80, 0.65)},
    "forehead": {"bounds": (0.25, 0.10, 0.75, 0.30)},
    "nose": {"bounds": (0.40, 0.38, 0.60, 0.55)},
    "lips": {"bounds": (0.35, 0.60, 0.65, 0.72)},
}


def _lab_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _pose_similarity(
    pose1: Optional[Tuple[float, float, float]],
    pose2: Optional[Tuple[float, float, float]],
) -> float:
    if pose1 is None or pose2 is None:
        return 0.7
    yaw_diff = abs(pose1[0] - pose2[0])
    pitch_diff = abs(pose1[1] - pose2[1])
    roll_diff = abs(pose1[2] - pose2[2])
    dist = np.sqrt(yaw_diff**2 + pitch_diff**2 + 0.5 * roll_diff**2)
    return float(np.exp(-dist / 25.0))


def _expression_similarity(e1: Optional[str], e2: Optional[str]) -> float:
    if e1 is None or e2 is None:
        return 0.7
    return 1.0 if e1 == e2 else 0.85


def _lighting_similarity(l1: Optional[str], l2: Optional[str]) -> float:
    if l1 is None or l2 is None:
        return 0.7
    return 1.0 if l1 == l2 else 0.9


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
        """Split image into low and high frequency components."""
        img_f = image.astype(np.float32)
        low = cv2.GaussianBlur(
            img_f,
            (self._kernel_size, self._kernel_size),
            self.low_pass_sigma,
        )
        high = img_f - low
        return low, high

    def reconstruct(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Reconstruct image from frequency components."""
        result = low + high
        return np.clip(result, 0, 255).astype(np.uint8)


class BeliefPixel:
    """Per-pixel belief distribution."""

    def __init__(self, h: int, w: int, channels: int = 3):
        self.best_low = np.zeros((h, w, channels), dtype=np.float32)
        self.best_high = np.zeros((h, w, channels), dtype=np.float32)

        self.quality_max = np.zeros((h, w), dtype=np.float32)
        self.quality_current = np.zeros((h, w), dtype=np.float32)

        self.observation_count = np.zeros((h, w), dtype=np.float32)
        self.variance = np.ones((h, w), dtype=np.float32) * 10.0

        self.pose_at_best = np.zeros((h, w, 3), dtype=np.float32)

        self.last_update_frame = np.full((h, w), -1, dtype=np.int32)
        self.frame_count = 0

        self.initialized = False

    def initialize(self, low: np.ndarray, high: np.ndarray, quality: np.ndarray) -> None:
        self.best_low = low.copy()
        self.best_high = high.copy()
        self.quality_max = quality.copy()
        self.quality_current = quality.copy()
        self.observation_count = np.ones_like(quality)
        self.variance = np.ones_like(quality) * 5.0
        self.last_update_frame[:] = 0
        self.initialized = True

    def update(
        self,
        low: np.ndarray,
        high: np.ndarray,
        quality: np.ndarray,
        pose: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.frame_count += 1

        if not self.initialized:
            self.initialize(low, high, quality)
            return

        # LOW FREQUENCY: EMA update
        base_rate = 0.15
        obs_factor = 1.0 / (1.0 + self.observation_count * 0.005)
        low_rate = base_rate * obs_factor * quality
        low_rate_3d = low_rate[:, :, np.newaxis]
        self.best_low = self.best_low * (1 - low_rate_3d) + low * low_rate_3d

        # HIGH FREQUENCY: BEST observation only
        better_mask = quality > self.quality_max
        better_mask_3d = better_mask[:, :, np.newaxis]
        self.best_high = np.where(better_mask_3d, high, self.best_high)
        self.quality_max = np.where(better_mask, quality, self.quality_max)

        if pose is not None:
            pose_arr = np.array(pose, dtype=np.float32)
            for c in range(3):
                self.pose_at_best[:, :, c] = np.where(
                    better_mask,
                    pose_arr[c],
                    self.pose_at_best[:, :, c],
                )

        # VARIANCE
        current_recon = self.best_low + self.best_high
        obs_diff = np.abs(low + high - current_recon)
        obs_variance = np.mean(obs_diff, axis=2)
        self.variance = self.variance * 0.95 + obs_variance * 0.05

        # OBSERVATION COUNT
        self.observation_count += quality
        self.quality_current = quality

        # Only update last_update_frame for pixels that were actually useful
        updated_mask = quality > 0.1
        self.last_update_frame = np.where(updated_mask, self.frame_count, self.last_update_frame)

    def reconstruct(self) -> np.ndarray:
        out = self.best_low + self.best_high
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_confidence(self) -> np.ndarray:
        obs_conf = np.clip(
            self.observation_count / max(cfg.memory.min_observations, 1),
            0,
            1,
        )
        var_conf = np.exp(-self.variance / 20.0)
        return (obs_conf * var_conf).astype(np.float32)

    def get_high_freq_confidence(self) -> np.ndarray:
        age = self.frame_count - self.last_update_frame
        age_conf = np.exp(-age.astype(np.float32) / 300.0)
        quality_conf = np.clip(self.quality_max, 0, 1)
        return (age_conf * quality_conf).astype(np.float32)


class IdentityHypothesis:
    """A hypothesis about what the face looks like."""

    def __init__(
        self,
        name: str,
        canonical_face: np.ndarray,
        quality: float,
        pose: Optional[Tuple[float, float, float]] = None,
        expression: Optional[str] = None,
        lighting: Optional[str] = None,
    ):
        self.name = name
        self.canonical_face = canonical_face.copy()
        self.quality = quality
        self.pose = pose
        self.expression = expression
        self.lighting = lighting

        self.low_freq: Optional[np.ndarray] = None
        self.high_freq: Optional[np.ndarray] = None

        self.support_count: int = 1
        self.contradiction_count: int = 0
        self.last_support_frame: int = 0

        self.confidence: float = quality

    def decompose(self, freq: FrequencyDecomposition) -> None:
        self.low_freq, self.high_freq = freq.decompose(self.canonical_face)

    def update_support(
        self,
        observation: np.ndarray,
        quality: float,
        frame_idx: int,
        similarity_threshold: float = 20.0,
    ) -> bool:
        if self.canonical_face.shape != observation.shape:
            return False

        hyp_lab = cv2.cvtColor(self.canonical_face, cv2.COLOR_BGR2LAB).astype(np.float32)
        obs_lab = cv2.cvtColor(observation, cv2.COLOR_BGR2LAB).astype(np.float32)

        hyp_mean = np.mean(hyp_lab, axis=(0, 1))
        obs_mean = np.mean(obs_lab, axis=(0, 1))
        distance = _lab_distance(hyp_mean, obs_mean)

        if distance < similarity_threshold:
            self.support_count += 1
            self.last_support_frame = frame_idx

            if quality > self.quality:
                self.canonical_face = observation.copy()
                self.quality = quality
                self.decompose(FrequencyDecomposition())

            total = self.support_count + self.contradiction_count
            self.confidence = min(1.0, self.support_count / max(total, 1))
            return True

        self.contradiction_count += 1
        total = self.support_count + self.contradiction_count
        self.confidence = min(1.0, self.support_count / max(total, 1))
        return False

    def get_age_penalty(self, current_frame: int, half_life: int = 300) -> float:
        age = current_frame - self.last_support_frame
        return float(np.exp(-age / half_life))

    def get_effective_confidence(self, current_frame: int) -> float:
        return self.confidence * self.get_age_penalty(current_frame)


class IdentityHypothesisSpace:
    """Maintains multiple hypotheses about face appearance."""

    def __init__(self, max_hypotheses: int = 10):
        self.max_hypotheses = max_hypotheses
        self.hypotheses: Dict[str, IdentityHypothesis] = {}
        self.frame_count: int = 0

    def update(
        self,
        canonical_face: np.ndarray,
        quality: float,
        pose: Optional[Tuple[float, float, float]] = None,
        expression: Optional[str] = None,
        lighting: Optional[str] = None,
    ) -> None:
        self.frame_count += 1

        key = self._make_key(pose, expression, lighting)

        matched = False
        for _, hyp in self.hypotheses.items():
            if hyp.update_support(canonical_face, quality, self.frame_count):
                matched = True
                break

        if not matched:
            self._create_hypothesis(key, canonical_face, quality, pose, expression, lighting)

        self._prune()

    def query(
        self,
        pose: Optional[Tuple[float, float, float]] = None,
        expression: Optional[str] = None,
        lighting: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        if not self.hypotheses:
            return None, 0.0

        best_hyp = None
        best_score = 0.0

        for _, hyp in self.hypotheses.items():
            score = hyp.get_effective_confidence(self.frame_count)

            if pose is not None and hyp.pose is not None:
                pose_sim = _pose_similarity(pose, hyp.pose)
                score *= (0.5 + 0.5 * pose_sim)

            score *= _expression_similarity(expression, hyp.expression)
            score *= _lighting_similarity(lighting, hyp.lighting)

            if score > best_score:
                best_score = score
                best_hyp = hyp

        if best_hyp is None:
            return None, 0.0

        return best_hyp.canonical_face, best_score

    def _make_key(
        self,
        pose: Optional[Tuple[float, float, float]] = None,
        expression: Optional[str] = None,
        lighting: Optional[str] = None,
    ) -> str:
        parts = []
        if pose is not None:
            parts.append(f"yaw{int(pose[0])}")
            parts.append(f"pit{int(pose[1])}")
        if expression:
            parts.append(expression)
        if lighting:
            parts.append(lighting)
        return "_".join(parts) if parts else f"hyp_{len(self.hypotheses)}"

    def _create_hypothesis(
        self,
        key: str,
        canonical_face: np.ndarray,
        quality: float,
        pose: Optional[Tuple[float, float, float]],
        expression: Optional[str],
        lighting: Optional[str],
    ) -> None:
        hyp = IdentityHypothesis(
            name=key,
            canonical_face=canonical_face,
            quality=quality,
            pose=pose,
            expression=expression,
            lighting=lighting,
        )
        hyp.decompose(FrequencyDecomposition())
        self.hypotheses[key] = hyp

    def _prune(self) -> None:
        if len(self.hypotheses) <= self.max_hypotheses:
            return

        sorted_hyps = sorted(
            self.hypotheses.items(),
            key=lambda x: x[1].get_effective_confidence(self.frame_count),
        )

        while len(self.hypotheses) > self.max_hypotheses and sorted_hyps:
            key, _ = sorted_hyps.pop(0)
            if key in self.hypotheses:
                del self.hypotheses[key]


class IdentityState:
    """The complete identity belief state."""

    def __init__(self, atlas_size: Tuple[int, int] = (256, 256)):
        self.atlas_size = atlas_size
        self.freq = FrequencyDecomposition(low_pass_sigma=2.0)
        self.belief: Optional[BeliefPixel] = None
        self._pose_history: List[Tuple[float, float, float]] = []

        # Module D: Identity Anchor
        self._anchor_low: Optional[np.ndarray] = None
        self._anchor_high: Optional[np.ndarray] = None
        self._anchor_lab: Optional[np.ndarray] = None
        self._anchor_strength = 0.35
        self._anchor_threshold = 25.0

        # Phase 4: Identity Hypotheses
        self.hypotheses = IdentityHypothesisSpace(max_hypotheses=10)

    def is_initialized(self) -> bool:
        return self.belief is not None and self.belief.initialized

    def set_anchor(self, reference_face: np.ndarray) -> None:
        low, high = self.freq.decompose(reference_face)
        self._anchor_low = low.copy()
        self._anchor_high = high.copy()
        self._anchor_lab = cv2.cvtColor(reference_face, cv2.COLOR_BGR2LAB).astype(np.float32)

    def _apply_anchor_correction(self) -> None:
        if self._anchor_low is None or self.belief is None:
            return

        current = self.belief.reconstruct()
        current_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32)

        anchor_mean = np.mean(self._anchor_lab, axis=(0, 1))
        current_mean = np.mean(current_lab, axis=(0, 1))
        drift = _lab_distance(anchor_mean, current_mean)

        if drift > 30:
            lambda_base = 0.95
        elif drift > 15:
            lambda_base = 0.80
        elif drift > 5:
            lambda_base = 0.60
        else:
            lambda_base = 0.40

        obs_count = np.mean(self.belief.observation_count)
        confidence_factor = 1.0 / (1.0 + obs_count * 0.01)
        lambda_conf = lambda_base * (0.7 + 0.3 * confidence_factor)
        lambda_clamped = np.clip(lambda_conf, 0.4, 0.95)

        self.belief.best_low = (
            (1 - lambda_clamped) * self.belief.best_low
            + lambda_clamped * self._anchor_low
        )

        lambda_high = lambda_clamped * 0.2
        self.belief.best_high = (
            (1 - lambda_high) * self.belief.best_high
            + lambda_high * self._anchor_high
        )

    def get_anchor_distance(self) -> float:
        if not self.is_initialized() or self._anchor_lab is None:
            return 0.0

        current = self.belief.reconstruct()
        current_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32)

        anchor_mean = np.mean(self._anchor_lab, axis=(0, 1))
        current_mean = np.mean(current_lab, axis=(0, 1))
        return _lab_distance(anchor_mean, current_mean)

    def update(
        self,
        canonical_face: np.ndarray,
        quality_map: np.ndarray,
        pose: Optional[Tuple[float, float, float]] = None,
        expression: Optional[str] = None,
        lighting: Optional[str] = None,
        region_mask: Optional[np.ndarray] = None,
    ) -> None:
        h, w = canonical_face.shape[:2]

        if self.belief is None:
            self.belief = BeliefPixel(h, w, 3)

        low, high = self.freq.decompose(canonical_face)
        self.belief.update(low, high, quality_map, pose)

        self._apply_anchor_correction()

        quality_mean = float(np.mean(quality_map))
        self.hypotheses.update(
            canonical_face=canonical_face,
            quality=quality_mean,
            pose=pose,
            expression=expression,
            lighting=lighting,
        )

        if pose is not None:
            self._pose_history.append(pose)

    def query(
        self,
        canonical_face: np.ndarray,
        quality_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_initialized():
            return canonical_face, np.ones(canonical_face.shape[:2], dtype=np.float32) * 0.5

        identity = self.belief.reconstruct()
        base_confidence = self.belief.get_confidence()

        current_quality = np.clip(quality_map, 0, 1).astype(np.float32)

        # Higher current quality -> slightly more trust in source; lower quality -> identity dominates
        confidence = base_confidence * (0.7 + 0.3 * current_quality)

        low_curr, high_curr = self.freq.decompose(canonical_face)
        low_id, high_id = self.freq.decompose(identity)

        low_blend = 0.5
        high_blend = 0.15

        conf_3d = confidence[:, :, np.newaxis]
        effective_low_blend = low_blend * conf_3d
        effective_high_blend = high_blend * conf_3d

        low_final = low_id * effective_low_blend + low_curr * (1 - effective_low_blend)
        high_final = high_id * effective_high_blend + high_curr * (1 - effective_high_blend)

        if self._anchor_low is not None and self._anchor_lab is not None:
            result_lab = cv2.cvtColor(
                np.clip(low_final + high_final, 0, 255).astype(np.uint8),
                cv2.COLOR_BGR2LAB,
            ).astype(np.float32)

            anchor_mean = np.mean(self._anchor_lab, axis=(0, 1))
            result_mean = np.mean(result_lab, axis=(0, 1))
            drift = _lab_distance(anchor_mean, result_mean)

            if drift > 30:
                lambda_base = 0.90
            elif drift > 15:
                lambda_base = 0.70
            elif drift > 5:
                lambda_base = 0.50
            else:
                lambda_base = 0.30

            lambda_clamped = np.clip(lambda_base, 0.1, 0.95)

            low_final = (1 - lambda_clamped) * low_final + lambda_clamped * self._anchor_low
            high_final = (1 - lambda_clamped * 0.2) * high_final + (
                lambda_clamped * 0.2
            ) * self._anchor_high

        result = low_final + high_final
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result, confidence

    def query_region(
        self,
        region_name: str,
        canonical_face: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        if not self.is_initialized():
            return None, 0.0

        h, w = canonical_face.shape[:2]
        identity = self.belief.reconstruct()
        confidence = self.belief.get_confidence()

        region_bounds = {
            "left_eye": (int(w * 0.20), int(h * 0.28), int(w * 0.42), int(h * 0.42)),
            "right_eye": (int(w * 0.58), int(h * 0.28), int(w * 0.80), int(h * 0.42)),
            "beard": (int(w * 0.25), int(h * 0.55), int(w * 0.75), int(h * 0.85)),
            "skin": (int(w * 0.20), int(h * 0.40), int(w * 0.80), int(h * 0.65)),
            "forehead": (int(w * 0.25), int(h * 0.10), int(w * 0.75), int(h * 0.30)),
            "nose": (int(w * 0.40), int(h * 0.38), int(w * 0.60), int(h * 0.55)),
            "lips": (int(w * 0.35), int(h * 0.60), int(w * 0.65), int(h * 0.72)),
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
        if not self.is_initialized():
            return None

        identity = self.belief.reconstruct()
        lab = cv2.cvtColor(identity, cv2.COLOR_BGR2LAB).astype(np.float32)
        return lab[:, :, 0]

    def compute_region_confidence(self) -> Dict[str, float]:
        if not self.is_initialized():
            return {name: 0.0 for name in REGION_DEFS}

        h, w = self.belief.best_low.shape[:2]
        confidence = self.belief.get_confidence()

        region_confidence = {}
        for name, rdef in REGION_DEFS.items():
            x1f, y1f, x2f, y2f = rdef["bounds"]
            x1, y1 = int(x1f * w), int(y1f * h)
            x2, y2 = int(x2f * w), int(y2f * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                region_confidence[name] = 0.0
                continue

            rc = confidence[y1:y2, x1:x2]
            region_confidence[name] = float(np.mean(rc))

        return region_confidence

    def get_pose_at_best(self) -> Optional[Tuple[float, float, float]]:
        if not self.is_initialized():
            return None

        weights = self.belief.quality_max
        if weights.sum() < 1:
            return None

        yaw = float(np.average(self.belief.pose_at_best[:, :, 0], weights=weights))
        pitch = float(np.average(self.belief.pose_at_best[:, :, 1], weights=weights))
        roll = float(np.average(self.belief.pose_at_best[:, :, 2], weights=weights))
        return yaw, pitch, roll

    def reset(self) -> None:
        self.belief = None
        self._pose_history.clear()
        # anchor is preserved
