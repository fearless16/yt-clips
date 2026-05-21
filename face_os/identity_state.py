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


# ─── Verification Gate ──────────────────────────────────────────────────────

class VerificationGate:
    """Gates identity memory updates behind identity/liveness checks.

    All checks must pass before identity_state.update() is called.
    Rejects: wrong identity, tiny faces, static posters, low quality.
    """

    def __init__(
        self,
        embedding_tolerance: float = 0.45,
        min_face_pixels: int = 4000,
        liveness_threshold: float = 0.5,
    ):
        self.embedding_tolerance = embedding_tolerance
        self.min_face_pixels = min_face_pixels
        self.liveness_threshold = liveness_threshold

        # Reference embedding for identity check
        self._reference_embedding: Optional[np.ndarray] = None

        # Landmark history for liveness check
        self._landmark_history: List[np.ndarray] = []
        self._max_history = 10

    def set_reference_embedding(self, embedding: np.ndarray) -> None:
        """Set reference embedding for identity matching."""
        self._reference_embedding = embedding

    def verify(
        self,
        canonical_face: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]],
        landmarks_pts: Optional[np.ndarray],
        embedding: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str]:
        """Run all verification checks.

        Args:
            canonical_face: Face image in canonical space
            face_bbox: (x, y, w, h) bounding box
            landmarks_pts: (N, 2) or (N, 3) landmark coordinates
            embedding: Face embedding vector

        Returns:
            (passed, reason) — True if all checks pass
        """
        # Gate 1: Face pixel count
        if face_bbox is not None:
            x, y, w, h = face_bbox
            face_pixels = w * h
            if face_pixels < self.min_face_pixels:
                return False, f"face_too_small: {face_pixels} < {self.min_face_pixels}"

        # Gate 2: Embedding identity check
        if self._reference_embedding is not None and embedding is not None:
            dist = self._embedding_distance(embedding, self._reference_embedding)
            if dist > self.embedding_tolerance:
                return False, f"identity_mismatch: {dist:.3f} > {self.embedding_tolerance}"

        # Gate 3: Liveness check (landmark jitter)
        if landmarks_pts is not None:
            # Use only x,y for jitter calculation
            pts_2d = landmarks_pts[:, :2] if landmarks_pts.shape[1] > 2 else landmarks_pts
            self._landmark_history.append(pts_2d.copy())
            if len(self._landmark_history) > self._max_history:
                self._landmark_history = self._landmark_history[-self._max_history:]

            if len(self._landmark_history) >= 2:
                jitter = self._compute_jitter()
                if jitter < self.liveness_threshold:
                    return False, f"static_poster: jitter={jitter:.4f} < {self.liveness_threshold}"

        return True, "passed"

    def _embedding_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute distance between two embeddings."""
        try:
            import face_recognition
            return float(face_recognition.face_distance([emb2], emb1)[0])
        except ImportError:
            # Fallback: cosine distance
            norm1 = emb1 / max(np.linalg.norm(emb1), 1e-6)
            norm2 = emb2 / max(np.linalg.norm(emb2), 1e-6)
            return float(1.0 - np.dot(norm1, norm2))

    def _compute_jitter(self) -> float:
        """Compute landmark jitter across history."""
        if len(self._landmark_history) < 2:
            return 1.0

        displacements = []
        for i in range(1, len(self._landmark_history)):
            prev = self._landmark_history[i - 1]
            curr = self._landmark_history[i]
            if prev.shape == curr.shape:
                disp = np.mean(np.sqrt(np.sum((curr - prev) ** 2, axis=1)))
                # Normalize by face size (assume ~200px)
                disp_norm = disp / 200.0
                displacements.append(disp_norm)

        if not displacements:
            return 1.0

        return float(np.mean(displacements))

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
        self.variance = np.ones_like(quality) * 2.0
        self.last_update_frame[:] = 0
        self.initialized = True

    def update(
        self,
        low: np.ndarray,
        high: np.ndarray,
        quality: np.ndarray,
        pose: Optional[Tuple[float, float, float]] = None,
        region_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.frame_count += 1

        if not self.initialized:
            self.initialize(low, high, quality)
            return

        # LOW FREQUENCY: EMA update (config-driven rate)
        base_rate = cfg.identity_state.low_freq_ema_rate
        obs_factor = 1.0 / (1.0 + self.observation_count * 0.005)
        low_rate = base_rate * obs_factor * quality

        # Region confidence modulation: faster learning for skin, slower for eyes
        if region_mask is not None:
            low_rate = low_rate * region_mask

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
        self._anchor_albedo: Optional[np.ndarray] = None  # I-05: lighting-invariant anchor
        self._anchor_strength = 0.35
        self._anchor_threshold = 25.0

        # Intrinsic belief snapshots — immutable per observation
        self._intrinsic_history: List[dict] = []
        self._intrinsic_history_limit = 32
        self._intrinsic_update_count = 0
        self._intrinsic_best_score = float("-inf")
        self._intrinsic_best_snapshot: Optional[dict] = None

        # Phase 4: Identity Hypotheses
        self.hypotheses = IdentityHypothesisSpace(max_hypotheses=10)

        # Verification Gate
        self._gate = VerificationGate(
            embedding_tolerance=cfg.identity.embedding_tolerance,
            min_face_pixels=cfg.verification_gate.min_face_pixels,
            liveness_threshold=cfg.verification_gate.liveness_threshold,
        )

        # V3: Intrinsic Decomposition
        from face_os.intrinsic_decomposition import IntrinsicDecomposer, IntrinsicComponents
        self._intrinsic_decomposer = IntrinsicDecomposer()
        self._intrinsic_components: Optional[IntrinsicComponents] = None

    def is_initialized(self) -> bool:
        return self.belief is not None and self.belief.initialized

    def set_anchor(self, reference_face: np.ndarray) -> None:
        low, high = self.freq.decompose(reference_face)
        self._anchor_low = low.copy()
        self._anchor_high = high.copy()
        self._anchor_lab = cv2.cvtColor(reference_face, cv2.COLOR_BGR2LAB).astype(np.float32)

        # V3: Compute intrinsic components for anchor
        reference_face_rgb = cv2.cvtColor(reference_face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self._anchor_intrinsic = self._intrinsic_decomposer.decompose(reference_face_rgb)

        # I-05: Store anchor albedo separately (lighting-invariant)
        self._anchor_albedo = self._normalize_white_balance(self._anchor_intrinsic.albedo)

    def _apply_anchor_correction(self) -> None:
        if self._anchor_low is None or self.belief is None:
            return

        current = self.belief.reconstruct()
        current_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32)

        anchor_mean = np.mean(self._anchor_lab, axis=(0, 1))
        current_mean = np.mean(current_lab, axis=(0, 1))
        drift = _lab_distance(anchor_mean, current_mean)

        if drift > 30:
            lambda_base = cfg.identity_state.anchor_lambda_max
        elif drift > 15:
            lambda_base = cfg.identity_state.anchor_lambda_max * 0.93
        elif drift > 5:
            lambda_base = cfg.identity_state.anchor_lambda_max * 0.87
        else:
            lambda_base = cfg.identity_state.anchor_lambda_max * 0.80

        obs_count = np.mean(self.belief.observation_count)
        confidence_factor = 1.0 / (1.0 + obs_count * 0.01)
        lambda_conf = lambda_base * (0.9 + 0.1 * confidence_factor)
        lambda_clamped = np.clip(lambda_conf, 0.60, cfg.identity_state.anchor_lambda_max)

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

    def _intrinsic_snapshot_score(
        self,
        snapshot: dict,
        quality_map: Optional[np.ndarray] = None,
    ) -> float:
        """Score an intrinsic snapshot without mutating state.

        Higher is better. We prefer:
        - higher confidence
        - lower reconstruction error
        - fresher observations
        - better current quality compatibility
        """
        components = snapshot.get("components")
        if components is None:
            return float("-inf")

        quality_mean = float(snapshot.get("quality_mean", 0.0))
        quality_map_mean = float(np.mean(quality_map)) if quality_map is not None else quality_mean
        intrinsic_conf = snapshot.get("confidence_scalar", 0.0)
        recon_error = float(getattr(components, "reconstruction_error", 1.0))
        age = max(self._intrinsic_update_count - int(snapshot.get("update_index", 0)), 0)
        age_factor = float(np.exp(-age / 24.0))

        # Blend quality, confidence, and reconstruction fidelity.
        base = 0.45 * quality_map_mean + 0.35 * float(intrinsic_conf) + 0.20 * max(0.0, 1.0 - min(recon_error, 1.0))
        return float(base * age_factor)

    def _store_intrinsic_snapshot(
        self,
        intrinsic_components,
        quality_map: np.ndarray,
        pose: Optional[Tuple[float, float, float]] = None,
        mesh_478: Optional[np.ndarray] = None,
        warp_M: Optional[np.ndarray] = None,
    ) -> None:
        """Store an immutable snapshot of intrinsic belief.

        This prevents later frames from silently overwriting the earlier,
        higher-confidence intrinsic state.
        """
        self._intrinsic_update_count += 1
        quality_mean = float(np.mean(quality_map)) if quality_map is not None else 0.0
        confidence_scalar = float(np.mean(getattr(intrinsic_components, "confidence", 0.0)))
        snapshot = {
            "update_index": self._intrinsic_update_count,
            "components": intrinsic_components,
            "quality_mean": quality_mean,
            "confidence_scalar": confidence_scalar,
            "pose": pose,
            "mesh_478": mesh_478,
            "warp_M": warp_M,
        }
        self._intrinsic_history.append(snapshot)
        if len(self._intrinsic_history) > self._intrinsic_history_limit:
            self._intrinsic_history = self._intrinsic_history[-self._intrinsic_history_limit:]

        score = self._intrinsic_snapshot_score(snapshot, quality_map)
        if score >= self._intrinsic_best_score:
            self._intrinsic_best_score = score
            self._intrinsic_best_snapshot = snapshot

    def _select_intrinsic_snapshot(
        self,
        quality_map: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """Select the best intrinsic snapshot for the current query.

        Never mutates the stored history.
        """
        if not self._intrinsic_history:
            return None

        best_snapshot = None
        best_score = float("-inf")
        for snapshot in self._intrinsic_history:
            score = self._intrinsic_snapshot_score(snapshot, quality_map)
            if score > best_score:
                best_score = score
                best_snapshot = snapshot

        return best_snapshot

    def update(
        self,
        canonical_face: np.ndarray,
        quality_map: np.ndarray,
        pose: Optional[Tuple[float, float, float]] = None,
        expression: Optional[str] = None,
        lighting: Optional[str] = None,
        region_mask: Optional[np.ndarray] = None,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        landmarks_pts: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
        mesh_478: Optional[np.ndarray] = None,
        warp_M: Optional[np.ndarray] = None,
    ) -> bool:
        """Update identity state with new observation.

        Args:
            canonical_face: Face in canonical space (256x256)
            quality_map: Per-pixel quality map
            pose: Head pose (yaw, pitch, roll)
            expression: Expression label
            lighting: Lighting label
            region_mask: Region-specific learning rates
            face_bbox: (x, y, w, h) bounding box for verification
            landmarks_pts: (N, 2) or (N, 3) landmark coordinates for liveness
            embedding: Face embedding for identity check
            mesh_478: Optional (478, 3) MediaPipe mesh for geometry normals
            warp_M: Optional (2, 3) forward warp for normal projection

        Returns:
            True if update was applied, False if rejected by verification gate
        """
        # VERIFICATION GATE: Check all gates before updating
        if face_bbox is not None or landmarks_pts is not None:
            ok, _ = self._gate.verify(canonical_face, face_bbox, landmarks_pts, embedding)
            if not ok:
                return False

        h, w = canonical_face.shape[:2]

        if self.belief is None:
            self.belief = BeliefPixel(h, w, 3)

        # Compute region confidence mask for low_rate modulation
        # Eyes: slower learning (preserve structure), Skin: faster (normalize lighting)
        computed_region_mask = np.ones((h, w), dtype=np.float32)
        for name, rdef in REGION_DEFS.items():
            x1f, y1f, x2f, y2f = rdef["bounds"]
            x1, y1 = int(x1f * w), int(y1f * h)
            x2, y2 = int(x2f * w), int(y2f * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                if "eye" in name:
                    computed_region_mask[y1:y2, x1:x2] = 0.5  # Slower for eyes
                elif name == "skin":
                    computed_region_mask[y1:y2, x1:x2] = 1.5  # Faster for skin
                elif name == "forehead":
                    computed_region_mask[y1:y2, x1:x2] = 1.3  # Faster for forehead

        # Use provided region_mask if available, otherwise use computed one
        final_region_mask = region_mask if region_mask is not None else computed_region_mask

        low, high = self.freq.decompose(canonical_face)
        self.belief.update(low, high, quality_map, pose, region_mask=final_region_mask)

        # V3: Compute intrinsic decomposition for this frame
        # Uses mesh-derived normals when mesh_478 + warp_M are available
        canonical_face_rgb = cv2.cvtColor(canonical_face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        intrinsic_components = self._intrinsic_decomposer.decompose(
            canonical_face_rgb, mesh_478=mesh_478, warp_M=warp_M,
        )
        self._intrinsic_components = intrinsic_components  # latest snapshot for backwards compatibility
        self._normal_source = self._intrinsic_decomposer._normal_source

        # Store immutable intrinsic snapshot so later frames do not clobber earlier belief.
        self._store_intrinsic_snapshot(
            intrinsic_components=intrinsic_components,
            quality_map=quality_map,
            pose=pose,
            mesh_478=mesh_478,
            warp_M=warp_M,
        )

        # Anchor correction removed from update - only apply at query time to preserve raw telemetry

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

        return True

    def query(
        self,
        canonical_face: np.ndarray,
        quality_map: np.ndarray,
        pose: Optional[Tuple[float, float, float]] = None,
        face_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_initialized():
            return canonical_face, np.ones(canonical_face.shape[:2], dtype=np.float32) * 0.5

        identity = self.belief.reconstruct()
        base_confidence = self.belief.get_confidence()

        # Phase 4: Check hypotheses for better pose-matched identity
        if pose is not None:
            hyp_face, hyp_score = self.hypotheses.query(pose=pose)
            if hyp_face is not None and hyp_score > 0.4:  # Lowered from 0.6
                # Use hypothesis as identity — it's pose-matched
                identity = hyp_face
                base_confidence = base_confidence * (0.7 + 0.3 * hyp_score)

        current_quality = np.clip(quality_map, 0, 1).astype(np.float32)

        # Higher current quality -> slightly more trust in source; lower quality -> identity dominates
        confidence = base_confidence * (0.9 + 0.1 * current_quality)

        # Mask confidence to face region only — prevent background reconstruction
        if face_mask is not None:
            confidence = confidence * face_mask

        low_curr, high_curr = self.freq.decompose(canonical_face)
        low_id, high_id = self.freq.decompose(identity)

        # Dynamic blend based on confidence
        # High confidence → more identity, less source
        # Low confidence → less identity, more source
        mean_conf = float(np.mean(confidence))
        low_blend = cfg.identity_state.low_blend_base + (1 - cfg.identity_state.low_blend_base) * mean_conf
        # High freq: use floor to preserve texture (beard, pores, skin grain)
        high_blend = max(cfg.identity_state.high_blend_base, 0.15)

        conf_3d = confidence[:, :, np.newaxis]
        effective_low_blend = low_blend * conf_3d
        # High freq: do NOT multiply by conf again — it's already low, double-dampening kills texture
        effective_high_blend = np.full_like(conf_3d, high_blend)

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
                lambda_base = cfg.identity_state.anchor_lambda_max
            elif drift > 15:
                lambda_base = cfg.identity_state.anchor_lambda_max * 0.93
            elif drift > 5:
                lambda_base = cfg.identity_state.anchor_lambda_max * 0.87
            else:
                lambda_base = cfg.identity_state.anchor_lambda_max * 0.80

            lambda_clamped = np.clip(lambda_base, 0.60, cfg.identity_state.anchor_lambda_max)

            low_final = (1 - lambda_clamped) * low_final + lambda_clamped * self._anchor_low
            high_final = (1 - lambda_clamped * 0.2) * high_final + (
                lambda_clamped * 0.2
            ) * self._anchor_high

        result = low_final + high_final
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result, confidence

    def query_identity(
        self,
        quality_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query raw identity memory without blending with source.

        Returns anchor-corrected identity + confidence.
        The compositor handles blending in source space.

        Args:
            quality_map: Per-pixel quality (H, W) float32

        Returns:
            (identity_face, confidence_map) both in canonical space
        """
        if not self.is_initialized():
            h, w = quality_map.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8), np.ones((h, w), dtype=np.float32) * 0.5

        identity = self.belief.reconstruct()
        base_confidence = self.belief.get_confidence()

        current_quality = np.clip(quality_map, 0, 1).astype(np.float32)
        confidence = base_confidence * (0.9 + 0.1 * current_quality)

        # Apply anchor correction to raw identity
        if self._anchor_low is not None and self._anchor_lab is not None:
            id_lab = cv2.cvtColor(identity, cv2.COLOR_BGR2LAB).astype(np.float32)
            anchor_mean = np.mean(self._anchor_lab, axis=(0, 1))
            id_mean = np.mean(id_lab, axis=(0, 1))
            drift = _lab_distance(anchor_mean, id_mean)

            if drift > 30:
                lambda_base = cfg.identity_state.anchor_lambda_max
            elif drift > 15:
                lambda_base = cfg.identity_state.anchor_lambda_max * 0.93
            elif drift > 5:
                lambda_base = cfg.identity_state.anchor_lambda_max * 0.87
            else:
                lambda_base = cfg.identity_state.anchor_lambda_max * 0.80

            lambda_clamped = np.clip(lambda_base, 0.60, cfg.identity_state.anchor_lambda_max)

            # Pull identity toward anchor
            id_lab[:, :, 0] = (1 - lambda_clamped) * id_lab[:, :, 0] + lambda_clamped * self._anchor_lab[:, :, 0]
            id_lab[:, :, 1] = (1 - lambda_clamped) * id_lab[:, :, 1] + lambda_clamped * self._anchor_lab[:, :, 1]
            id_lab[:, :, 2] = (1 - lambda_clamped) * id_lab[:, :, 2] + lambda_clamped * self._anchor_lab[:, :, 2]

            identity = cv2.cvtColor(np.clip(id_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        return identity, confidence

    def query_albedo(
        self,
        quality_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query lighting-invariant albedo identity.

        I-05: Returns albedo (lighting-invariant) with anchor correction applied
        in albedo space, NOT RGB space. This decouples identity from lighting.

        Anchor correction flow:
        1. Decompose current identity into intrinsic components
        2. Extract albedo (lighting-invariant)
        3. Apply white-balance normalization
        4. Pull albedo toward anchor_albedo (not anchor_rgb)
        5. Return corrected albedo + confidence

        Args:
            quality_map: Per-pixel quality (H, W) float32

        Returns:
            (albedo, confidence) both in canonical space, albedo in [0, 1] float32
        """
        if not self.is_initialized():
            h, w = quality_map.shape[:2]
            return (
                np.ones((h, w, 3), dtype=np.float32) * 0.5,
                np.ones((h, w), dtype=np.float32) * 0.5,
            )

        snapshot = self._select_intrinsic_snapshot(quality_map)
        if snapshot is None or snapshot.get("components") is None:
            h, w = quality_map.shape[:2]
            return (
                np.ones((h, w, 3), dtype=np.float32) * 0.5,
                np.ones((h, w), dtype=np.float32) * 0.5,
            )

        intrinsic = snapshot["components"]
        albedo = intrinsic.albedo.copy()

        # White-balance normalization (RULE 5)
        albedo = self._normalize_white_balance(albedo)

        # Anchor correction in albedo space (NOT RGB)
        if self._anchor_albedo is not None:
            anchor_mean = np.mean(self._anchor_albedo, axis=(0, 1))
            albedo_mean = np.mean(albedo, axis=(0, 1))
            drift = float(np.sqrt(np.sum((anchor_mean - albedo_mean) ** 2)))

            if drift > 0.15:
                lambda_base = cfg.identity_state.anchor_lambda_max
            elif drift > 0.08:
                lambda_base = cfg.identity_state.anchor_lambda_max * 0.93
            elif drift > 0.03:
                lambda_base = cfg.identity_state.anchor_lambda_max * 0.87
            else:
                lambda_base = cfg.identity_state.anchor_lambda_max * 0.80

            lambda_clamped = np.clip(lambda_base, 0.60, cfg.identity_state.anchor_lambda_max)

            # Pull albedo toward anchor albedo
            albedo = (1 - lambda_clamped) * albedo + lambda_clamped * self._anchor_albedo
            albedo = np.clip(albedo, 0, 1).astype(np.float32)

        # Confidence from belief
        base_confidence = self.belief.get_confidence()
        current_quality = np.clip(quality_map, 0, 1).astype(np.float32)
        confidence = base_confidence * (0.9 + 0.1 * current_quality)

        return albedo, confidence

    def query_intrinsic(
        self,
        quality_map: np.ndarray,
    ) -> Tuple[Optional['IntrinsicComponents'], np.ndarray]:
        """Query intrinsic decomposition components.

        Returns intrinsic components (albedo, shading, specular, normals) + confidence.
        Used by PhysicalRenderer for physically-based rendering.

        RULE 5: Returns albedo (lighting-invariant) separately from appearance.
        This decouples identity from lighting.

        I-05: Anchor correction is applied to albedo in albedo space (not RGB).

        Args:
            quality_map: Per-pixel quality (H, W) float32

        Returns:
            (intrinsic_components, confidence_map) or (None, default_confidence) if not available
        """
        if not self.is_initialized():
            h, w = quality_map.shape[:2]
            return None, np.ones((h, w), dtype=np.float32) * 0.5

        # Select the best intrinsic snapshot without mutating stored state.
        snapshot = self._select_intrinsic_snapshot(quality_map)
        if snapshot is None or snapshot.get("components") is None:
            h, w = quality_map.shape[:2]
            return None, np.ones((h, w), dtype=np.float32) * 0.5

        intrinsic = snapshot["components"]

        # RULE 5: Normalize albedo for white balance
        # This ensures identity is lighting-invariant
        normalized_albedo = self._normalize_white_balance(intrinsic.albedo)

        # I-05: Apply anchor correction in albedo space
        if self._anchor_albedo is not None:
            anchor_mean = np.mean(self._anchor_albedo, axis=(0, 1))
            albedo_mean = np.mean(normalized_albedo, axis=(0, 1))
            drift = float(np.sqrt(np.sum((anchor_mean - albedo_mean) ** 2)))

            if drift > 0.15:
                lambda_base = cfg.identity_state.anchor_lambda_max
            elif drift > 0.08:
                lambda_base = cfg.identity_state.anchor_lambda_max * 0.93
            elif drift > 0.03:
                lambda_base = cfg.identity_state.anchor_lambda_max * 0.87
            else:
                lambda_base = cfg.identity_state.anchor_lambda_max * 0.80

            lambda_clamped = np.clip(lambda_base, 0.60, cfg.identity_state.anchor_lambda_max)
            normalized_albedo = (1 - lambda_clamped) * normalized_albedo + lambda_clamped * self._anchor_albedo
            normalized_albedo = np.clip(normalized_albedo, 0, 1).astype(np.float32)

        # Create corrected intrinsic copy with normalized albedo
        from face_os.intrinsic_decomposition import IntrinsicComponents
        corrected = IntrinsicComponents(
            albedo=normalized_albedo,
            shading=intrinsic.shading,
            specular=intrinsic.specular,
            normal_map=intrinsic.normal_map,
            confidence=intrinsic.confidence,
            reconstruction_error=intrinsic.reconstruction_error,
            albedo_uncertainty=intrinsic.albedo_uncertainty,
            shading_uncertainty=intrinsic.shading_uncertainty,
            specular_uncertainty=intrinsic.specular_uncertainty,
            decomposition_quality=intrinsic.decomposition_quality,
        )

        # Compute confidence from quality map
        base_confidence = self.belief.get_confidence()
        current_quality = np.clip(quality_map, 0, 1).astype(np.float32)
        confidence = base_confidence * (0.9 + 0.1 * current_quality)

        return corrected, confidence

    def _normalize_white_balance(self, albedo: np.ndarray) -> np.ndarray:
        """Normalize albedo for white balance.

        RULE 5: Removes color cast from lighting to isolate identity.
        Uses gray-world assumption: mean of each channel should be equal.

        Args:
            albedo: Albedo map (H, W, 3) float32 [0, 1]

        Returns:
            White-balanced albedo (H, W, 3) float32 [0, 1]
        """
        # Gray-world white balance
        mean_per_channel = np.mean(albedo, axis=(0, 1))
        overall_mean = np.mean(mean_per_channel)
        scale = overall_mean / (mean_per_channel + 1e-8)
        normalized = albedo * scale[np.newaxis, np.newaxis, :]
        return np.clip(normalized, 0, 1).astype(np.float32)

    def _normalize_exposure(self, image: np.ndarray) -> np.ndarray:
        """Normalize exposure to standard luminance.

        RULE 5: Removes exposure variation from identity.

        Args:
            image: Image (H, W, 3) float32

        Returns:
            Exposure-normalized image (H, W, 3) float32
        """
        lab = cv2.cvtColor(
            (np.clip(image, 0, 1) * 255).astype(np.uint8),
            cv2.COLOR_BGR2LAB,
        ).astype(np.float32)
        L = lab[:, :, 0]
        mean_L = np.mean(L)
        target_L = 128.0  # Standard mid-gray
        scale = target_L / (mean_L + 1e-8)
        scale = np.clip(scale, 0.5, 2.0)
        lab[:, :, 0] = np.clip(L * scale, 0, 255)
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result.astype(np.float32) / 255.0

    def has_intrinsic(self) -> bool:
        """Check if intrinsic decomposition is available."""
        return self._intrinsic_components is not None

    def get_normal_source(self) -> str:
        """Get the source of surface normals from last decomposition.

        Returns:
            "mesh" if derived from 478-point mesh geometry,
            "shading_gradient" if derived from shading (circular fallback)
        """
        return getattr(self, '_normal_source', 'mesh')

    def get_anchor_intrinsic(self) -> Optional['IntrinsicComponents']:
        """Get intrinsic components for anchor face."""
        return getattr(self, '_anchor_intrinsic', None)

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
        self._intrinsic_components = None
        self._intrinsic_history.clear()
        self._intrinsic_update_count = 0
        self._intrinsic_best_score = float("-inf")
        self._intrinsic_best_snapshot = None
        # anchor is preserved (including _anchor_albedo)
