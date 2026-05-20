"""
appearance_field.py — The Appearance Field Engine (Phase 5).

CORE CONCEPT:
  Instead of: mesh → texture → render
  Learn: A(u,v,θ,L,t) → appearance

  Where:
    (u,v) = canonical coordinates
    θ = expression/pose
    L = lighting
    t = temporal memory state

  Outputs:
    - color
    - normals (approximated)
    - microdetail
    - reflectance (approximated)
    - dynamic texture behavior

THE APPEARANCE FIELD:
  The appearance field is a FUNCTION that maps from canonical coordinates
  + pose + lighting to appearance. It's NOT a static texture atlas.

  It's a DYNAMIC appearance function that:
  1. Learns from observations over time
  2. Interpolates between known poses/lighting
  3. Generates plausible appearance for unseen conditions
  4. Maintains identity consistency

IMPLEMENTATION:
  For now, we implement a simplified version that:
  1. Stores appearance as a function of pose and lighting
  2. Can query the best appearance for given conditions
  3. Supports interpolation between stored appearances
  4. Maintains a confidence map for each stored appearance

  This is similar to the patch database, but at a higher level -
  it's about the overall face appearance, not just individual patches.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config

cfg = get_config()


class AppearanceSample:
    """A single appearance observation.

    Stores the canonical face appearance at a specific (pose, lighting) condition.
    """

    def __init__(
        self,
        canonical_face: np.ndarray,
        pose: Tuple[float, float, float],
        lighting: Optional[np.ndarray] = None,
        quality: float = 0.5,
        frame_idx: int = 0,
    ):
        self.canonical_face = canonical_face.copy()
        self.pose = pose
        self.lighting = lighting
        self.quality = quality
        self.frame_idx = frame_idx

        # Derived features
        self.lab_mean: Optional[np.ndarray] = None
        self.feature_vector: Optional[np.ndarray] = None

        self._compute_features()

    def _compute_features(self):
        """Compute feature vector for this sample."""
        lab = cv2.cvtColor(self.canonical_face, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.lab_mean = np.mean(lab, axis=(0, 1))

        # Feature vector: [pose, lab_mean, quality]
        self.feature_vector = np.concatenate([
            np.array(self.pose, dtype=np.float32),
            self.lab_mean,
            np.array([self.quality], dtype=np.float32),
        ])

    def distance_to(self, other: 'AppearanceSample') -> float:
        """Compute distance to another sample."""
        if self.feature_vector is None or other.feature_vector is None:
            return float('inf')

        return float(np.sqrt(np.sum((self.feature_vector - other.feature_vector) ** 2)))


class AppearanceField:
    """The Appearance Field A(u,v,θ,L,t).

    Maintains a collection of appearance samples and can:
    1. Query the best appearance for given (pose, lighting) conditions
    2. Interpolate between samples for unseen conditions
    3. Track confidence over time
    4. Support temporal evolution

    This is the foundation for:
    - Dynamic UV flow (Phase 5)
    - Microdetail synthesis (Phase 5)
    - Appearance-state engine (future)
    """

    def __init__(self, max_samples: int = 50):
        self.max_samples = max_samples
        self.samples: List[AppearanceSample] = []

        # Cached query results
        self._cache: Dict[str, AppearanceSample] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Temporal state
        self.frame_count: int = 0
        self.last_query_pose: Optional[Tuple[float, float, float]] = None
        self.last_query_lighting: Optional[np.ndarray] = None

    def add_sample(
        self,
        canonical_face: np.ndarray,
        pose: Tuple[float, float, float],
        lighting: Optional[np.ndarray] = None,
        quality: float = 0.5,
        frame_idx: int = 0,
    ) -> None:
        """Add a new appearance sample.

        Args:
            canonical_face: Face in canonical space (H, W, 3) BGR
            pose: Head pose (yaw, pitch, roll)
            lighting: Lighting vector (optional)
            quality: Quality score (0-1)
            frame_idx: Frame index
        """
        self.frame_count += 1

        sample = AppearanceSample(
            canonical_face=canonical_face,
            pose=pose,
            lighting=lighting,
            quality=quality,
            frame_idx=frame_idx,
        )

        self.samples.append(sample)

        # Prune if too many
        if len(self.samples) > self.max_samples:
            self._prune()

        # Clear cache
        self._cache.clear()

    def query(
        self,
        pose: Tuple[float, float, float],
        lighting: Optional[np.ndarray] = None,
        k: int = 3,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Query the best appearance for given conditions.

        Uses k-nearest neighbor interpolation:
        1. Find k closest samples to (pose, lighting)
        2. Interpolate between them weighted by distance and quality
        3. Return interpolated appearance with confidence

        Args:
            pose: Head pose (yaw, pitch, roll)
            lighting: Lighting vector (optional)
            k: Number of nearest samples to interpolate

        Returns:
            (interpolated_face, confidence) or (None, 0.0)
        """
        if not self.samples:
            return None, 0.0

        # Check cache
        cache_key = self._make_cache_key(pose, lighting)
        if cache_key in self._cache:
            self._cache_hits += 1
            cached = self._cache[cache_key]
            return cached.canonical_face, cached.quality

        self._cache_misses += 1

        # Find k nearest samples
        distances = []
        for sample in self.samples:
            dist = self._sample_distance(sample, pose, lighting)
            distances.append((dist, sample))

        # Sort by distance
        distances.sort(key=lambda x: x[0])

        # Take k nearest
        nearest = distances[:k]

        if not nearest:
            return None, 0.0

        # If only one sample or very close to one sample, return it directly
        if len(nearest) == 1 or nearest[0][0] < 1.0:
            best_sample = nearest[0][1]
            return best_sample.canonical_face, best_sample.quality

        # Interpolate between k nearest samples
        interpolated, confidence = self._interpolate(nearest)

        # Cache result
        if interpolated is not None:
            cached_sample = AppearanceSample(
                canonical_face=interpolated,
                pose=pose,
                lighting=lighting,
                quality=confidence,
                frame_idx=self.frame_count,
            )
            self._cache[cache_key] = cached_sample

        return interpolated, confidence

    def _sample_distance(
        self,
        sample: AppearanceSample,
        pose: Tuple[float, float, float],
        lighting: Optional[np.ndarray],
    ) -> float:
        """Compute distance between sample and query conditions.

        Distance is weighted:
        - Pose distance (most important)
        - Lighting distance (if available)
        - Quality penalty (prefer high quality samples)
        """
        # Pose distance
        pose_dist = self._pose_distance(sample.pose, pose)

        # Lighting distance
        lighting_dist = 0.0
        if lighting is not None and sample.lighting is not None:
            lighting_dist = float(np.sqrt(np.sum((sample.lighting - lighting) ** 2)))
            lighting_dist /= 100.0  # Normalize

        # Quality penalty (prefer high quality)
        quality_penalty = 1.0 - sample.quality

        # Combined distance
        total_dist = (
            pose_dist * 0.6 +          # Pose is most important
            lighting_dist * 0.3 +       # Lighting is secondary
            quality_penalty * 0.1       # Quality is tertiary
        )

        return total_dist

    def _pose_distance(
        self,
        pose1: Tuple[float, float, float],
        pose2: Tuple[float, float, float],
    ) -> float:
        """Compute distance between two poses."""
        yaw_diff = abs(pose1[0] - pose2[0])
        pitch_diff = abs(pose1[1] - pose2[1])
        roll_diff = abs(pose1[2] - pose2[2])

        # Weighted distance (yaw is most important for appearance)
        dist = np.sqrt(
            (yaw_diff * 1.0) ** 2 +
            (pitch_diff * 0.8) ** 2 +
            (roll_diff * 0.5) ** 2
        )

        return dist / 30.0  # Normalize

    def _interpolate(
        self,
        nearest: List[Tuple[float, AppearanceSample]],
    ) -> Tuple[Optional[np.ndarray], float]:
        """Interpolate between nearest samples.

        Uses inverse distance weighting:
        weight_i = 1 / (distance_i + epsilon)

        Args:
            nearest: List of (distance, sample) pairs

        Returns:
            (interpolated_face, confidence)
        """
        if not nearest:
            return None, 0.0

        # Compute weights (inverse distance)
        epsilon = 0.01
        weights = []
        for dist, sample in nearest:
            w = 1.0 / (dist + epsilon)
            w *= sample.quality  # Weight by quality too
            weights.append(w)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight < 1e-6:
            return None, 0.0

        weights = [w / total_weight for w in weights]

        # Interpolate faces
        interpolated = np.zeros_like(nearest[0][1].canonical_face, dtype=np.float32)
        for (dist, sample), weight in zip(nearest, weights):
            interpolated += sample.canonical_face.astype(np.float32) * weight

        interpolated = np.clip(interpolated, 0, 255).astype(np.uint8)

        # Confidence is weighted average of qualities
        confidence = sum(s.quality * w for (_, s), w in zip(nearest, weights))

        return interpolated, confidence

    def _make_cache_key(
        self,
        pose: Tuple[float, float, float],
        lighting: Optional[np.ndarray],
    ) -> str:
        """Create cache key for query."""
        parts = [f'p{int(pose[0])}_{int(pose[1])}_{int(pose[2])}']
        if lighting is not None:
            parts.append(f'l{int(np.mean(lighting))}')
        return '_'.join(parts)

    def _prune(self) -> None:
        """Remove lowest quality samples."""
        if len(self.samples) <= self.max_samples:
            return

        # Sort by quality
        self.samples.sort(key=lambda s: s.quality, reverse=True)

        # Keep top max_samples
        self.samples = self.samples[:self.max_samples]

    def get_stats(self) -> Dict:
        """Get statistics about the appearance field."""
        return {
            'num_samples': len(self.samples),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            'frame_count': self.frame_count,
        }


class DynamicAppearanceField(AppearanceField):
    """Extended appearance field with dynamic UV flow support.

    Phase 5: Dynamic UV Flow
    - Expression-dependent pore movement
    - Skin stretching
    - Beard directional deformation

    This extends the basic appearance field with:
    1. UV deformation fields per expression
    2. Skin stretch model
    3. Microdetail synthesis
    """

    def __init__(self, max_samples: int = 50):
        super().__init__(max_samples)

        # UV deformation fields per expression
        self.uv_deformations: Dict[str, np.ndarray] = {}

        # Skin stretch model
        self.stretch_model: Optional[np.ndarray] = None

        # Microdetail cache
        self._microdetail_cache: Dict[str, np.ndarray] = {}

    def add_expression_deformation(
        self,
        expression: str,
        deformation_field: np.ndarray,
    ) -> None:
        """Add UV deformation field for an expression.

        Args:
            expression: Expression name (e.g., 'smile', 'talk')
            deformation_field: (H, W, 2) UV displacement field
        """
        self.uv_deformations[expression] = deformation_field.copy()

    def query_with_deformation(
        self,
        pose: Tuple[float, float, float],
        expression: Optional[str] = None,
        lighting: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Query appearance with UV deformation applied.

        1. Query base appearance
        2. Apply expression deformation if available
        3. Return deformed appearance

        Args:
            pose: Head pose
            expression: Expression name
            lighting: Lighting vector

        Returns:
            (deformed_face, confidence)
        """
        # Query base appearance
        base_face, confidence = self.query(pose, lighting)

        if base_face is None:
            return None, 0.0

        # Apply expression deformation if available
        if expression and expression in self.uv_deformations:
            deformed = self._apply_deformation(base_face, self.uv_deformations[expression])
            return deformed, confidence

        return base_face, confidence

    def _apply_deformation(
        self,
        face: np.ndarray,
        deformation: np.ndarray,
    ) -> np.ndarray:
        """Apply UV deformation to face.

        Args:
            face: (H, W, 3) face image
            deformation: (H, W, 2) UV displacement field

        Returns:
            Deformed face image
        """
        h, w = face.shape[:2]

        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Apply deformation
        map_x = x_coords + deformation[:, :, 0]
        map_y = y_coords + deformation[:, :, 1]

        # Warp face
        deformed = cv2.remap(
            face,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        return deformed

    def compute_stretch_model(
        self,
        canonical_shape: Tuple[int, int],
    ) -> None:
        """Compute skin stretch model.

        The stretch model maps from pose to UV displacement.
        This is a simplified version - in a full implementation,
        this would be learned from data.
        """
        h, w = canonical_shape

        # Simple stretch model: face stretches with expression
        # This is a placeholder - real implementation would learn from data
        self.stretch_model = np.zeros((h, w, 2), dtype=np.float32)

    def generate_microdetail(
        self,
        canonical_face: np.ndarray,
        detail_level: float = 0.5,
    ) -> np.ndarray:
        """Generate microdetail for the face.

        Phase 5: Microdetail Synthesis
        - Generate realistic skin texture
        - Add pores, wrinkles, fine details
        - Preserve identity while adding detail

        This is a simplified version - real implementation would use
        a neural network or procedural generation.
        """
        h, w = canonical_face.shape[:2]

        # Check cache
        cache_key = f'{h}x{w}_{detail_level}'
        if cache_key in self._microdetail_cache:
            return self._microdetail_cache[cache_key]

        # Generate procedural microdetail
        detail = self._procedural_microdetail(h, w, detail_level)

        # Cache result
        self._microdetail_cache[cache_key] = detail

        return detail

    def _procedural_microdetail(
        self,
        h: int,
        w: int,
        detail_level: float,
    ) -> np.ndarray:
        """Generate procedural microdetail.

        Creates a texture that mimics skin pores and fine details.
        """
        # Generate noise at multiple scales
        detail = np.zeros((h, w), dtype=np.float32)

        # Large scale (pores)
        noise_large = np.random.randn(h // 4, w // 4).astype(np.float32)
        noise_large = cv2.resize(noise_large, (w, h), interpolation=cv2.INTER_LINEAR)
        detail += noise_large * 0.3

        # Medium scale (fine wrinkles)
        noise_medium = np.random.randn(h // 2, w // 2).astype(np.float32)
        noise_medium = cv2.resize(noise_medium, (w, h), interpolation=cv2.INTER_LINEAR)
        detail += noise_medium * 0.2

        # Small scale (skin texture)
        noise_small = np.random.randn(h, w).astype(np.float32)
        detail += noise_small * 0.1

        # Smooth slightly
        detail = cv2.GaussianBlur(detail, (3, 3), 0.5)

        # Scale by detail level
        detail *= detail_level

        return detail
