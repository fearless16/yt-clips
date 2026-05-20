"""
neural_codec.py — Personalized Neural Codec (Phase 6).

CORE CONCEPT:
  The personalized neural codec learns a compact representation of
  the user's face that can be used to generate high-quality output.

  Instead of: source_frame → enhance → output
  Do: source_frame → encode → personalized_space → decode → output

  The personalized space is:
  - Compact (few dimensions)
  - Identity-preserving (same person every time)
  - Expression-aware (captures expressions)
  - Lighting-aware (captures lighting)

IMPLEMENTATION:
  For now, we implement a simplified version that:
  1. Learns a personalized face space from reference images
  2. Encodes source frames into this space
  3. Decodes from this space to generate output
  4. Maintains identity consistency

  This is a bridge between:
  - Current: pixel-based processing
  - Future: neural rendering
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config

cfg = get_config()


class PersonalizedSpace:
    """Learned personalized face space.

    The personalized space is a compact representation of the user's face.
    It captures:
    - Identity (who this person is)
    - Expression range (what expressions they make)
    - Lighting response (how lighting affects their face)

    This is learned from reference images and refined over time.
    """

    def __init__(self, dimensions: int = 64):
        """Initialize personalized space.

        Args:
            dimensions: Number of dimensions in the space
        """
        self.dimensions = dimensions

        # Basis vectors (learned from reference)
        self.basis: Optional[np.ndarray] = None  # (dimensions, H*W*3)
        self.mean_face: Optional[np.ndarray] = None  # (H, W, 3)

        # Learned parameters
        self.identity_centroid: Optional[np.ndarray] = None  # Center of identity
        self.expression_range: Optional[np.ndarray] = None  # Expression variation
        self.lighting_range: Optional[np.ndarray] = None  # Lighting variation

        # Training state
        self._training_samples: List[np.ndarray] = []
        self._is_trained: bool = False

    def train(self, reference_faces: List[np.ndarray]) -> None:
        """Train personalized space from reference faces.

        Uses PCA to learn the principal components of the face space.

        Args:
            reference_faces: List of canonical face images (H, W, 3)
        """
        if not reference_faces:
            return

        # Store mean face
        self.mean_face = np.mean(reference_faces, axis=0).astype(np.float32)

        # Flatten faces
        h, w, c = reference_faces[0].shape
        flat_faces = np.array([
            face.flatten().astype(np.float32) for face in reference_faces
        ])

        # Center faces
        centered = flat_faces - self.mean_face.flatten()

        # Compute PCA
        if len(reference_faces) >= self.dimensions:
            # Full PCA
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            self.basis = Vt[:self.dimensions]
        else:
            # Fewer samples than dimensions - use all
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            self.basis = Vt

        # Compute identity centroid (mean in PCA space)
        self.identity_centroid = np.mean(
            self._encode_faces(flat_faces), axis=0
        )

        # Compute expression range (variance in PCA space)
        encoded = self._encode_faces(flat_faces)
        self.expression_range = np.std(encoded, axis=0)

        # Compute lighting range (variance across different lighting)
        self.lighting_range = np.std(encoded, axis=0) * 0.5

        self._is_trained = True
        self._training_samples = reference_faces

    def encode(self, face: np.ndarray) -> np.ndarray:
        """Encode face into personalized space.

        Args:
            face: Canonical face image (H, W, 3)

        Returns:
            Encoded vector (dimensions,)
        """
        if not self._is_trained or self.basis is None:
            return np.zeros(self.dimensions)

        # Flatten and center
        flat = face.flatten().astype(np.float32)
        centered = flat - self.mean_face.flatten()

        # Project onto basis
        encoded = self.basis @ centered

        return encoded

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode from personalized space to face.

        Args:
            encoded: Encoded vector (dimensions,)

        Returns:
            Reconstructed face image (H, W, 3)
        """
        if not self._is_trained or self.basis is None:
            return np.zeros_like(self.mean_face)

        # Reconstruct
        reconstructed = self.basis.T @ encoded + self.mean_face.flatten()

        # Reshape
        h, w, c = self.mean_face.shape
        return reconstructed.reshape(h, w, c).astype(np.uint8)

    def _encode_faces(self, flat_faces: np.ndarray) -> np.ndarray:
        """Encode multiple faces."""
        centered = flat_faces - self.mean_face.flatten()
        return centered @ self.basis.T

    def get_identity_distance(self, face: np.ndarray) -> float:
        """Get distance from face to identity centroid.

        Args:
            face: Canonical face image

        Returns:
            Distance in PCA space
        """
        if not self._is_trained or self.identity_centroid is None:
            return float('inf')

        encoded = self.encode(face)
        distance = np.sqrt(np.sum((encoded - self.identity_centroid) ** 2))

        return float(distance)

    def is_trained(self) -> bool:
        """Check if space is trained."""
        return self._is_trained


class NeuralCodec:
    """Personalized Neural Codec.

    The codec learns a personalized representation of the user's face
    and uses it to generate high-quality output.

    Pipeline:
    1. Learn personalized space from reference images
    2. Encode source frames into personalized space
    3. Apply identity correction in personalized space
    4. Decode to generate output

    This is the foundation for:
    - Personalized rendering
    - Identity-preserving enhancement
    - Neural face synthesis
    """

    def __init__(self, dimensions: int = 64):
        """Initialize neural codec.

        Args:
            dimensions: Number of dimensions in personalized space
        """
        self.dimensions = dimensions
        self.space = PersonalizedSpace(dimensions)

        # Identity correction parameters
        self._correction_strength: float = 0.5
        self._correction_threshold: float = 10.0

    def train(self, reference_faces: List[np.ndarray]) -> None:
        """Train codec from reference faces.

        Args:
            reference_faces: List of canonical face images
        """
        self.space.train(reference_faces)

    def encode_and_correct(
        self,
        face: np.ndarray,
        correction_strength: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode face and apply identity correction.

        Args:
            face: Canonical face image
            correction_strength: Override correction strength

        Returns:
            (corrected_face, encoded_vector)
        """
        if not self.space.is_trained():
            return face, np.zeros(self.dimensions)

        # Encode
        encoded = self.space.encode(face)

        # Get identity distance
        distance = self.space.get_identity_distance(face)

        # Apply correction if too far from identity
        strength = correction_strength or self._correction_strength
        if distance > self._correction_threshold:
            # Pull toward identity centroid
            correction = (self.space.identity_centroid - encoded) * strength
            corrected_encoded = encoded + correction

            # Decode corrected
            corrected_face = self.space.decode(corrected_encoded)

            return corrected_face, corrected_encoded

        return face, encoded

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode from personalized space.

        Args:
            encoded: Encoded vector

        Returns:
            Reconstructed face image
        """
        return self.space.decode(encoded)

    def get_identity_score(self, face: np.ndarray) -> float:
        """Get identity match score (0-1, higher = better match).

        Args:
            face: Canonical face image

        Returns:
            Identity score
        """
        if not self.space.is_trained():
            return 0.0

        distance = self.space.get_identity_distance(face)

        # Convert distance to score (exponential decay)
        score = np.exp(-distance / 10.0)

        return float(np.clip(score, 0, 1))

    def is_trained(self) -> bool:
        """Check if codec is trained."""
        return self.space.is_trained()

    def get_stats(self) -> Dict:
        """Get codec statistics."""
        return {
            'is_trained': self.is_trained(),
            'dimensions': self.dimensions,
            'num_training_samples': len(self.space._training_samples),
            'correction_strength': self._correction_strength,
            'correction_threshold': self._correction_threshold,
        }


class IdentityOperatingSystem:
    """Full Identity Operating System.

    This is the top-level system that coordinates all modules:
    - Telemetry extraction
    - Canonical alignment
    - Patch memory
    - Identity anchor
    - Confidence engine
    - Temporal solve
    - Appearance field
    - Neural codec

    The Identity OS maintains a persistent belief about the user's
    identity and generates high-quality output that preserves this identity.
    """

    def __init__(self):
        """Initialize Identity OS."""
        # Core components
        self.codec = NeuralCodec(dimensions=64)

        # State
        self._is_initialized: bool = False
        self._frame_count: int = 0

        # Identity tracking
        self._identity_history: List[float] = []
        self._current_identity_score: float = 0.0

    def initialize(self, reference_faces: List[np.ndarray]) -> None:
        """Initialize Identity OS with reference faces.

        Args:
            reference_faces: List of canonical face images
        """
        # Train neural codec
        self.codec.train(reference_faces)

        self._is_initialized = True

    def process_frame(
        self,
        canonical_face: np.ndarray,
        quality: float = 0.5,
    ) -> Tuple[np.ndarray, float]:
        """Process a frame through the Identity OS.

        Args:
            canonical_face: Canonical face image
            quality: Quality score

        Returns:
            (processed_face, identity_score)
        """
        if not self._is_initialized:
            return canonical_face, 0.0

        self._frame_count += 1

        # Encode and correct
        corrected, encoded = self.codec.encode_and_correct(canonical_face)

        # Get identity score
        identity_score = self.codec.get_identity_score(corrected)

        # Track identity
        self._identity_history.append(identity_score)
        self._current_identity_score = identity_score

        # Keep history bounded
        if len(self._identity_history) > 1000:
            self._identity_history = self._identity_history[-1000:]

        return corrected, identity_score

    def get_identity_stability(self) -> float:
        """Get identity stability over recent frames.

        Returns:
            Stability score (0-1, higher = more stable)
        """
        if len(self._identity_history) < 10:
            return 0.0

        recent = self._identity_history[-50:]
        stability = 1.0 - np.std(recent)

        return float(np.clip(stability, 0, 1))

    def is_initialized(self) -> bool:
        """Check if Identity OS is initialized."""
        return self._is_initialized

    def get_stats(self) -> Dict:
        """Get Identity OS statistics."""
        return {
            'is_initialized': self._is_initialized,
            'frame_count': self._frame_count,
            'current_identity_score': self._current_identity_score,
            'identity_stability': self.get_identity_stability(),
            'codec_stats': self.codec.get_stats(),
        }
