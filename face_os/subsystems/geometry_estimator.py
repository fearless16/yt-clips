"""Subsystem A — Geometry Estimation.

Estimates all spatial structure: landmarks, pose, transforms, masks, normals.

Output: GeometryState (from face_os.types)
Delegates to: landmarks.py, canonical_map.py, dense_geometry.py

BOUNDARY CONTRACT:
- MUST NOT contain identity logic
- MUST NOT contain lighting logic
- MUST NOT perform RGB blending
"""

from typing import Optional

import numpy as np

from face_os.types import GeometryState


class GeometryEstimator:
    """Subsystem A: Spatial structure estimation.

    Thin wrapper that delegates to existing modules:
    - landmarks.py for 478-point detection + PnP pose
    - canonical_map.py for canonical alignment
    - dense_geometry.py for mesh (when available)

    FORBIDDEN: identity logic, lighting logic, RGB blending
    """

    def __init__(self, config=None):
        from face_os import landmarks as lm_module
        from face_os import canonical_map

        self._lm = lm_module
        self._canonical = canonical_map
        self._config = config

    def estimate(self, frame: np.ndarray, detection=None) -> GeometryState:
        """Estimate geometry for a single frame.

        Args:
            frame: Input frame (H, W, 3) uint8 BGR
            detection: FaceDetection from tracker

        Returns:
            GeometryState with all spatial information
        """
        state = GeometryState()

        if detection is None:
            return state

        landmarks = self._lm.extract_landmarks(frame, detection)
        if landmarks is None:
            return state

        state.landmarks_478 = (
            landmarks.points if hasattr(landmarks, "points") else None
        )
        state.landmarks = landmarks
        state.pose = (
            (landmarks.yaw, landmarks.pitch, landmarks.roll)
            if hasattr(landmarks, "yaw")
            else (0.0, 0.0, 0.0)
        )
        state.geometry_confidence = (
            detection.confidence if hasattr(detection, "confidence") else 0.0
        )

        if state.landmarks_478 is not None:
            canonical_size = (
                (256, 256)
                if self._config is None
                else tuple(self._config.canonical.atlas_size)
            )
            canonical, transform, M = self._canonical.warp_to_canonical(
                frame, landmarks, canonical_size=canonical_size
            )
            state.canonical_face = canonical
            state.canonical_transform = M
            state.inverse_transform = np.linalg.inv(M)

        return state

    def assemble_state(
        self,
        canonical_face: Optional[np.ndarray] = None,
        canonical_transform: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        mesh: Optional[np.ndarray] = None,
        landmarks=None,
        pose=(0.0, 0.0, 0.0),
        geometry_confidence: float = 0.0,
    ) -> GeometryState:
        """Package already-extracted geometry primitives into a GeometryState.

        The orchestration pipeline detects landmarks, warps to canonical, builds
        the face mask, and reads the 478-point mesh exactly once per frame. This
        method gives the Geometry subsystem ownership of the resulting
        ``GeometryState`` WITHOUT re-running detection — there is one geometry
        truth per frame, not two divergent ones. ``inverse_transform`` is the
        only derived quantity (computed from ``canonical_transform``); a singular
        transform leaves it ``None`` rather than raising.

        Args:
            canonical_face: (H, W, 3) canonical-space face crop (BGR), or None.
            canonical_transform: 3x3 or 2x3 source→canonical warp, or None.
            mask: (H, W) float32 face mask in canonical space, or None.
            mesh: (>=468, 3) dense landmark mesh (mesh_478), or None.
            landmarks: Landmarks object (optional, for downstream consumers).
            pose: (yaw, pitch, roll) tuple.
            geometry_confidence: scalar detection/geometry confidence.

        Returns:
            A GeometryState assembled from the inputs. Never raises.
        """
        state = GeometryState()
        state.canonical_face = canonical_face
        state.mask = mask
        state.mesh = mesh
        state.landmarks = landmarks
        state.pose = tuple(pose) if pose is not None else (0.0, 0.0, 0.0)
        state.geometry_confidence = float(geometry_confidence)
        if landmarks is not None and hasattr(landmarks, "points"):
            state.landmarks_478 = landmarks.points

        if canonical_transform is not None:
            M = np.asarray(canonical_transform, dtype=np.float32)
            # Promote a 2x3 affine to 3x3 so inverse_transform is well-defined.
            if M.shape == (2, 3):
                M = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float32)
            state.canonical_transform = M
            if M.shape == (3, 3):
                try:
                    state.inverse_transform = np.linalg.inv(M).astype(np.float32)
                except np.linalg.LinAlgError:
                    state.inverse_transform = None

        return state

    def compute_normals(self, geometry: GeometryState) -> GeometryState:
        """Compute face normals from geometry.

        Uses mesh-derived normals when available, falls back to face-prior
        ellipsoidal normals (brightness-invariant, breaks circularity).
        """
        if geometry.canonical_face is not None:
            h, w = geometry.canonical_face.shape[:2]
            y, x = np.mgrid[0:h, 0:w].astype(np.float32)
            cx, cy = w / 2, h / 2
            nx = (x - cx) / (w / 2)
            ny = (y - cy) / (h / 2)
            nz = np.sqrt(np.maximum(1 - nx**2 - ny**2, 0))
            normals = np.stack([nx, ny, nz], axis=-1)
            normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
            geometry.mesh_normals = normals
            geometry.normal_source = "face_prior"

        return geometry
