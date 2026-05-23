"""Dense Geometry Module.

Dense mesh fitting from sparse landmarks.

Approach:
    1. Start with parametric face model (icosphere template)
    2. Apply Anatomical Depth Prior (nose protrudes, eyes recede)
    3. Fit to 478 MediaPipe landmarks via Vectorized RBF deformation
    4. Output dense mesh + normals + UV coordinates

BEAST MODE FIXES:
    - Killed the fake "sphere projection" (balloon face).
    - Added analytical face depth prior (nose, cheeks, eye sockets).
    - Vectorized RBF interpolation (killed the Python for-loops).
    - Cached template and KDTree for real-time performance.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


@dataclass
class GeometryConfig:
    """Configuration for dense geometry estimation."""
    n_vertices: int = 3000
    n_faces: int = 6000
    landmark_weight: float = 10.0
    smoothness_weight: float = 0.1
    confidence_decay: float = 0.01
    uv_mode: str = 'spherical'
    rbf_neighbors: int = 15


@dataclass
class DenseGeometry:
    """Dense mesh geometry."""
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    uv_coords: np.ndarray
    confidence: np.ndarray
    landmark_correspondence: Dict[int, int]


@dataclass
class GeometryReport:
    """Per-frame geometry metrics."""
    frame_idx: int
    geometry: DenseGeometry
    fitting_error: float
    smoothness_score: float
    estimation_time_ms: float

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "fitting_error": self.fitting_error,
            "smoothness_score": self.smoothness_score,
            "estimation_time_ms": self.estimation_time_ms,
            "n_vertices": self.geometry.vertices.shape[0],
            "n_faces": self.geometry.faces.shape[0],
        }


class DenseGeometryEstimator:
    """Dense geometry estimator from sparse landmarks."""

    def __init__(self, config: Optional[GeometryConfig] = None):
        self.config = config or GeometryConfig()
        self._template_vertices = None
        self._template_faces = None
        self._landmark_indices = None
        self._template_tree = None

    def estimate(
        self,
        landmarks: np.ndarray,
        config: Optional[GeometryConfig] = None,
    ) -> DenseGeometry:
        if config is not None:
            self.config = config

        start_time = time.perf_counter()

        if self._template_vertices is None:
            self._initialize_template()

        vertices = self._fit_to_landmarks(landmarks)
        normals = self._compute_normals(vertices, self._template_faces)
        uv_coords = self._compute_uv(vertices)
        confidence = self._compute_confidence(vertices, landmarks)
        landmark_correspondence = self._create_correspondence(landmarks)

        return DenseGeometry(
            vertices=vertices.astype(np.float32),
            faces=self._template_faces.copy(),
            normals=normals.astype(np.float32),
            uv_coords=uv_coords.astype(np.float32),
            confidence=confidence.astype(np.float32),
            landmark_correspondence=landmark_correspondence,
        )

    def _initialize_template(self):
        """Initialize and cache the sphere template mesh."""
        self._template_vertices, self._template_faces = self._create_icosphere(n_subdivisions=3)

        n_vertices = self._template_vertices.shape[0]
        
        # Scale to face-like proportions
        self._template_vertices[:, 0] *= 1.2  
        self._template_vertices[:, 1] *= 1.5  
        self._template_vertices[:, 2] *= 0.8  

        # Anatomical landmark mapping: key MediaPipe indices → icosphere regions
        key_anchors = {
            1: (0.0, -0.1),      # nose tip
            10: (0.0, 0.55),     # forehead
            33: (-0.22, 0.12),   # left eye inner
            133: (-0.35, 0.12),  # left eye outer
            263: (0.22, 0.12),   # right eye inner
            362: (0.35, 0.12),   # right eye outer
            61: (-0.12, -0.35),  # left mouth corner
            291: (0.12, -0.35),  # right mouth corner
            152: (0.0, -0.65),   # chin
            234: (-0.55, 0.0),   # left ear
            454: (0.55, 0.0),    # right ear
            168: (0.0, 0.2),     # nose bridge
            4: (0.0, -0.2),      # nose bottom
            70: (-0.15, -0.15),  # left nostril
            300: (0.15, -0.15),  # right nostril
        }

        template_xy = self._template_vertices[:, :2]
        anchor_tree = cKDTree(template_xy)

        self._landmark_indices = np.zeros(478, dtype=int)
        assigned = set()

        for mp_idx, (ax, ay) in key_anchors.items():
            if mp_idx < 478:
                target = np.array([ax * 1.2, ay * 1.5])
                _, nearest_idx = anchor_tree.query(target)
                self._landmark_indices[mp_idx] = nearest_idx
                assigned.add(mp_idx)

        remaining = [i for i in range(478) if i not in assigned]
        used = set(self._landmark_indices[list(assigned)])
        available = np.array(sorted(set(range(n_vertices)) - used))

        if len(available) >= len(remaining):
            spacing = np.linspace(0, len(available) - 1, len(remaining), dtype=int)
            for i, mp_idx in enumerate(remaining):
                self._landmark_indices[mp_idx] = available[spacing[i]]
        else:
            spacing = np.linspace(0, n_vertices - 1, len(remaining), dtype=int)
            for i, mp_idx in enumerate(remaining):
                self._landmark_indices[mp_idx] = spacing[i]
        
        # Cache KDTree for RBF
        self._template_tree = cKDTree(self._template_vertices[self._landmark_indices])

    def _create_icosphere(self, n_subdivisions: int = 3):
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=np.float64)
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=np.int32)
        
        for _ in range(n_subdivisions):
            vertices, faces = self._subdivide_mesh(vertices, faces)
        
        return vertices, faces

    def _subdivide_mesh(self, vertices, faces):
        midpoint_cache = {}
        new_vertices = list(vertices)
        
        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in midpoint_cache:
                return midpoint_cache[key]
            mid = (vertices[i] + vertices[j]) / 2
            mid = mid / np.linalg.norm(mid)
            idx = len(new_vertices)
            new_vertices.append(mid)
            midpoint_cache[key] = idx
            return idx
        
        new_faces = []
        for face in faces:
            v0, v1, v2 = face
            m0 = get_midpoint(v0, v1)
            m1 = get_midpoint(v1, v2)
            m2 = get_midpoint(v2, v0)
            new_faces.append([v0, m0, m2])
            new_faces.append([v1, m1, m0])
            new_faces.append([v2, m2, m1])
            new_faces.append([m0, m1, m2])
        
        return np.array(new_vertices), np.array(new_faces, dtype=np.int32)

    def _get_anatomical_depth(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """BEAST MODE: Analytical face depth prior.
        Nose sticks out, eye sockets recede, cheeks bulge.
        Stops the face from looking like a smooth balloon.
        """
        z = np.zeros_like(x)
        
        # Nose bridge and tip (protrudes)
        z += 0.35 * np.exp(-((x / 0.15)**2 + ((y + 0.1) / 0.25)**2))
        z += 0.20 * np.exp(-((x / 0.25)**2 + ((y + 0.35) / 0.15)**2))
        
        # Eye sockets (recede)
        z -= 0.15 * np.exp(-(((x - 0.3) / 0.15)**2 + ((y - 0.15) / 0.1)**2))
        z -= 0.15 * np.exp(-(((x + 0.3) / 0.15)**2 + ((y - 0.15) / 0.1)**2))
        
        # Cheeks (bulge slightly)
        z += 0.10 * np.exp(-(((x - 0.45) / 0.25)**2 + ((y + 0.2) / 0.2)**2))
        z += 0.10 * np.exp(-(((x + 0.45) / 0.25)**2 + ((y + 0.2) / 0.2)**2))
        
        # Chin (protrudes slightly)
        z += 0.15 * np.exp(-((x / 0.2)**2 + ((y + 0.65) / 0.15)**2))
        
        # Forehead (smooth curve)
        z += 0.10 * np.exp(-((x / 0.6)**2 + ((y - 0.5) / 0.3)**2))
        
        return z

    def _fit_to_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        vertices = self._template_vertices.copy()
        n_vertices = vertices.shape[0]

        landmarks_3d = self._landmarks_to_3d(landmarks)
        template_landmarks = vertices[self._landmark_indices]
        displacements = landmarks_3d - template_landmarks

        # BEAST MODE: Vectorized RBF (Killed the Python for-loop)
        k = min(self.config.rbf_neighbors, len(self._landmark_indices))
        distances, indices = self._template_tree.query(vertices, k=k)
        
        eps = 1e-6
        sigma = self.config.smoothness_weight + eps
        weights = np.exp(-distances**2 / (2 * sigma**2))
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + eps)

        # Fully vectorized interpolation: (N, K, 1) * (N, K, 3) -> sum over K
        interpolated_displacements = np.sum(
            weights[..., np.newaxis] * displacements[indices], axis=1
        )

        vertices = vertices + interpolated_displacements
        return np.nan_to_num(vertices, nan=0.0, posinf=1.0, neginf=-1.0)

    def _landmarks_to_3d(self, landmarks: np.ndarray) -> np.ndarray:
        center = np.mean(landmarks, axis=0)
        scale = np.max(np.abs(landmarks - center))
        landmarks_norm = (landmarks - center) / (scale + 1e-8)

        x = landmarks_norm[:, 0]
        y = landmarks_norm[:, 1]
        
        # BEAST MODE: Use anatomical depth prior instead of dumb sphere projection
        z = self._get_anatomical_depth(x, y)

        landmarks_3d = np.stack([x, y, z], axis=1)
        landmarks_3d = landmarks_3d * scale + np.array([center[0], center[1], 0])

        return landmarks_3d

    def _compute_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        n_vertices = vertices.shape[0]
        normals = np.zeros((n_vertices, 3))

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)

        for i in range(3):
            np.add.at(normals, faces[:, i], face_normals)

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-8)

        if np.mean(normals[:, 2]) < 0:
            normals = -normals

        return normals

    def _compute_uv(self, vertices: np.ndarray) -> np.ndarray:
        center = np.mean(vertices, axis=0)
        vertices_centered = vertices - center
        scale = np.max(np.abs(vertices_centered))
        vertices_norm = vertices_centered / (scale + 1e-8)

        x, y, z = vertices_norm[:, 0], vertices_norm[:, 1], vertices_norm[:, 2]
        u = np.arctan2(x, z) / (2 * np.pi) + 0.5
        v = np.arcsin(np.clip(y, -1, 1)) / np.pi + 0.5

        uv = np.clip(np.stack([u, v], axis=1), 0, 1)
        return uv

    def _compute_confidence(self, vertices: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        landmarks_3d = self._landmarks_to_3d(landmarks)
        tree = cKDTree(landmarks_3d)
        distances, _ = tree.query(vertices)
        confidence = np.exp(-self.config.confidence_decay * distances)
        return np.clip(confidence, 0, 1)

    def _create_correspondence(self, landmarks: np.ndarray) -> Dict[int, int]:
        return {i: int(idx) for i, idx in enumerate(self._landmark_indices)}

    def estimate_from_landmarks(
        self,
        landmarks_478: np.ndarray,
        image_shape: tuple,
    ) -> tuple:
        landmarks = np.asarray(landmarks_478, dtype=np.float32)
        if landmarks.ndim != 2 or landmarks.shape[1] < 2:
            raise ValueError(f"Expected (N, 2+) landmarks, got {landmarks.shape}")
            
        # Ensure we only pass x, y
        if landmarks.shape[1] > 2:
            landmarks = landmarks[:, :2]

        geometry = self.estimate(landmarks)
        return geometry.vertices, geometry.faces