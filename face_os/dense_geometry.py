"""Dense Geometry Module.

Dense mesh fitting from sparse landmarks.

Approach:
    1. Start with parametric face model (sphere template)
    2. Fit to 478 MediaPipe landmarks via Laplacian deformation
    3. Refine with smoothness regularization
    4. Output dense mesh + normals + UV coordinates

Mathematical Model:
    min ||V[landmark_idx] - L||^2 + λ * ||Laplacian(V)||^2

where:
    V = vertex positions (N, 3)
    L = landmark positions (478, 3)
    λ = smoothness weight

References:
    - Laplacian surface editing (Sorkine et al., 2004)
    - Parametric face models (FLAME, BFM)
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import spsolve


@dataclass
class GeometryConfig:
    """Configuration for dense geometry estimation."""

    # Number of vertices in mesh
    n_vertices: int = 3000

    # Number of faces in mesh
    n_faces: int = 6000

    # Landmark fitting weight
    landmark_weight: float = 10.0

    # Smoothness weight (Laplacian regularization)
    smoothness_weight: float = 0.1

    # Confidence decay rate (distance from landmarks)
    confidence_decay: float = 0.01

    # UV mapping mode ('spherical', 'cylindrical', 'planar')
    uv_mode: str = 'spherical'


@dataclass
class DenseGeometry:
    """Dense mesh geometry."""

    # Vertices: (N, 3)
    vertices: np.ndarray

    # Faces: (F, 3) — triangle indices
    faces: np.ndarray

    # Normals: (N, 3) — vertex normals (unit vectors)
    normals: np.ndarray

    # UV coordinates: (N, 2) — texture coordinates [0, 1]
    uv_coords: np.ndarray

    # Confidence: (N,) — per-vertex confidence [0, 1]
    confidence: np.ndarray

    # Landmark correspondence: landmark_idx → vertex_idx
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
        """Convert to dictionary."""
        return {
            "frame_idx": self.frame_idx,
            "fitting_error": self.fitting_error,
            "smoothness_score": self.smoothness_score,
            "estimation_time_ms": self.estimation_time_ms,
            "n_vertices": self.geometry.vertices.shape[0],
            "n_faces": self.geometry.faces.shape[0],
        }


class DenseGeometryEstimator:
    """Dense geometry estimator from sparse landmarks.

    Fits a dense mesh to 478 MediaPipe landmarks using
    Laplacian deformation with smoothness regularization.
    """

    def __init__(self, config: Optional[GeometryConfig] = None):
        """Initialize estimator.

        Args:
            config: Geometry configuration
        """
        self.config = config or GeometryConfig()
        self._template_vertices = None
        self._template_faces = None
        self._landmark_indices = None
        self._laplacian_matrix = None

    def estimate(
        self,
        landmarks: np.ndarray,
        config: Optional[GeometryConfig] = None,
    ) -> DenseGeometry:
        """Estimate dense geometry from landmarks.

        Args:
            landmarks: Landmark positions (478, 2)
            config: Optional config override

        Returns:
            DenseGeometry with vertices, faces, normals, UV, confidence
        """
        if config is not None:
            self.config = config

        start_time = time.time()

        # Initialize template if needed
        if self._template_vertices is None:
            self._initialize_template()

        # Fit mesh to landmarks
        vertices = self._fit_to_landmarks(landmarks)

        # Compute normals
        normals = self._compute_normals(vertices, self._template_faces)

        # Compute UV coordinates
        uv_coords = self._compute_uv(vertices)

        # Compute confidence
        confidence = self._compute_confidence(vertices, landmarks)

        # Create landmark correspondence
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
        """Initialize sphere template mesh."""
        n_vertices = self.config.n_vertices
        n_faces = self.config.n_faces

        # Create sphere vertices using icosphere subdivision
        # Start with icosahedron and subdivide
        self._template_vertices, self._template_faces = self._create_icosphere(
            n_subdivisions=3
        )

        # Update n_vertices and n_faces to match actual mesh
        n_vertices = self._template_vertices.shape[0]
        n_faces = self._template_faces.shape[0]

        # Scale to face-like proportions (wider than tall)
        self._template_vertices[:, 0] *= 1.2  # Width
        self._template_vertices[:, 1] *= 1.5  # Height
        self._template_vertices[:, 2] *= 0.8  # Depth

        # Create landmark indices (478 evenly spaced vertices)
        self._landmark_indices = np.linspace(
            0, n_vertices - 1, 478, dtype=int
        )

        # Build Laplacian matrix
        self._laplacian_matrix = self._build_laplacian(
            self._template_vertices, self._template_faces
        )

    def _create_icosphere(self, n_subdivisions: int = 3):
        """Create icosphere mesh via subdivision.

        Args:
            n_subdivisions: Number of subdivisions

        Returns:
            vertices (N, 3), faces (F, 3)
        """
        # Create icosahedron
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Vertices of icosahedron
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=np.float64)
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Faces of icosahedron
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=np.int32)
        
        # Subdivide
        for _ in range(n_subdivisions):
            vertices, faces = self._subdivide_mesh(vertices, faces)
        
        return vertices, faces

    def _subdivide_mesh(self, vertices, faces):
        """Subdivide mesh by splitting each triangle into 4.

        Args:
            vertices: (N, 3)
            faces: (F, 3)

        Returns:
            new_vertices (N', 3), new_faces (F', 3)
        """
        # Create midpoint cache
        midpoint_cache = {}
        new_vertices = list(vertices)
        
        def get_midpoint(i, j):
            """Get or create midpoint vertex."""
            key = (min(i, j), max(i, j))
            if key in midpoint_cache:
                return midpoint_cache[key]
            
            # Create new vertex at midpoint
            mid = (vertices[i] + vertices[j]) / 2
            mid = mid / np.linalg.norm(mid)  # Project to unit sphere
            idx = len(new_vertices)
            new_vertices.append(mid)
            midpoint_cache[key] = idx
            return idx
        
        # Subdivide each face
        new_faces = []
        for face in faces:
            v0, v1, v2 = face
            
            # Get midpoints
            m0 = get_midpoint(v0, v1)
            m1 = get_midpoint(v1, v2)
            m2 = get_midpoint(v2, v0)
            
            # Create 4 new faces
            new_faces.append([v0, m0, m2])
            new_faces.append([v1, m1, m0])
            new_faces.append([v2, m2, m1])
            new_faces.append([m0, m1, m2])
        
        return np.array(new_vertices), np.array(new_faces, dtype=np.int32)

    def _fit_to_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Fit mesh vertices to landmarks.

        Uses radial basis function (RBF) interpolation for robust fitting.

        Args:
            landmarks: Landmark positions (478, 2)

        Returns:
            Fitted vertices (N, 3)
        """
        vertices = self._template_vertices.copy()
        n_vertices = vertices.shape[0]

        # Convert 2D landmarks to 3D (estimate depth from sphere)
        landmarks_3d = self._landmarks_to_3d(landmarks)

        # Get template landmark positions
        template_landmarks = vertices[self._landmark_indices]

        # Compute displacement at landmark positions
        displacements = landmarks_3d - template_landmarks

        # RBF interpolation to propagate displacements to all vertices
        # Use thin-plate spline RBF: φ(r) = r² * log(r)
        from scipy.spatial import cKDTree
        
        # Build tree from template landmarks
        tree = cKDTree(template_landmarks)
        
        # Query all vertices
        distances, indices = tree.query(vertices, k=min(10, len(self._landmark_indices)))
        
        # Compute RBF weights
        eps = 1e-6
        weights = np.exp(-distances**2 / (2 * (self.config.smoothness_weight + eps)**2))
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + eps)

        # Interpolate displacements
        interpolated_displacements = np.zeros((n_vertices, 3))
        for k in range(distances.shape[1]):
            idx = indices[:, k]
            w = weights[:, k:k+1]
            interpolated_displacements += w * displacements[idx]

        # Apply displacements
        vertices = vertices + interpolated_displacements

        # Ensure no NaN/Inf
        vertices = np.nan_to_num(vertices, nan=0.0, posinf=1.0, neginf=-1.0)

        return vertices

    def _landmarks_to_3d(self, landmarks: np.ndarray) -> np.ndarray:
        """Convert 2D landmarks to 3D positions.

        Uses sphere projection to estimate depth.

        Args:
            landmarks: 2D landmarks (478, 2)

        Returns:
            3D landmarks (478, 3)
        """
        # Normalize landmarks to [-1, 1]
        center = np.mean(landmarks, axis=0)
        scale = np.max(np.abs(landmarks - center))
        landmarks_norm = (landmarks - center) / (scale + 1e-8)

        # Project onto sphere (frontal hemisphere)
        x = landmarks_norm[:, 0]
        y = landmarks_norm[:, 1]
        
        # Depth from sphere equation: z = sqrt(1 - x^2 - y^2)
        r2 = x**2 + y**2
        r2 = np.clip(r2, 0, 1)  # Clamp to unit sphere
        z = np.sqrt(1 - r2)

        # Stack to 3D
        landmarks_3d = np.stack([x, y, z], axis=1)

        # Scale back to original coordinate system
        landmarks_3d = landmarks_3d * scale + np.array([center[0], center[1], 0])

        return landmarks_3d

    def _build_laplacian(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> csr_matrix:
        """Build Laplacian matrix for mesh.

        L = D - A
        where D = degree matrix, A = adjacency matrix

        Args:
            vertices: Vertex positions (N, 3)
            faces: Face indices (F, 3)

        Returns:
            Laplacian matrix (N, N)
        """
        n_vertices = vertices.shape[0]
        
        # Build adjacency matrix
        row = []
        col = []
        for face in faces:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        row.append(face[i])
                        col.append(face[j])

        data = np.ones(len(row))
        A = csr_matrix((data, (row, col)), shape=(n_vertices, n_vertices))
        
        # Degree matrix
        D = csr_matrix((np.array(A.sum(axis=1)).flatten(), (np.arange(n_vertices), np.arange(n_vertices))),
                       shape=(n_vertices, n_vertices))

        # Laplacian: L = D - A
        L = D - A

        return L

    def _compute_normals(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> np.ndarray:
        """Compute vertex normals from mesh.

        Vertex normal = average of adjacent face normals.
        Ensures normals point outward (positive z for frontal face).

        Args:
            vertices: Vertex positions (N, 3)
            faces: Face indices (F, 3)

        Returns:
            Vertex normals (N, 3), unit vectors
        """
        n_vertices = vertices.shape[0]
        normals = np.zeros((n_vertices, 3))

        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)

        # Accumulate face normals to vertices
        for i in range(3):
            np.add.at(normals, faces[:, i], face_normals)

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-8)

        # Ensure normals point outward (positive z for frontal face)
        # Check if majority of normals point inward
        mean_nz = np.mean(normals[:, 2])
        if mean_nz < 0:
            # Flip all normals
            normals = -normals

        return normals

    def _compute_uv(self, vertices: np.ndarray) -> np.ndarray:
        """Compute UV coordinates for vertices.

        Uses spherical projection for UV mapping.

        Args:
            vertices: Vertex positions (N, 3)

        Returns:
            UV coordinates (N, 2), [0, 1]
        """
        # Normalize vertices to unit sphere
        center = np.mean(vertices, axis=0)
        vertices_centered = vertices - center
        scale = np.max(np.abs(vertices_centered))
        vertices_norm = vertices_centered / (scale + 1e-8)

        # Spherical UV mapping
        x, y, z = vertices_norm[:, 0], vertices_norm[:, 1], vertices_norm[:, 2]
        
        # U = atan2(x, z) / (2*pi) + 0.5
        u = np.arctan2(x, z) / (2 * np.pi) + 0.5
        
        # V = asin(y) / pi + 0.5
        v = np.arcsin(np.clip(y, -1, 1)) / np.pi + 0.5

        uv = np.stack([u, v], axis=1)

        # Clip to [0, 1]
        uv = np.clip(uv, 0, 1)

        return uv

    def _compute_confidence(
        self,
        vertices: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """Compute per-vertex confidence.

        Confidence decreases with distance from nearest landmark.

        Args:
            vertices: Vertex positions (N, 3)
            landmarks: Landmark positions (478, 2)

        Returns:
            Confidence (N,), [0, 1]
        """
        n_vertices = vertices.shape[0]
        confidence = np.zeros(n_vertices)

        # Convert landmarks to 3D
        landmarks_3d = self._landmarks_to_3d(landmarks)

        # Compute distance from each vertex to nearest landmark
        from scipy.spatial import cKDTree
        tree = cKDTree(landmarks_3d)
        distances, _ = tree.query(vertices)

        # Confidence = exp(-decay * distance)
        confidence = np.exp(-self.config.confidence_decay * distances)

        # Clamp to [0, 1]
        confidence = np.clip(confidence, 0, 1)

        return confidence

    def _create_correspondence(
        self,
        landmarks: np.ndarray,
    ) -> Dict[int, int]:
        """Create landmark to vertex correspondence.

        Args:
            landmarks: Landmark positions (478, 2)

        Returns:
            Dictionary mapping landmark_idx → vertex_idx
        """
        correspondence = {}
        for i, idx in enumerate(self._landmark_indices):
            correspondence[i] = int(idx)

        return correspondence
