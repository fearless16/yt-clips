"""
Mesh-derived semantic masking for Face OS.

This module implements the geometry-first mask required by ``face_os/arch.md``:
landmark triangles are rasterized directly, and soft edges are derived from a
signed distance field. No image intensity, threshold masks, ellipses, or
Gaussian feathering are used.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, Optional, Tuple
import importlib

import cv2
import numpy as np

from face_os.types import SemanticMeshMask


TriangleArray = np.ndarray


_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
_NOSE = [1, 168, 6, 197, 195, 5, 4, 45, 275, 220, 440]

_REGION_CONTOURS: Dict[str, Tuple[int, ...]] = {
    "face_oval": tuple(_FACE_OVAL),
    "left_eye": tuple(_LEFT_EYE),
    "right_eye": tuple(_RIGHT_EYE),
    "mouth": tuple(_LIPS),
    "nose": tuple(_NOSE),
}


def build_semantic_mesh_mask(
    mesh_478: np.ndarray,
    image_shape: Tuple[int, int] | Tuple[int, int, int],
    triangles: Optional[Iterable[Iterable[int]]] = None,
    feather_px: float = 4.0,
) -> SemanticMeshMask:
    """Rasterize a 468/478 point face mesh into a semantic mask.

    Args:
        mesh_478: ``(N, 2+)`` landmark array in the target image coordinate
            space. ``N`` is normally 468 or 478, but the function is robust to
            smaller synthetic meshes for deterministic unit tests.
        image_shape: target ``(H, W)`` or image ``(H, W, C)`` shape.
        triangles: optional triangle index array. Supplied winding is preserved
            so inverted triangles can be counted.
        feather_px: SDF feather half-width in pixels. ``0`` returns a hard
            binary rasterization.

    Returns:
        ``SemanticMeshMask`` with a soft SDF-derived alpha mask, per-region
        masks, the signed distance field, and topology metrics.
    """
    h, w = _shape_hw(image_shape)
    points = _as_points(mesh_478)
    source = f"mesh_{len(points)}"

    if h <= 0 or w <= 0 or len(points) < 3:
        empty = np.zeros((max(h, 0), max(w, 0)), dtype=np.float32)
        return SemanticMeshMask(
            mask=empty,
            regions={"face": empty.copy()},
            sdf=empty.copy(),
            triangle_count=0,
            inverted_triangles=0,
            coverage=0.0,
            source=f"{source}:empty_geometry",
        )

    raw_triangles: TriangleArray
    normalize_winding = False
    topology_source = "supplied_triangles"
    if triangles is None:
        raw_triangles, topology_source, normalize_winding = default_triangles(points)
    else:
        raw_triangles = np.asarray(triangles, dtype=np.int32)

    valid_triangles, inverted = _sanitize_triangles(
        raw_triangles,
        points,
        normalize_winding=normalize_winding,
    )
    hard_face = _rasterize_triangles(points, valid_triangles, (h, w))
    sdf = signed_distance_field(hard_face)
    mask = mask_from_sdf(sdf, feather_px=feather_px)
    coverage = float(np.mean(mask)) if mask.size else 0.0

    regions = _semantic_regions(points, valid_triangles, hard_face, (h, w), feather_px)

    return SemanticMeshMask(
        mask=mask,
        regions=regions,
        sdf=sdf,
        triangle_count=int(len(valid_triangles)),
        inverted_triangles=int(inverted),
        coverage=coverage,
        source=f"{source}:{topology_source}",
    )


def rasterize_mesh_mask(
    mesh_478: np.ndarray,
    image_shape: Tuple[int, int] | Tuple[int, int, int],
    triangles: Optional[Iterable[Iterable[int]]] = None,
    feather_px: float = 4.0,
) -> SemanticMeshMask:
    """Alias kept intentionally small for call sites that prefer verb phrasing."""
    return build_semantic_mesh_mask(mesh_478, image_shape, triangles, feather_px)


class SemanticMeshMaskBuilder:
    """Small deterministic builder wrapper for pipeline integration."""

    def __init__(
        self,
        triangles: Optional[Iterable[Iterable[int]]] = None,
        feather_px: float = 4.0,
    ) -> None:
        self.triangles = None if triangles is None else np.asarray(triangles, dtype=np.int32)
        self.feather_px = float(feather_px)

    def build(
        self,
        mesh_478: np.ndarray,
        image_shape: Tuple[int, int] | Tuple[int, int, int],
    ) -> SemanticMeshMask:
        return build_semantic_mesh_mask(
            mesh_478,
            image_shape,
            triangles=self.triangles,
            feather_px=self.feather_px,
        )

    def __call__(
        self,
        mesh_478: np.ndarray,
        image_shape: Tuple[int, int] | Tuple[int, int, int],
    ) -> SemanticMeshMask:
        return self.build(mesh_478, image_shape)


def default_triangles(points: Optional[np.ndarray] = None) -> Tuple[TriangleArray, str, bool]:
    """Return deterministic default face triangles and their provenance.

    MediaPipe publishes tessellation as edges. When that local API is available
    we recover triangles as 3-cliques in the edge graph. If not, a convex-hull
    fan is used and reported explicitly as ``fallback_convex_hull_fan``.
    """
    triangles, source = _mediapipe_tessellation_triangles()
    if triangles.size:
        return triangles, source, True
    if points is None:
        return np.empty((0, 3), dtype=np.int32), "fallback_convex_hull_fan_no_points", True
    return _fallback_convex_hull_fan(_as_points(points)), "fallback_convex_hull_fan", True


def signed_distance_field(hard_mask: np.ndarray) -> np.ndarray:
    """Compute a finite signed distance field from a hard mask.

    Positive values are inside the mask, negative values are outside.
    """
    inside = (np.asarray(hard_mask) > 0).astype(np.uint8)
    h, w = inside.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((h, w), dtype=np.float32)
    if not np.any(inside):
        return np.full((h, w), -float(max(h, w)), dtype=np.float32)
    outside = (inside == 0).astype(np.uint8)
    if not np.any(outside):
        return np.full((h, w), float(max(h, w)), dtype=np.float32)

    dist_in = cv2.distanceTransform(inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_out = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sdf = dist_in.astype(np.float32) - dist_out.astype(np.float32)
    return np.nan_to_num(sdf, nan=0.0, posinf=float(max(h, w)), neginf=-float(max(h, w)))


def mask_from_sdf(sdf: np.ndarray, feather_px: float = 4.0) -> np.ndarray:
    """Convert an SDF to a soft alpha mask without blur."""
    sdf_f = np.nan_to_num(np.asarray(sdf, dtype=np.float32), nan=0.0)
    if feather_px <= 0:
        return (sdf_f > 0).astype(np.float32)
    width = max(float(feather_px), 1e-6)
    mask = 0.5 + sdf_f / (2.0 * width)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


@lru_cache(maxsize=1)
def _mediapipe_tessellation_triangles() -> Tuple[TriangleArray, str]:
    candidates = (
        (
            "mediapipe.tasks.python.vision.face_landmarker",
            "FaceLandmarksConnections",
            "FACE_LANDMARKS_TESSELATION",
            "mediapipe_tasks_face_landmarks_tessellation_cliques",
        ),
        (
            "mediapipe.solutions.face_mesh_connections",
            None,
            "FACEMESH_TESSELATION",
            "mediapipe_solutions_facemesh_tessellation_cliques",
        ),
        (
            "mediapipe.python.solutions.face_mesh_connections",
            None,
            "FACEMESH_TESSELATION",
            "mediapipe_python_facemesh_tessellation_cliques",
        ),
    )
    for module_name, owner_name, attr_name, source in candidates:
        try:
            module = importlib.import_module(module_name)
            owner = getattr(module, owner_name) if owner_name else module
            edges = getattr(owner, attr_name)
            triangles = _triangles_from_edges(edges)
        except Exception:
            continue
        if triangles.size:
            return triangles, source
    return np.empty((0, 3), dtype=np.int32), "fallback_convex_hull_fan"


def _triangles_from_edges(edges: Iterable[object]) -> TriangleArray:
    edge_set = set()
    for edge in edges:
        if hasattr(edge, "start") and hasattr(edge, "end"):
            a, b = int(edge.start), int(edge.end)
        else:
            a, b = edge
            a, b = int(a), int(b)
        if a == b:
            continue
        edge_set.add((min(a, b), max(a, b)))

    adjacency: Dict[int, set[int]] = {}
    for a, b in edge_set:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    triangles = []
    for a in sorted(adjacency):
        for b in sorted(v for v in adjacency[a] if v > a):
            for c in sorted(v for v in adjacency[a].intersection(adjacency[b]) if v > b):
                triangles.append((a, b, c))
    return np.asarray(triangles, dtype=np.int32)


def _fallback_convex_hull_fan(points: np.ndarray) -> TriangleArray:
    finite = np.all(np.isfinite(points), axis=1)
    finite_indices = np.flatnonzero(finite)
    if len(finite_indices) < 3:
        return np.empty((0, 3), dtype=np.int32)

    finite_points = points[finite_indices].astype(np.float32)
    hull_rel = cv2.convexHull(finite_points, returnPoints=False).reshape(-1)
    hull = finite_indices[hull_rel]
    if len(hull) < 3:
        return np.empty((0, 3), dtype=np.int32)

    centroid = np.mean(points[hull], axis=0)
    center_rel = int(np.argmin(np.linalg.norm(finite_points - centroid, axis=1)))
    center = int(finite_indices[center_rel])

    triangles = []
    if center not in set(int(i) for i in hull):
        for i in range(len(hull)):
            triangles.append((center, int(hull[i]), int(hull[(i + 1) % len(hull)])))
    else:
        anchor = int(hull[0])
        for i in range(1, len(hull) - 1):
            triangles.append((anchor, int(hull[i]), int(hull[i + 1])))
    return np.asarray(triangles, dtype=np.int32)


def _sanitize_triangles(
    triangles: TriangleArray,
    points: np.ndarray,
    normalize_winding: bool,
) -> Tuple[TriangleArray, int]:
    tri = np.asarray(triangles, dtype=np.int32)
    if tri.size == 0:
        return np.empty((0, 3), dtype=np.int32), 0
    tri = tri.reshape(-1, 3)

    n = len(points)
    in_bounds = np.all((tri >= 0) & (tri < n), axis=1)
    tri = tri[in_bounds]
    if tri.size == 0:
        return np.empty((0, 3), dtype=np.int32), 0

    finite = np.all(np.isfinite(points[tri]), axis=(1, 2))
    tri = tri[finite]
    if tri.size == 0:
        return np.empty((0, 3), dtype=np.int32), 0

    areas = _signed_areas(points, tri)
    nondegenerate = np.abs(areas) > 1e-5
    tri = tri[nondegenerate]
    areas = areas[nondegenerate]
    inverted = int(np.count_nonzero(areas < 0.0))

    if normalize_winding and len(tri):
        flip = areas < 0.0
        tri = tri.copy()
        tri[flip, 1], tri[flip, 2] = tri[flip, 2].copy(), tri[flip, 1].copy()
        inverted = 0

    return tri.astype(np.int32), inverted


def _signed_areas(points: np.ndarray, triangles: TriangleArray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])) * 0.5


def _rasterize_triangles(points: np.ndarray, triangles: TriangleArray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    hard = np.zeros((h, w), dtype=np.uint8)
    if len(triangles) == 0:
        return hard
    polygons = [np.rint(points[t]).astype(np.int32).reshape(-1, 1, 2) for t in triangles]
    cv2.fillPoly(hard, polygons, 255, lineType=cv2.LINE_8)
    return hard


def _semantic_regions(
    points: np.ndarray,
    triangles: TriangleArray,
    hard_face: np.ndarray,
    shape: Tuple[int, int],
    feather_px: float,
) -> Dict[str, np.ndarray]:
    regions: Dict[str, np.ndarray] = {
        "face": mask_from_sdf(signed_distance_field(hard_face), feather_px=feather_px),
    }
    for name, contour_indices in _REGION_CONTOURS.items():
        valid_indices = [idx for idx in contour_indices if idx < len(points) and np.all(np.isfinite(points[idx]))]
        if len(valid_indices) < 3:
            continue
        selected = _triangles_inside_contour(points, triangles, np.asarray(valid_indices, dtype=np.int32))
        if len(selected) == 0:
            continue
        hard = _rasterize_triangles(points, selected, shape)
        if np.any(hard):
            regions[name] = mask_from_sdf(signed_distance_field(hard), feather_px=feather_px)

    skin = regions["face"].copy()
    for name in ("left_eye", "right_eye", "mouth"):
        if name in regions:
            skin = np.clip(skin - regions[name], 0.0, 1.0)
    regions["skin"] = skin.astype(np.float32)
    return regions


def _triangles_inside_contour(
    points: np.ndarray,
    triangles: TriangleArray,
    contour_indices: np.ndarray,
) -> TriangleArray:
    if len(triangles) == 0:
        return np.empty((0, 3), dtype=np.int32)

    contour = points[contour_indices].astype(np.float32)
    if len(contour) < 3:
        return np.empty((0, 3), dtype=np.int32)

    selected = []
    centroids = np.mean(points[triangles], axis=1)
    for tri, centroid in zip(triangles, centroids):
        inside = cv2.pointPolygonTest(contour, (float(centroid[0]), float(centroid[1])), False) >= 0
        if inside:
            selected.append(tuple(int(v) for v in tri))
    return np.asarray(selected, dtype=np.int32).reshape(-1, 3) if selected else np.empty((0, 3), dtype=np.int32)


def _shape_hw(image_shape: Tuple[int, int] | Tuple[int, int, int]) -> Tuple[int, int]:
    if len(image_shape) < 2:
        raise ValueError(f"image_shape must include height and width, got {image_shape}")
    return int(image_shape[0]), int(image_shape[1])


def _as_points(mesh_478: np.ndarray) -> np.ndarray:
    arr = np.asarray(mesh_478, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"mesh_478 must have shape (N, 2+), got {arr.shape}")
    return arr[:, :2].astype(np.float32, copy=False)

