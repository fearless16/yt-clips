"""Subsystem D — Physically Consistent Rendering.

Generates output frames using physically-based rendering.

Equation: Y = M ⊙ Y_face + (1 - M) ⊙ Y_bg
where Y_face = R(G, A, L, V) with Lambertian + Blinn-Phong

Delegates to: physical_renderer.py, compositor.py, face_enhance.py

BOUNDARY CONTRACT:
- MUST NOT perform RGB-space rescue compositing
- MUST NOT use heuristic blending (except as documented fallback)
- MUST NOT estimate geometry or identity
"""

from typing import Optional

import numpy as np


class FaceRenderer:
    """Subsystem D: Physically consistent rendering.

    Thin wrapper that delegates to physical_renderer.py.
    Enforces linear-light compositing and consistent sharpening.

    FORBIDDEN: RGB-space rescue compositing, heuristic blending,
               geometry estimation, identity estimation
    """

    def __init__(self, physical_renderer, config=None):
        """Args:
        physical_renderer: PhysicallyInspiredRenderer instance
        config: pipeline config
        """
        self._renderer = physical_renderer
        self._config = config

    def render(self, albedo, normal_map, shading, lighting) -> Optional[np.ndarray]:
        """Render using physical renderer with intrinsic normal map.

        Args:
            albedo: (H, W, 3) float32 [0,1]
            normal_map: (H, W, 3) float32
            shading: (H, W) or (H, W, 1) float32
            lighting: LightingModel

        Returns:
            Rendered frame (H, W, 3) float32 [0,1] or None
        """
        if self._renderer is None:
            return None

        try:
            result = self._renderer.render(
                albedo=albedo,
                normal_map=normal_map,
                shading=shading,
                lighting=lighting,
            )
            return result.rendered
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Render failed: %s", e)
            return None

    def render_with_mesh(self, albedo, mesh_vertices, mesh_faces, shading, lighting,
                         image_shape: tuple) -> Optional[np.ndarray]:
        """Render using mesh-derived normals (true geometry).

        Args:
            albedo: (H, W, 3) float32 [0,1]
            mesh_vertices: (N, 3) mesh vertex positions
            mesh_faces: (F, 3) face indices
            shading: (H, W) or (H, W, 1) float32
            lighting: LightingModel
            image_shape: (H, W) output size

        Returns:
            Rendered frame (H, W, 3) float32 [0,1] or None
        """
        if self._renderer is None:
            return None

        try:
            result = self._renderer.render_with_mesh(
                albedo=albedo,
                mesh_vertices=mesh_vertices,
                mesh_faces=mesh_faces,
                shading=shading,
                lighting=lighting,
                image_size=image_shape,
            )
            return result.rendered if hasattr(result, 'rendered') else result
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Mesh render failed: %s", e)
            return None
