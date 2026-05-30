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

    def render_from_latent(self, components, geometry, lighting, view_direction=None):
        """Synthesize the stored identity under current geometry + lighting.

        PRIMARY latent render path (D-05 Phase 2). Takes the ``IntrinsicComponents``
        produced by ``IdentityEstimator.synthesize_identity`` (latent albedo +
        microdetail + geometry normals + a NEUTRAL unit shading field) and shades
        them under the ``lighting`` ESTIMATED from the current frame. It NEVER
        reads the source crop — there is deliberately no ``observed``/``source``
        parameter, so the legacy paste-then-relight (A-3) and source-HF
        reinjection (A-5) paths cannot be reached from here. Illumination is
        applied at render time, never baked into the latent (A-1).

        The B->D boundary is a HARD contract (A-10): a malformed ``components``
        (e.g. multi-channel shading) RAISES ``ContractViolation`` rather than
        being silently sanitized. This assertion runs BEFORE the renderer's own
        try/except so it can never be swallowed.

        Args:
            components: ``IntrinsicComponents`` from ``synthesize_identity``
                (latent-only provenance).
            geometry: ``GeometryState`` — provenance of the normals/mesh/mask
                already warped into ``components`` by ``synthesize_identity``.
            lighting: ``LightingModel`` estimated from the OBSERVATION (not the
                latent).
            view_direction: optional (3,) view vector; the renderer defaults to
                ``[0, 0, 1]`` when ``None``.

        Returns:
            ``(H, W, 3)`` float32 linear-light render in ``[0, 1]`` (face
            interior). Lighting is reproducible (deterministic) for fixed inputs.
        """
        from face_os.intrinsic_decomposition import assert_intrinsic_contract

        # HARD assertion at the B->D boundary (fatal): contract violations must
        # surface, never be clamped. Validate the components' internal geometry
        # against their own albedo size.
        expect_hw = (int(components.albedo.shape[0]), int(components.albedo.shape[1]))
        assert_intrinsic_contract(components, expect_hw=expect_hw, mode="fatal")

        # observed=None => the renderer's detail path uses identity microdetail
        # only and never mixes in source high-frequency (no A-5 leak).
        out = self._renderer.render_with_intrinsic(
            components,
            lighting=lighting,
            view_direction=view_direction,
            observed=None,
        )
        return out.rendered
