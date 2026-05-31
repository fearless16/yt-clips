"""§16.6 Visibility / Occlusion Field.

arch.md §16.6:
    V(u,v,t) ∈ [0,1]

A geometry-derived self-occlusion factor in canonical UV space, NOT a 2D
sharpness proxy. The camera views the canonical face down the +Z axis
(physical_renderer.py: view_direction = [0, 0, 1]), so a surface point is
visible to the camera in proportion to how much its normal faces the camera:

    V(u,v) = clip( N(u,v) · view , 0 , 1 )

Front-facing normals (N·view → 1) are fully visible; back-facing or grazing
normals (N·view ≤ 0) are occluded (V = 0). This is the ``Visibility`` factor of
the §16.8 composite ``C_recon = C_obs · Coverage_pose · Coverage_light ·
Visibility`` and the gate for the §16.6 memory-update rule
``C_new(u,v) = C_old(u,v) + q_t · V(u,v,t)``.

The normal field is the per-pixel canonical normal map already produced at
runtime (``IntrinsicComponents.normal_map``); no new geometry is needed.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# Camera view direction in canonical space (matches physical_renderer.py:362).
_DEFAULT_VIEW: Tuple[float, float, float] = (0.0, 0.0, 1.0)


def compute_visibility(
    normal_map: np.ndarray,
    view_direction: Tuple[float, float, float] = _DEFAULT_VIEW,
) -> np.ndarray:
    """Per-UV geometric visibility ``V = clip(N · view, 0, 1)`` (§16.6).

    Args:
        normal_map: ``(H, W, 3)`` float surface normals in canonical UV space
            (e.g. ``IntrinsicComponents.normal_map``). Need not be perfectly
            unit-length; degenerate (zero) normals yield ``V = 0``.
        view_direction: camera view direction; normalized internally. Defaults
            to ``+Z`` (the canonical-render camera axis).

    Returns:
        ``(H, W)`` float32 visibility in ``[0, 1]``. Back-facing / grazing /
        zero normals → 0; a normal pointing straight at the camera → 1; oblique
        normals return the cosine ``N · view`` (a soft factor, not a hard mask).
    """
    n = np.asarray(normal_map, dtype=np.float32)
    if n.ndim != 3 or n.shape[2] != 3:
        raise ValueError(
            f"normal_map must be (H, W, 3); got shape {getattr(n, 'shape', None)}"
        )

    view = np.asarray(view_direction, dtype=np.float32)
    view_norm = float(np.linalg.norm(view))
    if view_norm < 1e-8:
        raise ValueError("view_direction must be non-zero")
    view = view / view_norm

    # N · view per texel, clamped to [0, 1] (the front-facing cosine).
    dot = n @ view
    return np.clip(dot, 0.0, 1.0).astype(np.float32)
