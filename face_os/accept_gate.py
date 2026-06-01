"""Central architecture accept/reject gate.

The gate is intentionally small and deterministic: it decides whether a frame is
allowed to update/render latent identity from explicit geometry, identity,
temporal, and lighting signals. A rejection is not an exception; it is a named
contract-preserving fallback reason.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from face_os.types import AcceptDecision, GeometryState, TemporalState


def evaluate_acceptance(
    geometry: Optional[GeometryState],
    identity_confidence: float = 0.0,
    temporal: Optional[TemporalState] = None,
    lighting=None,
    min_geometry_confidence: float = 0.05,
    min_identity_confidence: float = 0.01,
    min_temporal_confidence: float = 0.05,
) -> AcceptDecision:
    """Return the single frame accept/reject decision.

    A frame is accepted only when the required subsystems provide usable,
    finite, bounded signals. The thresholds are deliberately low because this
    gate is a hard architectural guard, not a quality preference switch.
    """
    geometry_ok = _geometry_ok(geometry, min_geometry_confidence)
    identity_ok = bool(np.isfinite(identity_confidence) and identity_confidence >= min_identity_confidence)
    temporal_conf = 1.0 if temporal is None else float(getattr(temporal, "temporal_confidence", 0.0) or 0.0)
    temporal_ok = bool(np.isfinite(temporal_conf) and temporal_conf >= min_temporal_confidence)
    lighting_ok = _lighting_ok(lighting)

    score = float(np.clip(
        min(
            float(getattr(geometry, "geometry_confidence", 0.0) or 0.0) if geometry is not None else 0.0,
            float(identity_confidence),
            temporal_conf,
            1.0 if lighting_ok else 0.0,
        ),
        0.0,
        1.0,
    ))

    reason = None
    if not geometry_ok:
        reason = "accept_reject_geometry"
    elif not identity_ok:
        reason = "accept_reject_identity"
    elif not temporal_ok:
        reason = "accept_reject_temporal"
    elif not lighting_ok:
        reason = "accept_reject_lighting"

    return AcceptDecision(
        accept=reason is None,
        reason=reason,
        geometry_ok=geometry_ok,
        identity_ok=identity_ok,
        temporal_ok=temporal_ok,
        lighting_ok=lighting_ok,
        score=score,
    )


def _geometry_ok(geometry: Optional[GeometryState], min_confidence: float) -> bool:
    if geometry is None:
        return False
    conf = float(getattr(geometry, "geometry_confidence", 0.0) or 0.0)
    if not np.isfinite(conf) or conf < min_confidence:
        return False
    graph = getattr(geometry, "transform_graph", None)
    if graph is not None and getattr(graph, "edges", None):
        if not graph.valid:
            return False
    mask = getattr(geometry, "mask", None)
    if mask is not None:
        arr = np.asarray(mask)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return False
    return True


def _lighting_ok(lighting) -> bool:
    if lighting is None:
        return True
    for name in ("ambient", "diffuse_intensity", "specular_intensity"):
        value = getattr(lighting, name, 0.0)
        if not np.isfinite(float(value)):
            return False
    direction = getattr(lighting, "diffuse_direction", None)
    if direction is not None:
        arr = np.asarray(direction, dtype=np.float32)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return False
    return True
