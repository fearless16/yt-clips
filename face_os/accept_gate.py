"""Central architecture accept/reject gate.

The gate is intentionally small and deterministic: it decides whether a frame is
allowed to update/render latent identity from explicit geometry, identity,
temporal, and lighting signals. A rejection is not an exception; it is a named
contract-preserving fallback reason.

Phase 2B (§19): the gate's identity decision now consumes the §16.8 composite
C_recon = C_obs · Coverage_pose · Coverage_light · Visibility. Callers opt in
via ``use_c_recon_gate=True``; the legacy ``identity_confidence`` path remains
the default for backward compatibility. The composite is naturally small
(<< 0.01 in cold start), so the floor is correspondingly low.
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
    c_recon: Optional[float] = None,
    use_c_recon_gate: bool = False,
    min_c_recon: float = 0.0005,
) -> AcceptDecision:
    """Return the single frame accept/reject decision.

    A frame is accepted only when the required subsystems provide usable,
    finite, bounded signals. The thresholds are deliberately low because this
    gate is a hard architectural guard, not a quality preference switch.

    Phase 2B identity selection:
      - If ``use_c_recon_gate=True`` AND ``c_recon`` is a finite number,
        identity_ok is governed by ``c_recon >= min_c_recon``. The §16.8
        composite already incorporates pose/lighting/visibility coverage, so
        this is the architectural trust signal.
      - Otherwise, the legacy ``identity_confidence >= min_identity_confidence``
        rule applies.
      - The chosen source is recorded in ``AcceptDecision.trust_source``
        (``"c_recon"`` | ``"c_obs"``) and the observed c_recon is attached.
    """
    geometry_ok = _geometry_ok(geometry, min_geometry_confidence)
    temporal_conf = 1.0 if temporal is None else float(getattr(temporal, "temporal_confidence", 0.0) or 0.0)
    temporal_ok = bool(np.isfinite(temporal_conf) and temporal_conf >= min_temporal_confidence)
    lighting_ok = _lighting_ok(lighting)

    # ── Phase 2B: identity_ok source selection ──────────────────────────
    c_recon_finite = (
        c_recon is not None
        and np.isfinite(float(c_recon))
        and float(c_recon) >= 0.0
    )
    if use_c_recon_gate and c_recon_finite:
        cr = float(c_recon)
        identity_ok = bool(cr >= min_c_recon)
        trust_source = "c_recon"
    else:
        identity_ok = bool(
            np.isfinite(identity_confidence) and identity_confidence >= min_identity_confidence
        )
        trust_source = "c_obs"
        cr = None

    # Score = min of the 4 scalar signals, clipped to [0, 1]
    geom_score = float(getattr(geometry, "geometry_confidence", 0.0) or 0.0) if geometry is not None else 0.0
    if use_c_recon_gate and cr is not None:
        identity_score = cr
    else:
        identity_score = float(identity_confidence)
    score = float(np.clip(
        min(
            geom_score,
            identity_score,
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
        trust_source=trust_source,
        c_recon=cr,
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
