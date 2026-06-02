import numpy as np

from face_os.accept_gate import evaluate_acceptance
from face_os.types import (
    CoordinateSpace,
    GeometryState,
    TransformEdge,
    TransformGraph,
)


def _geometry(confidence=0.9, matrix=None):
    graph = TransformGraph()
    graph.add(
        TransformEdge.from_matrix(
            np.eye(3, dtype=np.float32) if matrix is None else matrix,
            CoordinateSpace.SOURCE_FRAME,
            CoordinateSpace.CANONICAL_UV,
        )
    )
    return GeometryState(
        canonical_transform=np.eye(3, dtype=np.float32),
        mask=np.ones((16, 16), dtype=np.float32),
        geometry_confidence=confidence,
        transform_graph=graph,
    )


def test_accept_gate_accepts_valid_signals():
    decision = evaluate_acceptance(
        _geometry(),
        identity_confidence=0.8,
    )

    assert decision.accept is True
    assert decision.reason is None
    assert decision.to_dict()["geometry_ok"] is True
    # Phase 2B default: c_obs path
    assert decision.trust_source == "c_obs"
    assert decision.c_recon is None


def test_accept_gate_rejects_invalid_transform_graph():
    singular = np.zeros((3, 3), dtype=np.float32)

    decision = evaluate_acceptance(
        _geometry(matrix=singular),
        identity_confidence=0.8,
    )

    assert decision.accept is False
    assert decision.reason == "accept_reject_geometry"


def test_accept_gate_rejects_low_identity_confidence():
    decision = evaluate_acceptance(
        _geometry(),
        identity_confidence=0.0,
    )

    assert decision.accept is False
    assert decision.reason == "accept_reject_identity"


# ── Phase 2B: §16.8 C_recon as identity trust signal ──────────────────────


def test_phase_2b_c_recon_accepts_when_above_floor():
    decision = evaluate_acceptance(
        _geometry(),
        identity_confidence=0.0,            # raw C_obs would reject
        c_recon=0.05,                       # composite above min_c_recon floor
        use_c_recon_gate=True,
    )

    assert decision.accept is True
    assert decision.reason is None
    assert decision.identity_ok is True
    assert decision.trust_source == "c_recon"
    assert decision.c_recon == 0.05


def test_phase_2b_c_recon_rejects_below_floor():
    decision = evaluate_acceptance(
        _geometry(),
        identity_confidence=0.99,           # raw C_obs would accept
        c_recon=1e-6,                       # composite below min_c_recon floor
        use_c_recon_gate=True,
    )

    assert decision.accept is False
    assert decision.reason == "accept_reject_identity"
    assert decision.trust_source == "c_recon"
    assert decision.c_recon == 1e-6


def test_phase_2b_c_recon_none_falls_back_to_c_obs():
    decision = evaluate_acceptance(
        _geometry(),
        identity_confidence=0.8,
        c_recon=None,
        use_c_recon_gate=True,
    )

    assert decision.accept is True
    assert decision.trust_source == "c_obs"           # graceful fallback
    assert decision.c_recon is None


def test_phase_2b_c_recon_flag_off_keeps_legacy_path():
    decision = evaluate_acceptance(
        _geometry(),
        identity_confidence=0.8,
        c_recon=0.0001,                     # would reject under c_recon path
        use_c_recon_gate=False,             # but flag is off
    )

    assert decision.accept is True
    assert decision.trust_source == "c_obs"


def test_phase_2b_c_recon_score_uses_composite():
    decision = evaluate_acceptance(
        _geometry(confidence=0.9),
        identity_confidence=0.99,           # high raw
        c_recon=0.01,                       # but composite is small
        use_c_recon_gate=True,
    )

    # Score is min(geom, identity, temporal, lighting) = min(0.9, 0.01, 1.0, 1.0) = 0.01
    assert abs(decision.score - 0.01) < 1e-9
    assert decision.to_dict()["trust_source"] == "c_recon"
    assert decision.to_dict()["c_recon"] == 0.01


def test_phase_2b_c_recon_non_finite_treated_as_missing():
    decision = evaluate_acceptance(
        _geometry(),
        identity_confidence=0.8,
        c_recon=float("nan"),
        use_c_recon_gate=True,
    )

    assert decision.accept is True
    assert decision.trust_source == "c_obs"           # falls back gracefully
    assert decision.c_recon is None


def test_phase_2b_c_recon_dict_round_trip():
    decision = evaluate_acceptance(
        _geometry(),
        identity_confidence=0.5,
        c_recon=0.02,
        use_c_recon_gate=True,
    )
    d = decision.to_dict()
    assert d["trust_source"] == "c_recon"
    assert d["c_recon"] == 0.02
    assert d["accept"] is True
