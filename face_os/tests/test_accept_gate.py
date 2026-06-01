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
