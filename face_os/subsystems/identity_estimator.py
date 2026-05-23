"""Subsystem B — Identity Estimation.

Estimates stable identity representation.

Output: IdentityEstimatorState (from face_os.types)
Delegates to: identity_state.py, intrinsic_decomposition.py

BOUNDARY CONTRACT:
- MUST NOT perform RGB EMA blending
- MUST NOT accumulate raw frames
- MUST NOT handle geometry estimation
"""

import numpy as np

from face_os.types import IdentityEstimatorState


class IdentityEstimator:
    """Subsystem B: Stable identity representation.

    Thin wrapper that delegates to existing identity_state.py.
    Enforces lighting-invariant identity via query_albedo().

    FORBIDDEN: RGB EMA blending, raw frame accumulation, geometry estimation
    """

    def __init__(self, identity_state):
        """Args:
        identity_state: IdentityEstimatorState instance from identity_state.py
        """
        self._state = identity_state

    def query(self, quality_map: np.ndarray) -> IdentityEstimatorState:
        """Query lighting-invariant identity.

        Uses query_albedo (not query_identity) for lighting invariance.

        Args:
            quality_map: Per-pixel quality (H, W) float32

        Returns:
            IdentityEstimatorState with albedo-based identity
        """
        if not self._state.is_initialized():
            return IdentityEstimatorState()

        albedo, albedo_conf = self._state.query_albedo(quality_map)
        rgb_face, rgb_conf = self._state.query_identity(quality_map)
        intrinsic, intrinsic_conf = self._state.query_intrinsic(quality_map)

        return IdentityEstimatorState(
            appearance_latent=rgb_face,
            anchor_basis=[self._state._anchor_albedo]
            if hasattr(self._state, "_anchor_albedo") and self._state._anchor_albedo is not None
            else [],
            identity_uncertainty=(
                1.0 - float(np.mean(albedo_conf))
                if albedo_conf is not None
                else 1.0
            ),
            initialized=True,
        )

    def query_albedo(self, quality_map: np.ndarray):
        """Query lighting-invariant albedo directly.

        Args:
            quality_map: Per-pixel quality (H, W) float32

        Returns:
            (albedo_face, albedo_conf) tuple
        """
        if not self._state.is_initialized():
            return None, None
        return self._state.query_albedo(quality_map)

    def query_intrinsic(self, quality_map: np.ndarray):
        """Query intrinsic decomposition components.

        Args:
            quality_map: Per-pixel quality (H, W) float32

        Returns:
            (intrinsic_components, intrinsic_conf) tuple
        """
        if not self._state.is_initialized():
            return None, None
        return self._state.query_intrinsic(quality_map)
