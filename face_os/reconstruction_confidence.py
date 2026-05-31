"""§16.8 Reconstruction Confidence — the composite trust signal.

arch §16.8:
    C_recon = C_obs · Coverage_pose · Coverage_light · Visibility

The COMPOSITE confidence the Phase-2B production gate consumes. Product of
raw observation confidence and all three coverage/visibility factors:
  - C_obs: raw latent confidence (mean_confidence())
  - Coverage_pose: |observed pose bins| / |total pose bins| (§16.7)
  - Coverage_light: |observed lighting bins| / |total lighting bins| (§16.7)
  - Visibility: mean_visibility (§16.6, scalar proxy for V(u,v,t))

Invariant: C_recon ≤ C_obs always (coverage/visibility can only REDUCE trust,
never add it). All inputs are clamped to [0, 1].

This is the Phase-2B gate input (arch §19 mandatory order: 2A proved pixels →
2B calibrated production gate → 2C graceful fallback). Once all factors exist
(they now do), this composite is the single number the gate reads.
"""
from __future__ import annotations


def compute_reconstruction_confidence(
    c_obs: float,
    coverage_pose: float,
    coverage_light: float,
    mean_visibility: float,
) -> float:
    """§16.8 composite ``C_recon = C_obs · Coverage_pose · Coverage_light · Visibility``.

    Properties (all tested):
      - result ≤ c_obs always (the §16.8 invariant)
      - any factor at 0 zeros the composite
      - all factors at 1 leaves c_obs unchanged
      - monotonic non-decreasing in each factor
      - inputs clamped to [0, 1]
    """
    c = min(max(float(c_obs), 0.0), 1.0)
    cp = min(max(float(coverage_pose), 0.0), 1.0)
    cl = min(max(float(coverage_light), 0.0), 1.0)
    mv = min(max(float(mean_visibility), 0.0), 1.0)
    return float(c * cp * cl * mv)
