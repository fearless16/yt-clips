"""Face OS V2 Subsystem Architecture.

4 isolated subsystems with explicit interfaces and boundary contracts.

Each subsystem:
- Has a clear input/output contract
- Is forbidden from certain operations (boundary enforcement)
- Delegates to existing modules (no logic duplication)
"""

from face_os.subsystems.geometry_estimator import GeometryEstimator
from face_os.subsystems.identity_estimator import IdentityEstimator
from face_os.subsystems.temporal_estimator import TemporalEstimator
from face_os.subsystems.renderer import FaceRenderer

__all__ = [
    "GeometryEstimator",
    "IdentityEstimator",
    "TemporalEstimator",
    "FaceRenderer",
]
