"""Subsystem boundary enforcement.

Validates that subsystems don't violate their contracts.
Used for testing and runtime assertions.
"""


class BoundaryViolation(Exception):
    """Raised when a subsystem violates its contract."""


FORBIDDEN_IMPORTS = {
    "geometry_estimator": ["identity_state", "appearance_field", "neural_codec"],
    "identity_estimator": ["landmarks", "canonical_map", "crop_planner"],
    "temporal_estimator": ["identity_state", "landmarks", "canonical_map"],
    "renderer": ["landmarks", "identity_state", "detect_track"],
}


def check_boundary(subsystem: str, module_globals: dict):
    """Check that a subsystem doesn't import forbidden modules.

    Args:
        subsystem: One of "geometry_estimator", "identity_estimator",
                   "temporal_estimator", "renderer"
        module_globals: The module's globals() dict

    Raises:
        BoundaryViolation if forbidden import detected
    """
    forbidden = FORBIDDEN_IMPORTS.get(subsystem, [])
    for name, obj in module_globals.items():
        if hasattr(obj, "__module__") and obj.__module__:
            for fb in forbidden:
                if fb in obj.__module__:
                    raise BoundaryViolation(
                        f"{subsystem} must not import {fb} "
                        f"(found: {name} from {obj.__module__})"
                    )
