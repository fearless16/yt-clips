"""
config.py — Face OS configuration.

Loads from face_os_config.yaml with sensible defaults.
All tuning parameters live here — no magic numbers in modules.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


_DEFAULTS: Dict[str, Any] = {
    # ─── Identity ──────────────────────────────────────────────────────────
    "identity": {
        "reference_dir": "photos/",           # Directory with reference face photos
        "reference_image": "expectation.png", # Primary reference for appearance
        "embedding_tolerance": 0.50,          # Face matching threshold (lower = stricter)
        "max_embeddings": 50,                 # Max reference embeddings to store
    },

    # ─── Detection & Tracking ──────────────────────────────────────────────
    "detection": {
        "model": "mediapipe",                  # mediapipe (V4, no dlib)
        "min_face_size": 60,                  # Minimum face size in pixels
        "detection_interval": 5,              # Detect every N frames (track in between)
        "max_lost_frames": 30,                # Frames before declaring face LOST
        "smoothing_alpha": 0.3,               # EMA smoothing for bbox (0 = no smoothing)
    },

    # ─── Landmarks ─────────────────────────────────────────────────────────
    "landmarks": {
        "model": "mediapipe_478",                 # mediapipe_478 (V4, no dlib)
        "pose_smoothing": 0.4,                # EMA for head pose angles
    },

    # ─── Canonical Face Mapping ────────────────────────────────────────────
    "canonical": {
        "atlas_size": [256, 256],             # Canonical face resolution [W, H]
        "alignment_mode": "similarity",       # similarity | affine | perspective
        "enrollment_frames": 30,              # Frames to average for enrollment
        "min_confidence_for_update": 0.7,     # Min detection confidence to update atlas
    },

    # ─── Crop Planner ──────────────────────────────────────────────────────
    "crop": {
        "output_size": [1080, 1920],          # Target output [W, H]
        "headroom_ratio": 0.30,               # Fraction of output above face center
        "face_target_width": 270,             # Target face width in output (pixels)
        "smoothing_alpha": 0.25,              # EMA for crop position
        "max_crop_velocity": 50,              # Max pixels crop can move per frame
        "protect_forehead": True,             # Never crop above forehead
        "allow_bottom_crop": True,            # OK to crop below chin
    },

    # ─── Temporal Stabilizer ───────────────────────────────────────────────
    "temporal": {
        "identity_inertia": 0.85,             # How much identity resists change (0-1)
        "flicker_threshold": 15.0,            # LAB distance to trigger stabilization
        "temporal_window": 5,                 # Frames to average for stabilization
        "use_motion_compensation": True,      # Compensate for camera motion
    },

    # ─── Face Enhancement ──────────────────────────────────────────────────
    "enhance": {
        "eye_boost": 1.5,                     # Enhancement multiplier for eyes
        "brow_boost": 1.3,                    # Enhancement multiplier for brows
        "beard_boost": 1.2,                   # Enhancement multiplier for beard
        "contour_boost": 1.2,                 # Enhancement multiplier for face contour
        "skin_smoothing": 0.3,                # Skin smoothing strength (0 = none)
        "sharpen_amount": 0.3,                # Sharpening strength
        "sharpen_radius": 1.0,                # Sharpening radius
        "use_cinematic_noise": True,          # Add subtle sensor grain
        "noise_strength": 0.02,               # Grain intensity (0 = none)
    },

    # ─── Identity Memory (Photic Memory) ───────────────────────────────────
    "memory": {
        "accumulation_rate": 0.1,             # How fast confidence accumulates
        "decay_rate": 0.01,                   # How fast confidence decays without observation
        "min_observations": 5,                # Min observations before using memory
        "max_age_frames": 300,                # Max frames to keep in memory
        "use_pose_weighting": True,           # Weight observations by pose similarity
    },

    # ─── Compositor ────────────────────────────────────────────────────────
    "compositor": {
        "confidence_threshold": 0.3,          # Below this, use source pixels
        "blend_mode": "poisson",              # poisson | laplacian | alpha
        "feather_pixels": 10,                 # Edge feathering width
        "use_light_matching": True,           # Match lighting between face and background
    },

    # ─── Export ────────────────────────────────────────────────────────────
    "export": {
        "codec": "libx264",
        "crf": 18,
        "preset": "slow",
        "bitrate": "25M",
        "audio_bitrate": "320k",
        "fps": 30,
        "pixel_format": "yuv420p",
        "timeout_seconds": 900,
        "fade_in": 0.5,
        "fade_out": 0.5,
    },

    # ─── QC ────────────────────────────────────────────────────────────────
    "qc": {
        "min_face_detection_rate": 0.80,      # Min % of frames with face detected
        "max_identity_drift": 20.0,           # Max LAB distance from reference
        "max_flicker_score": 5.0,             # Max frame-to-frame variance
        "min_sharpness": 10.0,                # Min Laplacian variance
        "check_av_sync": True,                # Validate audio-video sync
    },

    # ─── Paths ─────────────────────────────────────────────────────────────
    "paths": {
        "temp": "temp/",
        "output": "output/face_os/",
        "logs": "logs/",
    },

    # ─── Logging ───────────────────────────────────────────────────────────
    "logging": {
        "level": "INFO",
        "log_file": "logs/face_os.log",
    },
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge override into base. Override wins on conflicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class FaceOSConfig:
    """Configuration for the Face OS pipeline.

    Loads from face_os_config.yaml, falls back to defaults.
    Access via dot notation: cfg.identity.reference_dir
    """

    def __init__(self, config_path: Optional[str] = None):
        self._data = _DEFAULTS.copy()

        # Try to load config file
        paths_to_try = [
            config_path,
            "face_os_config.yaml",
            "face_os_config.yml",
        ]
        for p in paths_to_try:
            if p and Path(p).exists():
                with open(p, "r") as f:
                    user_cfg = yaml.safe_load(f) or {}
                self._data = _deep_merge(self._data, user_cfg)
                break

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no section '{name}'")
        if isinstance(value, dict):
            return _ConfigSection(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def section(self, name: str) -> Dict:
        return self._data.get(name, {})

    def to_dict(self) -> Dict:
        return self._data.copy()


class _ConfigSection:
    """Nested config section with dot-notation access."""

    def __init__(self, data: Dict):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(f"Config section has no key '{name}'")
        if isinstance(value, dict):
            return _ConfigSection(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __repr__(self) -> str:
        return repr(self._data)


# Singleton
_cfg: Optional[FaceOSConfig] = None


def get_config(config_path: Optional[str] = None) -> FaceOSConfig:
    """Get or create the global config singleton."""
    global _cfg
    if _cfg is None:
        _cfg = FaceOSConfig(config_path)
    return _cfg
