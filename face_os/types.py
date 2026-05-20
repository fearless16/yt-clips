"""
types.py — Core data structures for the Face OS pipeline.

Every module communicates through these typed structures.
No raw dicts flowing between modules — everything is explicit.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Enums ──────────────────────────────────────────────────────────────────

class FaceState(Enum):
    """Tracking state for a face across frames."""
    DETECTED = auto()      # Face found and matched to identity
    TRACKED = auto()       # Face predicted by tracker (no detection)
    OCCLUDED = auto()      # Face temporarily hidden
    LOST = auto()          # Face not found for N frames


class CropStrategy(Enum):
    """How to crop the 16:9 source into 9:16."""
    FACE_LOCKED = auto()   # Face detected — crop follows face
    CENTER = auto()        # No face — fall back to center crop
    LAST_KNOWN = auto()    # Use last valid face position


class EnhancementLevel(Enum):
    """Per-region enhancement intensity."""
    FULL = auto()          # Eyes, brows, beard — max quality
    STANDARD = auto()      # Skin, nose — normal processing
    LIGHT = auto()         # Cheeks, forehead — gentle processing
    SKIP = auto()          # Background — no enhancement


# ─── Frame-level structures ─────────────────────────────────────────────────

@dataclass
class FaceDetection:
    """A single face detection in one frame."""
    bbox: Tuple[int, int, int, int]          # (x, y, w, h)
    confidence: float                         # Detection confidence [0, 1]
    is_target: bool                           # Is this the target identity?
    embedding: Optional[np.ndarray] = None    # Face embedding vector
    distance: float = 1.0                     # Distance to reference embedding


@dataclass
class Landmarks:
    """Facial landmarks + derived measurements (V4: MediaPipe 478-point)."""
    points: np.ndarray                        # Shape (N, 2) — pixel coords
    # Derived pose
    yaw: float = 0.0                          # Head rotation left/right (degrees)
    pitch: float = 0.0                        # Head rotation up/down (degrees)
    roll: float = 0.0                         # Head tilt (degrees)
    # Derived regions (indices into points)
    left_eye_center: Tuple[float, float] = (0.0, 0.0)
    right_eye_center: Tuple[float, float] = (0.0, 0.0)
    nose_tip: Tuple[float, float] = (0.0, 0.0)
    mouth_center: Tuple[float, float] = (0.0, 0.0)
    # Quality
    landmark_confidence: float = 0.0          # Average landmark confidence


@dataclass
class FaceTrack:
    """Temporal face track — a face identity across multiple frames."""
    track_id: int                             # Unique track ID
    state: FaceState = FaceState.DETECTED
    frames_visible: int = 0                   # How many frames this track has been active
    frames_lost: int = 0                      # How many frames since last detection
    # Current frame data
    detection: Optional[FaceDetection] = None
    landmarks: Optional[Landmarks] = None
    # Smoothed position (EMA)
    smooth_bbox: Optional[Tuple[int, int, int, int]] = None
    # History
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    landmark_history: List[np.ndarray] = field(default_factory=list)
    # V4: Face mesh (478 points) from MediaPipe FaceLandmarker
    mesh_478: Optional[np.ndarray] = None     # (478, 3) pixel coords (x, y, z)
    # Quality gate metrics
    quality_metrics: Dict[str, float] = field(default_factory=dict)


# ─── Appearance field structures ────────────────────────────────────────────

@dataclass
class AppearanceField:
    """Dynamic appearance function A(u,v,θ,L,t).

    Instead of rendering from mesh+texture, we learn the appearance
    directly as a function of UV coordinates, pose, lighting, and time.

    For overfit mode (one person), this is a dense per-pixel cache
    that accumulates confidence over time.
    """
    # Canonical UV atlas (face in neutral pose, frontal)
    atlas_rgb: Optional[np.ndarray] = None        # (H, W, 3) canonical face
    atlas_lab: Optional[np.ndarray] = None        # (H, W, 3) LAB space
    atlas_confidence: Optional[np.ndarray] = None  # (H, W) per-pixel confidence
    atlas_normals: Optional[np.ndarray] = None     # (H, W, 3) surface normals

    # Dynamic UV flow — how UVs deform with expression
    uv_flow: Optional[np.ndarray] = None           # (H, W, 2) displacement field

    # Identity residual — what makes THIS face unique
    identity_residual: Optional[np.ndarray] = None  # (H, W, 3) correction over base

    # Metadata
    enrollment_frames: int = 0                       # Frames used to build atlas
    last_update_frame: int = 0                       # Last frame that updated atlas


@dataclass
class CanonicalMapping:
    """Maps a detected face to the canonical atlas space."""
    # Transform from source frame to canonical
    transform_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float32)
    )
    # Inverse transform from canonical to source
    inverse_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float32)
    )
    # Canonical face size (pixels)
    canonical_size: Tuple[int, int] = (256, 256)
    # Pose at enrollment
    enrolled_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# ─── Crop and composition structures ───────────────────────────────────────

@dataclass
class CropPlan:
    """Planned crop from 16:9 source to 9:16 output."""
    strategy: CropStrategy = CropStrategy.CENTER
    # Source crop region
    src_x: int = 0
    src_y: int = 0
    src_w: int = 0
    src_h: int = 0
    # Target output size
    dst_w: int = 1080
    dst_h: int = 1920
    # Face position in output (for downstream modules)
    face_center_out: Optional[Tuple[int, int]] = None
    # Headroom fraction (0.0 = no headroom, 0.35 = 35% above face)
    headroom_ratio: float = 0.30
    # Confidence in this crop plan
    confidence: float = 1.0


@dataclass
class EnhancementMask:
    """Per-pixel enhancement intensity map."""
    # Regions and their enhancement levels
    face_mask: Optional[np.ndarray] = None         # (H, W) float [0, 1]
    eye_mask: Optional[np.ndarray] = None          # (H, W) float [0, 1]
    brow_mask: Optional[np.ndarray] = None         # (H, W) float [0, 1]
    beard_mask: Optional[np.ndarray] = None        # (H, W) float [0, 1]
    contour_mask: Optional[np.ndarray] = None      # (H, W) float [0, 1]
    skin_mask: Optional[np.ndarray] = None         # (H, W) float [0, 1]
    background_mask: Optional[np.ndarray] = None   # (H, W) float [0, 1]


@dataclass
class ConfidenceMap:
    """Per-pixel confidence from photic memory accumulation."""
    # Spatial confidence [0, 1] — higher = more observations
    spatial_confidence: Optional[np.ndarray] = None   # (H, W)
    # Temporal confidence — how stable this pixel is across frames
    temporal_confidence: Optional[np.ndarray] = None  # (H, W)
    # Combined confidence used for compositing
    combined: Optional[np.ndarray] = None             # (H, W)
    # Per-region quality scores
    eye_quality: float = 0.0
    skin_quality: float = 0.0
    contour_quality: float = 0.0


# ─── Pipeline structures ────────────────────────────────────────────────────

@dataclass
class FrameData:
    """All data for a single frame flowing through the pipeline."""
    frame_idx: int
    timestamp: float                              # Seconds
    source_frame: Optional[np.ndarray] = None     # Original BGR frame
    # Detection + tracking
    face_track: Optional[FaceTrack] = None
    # Landmarks
    landmarks: Optional[Landmarks] = None
    # Canonical mapping
    canonical_map: Optional[CanonicalMapping] = None
    # Crop plan
    crop_plan: Optional[CropPlan] = None
    cropped_frame: Optional[np.ndarray] = None    # After crop
    # Enhancement
    enhancement_mask: Optional[EnhancementMask] = None
    enhanced_frame: Optional[np.ndarray] = None   # After enhancement
    # Confidence
    confidence: Optional[ConfidenceMap] = None
    # Final output
    output_frame: Optional[np.ndarray] = None


@dataclass
class VideoMeta:
    """Metadata for an ingested video."""
    path: str
    width: int = 0
    height: int = 0
    fps: float = 30.0
    total_frames: int = 0
    duration: float = 0.0
    has_audio: bool = False
    codec: str = ""
    audio_codec: str = ""


@dataclass
class IdentityProfile:
    """The target identity's profile — built from reference images/video."""
    name: str = "target"
    # Reference embeddings
    embeddings: List[np.ndarray] = field(default_factory=list)
    # Canonical appearance
    appearance: AppearanceField = field(default_factory=AppearanceField)
    # Reference images used
    reference_paths: List[str] = field(default_factory=list)
    # Enrollment state
    enrolled: bool = False
    enrollment_frames: int = 0
