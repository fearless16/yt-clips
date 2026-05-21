"""
types.py — Core data structures for the Face OS pipeline.

Every module communicates through these typed structures.
No raw dicts flowing between modules — everything is explicit.

Phase 0: Contract Lockdown — FrameContract, EnergyReport, RendererReport, PassReport
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

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


# ─── Geometry Estimator structures ─────────────────────────────────────────

@dataclass
class GeometryState:
    """Geometry state from Geometry Estimator subsystem."""
    landmarks_478: Optional[np.ndarray] = None          # (478, 3) MediaPipe mesh
    landmarks: Optional[Landmarks] = None               # Extracted landmarks with pose
    pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (yaw, pitch, roll)
    canonical_transform: Optional[np.ndarray] = None    # Transform to canonical space
    inverse_transform: Optional[np.ndarray] = None      # Transform from canonical space  
    crop_transform: Optional[CropPlan] = None           # Crop plan
    mesh: Optional[np.ndarray] = None                   # Face mesh
    semantic_regions: Optional[Dict[str, np.ndarray]] = None  # Region masks
    mask: Optional[np.ndarray] = None                   # Geometry-based face mask
    geometry_confidence: float = 0.0                    # Overall geometry confidence
    canonical_face: Optional[np.ndarray] = None         # Frame warped to canonical space


# ─── Identity Estimator structures ─────────────────────────────────────────

@dataclass
class IdentityState:
    """Identity state from Identity Estimator subsystem."""
    anchor_basis: list = field(default_factory=list)           # List of anchor states
    anchor_weights: list = field(default_factory=list)         # Weights for anchors
    appearance_latent: Optional[np.ndarray] = None             # Current identity appearance
    region_confidence: Dict[str, float] = field(default_factory=dict)  # Per-region confidence
    identity_uncertainty: float = 1.0                          # Overall uncertainty (0-1)
    initialized: bool = False                                  # Whether initialized


# ─── Temporal Estimator structures ─────────────────────────────────────────

@dataclass
class TemporalState:
    """Temporal state from Temporal Estimator subsystem."""
    motion_field: Optional[np.ndarray] = None                  # Optical flow field (H, W, 2)
    temporal_confidence: float = 1.0                          # Temporal consistency confidence
    drift_score: float = 0.0                                  # Identity drift from anchor
    continuity_score: float = 1.0                             # Temporal smoothness score  
    smoothing_constraints: Dict[str, float] = field(default_factory=dict)  # Smoothing limits
    pose: Optional[Tuple[float, float, float]] = None         # Pose for continuity tracking


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


# ─── Phase 0: Contract Lockdown Structures ──────────────────────────────────

@dataclass
class FrameContract:
    """Frame output contract — every output frame must satisfy these invariants."""
    # Shape contract
    expected_height: int = 1920
    expected_width: int = 1080
    expected_channels: int = 3
    # Dtype contract
    expected_dtype: str = "uint8"
    # Value range contract
    min_value: int = 0
    max_value: int = 255
    # Stability contract
    allow_nan: bool = False
    allow_inf: bool = False

    def validate(self, frame: np.ndarray) -> Tuple[bool, str]:
        """Validate a frame against this contract.

        Returns:
            (passed, reason) — True if all checks pass
        """
        if frame.shape != (self.expected_height, self.expected_width, self.expected_channels):
            return False, f"shape_mismatch: {frame.shape} != ({self.expected_height}, {self.expected_width}, {self.expected_channels})"

        if str(frame.dtype) != self.expected_dtype:
            return False, f"dtype_mismatch: {frame.dtype} != {self.expected_dtype}"

        if not self.allow_nan and np.any(np.isnan(frame)):
            return False, "nan_detected"

        if not self.allow_inf and np.any(np.isinf(frame)):
            return False, "inf_detected"

        if frame.min() < self.min_value or frame.max() > self.max_value:
            return False, f"value_range: [{frame.min()}, {frame.max()}] not in [{self.min_value}, {self.max_value}]"

        return True, "passed"


@dataclass
class GeometryMetrics:
    """Geometry state metrics for parameter-wise visibility."""
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    det_A: float = 1.0                      # Determinant of transform matrix
    mask_coverage_pct: float = 0.0          # Mask coverage percentage
    transform_stability: float = 1.0        # Frame-to-frame transform stability
    geometry_confidence: float = 0.0
    landmark_count: int = 0
    pose_magnitude: float = 0.0             # |yaw| + |pitch| + |roll|


@dataclass
class IdentityMetrics:
    """Identity state metrics for parameter-wise visibility."""
    anchor_weights: List[float] = field(default_factory=list)
    uncertainty: float = 1.0
    region_confidence: Dict[str, float] = field(default_factory=dict)
    appearance_latent_norm: float = 0.0     # L2 norm of appearance_latent
    anchor_distance_lab: float = 0.0        # LAB distance from anchor
    observation_count: float = 0.0          # Mean observation count


@dataclass
class TemporalMetrics:
    """Temporal state metrics for parameter-wise visibility."""
    temporal_confidence: float = 1.0
    drift_score: float = 0.0
    continuity_score: float = 1.0
    motion_field_norm: float = 0.0          # Mean magnitude of motion field
    covariance_trace: float = 0.0           # Trace of belief covariance (Phase 3)
    uncertainty_mean: float = 0.0           # Mean uncertainty (Phase 3)


@dataclass
class EnergyTerms:
    """Energy function terms — each term is a measurable float."""
    E_geom: float = 0.0                     # Geometry energy
    E_identity: float = 0.0                 # Identity energy
    E_temporal: float = 0.0                 # Temporal energy
    E_photometric: float = 0.0              # Photometric energy
    E_smoothness: float = 0.0               # Smoothness energy
    E_total: float = 0.0                    # Sum of all terms
    _normalized: bool = False               # Whether E_total is normalized
    _raw_total: float = 0.0                 # Raw sum before normalization

    def to_dict(self) -> Dict[str, float]:
        return {
            "E_geom": self.E_geom,
            "E_identity": self.E_identity,
            "E_temporal": self.E_temporal,
            "E_photometric": self.E_photometric,
            "E_smoothness": self.E_smoothness,
            "E_total": self.E_total,
            "E_total_raw": self._raw_total,
            "normalized": self._normalized,
        }


@dataclass
class RendererMetrics:
    """Renderer metrics for parameter-wise visibility."""
    M_mean: float = 0.0                     # Mean mask value
    M_min: float = 0.0                      # Min mask value
    M_max: float = 0.0                      # Max mask value
    Y_face_range: Tuple[int, int] = (0, 255)  # Face pixel range
    Y_bg_range: Tuple[int, int] = (0, 255)    # Background pixel range
    blend_weight_min: float = 0.0
    blend_weight_mean: float = 0.0
    blend_weight_max: float = 0.0
    temporal_confidence: float = 1.0
    output_shape: Tuple[int, int, int] = (1920, 1080, 3)
    output_dtype: str = "uint8"


@dataclass
class EnergyReport:
    """Per-frame energy report — all energy terms as measurable floats."""
    frame_idx: int = 0
    terms: EnergyTerms = field(default_factory=EnergyTerms)
    geometry: GeometryMetrics = field(default_factory=GeometryMetrics)
    identity: IdentityMetrics = field(default_factory=IdentityMetrics)
    temporal: TemporalMetrics = field(default_factory=TemporalMetrics)
    renderer: RendererMetrics = field(default_factory=RendererMetrics)
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "status": self.status,
            "energy_terms": self.terms.to_dict(),
            "geometry": {
                "yaw": self.geometry.yaw,
                "pitch": self.geometry.pitch,
                "roll": self.geometry.roll,
                "det_A": self.geometry.det_A,
                "mask_coverage_pct": self.geometry.mask_coverage_pct,
                "transform_stability": self.geometry.transform_stability,
                "geometry_confidence": self.geometry.geometry_confidence,
            },
            "identity": {
                "anchor_weights": self.identity.anchor_weights,
                "uncertainty": self.identity.uncertainty,
                "region_confidence": self.identity.region_confidence,
                "appearance_latent_norm": self.identity.appearance_latent_norm,
                "anchor_distance_lab": self.identity.anchor_distance_lab,
            },
            "temporal": {
                "temporal_confidence": self.temporal.temporal_confidence,
                "drift_score": self.temporal.drift_score,
                "continuity_score": self.temporal.continuity_score,
                "covariance_trace": self.temporal.covariance_trace,
            },
            "renderer": {
                "M_mean": self.renderer.M_mean,
                "Y_face_range": list(self.renderer.Y_face_range),
                "Y_bg_range": list(self.renderer.Y_bg_range),
                "blend_weight_stats": {
                    "min": self.renderer.blend_weight_min,
                    "mean": self.renderer.blend_weight_mean,
                    "max": self.renderer.blend_weight_max,
                },
            },
        }


@dataclass
class RendererReport:
    """Per-frame renderer report — output contract validation."""
    frame_idx: int = 0
    output_shape: Tuple[int, int, int] = (1920, 1080, 3)
    output_dtype: str = "uint8"
    nan_count: int = 0
    inf_count: int = 0
    value_range: Tuple[int, int] = (0, 255)
    contract_passed: bool = True
    contract_reason: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "value_range": list(self.value_range),
            "contract_passed": self.contract_passed,
            "contract_reason": self.contract_reason,
        }


@dataclass
class PassReport:
    """Per-pass report with before/after/delta metrics.

    This is the MANDATORY visibility format for every change.
    If this report is missing, the change must be rejected.
    """
    pass_id: str = ""                       # e.g. "phase2_transform_hardening"
    frame_id: int = 0                       # Frame index
    status: str = "pending"                 # accepted / rejected / skipped
    before: Dict[str, float] = field(default_factory=dict)
    after: Dict[str, float] = field(default_factory=dict)
    delta: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "frame_id": self.frame_id,
            "status": self.status,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "metrics": self.metrics,
        }

    def compute_delta(self) -> None:
        """Compute delta from before and after."""
        for key in self.before:
            if key in self.after:
                before_val = self.before[key]
                after_val = self.after[key]
                if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                    self.delta[key] = after_val - before_val


@dataclass
class PhaseState:
    """Current phase state for the pipeline."""
    current_phase: int = 0                  # 0-6
    phase_name: str = "phase0_contract_lockdown"
    phase_status: str = "in_progress"       # in_progress / passed / failed
    invariants_passed: int = 0
    invariants_total: int = 0
    energy_reports: List[EnergyReport] = field(default_factory=list)
    pass_reports: List[PassReport] = field(default_factory=list)
