"""
types.py — Core data structures for the Face OS pipeline.

BEAST MODE FIXES:
- Renamed IdentityState -> IdentityEstimatorState to prevent namespace collision with identity_state.py.
- Fixed np.isnan TypeError crash on uint8 arrays in FrameContract.validate.
- Fixed isinstance() blindness to np.float32 in PassReport.compute_delta using numbers.Real.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import numbers

import numpy as np


# ─── Enums ──────────────────────────────────────────────────────────────────

class FaceState(Enum):
    DETECTED = auto()
    TRACKED = auto()
    OCCLUDED = auto()
    LOST = auto()


class CropStrategy(Enum):
    FACE_LOCKED = auto()
    CENTER = auto()
    LAST_KNOWN = auto()


class EnhancementLevel(Enum):
    FULL = auto()
    STANDARD = auto()
    LIGHT = auto()
    SKIP = auto()


# ─── Frame-level structures ─────────────────────────────────────────────────

@dataclass
class FaceDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    is_target: bool
    embedding: Optional[np.ndarray] = None
    distance: float = 1.0


@dataclass
class Landmarks:
    points: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    left_eye_center: Tuple[float, float] = (0.0, 0.0)
    right_eye_center: Tuple[float, float] = (0.0, 0.0)
    nose_tip: Tuple[float, float] = (0.0, 0.0)
    mouth_center: Tuple[float, float] = (0.0, 0.0)
    landmark_confidence: float = 0.0


@dataclass
class FaceTrack:
    track_id: int
    state: FaceState = FaceState.DETECTED
    frames_visible: int = 0
    frames_lost: int = 0
    detection: Optional[FaceDetection] = None
    landmarks: Optional[Landmarks] = None
    smooth_bbox: Optional[Tuple[int, int, int, int]] = None
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    landmark_history: List[np.ndarray] = field(default_factory=list)
    mesh_478: Optional[np.ndarray] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)


# ─── Appearance field structures ────────────────────────────────────────────

@dataclass
class AppearanceField:
    atlas_rgb: Optional[np.ndarray] = None
    atlas_lab: Optional[np.ndarray] = None
    atlas_confidence: Optional[np.ndarray] = None
    atlas_normals: Optional[np.ndarray] = None
    uv_flow: Optional[np.ndarray] = None
    identity_residual: Optional[np.ndarray] = None
    enrollment_frames: int = 0
    last_update_frame: int = 0


@dataclass
class CanonicalMapping:
    transform_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    inverse_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    canonical_size: Tuple[int, int] = (256, 256)
    enrolled_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# ─── Crop and composition structures ───────────────────────────────────────

@dataclass
class CropPlan:
    strategy: CropStrategy = CropStrategy.CENTER
    src_x: int = 0
    src_y: int = 0
    src_w: int = 0
    src_h: int = 0
    dst_w: int = 1080
    dst_h: int = 1920
    face_center_out: Optional[Tuple[int, int]] = None
    headroom_ratio: float = 0.30
    confidence: float = 1.0


@dataclass
class EnhancementMask:
    face_mask: Optional[np.ndarray] = None
    eye_mask: Optional[np.ndarray] = None
    brow_mask: Optional[np.ndarray] = None
    beard_mask: Optional[np.ndarray] = None
    contour_mask: Optional[np.ndarray] = None
    skin_mask: Optional[np.ndarray] = None
    background_mask: Optional[np.ndarray] = None


@dataclass
class ConfidenceMap:
    spatial_confidence: Optional[np.ndarray] = None
    temporal_confidence: Optional[np.ndarray] = None
    combined: Optional[np.ndarray] = None
    eye_quality: float = 0.0
    skin_quality: float = 0.0
    contour_quality: float = 0.0


# ─── Subsystem structures ───────────────────────────────────────────────────

@dataclass
class GeometryState:
    landmarks_478: Optional[np.ndarray] = None
    landmarks: Optional[Landmarks] = None
    pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    canonical_transform: Optional[np.ndarray] = None
    inverse_transform: Optional[np.ndarray] = None
    crop_transform: Optional[CropPlan] = None
    mesh: Optional[np.ndarray] = None
    semantic_regions: Optional[Dict[str, np.ndarray]] = None
    mask: Optional[np.ndarray] = None
    geometry_confidence: float = 0.0
    canonical_face: Optional[np.ndarray] = None


@dataclass
class IdentityEstimatorState:  # BEAST MODE FIX: Renamed from IdentityState to prevent collision
    """Identity state from Identity Estimator subsystem."""
    anchor_basis: list = field(default_factory=list)
    anchor_weights: list = field(default_factory=list)
    appearance_latent: Optional[np.ndarray] = None
    region_confidence: Dict[str, float] = field(default_factory=dict)
    identity_uncertainty: float = 1.0
    initialized: bool = False


@dataclass
class TemporalState:
    motion_field: Optional[np.ndarray] = None
    temporal_confidence: float = 1.0
    drift_score: float = 0.0
    continuity_score: float = 1.0
    smoothing_constraints: Dict[str, float] = field(default_factory=dict)
    pose: Optional[Tuple[float, float, float]] = None


# ─── Pipeline structures ────────────────────────────────────────────────────

@dataclass
class FrameData:
    frame_idx: int
    timestamp: float
    source_frame: Optional[np.ndarray] = None
    face_track: Optional[FaceTrack] = None
    landmarks: Optional[Landmarks] = None
    canonical_map: Optional[CanonicalMapping] = None
    crop_plan: Optional[CropPlan] = None
    cropped_frame: Optional[np.ndarray] = None
    enhancement_mask: Optional[EnhancementMask] = None
    enhanced_frame: Optional[np.ndarray] = None
    confidence: Optional[ConfidenceMap] = None
    output_frame: Optional[np.ndarray] = None


@dataclass
class VideoMeta:
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
    name: str = "target"
    embeddings: List[np.ndarray] = field(default_factory=list)
    appearance: AppearanceField = field(default_factory=AppearanceField)
    reference_paths: List[str] = field(default_factory=list)
    enrolled: bool = False
    enrollment_frames: int = 0


# ─── Phase 0: Contract Lockdown Structures ──────────────────────────────────

@dataclass
class FrameContract:
    expected_height: int = 1920
    expected_width: int = 1080
    expected_channels: int = 3
    expected_dtype: str = "uint8"
    min_value: int = 0
    max_value: int = 255
    allow_nan: bool = False
    allow_inf: bool = False

    def validate(self, frame: np.ndarray) -> Tuple[bool, str]:
        if frame.shape != (self.expected_height, self.expected_width, self.expected_channels):
            return False, f"shape_mismatch: {frame.shape} != ({self.expected_height}, {self.expected_width}, {self.expected_channels})"

        if str(frame.dtype) != self.expected_dtype:
            return False, f"dtype_mismatch: {frame.dtype} != {self.expected_dtype}"

        # BEAST MODE FIX: np.isnan crashes on uint8. Only check floating point types.
        if np.issubdtype(frame.dtype, np.floating):
            if not self.allow_nan and np.any(np.isnan(frame)):
                return False, "nan_detected"
            if not self.allow_inf and np.any(np.isinf(frame)):
                return False, "inf_detected"

        if frame.min() < self.min_value or frame.max() > self.max_value:
            return False, f"value_range: [{frame.min()}, {frame.max()}] not in [{self.min_value}, {self.max_value}]"

        return True, "passed"


@dataclass
class GeometryMetrics:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    det_A: float = 1.0
    mask_coverage_pct: float = 0.0
    transform_stability: float = 1.0
    geometry_confidence: float = 0.0
    landmark_count: int = 0
    pose_magnitude: float = 0.0


@dataclass
class IdentityMetrics:
    anchor_weights: List[float] = field(default_factory=list)
    uncertainty: float = 1.0
    region_confidence: Dict[str, float] = field(default_factory=dict)
    appearance_latent_norm: float = 0.0
    anchor_distance_lab: float = 0.0
    observation_count: float = 0.0


@dataclass
class TemporalMetrics:
    temporal_confidence: float = 1.0
    drift_score: float = 0.0
    continuity_score: float = 1.0
    motion_field_norm: float = 0.0
    covariance_trace: float = 0.0
    uncertainty_mean: float = 0.0


@dataclass
class EnergyTerms:
    E_geom: float = 0.0
    E_identity: float = 0.0
    E_temporal: float = 0.0
    E_photometric: float = 0.0
    E_smoothness: float = 0.0
    E_total: float = 0.0
    _normalized: bool = False
    _raw_total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "E_geom": self.E_geom, "E_identity": self.E_identity, "E_temporal": self.E_temporal,
            "E_photometric": self.E_photometric, "E_smoothness": self.E_smoothness, "E_total": self.E_total,
            "E_total_raw": self._raw_total, "normalized": self._normalized,
        }


@dataclass
class RendererMetrics:
    M_mean: float = 0.0
    M_min: float = 0.0
    M_max: float = 0.0
    Y_face_range: Tuple[int, int] = (0, 255)
    Y_bg_range: Tuple[int, int] = (0, 255)
    blend_weight_min: float = 0.0
    blend_weight_mean: float = 0.0
    blend_weight_max: float = 0.0
    temporal_confidence: float = 1.0
    output_shape: Tuple[int, int, int] = (1920, 1080, 3)
    output_dtype: str = "uint8"


@dataclass
class EnergyReport:
    frame_idx: int = 0
    terms: EnergyTerms = field(default_factory=EnergyTerms)
    geometry: GeometryMetrics = field(default_factory=GeometryMetrics)
    identity: IdentityMetrics = field(default_factory=IdentityMetrics)
    temporal: TemporalMetrics = field(default_factory=TemporalMetrics)
    renderer: RendererMetrics = field(default_factory=RendererMetrics)
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx, "status": self.status, "energy_terms": self.terms.to_dict(),
            "geometry": {
                "yaw": self.geometry.yaw, "pitch": self.geometry.pitch, "roll": self.geometry.roll,
                "det_A": self.geometry.det_A, "mask_coverage_pct": self.geometry.mask_coverage_pct,
                "transform_stability": self.geometry.transform_stability, "geometry_confidence": self.geometry.geometry_confidence,
            },
            "identity": {
                "anchor_weights": self.identity.anchor_weights, "uncertainty": self.identity.uncertainty,
                "region_confidence": self.identity.region_confidence, "appearance_latent_norm": self.identity.appearance_latent_norm,
                "anchor_distance_lab": self.identity.anchor_distance_lab,
            },
            "temporal": {
                "temporal_confidence": self.temporal.temporal_confidence, "drift_score": self.temporal.drift_score,
                "continuity_score": self.temporal.continuity_score, "covariance_trace": self.temporal.covariance_trace,
            },
            "renderer": {
                "M_mean": self.renderer.M_mean, "Y_face_range": list(self.renderer.Y_face_range),
                "Y_bg_range": list(self.renderer.Y_bg_range),
                "blend_weight_stats": {"min": self.renderer.blend_weight_min, "mean": self.renderer.blend_weight_mean, "max": self.renderer.blend_weight_max},
            },
        }


@dataclass
class RendererReport:
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
            "frame_idx": self.frame_idx, "output_shape": list(self.output_shape), "output_dtype": self.output_dtype,
            "nan_count": self.nan_count, "inf_count": self.inf_count, "value_range": list(self.value_range),
            "contract_passed": self.contract_passed, "contract_reason": self.contract_reason,
        }


@dataclass
class PassReport:
    pass_id: str = ""
    frame_id: int = 0
    status: str = "pending"
    before: Dict[str, float] = field(default_factory=dict)
    after: Dict[str, float] = field(default_factory=dict)
    delta: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass_id": self.pass_id, "frame_id": self.frame_id, "status": self.status,
            "before": self.before, "after": self.after, "delta": self.delta, "metrics": self.metrics,
        }

    def compute_delta(self) -> None:
        for key in self.before:
            if key in self.after:
                before_val = self.before[key]
                after_val = self.after[key]
                # BEAST MODE FIX: isinstance(float) ignores np.float32. Use numbers.Real.
                if isinstance(before_val, (int, float, numbers.Real)) and isinstance(after_val, (int, float, numbers.Real)):
                    self.delta[key] = float(after_val) - float(before_val)


@dataclass
class PhaseState:
    current_phase: int = 0
    phase_name: str = "phase0_contract_lockdown"
    phase_status: str = "in_progress"
    invariants_passed: int = 0
    invariants_total: int = 0
    energy_reports: List[EnergyReport] = field(default_factory=list)
    pass_reports: List[PassReport] = field(default_factory=list)


# ─── Latent Identity Rendering (D-05) — Phase 0 additive structures ──────────

@dataclass
class IdentityLatent:
    """Lighting-invariant identity latent in CANONICAL UV space.

    This is the renderer's PRIMARY input source (once the latent path is live).
    It stores reflectance and structure ONLY — NEVER illumination, NEVER raw
    RGB frames.

    Fields default to ``None`` so an empty (uninitialized) latent can be
    constructed cheaply; ``IdentityEstimator`` populates them on the first
    observation. The invariants below hold ONCE ``initialized is True``:

      - ``albedo``: (H, W, 3) float32 in [0, 1], white-balance normalized
        against ``wb_reference``, in canonical UV (``atlas_size``).
      - ``appearance_code``: (D,) float32, D = ManifoldConfig.dimension (16).
      - ``microdetail``: (H, W, 3) float32 zero-mean HF residual,
        best-observation-only (NEVER an EMA of pixels).
      - ``wb_reference``: (3,) float32 white-balance reference.
      - ``albedo_uncertainty`` / ``microdetail_uncertainty``: (H, W) float32 in
        [0, 1], same HxW as their data field.
      - ``appearance_uncertainty``: scalar [0, 1] (epistemic, from manifold).
      - ``observation_count``: (H, W) float32 — accumulated quality, for
        confidence.
    """
    atlas_size: Tuple[int, int] = (256, 256)          # (H, W) canonical UV

    albedo: Optional[np.ndarray] = None               # (H, W, 3) float32 [0,1]
    appearance_code: Optional[np.ndarray] = None      # (D,) float32
    microdetail: Optional[np.ndarray] = None          # (H, W, 3) float32 zero-mean
    wb_reference: Optional[np.ndarray] = None         # (3,) float32

    albedo_uncertainty: Optional[np.ndarray] = None        # (H, W) float32 [0,1]
    appearance_uncertainty: float = 1.0                    # scalar [0,1]
    microdetail_uncertainty: Optional[np.ndarray] = None   # (H, W) float32 [0,1]

    observation_count: Optional[np.ndarray] = None    # (H, W) float32
    initialized: bool = False

    def mean_confidence(self) -> float:
        """Mean latent confidence = 1 - mean(albedo_uncertainty), clamped [0,1].

        Returns 0.0 when uncertainty is unavailable (None or empty), so an
        uninitialized latent reads as zero-confidence rather than crashing.
        """
        unc = self.albedo_uncertainty
        if unc is None:
            return 0.0
        arr = np.asarray(unc, dtype=np.float32)
        if arr.size == 0:
            return 0.0
        conf = 1.0 - float(np.mean(arr))
        return float(np.clip(conf, 0.0, 1.0))


@dataclass
class LatentRenderTelemetry:
    """Per-frame proof that the latent (not the source crop) drove the render.

    Emitted once per frame by the Telemetry_System. On legacy frames it
    documents the current truth (``latent_primary=False``,
    ``source_pixel_fraction≈1.0``).
    """
    frame_idx: int = 0
    render_path: str = "physical_legacy"   # 'latent' | 'physical_legacy' | 'alpha' | 'enhancement'
    latent_primary: bool = False           # True iff face interior synthesized from latent
    source_pixel_fraction: float = 1.0     # fraction of face-mask pixels traceable to source
    latent_confidence: float = 0.0
    albedo_drift_from_anchor: float = 0.0
    uncertainty_mean: float = 0.0
    contract_assertions_passed: bool = True
    gate_state: str = "disabled"           # Phase 2B gate decision: 'engaged' |
    #   'below_floor' | 'confidence_spike' | 'uninitialized' | 'disabled'
    hybrid_alpha_mean: float = 1.0         # Phase 2B: mean per-pixel LATENT
    #   authority (1.0 = pure latent; <1 = low-freq observation blended in where
    #   uncertain). Proves the uncertainty hybrid actually engaged.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "render_path": self.render_path,
            "latent_primary": self.latent_primary,
            "source_pixel_fraction": self.source_pixel_fraction,
            "latent_confidence": self.latent_confidence,
            "albedo_drift_from_anchor": self.albedo_drift_from_anchor,
            "uncertainty_mean": self.uncertainty_mean,
            "contract_assertions_passed": self.contract_assertions_passed,
            "gate_state": self.gate_state,
            "hybrid_alpha_mean": self.hybrid_alpha_mean,
        }
