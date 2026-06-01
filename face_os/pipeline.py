"""
pipeline.py — Face OS Pipeline Orchestrator v2.

THE MENTAL SHIFT:
  OLD: "how do I enhance this frame?"
  NEW: "what does this person's face usually look like?"

FLOW PER FRAME:
  1. Detect & track face (telemetry extraction)
  2. Extract landmarks + pose (face telemetry)
  3. Canonical alignment (convert to standard face space)
  4. Query identity state (what does this region usually look like?)
  5. Query patch memory (pose-conditioned best patches)
  6. Plan 9:16 crop (face-locked with headroom)
  7. Render face (structure-preserving, NOT enhancing)
  8. Composite (confidence-weighted, frequency-aware)
  9. Write output

OFFLINE SUPERPOWER (bidirectional):
  Forward pass: collect all frames + quality
  Solve: future frames repair past frames
  Final pass: query solved identity for each frame

CORE EQUATION:
  FINAL = source * (1 - confidence) + identity_memory * confidence

  But FREQUENCY-AWARE:
    Low freq: trust identity more (skin tone is stable)
    High freq: trust source more for current pose
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import (
    ConfidenceMap,
    CropPlan,
    EnhancementMask,
    FaceTrack,
    FrameData,
    GeometryState,
    IdentityProfile,
    Landmarks,
    LatentRenderTelemetry,
    VideoMeta,
)

# Module imports
from face_os import ingest
from face_os import detect_track
from face_os import landmarks as lm_module
from face_os import canonical_map
from face_os import crop_planner
from face_os import face_enhance
from face_os import compositor
from face_os.compositor import _blend_linear, multiband_blend, _srgb_to_linear, _linear_to_srgb
from face_os.photometric import photometric_lock, reset_photometric_lock
from face_os import export_qc

# NEW modules
from face_os.identity_state import IdentityState
from face_os.patch_memory import PatchMemory
from face_os.dense_geometry import DenseGeometryEstimator
from face_os.temporal_solve import TemporalRepairEngine, FrameQuality

# V3 modules
from face_os.physical_renderer import PhysicalRenderer, LightingModel
from face_os.intrinsic_decomposition import IntrinsicComponents, assert_intrinsic_contract
from face_os.lie_group import SIM2Transform, interpolate_sim2
from face_os.renderer_mode import RendererMode, RendererModeState
from face_os.state_evolution import StateEvolution
from face_os.energy_scaling import EnergyScaler

# D-10: Subsystem wrappers (thin delegation, boundary enforcement)
from face_os.subsystems import IdentityEstimator, TemporalEstimator, FaceRenderer, GeometryEstimator


cfg = get_config()


cfg = get_config()

# ─── Feature flags ──────────────────────────────────────────────────────────
# Set USE_IDENTITY=False to disable identity memory and get simple enhancement
# (crop + sharpen + denoise). No ghosting, no background bleed, no plastic skin.
USE_IDENTITY = True


class FaceOSPipeline:
    """The Face OS processing pipeline v2.

    Philosophy:
      - Source video is TELEMETRY, not ground truth
      - Each frame is a noisy photon observation
      - Maintain IDENTITY BELIEF STATE
      - Query memory, don't enhance pixels
      - Frequency decomposition: low freq smooth, high freq best-only
      - Per-region independent dynamics
      - Bidirectional temporal solve (offline superpower)

    Pipeline flow per frame:
      1. Detect & track (telemetry)
      2. Landmarks + pose (face telemetry)
      3. Canonical alignment (standard face space)
      4. Quality map computation (per-pixel quality)
      5. Identity state update (belief update)
      6. Patch memory update (per-region)
      7. Query identity (what does this region usually look like?)
      8. Crop planning (face-locked 9:16)
      9. Render (structure-preserving)
      10. Composite (confidence-weighted)
    """

    def __init__(self, use_bidirectional: bool = True):
        # Core modules
        self.tracker: Optional[detect_track.FaceTracker] = None
        self.appearance_builder: Optional[canonical_map.AppearanceFieldBuilder] = None
        self.crop: Optional[crop_planner.CropPlanner] = None
        self.compositor: Optional[compositor.Compositor] = None

        # NEW: Identity belief state
        self.identity_state: Optional[IdentityState] = None
        self.patch_memory: Optional[PatchMemory] = None

        # NEW: Bidirectional solver
        self.use_bidirectional = use_bidirectional
        self.temporal_solver: Optional[TemporalRepairEngine] = None

        # Identity profile
        self.identity: Optional[IdentityProfile] = None

        # State
        self._enrolled = False
        self._frame_count = 0

        # Face lock state machine
        self._face_state = "LOST_FACE"  # FACE_LOCKED, LOST_FACE, RECOVERY
        self._lost_frame_count = 0
        self._recovery_frame_count = 0

        # M_inv smoothing for floating mask fix
        self._last_M_inv: Optional[np.ndarray] = None

        # Last good crop plan for fallback paths (prevents frame size change)
        self._last_good_crop_plan: Optional[CropPlan] = None

        # V3: Physical renderer
        self.physical_renderer: Optional[PhysicalRenderer] = None

        # V3: Renderer mode state
        self.renderer_mode_state: Optional[RendererModeState] = None

        # V3: State evolution model
        self.state_evolution: Optional[StateEvolution] = None
        self._latent_state: Optional[np.ndarray] = None
        self._latent_covariance: Optional[np.ndarray] = None

        # D-06: Per-frame belief storage (not global mutable cache)
        self._frame_beliefs: dict = {}

        # V3: LieGroup transform state
        self._last_SIM2: Optional[SIM2Transform] = None
        self._prev_SIM2: Optional[SIM2Transform] = None
        self._predicted_SIM2: Optional[SIM2Transform] = None

        # V3.1: Energy scaler for normalized energy terms (I-07: default-on)
        energy_cfg = getattr(cfg, 'energy', None)
        self._normalize_energy = getattr(energy_cfg, 'normalize_energy', True) if energy_cfg else True
        self._energy_method = getattr(energy_cfg, 'normalization_method', 'zscore') if energy_cfg else 'zscore'
        from face_os.energy_scaling import EnergyScalingConfig
        self.energy_scaler = EnergyScaler(EnergyScalingConfig(
            normalization_method=self._energy_method if self._normalize_energy else 'none',
        ))

        # V3: Runtime telemetry — tracks which paths are actually used
        self._telemetry = {
            "total_frames": 0,
            "physical_render_frames": 0,      # Frames using PhysicalRenderer
            "alpha_fallback_frames": 0,        # Frames using alpha compositing
            "intrinsic_success_frames": 0,     # Frames where intrinsic decomposition succeeded
            "intrinsic_failure_frames": 0,     # Frames where intrinsic decomposition failed
            "renderer_mode_transitions": 0,    # Number of renderer mode changes
            # Activation details
            "intrinsic_failure_reasons": {},   # Why intrinsic failed
            "renderer_mode_distribution": {    # Time in each mode
                "physical": 0,
                "hybrid": 0,
                "alpha": 0,
            },
            # Confidence distributions
            "intrinsic_confidence_sum": 0.0,
            "intrinsic_confidence_count": 0,
            "decomposition_error_sum": 0.0,
            "decomposition_error_count": 0,
            "mesh_normal_frames": 0,
            "shading_normal_frames": 0,
            # RULE 8: Fallback reason tracking
            "fallback_reason_distribution": {},
            # D-01: Identity path failure tracking
            "identity_path_failures": 0,
            # RULE 8: Timing telemetry
            "render_time_sum_ms": 0.0,
            "render_time_count": 0,
        }

        # D-08: Per-frame telemetry log (every frame exposed as JSON)
        self._frame_telemetry_log: list = []
        # D-05 Phase 0: Per-frame LatentRenderTelemetry log (one dict per frame),
        # mirrors _frame_telemetry_log. Exposed via get_latent_telemetry().
        self._latent_telemetry_log: list = []
        self._last_geometry_source = "none"
        self._last_transform_det = 1.0

        # D-05 Phase 0: Warn-only IntrinsicComponents contract mode.
        # Default 'warn' (logs, no clamp, no raise) so the legacy path is
        # behavior-preserving. An explicitly-configured fatal mode is honored
        # during legacy migration (Requirement 3.5): read from cfg if present.
        latent_cfg = getattr(cfg, 'latent', None)
        contract_mode = getattr(latent_cfg, 'contract_mode', None) if latent_cfg else None
        self._contract_mode = contract_mode if contract_mode in ('warn', 'fatal') else 'warn'

        # D-05 Phase 0: Per-frame latent telemetry state. Reset each frame so a
        # record reflects only the current frame (Requirement 8.3/8.4 — no
        # carryover). Phase 0 is legacy/shadow: latent does not drive rendering.
        self._last_contract_passed = True
        self._last_latent_confidence = 0.0
        self._last_albedo_drift = 0.0
        self._last_uncertainty_mean = 0.0
        # D-05 Task 2.6: kill-switch for the shadow-mode latent update. A shadow
        # subsystem must be toggleable (operational off-switch + clean A/B). On
        # by default; honors cfg.latent.shadow_enabled when present.
        self._latent_shadow_enabled = bool(
            getattr(latent_cfg, 'shadow_enabled', True) if latent_cfg else True
        )

        # D-05 Phase 2: render-path selector. 'legacy' = the existing
        # paste-then-relight path (A-2/A-3/A-5); 'latent' = the latent drives the
        # face interior via synthesize_identity + estimate_lighting +
        # render_from_latent (no source crop). Default 'legacy' so existing
        # behavior is untouched until a caller opts in (A/B). Honors
        # cfg.latent.render_source when present.
        render_source = getattr(latent_cfg, 'render_source', None) if latent_cfg else None
        self.render_source: str = render_source if render_source in ('legacy', 'latent') else 'legacy'

        # D-05 Phase 2A: gate policy selector. 'production' = Option 1
        # (relative-to-floor confidence gate, the existing _evaluate_latent_gate);
        # 'forced_latent' = Option 3 (engage whenever latent is initialized,
        # for A/B proving the path works). Option 2 (per-pixel blend) is a
        # future refinement. Honors cfg.latent.gate_policy when present.
        gate_policy = getattr(latent_cfg, 'gate_policy', None) if latent_cfg else None
        self._gate_policy: str = gate_policy if gate_policy in ('production', 'forced_latent') else 'production'
        # Fraction of the rendered crop still driven by source pixels (1.0 =
        # fully source/legacy). The latent render path lowers this to its face
        # coverage complement; read into per-frame telemetry.
        self._last_source_pixel_fraction: float = 1.0

        # D-05 Phase 2B production gate state (read into per-frame telemetry).
        # gate_state labels WHY the latent did or didn't drive the frame; the
        # default 'disabled' applies whenever render_source != 'latent'. The
        # confidence floor is the enrollment-seed confidence (the relative-to-
        # floor baseline), captured once at enroll(); prev tracks last frame's
        # confidence for spike detection.
        self._last_gate_state: str = "disabled"
        self._prev_latent_confidence: float = 0.0
        self._latent_confidence_floor: float = 0.0

        # D-05 Phase 2B per-pixel uncertainty HYBRID. blend_max CAPS how much of
        # the (low-frequency) observation may cross where the latent is fully
        # uncertain. 0.5 keeps the latent's synthesized identity at >=50%
        # authority on every pixel. Safe at 0.5 because the hybrid is RESTRICTED
        # to the solid mask interior (feathered_mask>0.99): measurement PROVED
        # 100% of hybrid-induced source-leak lived in the feather transition band
        # (where the multiband composite already mixes source) — restricted to
        # the solid interior, leak == pure-latent (<0.01) even at blend_max=0.5.
        # 0.0 disables the hybrid. Honors cfg.latent.hybrid_blend_max when present.
        hybrid_bm = getattr(latent_cfg, 'hybrid_blend_max', None) if latent_cfg else None
        self._hybrid_blend_max: float = (
            float(hybrid_bm) if isinstance(hybrid_bm, (int, float)) and 0.0 <= hybrid_bm <= 1.0
            else 0.5
        )
        self._last_hybrid_alpha_mean: float = 1.0
        self._last_effective_blend_max: float = self._hybrid_blend_max
        self._last_appearance_uncertainty: float = 0.0
        self._last_deform_max: float = 0.0
        self._last_deform_mean: float = 0.0

        # DIAGNOSTIC ONLY (default OFF, zero cost when off): when enabled, the
        # latent render path stashes its pre-composite rendered face, the actual
        # crop_mask, and the source crop into _last_latent_debug so an external
        # A/B report can measure latent-vs-legacy-vs-source INSIDE the real face
        # mask (not the diluted landmark bbox). Never read by the runtime.
        self._capture_latent_debug: bool = False
        self._last_latent_debug: Optional[dict] = None

        # D-02: A/B validation — render mode override (None = use default, 'alpha' = force alpha)
        self.render_mode_override: Optional[str] = None

        # D-10: Subsystem wrappers (thin delegation, not replacement)
        self._identity_estimator = IdentityEstimator(self.identity_state) if self.identity_state else None
        self._temporal_estimator = TemporalEstimator(self.state_evolution) if self.state_evolution else None
        self._face_renderer = FaceRenderer(self.physical_renderer, config=cfg)
        self._dense_geometry = DenseGeometryEstimator()
        # A-7: Geometry subsystem gets a real runtime instance. It does NOT
        # re-detect — assemble_state() packages the geometry the frame loop
        # already extracted into a single GeometryState (one geometry truth).
        self._geometry_estimator = GeometryEstimator(config=cfg)

        self._start_visibility_run()
        self._log_event(
            "pipeline_init",
            use_identity=USE_IDENTITY,
            use_bidirectional=self.use_bidirectional,
        )

    @staticmethod
    def _affine_to_sim2(M_inv_2x3: np.ndarray) -> SIM2Transform:
        """Convert 2x3 affine matrix to SIM2Transform.

        Decomposes affine matrix into rotation, translation, scale.

        Args:
            M_inv_2x3: 2x3 affine matrix

        Returns:
            SIM2Transform
        """
        # Extract rotation and scale from 2x2 part
        R = M_inv_2x3[:, :2]
        # Compute scale as average of column norms
        scale = (np.linalg.norm(R[:, 0]) + np.linalg.norm(R[:, 1])) / 2.0
        # Normalize rotation matrix
        R_normalized = R / (scale + 1e-8)
        # Extract rotation angle
        theta = np.arctan2(R_normalized[1, 0], R_normalized[0, 0])
        # Extract translation
        tx = M_inv_2x3[0, 2]
        ty = M_inv_2x3[1, 2]

        return SIM2Transform(theta=theta, tx=tx, ty=ty, scale=scale)

    @staticmethod
    def _sim2_to_affine(T: SIM2Transform) -> np.ndarray:
        """Convert SIM2Transform to 2x3 affine matrix.

        Args:
            T: SIM2Transform

        Returns:
            2x3 affine matrix
        """
        c, s = np.cos(T.theta), np.sin(T.theta)
        return np.array([
            [T.scale * c, -T.scale * s, T.tx],
            [T.scale * s,  T.scale * c, T.ty]
        ], dtype=np.float64)


    @staticmethod
    def _safe_mean_confidence(intrinsic_conf: Optional[np.ndarray]) -> float:
        """Return a scalar confidence value from a confidence map or scalar."""
        if intrinsic_conf is None:
            return 0.0
        arr = np.asarray(intrinsic_conf, dtype=np.float32)
        if arr.size == 0:
            return 0.0
        return float(np.mean(arr))

    # ─── Visibility / Logging Helpers ──────────────────────────────────────

    def _start_visibility_run(self) -> None:
        """Start a fresh per-clip visibility/logging run."""
        self._run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._visibility_dir = Path("output/face_os/visibility")
        self._visibility_dir.mkdir(parents=True, exist_ok=True)
        self._run_log_path = self._visibility_dir / f"pipeline_{self._run_id}.jsonl"
        self._summary_log_path = self._visibility_dir / f"pipeline_{self._run_id}_summary.json"

        self._logger = logging.getLogger("face_os.pipeline")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    def _json_default(self, obj):
        """Safe JSON serialization for numpy / custom objects."""
        if isinstance(obj, np.ndarray):
            return {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "min": float(np.min(obj)) if obj.size else 0.0,
                "max": float(np.max(obj)) if obj.size else 0.0,
            }
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _append_visibility_record(self, record: dict) -> None:
        """Append one JSONL record to the run log."""
        line = json.dumps(record, default=self._json_default, ensure_ascii=False)
        with self._run_log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _log_event(self, event: str, **payload) -> None:
        """Human log + JSONL log."""
        record = {
            "ts": time.time(),
            "run_id": getattr(self, "_run_id", "unknown"),
            "event": event,
            **payload,
        }
        self._append_visibility_record(record)
        self._logger.info("%s | %s", event, json.dumps(payload, default=self._json_default, ensure_ascii=False))

    def _write_run_summary(self, status: str, output_path: Optional[str] = None) -> None:
        """Write a final per-run summary JSON."""
        summary = {
            "ts": time.time(),
            "run_id": getattr(self, "_run_id", "unknown"),
            "status": status,
            "output_path": output_path,
            "telemetry": self.get_telemetry_report(),
            "frames_logged": len(self._frame_telemetry_log),
        }
        self._summary_log_path.write_text(
            json.dumps(summary, indent=2, default=self._json_default, ensure_ascii=False),
            encoding="utf-8",
        )
        self._logger.info("run_summary_written | %s", str(self._summary_log_path))

    def _inject_detail_residual(
        self,
        rendered_bgr: np.ndarray,
        intrinsic_components: Optional["IntrinsicComponents"],
        face_mask: Optional[np.ndarray] = None,
        strength: float = 0.30,
    ) -> np.ndarray:
        """
        Re-inject high-frequency detail from intrinsic decomposition.
        This preserves sharpness without breaking the smooth base render.
        """
        if intrinsic_components is None or getattr(intrinsic_components, "detail_residual", None) is None:
            return rendered_bgr

        base = np.asarray(rendered_bgr, dtype=np.float32)
        if base.max(initial=0.0) > 1.5:
            base = base / 255.0
        base = np.clip(base, 0.0, 1.0)

        detail = np.asarray(intrinsic_components.detail_residual, dtype=np.float32)
        if detail.shape[:2] != base.shape[:2]:
            detail = cv2.resize(detail, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)

        if face_mask is not None and face_mask.shape[:2] == base.shape[:2]:
            mask = np.asarray(face_mask, dtype=np.float32)
            mask = np.clip(mask, 0.0, 1.0)[..., np.newaxis]
            detail = detail * mask

        out = np.clip(base + strength * detail, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8)

    def _reinject_source_hf(
        self,
        rendered_bgr: np.ndarray,
        source_bgr: np.ndarray,
        face_mask: Optional[np.ndarray],
        strength: float = 0.35,
    ) -> np.ndarray:
        """Re-inject source high-frequency texture into rendered output.

        Mathematical model:
            source_hf = source - GaussianBlur(source, σ=1.5)
            output = rendered + strength * source_hf * face_mask

        The source frame has HF at native resolution (no warp loss).
        The rendered output has correct low-freq identity + smooth shading.
        Their sum gives identity-accurate LF + source-authentic HF.

        This directly addresses D-01: 'frequency destruction' from single-
        resample canonical warp (256→1920 = ×7.5 magnification).
        """
        if face_mask is not None and face_mask.max() < 0.01:
            # Mask present but empty — no face region to reinject into
            return rendered_bgr

        # Resize source to match rendered if needed
        rh, rw = rendered_bgr.shape[:2]
        src = source_bgr
        if src.shape[:2] != (rh, rw):
            src = cv2.resize(src, (rw, rh), interpolation=cv2.INTER_LINEAR)

        # Extract source HF: σ=1.5 separates texture-scale HF from illumination LF
        src_f32 = src.astype(np.float32)
        src_blur = cv2.GaussianBlur(src_f32, (0, 0), sigmaX=1.5, sigmaY=1.5,
                                    borderType=cv2.BORDER_REFLECT)
        src_hf = src_f32 - src_blur  # zero-mean HF band

        if face_mask is None:
            # No mask: apply HF reinject uniformly across the full crop.
            # This is the correct mode for enhancement path where face_mask is None.
            mask3 = np.ones((rendered_bgr.shape[0], rendered_bgr.shape[1], 1), dtype=np.float32)
        else:
            mask3 = np.clip(face_mask, 0.0, 1.0)[:, :, np.newaxis].astype(np.float32)

        rendered_f32 = rendered_bgr.astype(np.float32)
        result = rendered_f32 + strength * src_hf * mask3
        return np.clip(result, 0, 255).astype(np.uint8)


    def _commit_renderer_mode(
        self,
        intrinsic_components: Optional['IntrinsicComponents'],
        intrinsic_conf: Optional[np.ndarray],
    ) -> Optional[RendererMode]:
        """Commit RendererMode exactly once from orchestration code.

        Render functions must remain pure. This is the single source of truth
        for renderer-mode transitions and telemetry accounting.
        """
        if not USE_IDENTITY or self.renderer_mode_state is None:
            return None

        intrinsic_available = intrinsic_components is not None
        avg_confidence = self._safe_mean_confidence(intrinsic_conf)
        decomposition_error = (
            intrinsic_components.reconstruction_error
            if intrinsic_components is not None
            else 1.0
        )

        prev_mode = self.renderer_mode_state.current_mode
        new_mode = self.renderer_mode_state.update(
            intrinsic_available=intrinsic_available,
            intrinsic_confidence=avg_confidence,
            decomposition_error=decomposition_error,
        )

        if new_mode != prev_mode:
            self._telemetry["renderer_mode_transitions"] = self.renderer_mode_state.transition_count

        self._telemetry["renderer_mode_distribution"][new_mode.value] += 1
        return new_mode

    def _merge_intrinsic_beliefs(
        self,
        current_components: Optional['IntrinsicComponents'],
        current_conf: Optional[np.ndarray],
        reference_components: Optional['IntrinsicComponents'] = None,
        reference_conf: Optional[np.ndarray] = None,
    ):
        """Select the stronger intrinsic belief without mutating either source."""
        current_scalar = self._safe_mean_confidence(current_conf)
        reference_scalar = self._safe_mean_confidence(reference_conf)

        if current_components is None and reference_components is None:
            return None, None

        if current_components is None:
            return reference_components, reference_conf

        if reference_components is None:
            return current_components, current_conf

        if reference_scalar > current_scalar:
            return reference_components, reference_conf

        return current_components, current_conf


    def get_telemetry_report(self) -> dict:
        """Get runtime telemetry report.

        Returns:
            Dictionary with telemetry data and derived metrics
        """
        total = self._telemetry["total_frames"]
        if total == 0:
            return {
                **self._telemetry,
                "physical_render_rate": 0.0,
                "alpha_fallback_rate": 0.0,
                "intrinsic_success_rate": 0.0,
                "intrinsic_failure_rate": 0.0,
                "avg_intrinsic_confidence": 0.0,
                "avg_decomposition_error": 0.0,
                "mesh_normal_rate": 0.0,
                "shading_normal_rate": 0.0,
            }

        # Compute averages
        avg_intrinsic_confidence = 0.0
        if self._telemetry["intrinsic_confidence_count"] > 0:
            avg_intrinsic_confidence = (
                self._telemetry["intrinsic_confidence_sum"]
                / self._telemetry["intrinsic_confidence_count"]
            )

        avg_decomposition_error = 0.0
        if self._telemetry["decomposition_error_count"] > 0:
            avg_decomposition_error = (
                self._telemetry["decomposition_error_sum"]
                / self._telemetry["decomposition_error_count"]
            )

        # Compute fallback reason distribution as fractions
        physical = self._telemetry["physical_render_frames"]
        alpha = self._telemetry["alpha_fallback_frames"]
        total_render = physical + alpha if (physical + alpha) > 0 else 1

        mesh_normal = self._telemetry["mesh_normal_frames"]
        shading_normal = self._telemetry["shading_normal_frames"]
        total_normal = mesh_normal + shading_normal if (mesh_normal + shading_normal) > 0 else 1

        return {
            **self._telemetry,
            "physical_render_rate": physical / total,
            "alpha_fallback_rate": alpha / total,
            "intrinsic_success_rate": self._telemetry["intrinsic_success_frames"] / total,
            "intrinsic_failure_rate": self._telemetry["intrinsic_failure_frames"] / total,
            "avg_intrinsic_confidence": avg_intrinsic_confidence,
            "avg_decomposition_error": avg_decomposition_error,
            "physical_render_fraction": physical / total_render,
            "alpha_fallback_fraction": alpha / total_render,
            "mesh_normal_rate": mesh_normal / total_normal,
            "shading_normal_rate": shading_normal / total_normal,
            # RULE 8: Timing telemetry
            "avg_render_time_ms": (
                self._telemetry["render_time_sum_ms"]
                / max(self._telemetry["render_time_count"], 1)
            ),
            # RULE 7: Energy scaler stats
            "energy_scaler_stats": self.energy_scaler.get_stats(),
        }

    def get_frame_telemetry(self) -> list:
        """Get per-frame telemetry log.

        D-08: Every frame exposes render_path, renderer_mode, fallback_reason,
        intrinsic_used, geometry_source, resample_count, energy_terms, transform_det.

        Returns:
            List of per-frame telemetry dicts
        """
        return self._frame_telemetry_log

    def get_latent_telemetry(self) -> list:
        """Get per-frame LatentRenderTelemetry log.

        D-05 Phase 0: one LatentRenderTelemetry dict per frame (mirrors
        get_frame_telemetry). On legacy frames each record documents the
        current truth (latent_primary=False, source_pixel_fraction=1.0).

        Returns:
            List of per-frame latent telemetry dicts
        """
        return self._latent_telemetry_log

    def enroll(
        self,
        reference_image: str = "expectation.png",
        reference_dir: str = "photos/",
    ) -> bool:
        """Enroll the target identity from reference images.

        1. Load reference images
        2. Extract face embeddings for identity matching
        3. Build initial canonical appearance atlas
        4. Initialize identity belief state
        5. Initialize patch memory
        """
        print("=== FACE OS ENROLLMENT ===")

        # Load references
        primary, all_refs = ingest.load_reference_images(reference_dir, reference_image)
        if primary is None:
            print(f"ERROR: Cannot load reference image: {reference_image}")
            return False

        print(f"  Loaded {len(all_refs)} reference image(s)")

        # Build identity profile
        paths = [reference_image] + [
            str(p) for p in Path(reference_dir).glob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        ]
        self.identity = canonical_map.build_identity_profile(all_refs, paths)

        print(f"  Embeddings: {len(self.identity.embeddings)}")
        print(f"  Atlas enrolled: {self.identity.enrolled}")

        # Initialize modules
        self.tracker = detect_track.FaceTracker(self.identity.embeddings)
        self.appearance_builder = canonical_map.AppearanceFieldBuilder()
        self.crop = crop_planner.CropPlanner(reference_image=reference_image)
        self.compositor = compositor.Compositor()

        # V3: Initialize physical renderer
        self.physical_renderer = PhysicalRenderer()
        # D-10: Wire renderer subsystem wrapper
        self._face_renderer = FaceRenderer(self.physical_renderer, config=cfg)

        # V3: Initialize renderer mode state
        self.renderer_mode_state = RendererModeState()

        # V3: Initialize state evolution model
        self.state_evolution = StateEvolution()
        self._latent_state = np.zeros(11)  # Initial latent state
        self._latent_covariance = np.eye(11)  # Initial covariance
        # D-10: Wire temporal subsystem wrapper
        self._temporal_estimator = TemporalEstimator(self.state_evolution)

        # Extract reference mesh for quality gates
        ref_mesh = detect_track.extract_face_mesh(primary)
        if ref_mesh is not None:
            self.tracker.set_reference_mesh(ref_mesh)
            print(f"  Reference mesh: {ref_mesh.shape[0]} landmarks")
        else:
            print("  WARNING: Could not extract reference mesh — quality gates will be relaxed")

        # NEW: Initialize identity belief state
        if USE_IDENTITY:
            atlas_size = tuple(cfg.canonical.atlas_size) if hasattr(cfg.canonical, 'atlas_size') else (512, 512)
            self.identity_state = IdentityState(atlas_size=atlas_size)
            self.patch_memory = PatchMemory()
            # D-10: Wire identity subsystem wrapper
            self._identity_estimator = IdentityEstimator(self.identity_state)
        else:
            self.identity_state = None
            self.patch_memory = None
            self._identity_estimator = None
            print("  Identity: DISABLED (simple enhancement mode)")

        # Set reference embedding on verification gate
        if USE_IDENTITY and self.identity.embeddings:
            self.identity_state._gate.set_reference_embedding(
                self.identity.embeddings[0]
            )
            # FIX: Read from config dynamically
            print(f"  Verification gate: embedding_tolerance={cfg.identity.embedding_tolerance}, min_face_pixels={cfg.verification_gate.min_face_pixels}, liveness_threshold={cfg.verification_gate.liveness_threshold}")

        # Pre-populate from reference
        if USE_IDENTITY and self.identity.enrolled and self.identity.appearance.atlas_rgb is not None:
            self.appearance_builder.atlas = self.identity.appearance

            # Initialize identity state from reference
            ref_rgb = self.identity.appearance.atlas_rgb
            if ref_rgb is not None:
                h, w = ref_rgb.shape[:2]
                target_h, target_w = atlas_size[1], atlas_size[0]

                # D-01: Resize reference atlas to match config atlas_size
                if (h, w) != (target_h, target_w):
                    ref_rgb = cv2.resize(ref_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                    h, w = target_h, target_w

                quality = np.ones((h, w), dtype=np.float32) * 0.9
                ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)

                # Module D: Set identity anchor from reference
                # This prevents identity drift — all reconstructions must stay close to anchor
                self.identity_state.set_anchor(ref_bgr)
                print(f"  Anchor set from reference (LAB distance threshold: {self.identity_state._anchor_threshold})")

                # D-05 Task 2.1/2.6: Seed the lighting-invariant latent from the
                # same enrollment reference (shadow mode — does not drive render).
                if self._identity_estimator is not None:
                    self._identity_estimator.set_anchor(ref_bgr, enrollment_mesh=ref_mesh)
                    self._last_latent_confidence = float(
                        self._identity_estimator.latent().mean_confidence()
                    )
                    # D-05 Phase 2B: the enrollment-seed confidence IS the gate's
                    # relative-to-floor baseline. The latent only earns the right
                    # to drive the render once it absorbs real-video evidence and
                    # rises above this seed (+margin). Seed prev = floor so the
                    # first frame's spike check sees no artificial drop.
                    self._latent_confidence_floor = self._last_latent_confidence
                    self._prev_latent_confidence = self._last_latent_confidence
                    print(f"  Latent anchor seeded (confidence: {self._last_latent_confidence:.3f})")

                # Pre-populate identity state with MULTIPLE reference observations
                # This gives the identity state a strong starting point
                # Like a Bayesian prior — strong belief from reference
                for _ in range(100):
                    self.identity_state.update(ref_bgr, quality, pose=(0, 0, 0))

                print(f"  Identity pre-populated with 100 reference observations")

        self._enrolled = True
        print("  Enrollment complete.")
        return True

    def process(
        self,
        video_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
    ) -> Optional[str]:
        """Process a video through the full pipeline.

        If use_bidirectional is True:
          1. Forward pass: collect all frames + quality metrics
          2. Bidirectional solve: future frames repair past frames
          3. Final pass: query solved identity for each frame

        If use_bidirectional is False:
          Standard forward-only pass
        """
        if not self._enrolled:
            print("ERROR: Must enroll before processing.")
            return None

        self._log_event(
            "process_start",
            video_path=video_path,
            output_path=output_path,
            max_frames=max_frames,
            bidirectional=self.use_bidirectional,
        )

        print(f"\n=== FACE OS PROCESSING ===")
        print(f"  Input: {video_path}")
        print(f"  Output: {output_path}")
        print(f"  Bidirectional: {self.use_bidirectional}")

        meta = ingest.load_video_meta(video_path)
        print(f"  Video: {meta.width}x{meta.height} @ {meta.fps:.1f}fps ({meta.total_frames} frames)")

        # Reset per-clip state
        self._reset_state()

        try:
            if not USE_IDENTITY:
                print("  Mode: SIMPLE ENHANCEMENT (no identity, no bidirectional)")
                result = self._process_forward(video_path, output_path, max_frames, meta)
            elif self.use_bidirectional:
                result = self._process_bidirectional(video_path, output_path, max_frames, meta)
            else:
                result = self._process_forward(video_path, output_path, max_frames, meta)

            self._log_event("process_end", result=result)
            self._write_run_summary("completed", output_path=result)
            return result
        except Exception as e:
            self._log_event("process_error", error=str(e))
            self._write_run_summary("failed", output_path=output_path)
            raise

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> dict:
        """Process a single frame and return result dict.

        Public API for A/B validation and testing.
        Wraps _process_frame_v2 with face detection.

        Args:
            frame: Input frame (H, W, 3) uint8 BGR
            frame_idx: Frame index

        Returns:
            Dict with keys: 'frame' (output ndarray), 'landmarks', 'transform', 'render_path'
        """
        # Process through forward path v2
        timestamp = float(frame_idx) / 30.0
        output = self._process_frame_v2(frame, frame_idx, timestamp)
        self._telemetry["total_frames"] += 1

        # Get landmarks from tracker if target track exists
        face_track = self.tracker._get_target_track()
        landmarks = face_track.mesh_478 if face_track and hasattr(face_track, 'mesh_478') else None

        # Get last render path from telemetry
        render_path = "enhancement"
        if self._frame_telemetry_log:
            render_path = self._frame_telemetry_log[-1].get("render_path", "enhancement")

        return {
            'frame': output if output is not None else frame,
            'landmarks': landmarks,
            'transform': self._last_SIM2,
            'render_path': render_path,
        }

    def _process_forward(
        self,
        video_path: str,
        output_path: str,
        max_frames: Optional[int],
        meta: VideoMeta,
    ) -> Optional[str]:
        """Standard forward-only processing."""
        # Open exporter
        output_w = cfg.crop.output_size[0] if hasattr(cfg.crop, 'output_size') else 1080
        output_h = cfg.crop.output_size[1] if hasattr(cfg.crop, 'output_size') else 1920
        exporter = export_qc.VideoExporter(
            output_path, fps=cfg.export.fps,
            width=output_w, height=output_h,
            source_path=video_path,
        )

        face_detected_frames = 0
        total_frames = 0
        all_frames = []
        t_start = time.perf_counter()

        try:
            for frame_idx, timestamp, source_frame in ingest.frame_reader(video_path):
                if max_frames and total_frames >= max_frames:
                    break

                result = self._process_frame_v2(source_frame, frame_idx, timestamp)
                self._telemetry["total_frames"] += 1

                if result is not None:
                    exporter.write_frame(result)
                    all_frames.append(result)
                    if self.tracker and self.tracker.tracks:
                        face_detected_frames += 1

                total_frames += 1

                if total_frames % 100 == 0:
                    elapsed = time.perf_counter() - t_start
                    fps_actual = total_frames / max(elapsed, 0.001)
                    print(f"  {total_frames} frames ({fps_actual:.0f} fps)")

        except Exception as e:
            print(f"  ERROR at frame {total_frames}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            exporter.close()

        elapsed = time.perf_counter() - t_start

        # Post-processing
        self._post_process(output_path, video_path, all_frames, face_detected_frames, total_frames, elapsed)
        return output_path

    def _process_bidirectional(
        self,
        video_path: str,
        output_path: str,
        max_frames: Optional[int],
        meta: VideoMeta,
    ) -> Optional[str]:
        """Bidirectional processing — the offline superpower.

        Pass 1 (forward): Collect all canonical faces + quality metrics
        Pass 2 (solve): Bidirectional temporal solve
        Pass 3 (render): Query solved identity for each frame
        """
        self.temporal_solver = TemporalRepairEngine(
            lookback=cfg.temporal.temporal_window if hasattr(cfg.temporal, 'temporal_window') else 10,
            lookahead=cfg.temporal.temporal_window if hasattr(cfg.temporal, 'temporal_window') else 10,
        )

        # === PASS 1: Forward collection ===
        print("  Pass 1/3: Forward collection...")
        canonical_faces = {}  # frame_idx → canonical face
        quality_maps = {}     # frame_idx → quality map
        frame_data = {}       # frame_idx → (source_frame, face_track, landmarks, crop_plan)
        total_frames = 0
        t_start = time.perf_counter()

        for frame_idx, timestamp, source_frame in ingest.frame_reader(video_path):
            if max_frames and total_frames >= max_frames:
                break

            # Detect + landmarks + canonical
            face_track = self.tracker.process_frame(source_frame, frame_idx)
            landmarks = None
            if face_track and face_track.smooth_bbox:
                landmarks = lm_module.extract_landmarks(source_frame, face_track.mesh_478)
                face_track.landmarks = landmarks

            crop_plan = self.crop.plan_crop(source_frame.shape[:2], face_track, landmarks)

            # Track last good crop plan for fallback in pass 3
            if crop_plan is not None:
                self._last_good_crop_plan = crop_plan

            if landmarks and face_track.detection:
                # Warp to canonical space
                try:
                    warped_rgb, warped_lab, M = canonical_map.warp_to_canonical(
                        source_frame, landmarks,
                        canonical_size=tuple(cfg.canonical.atlas_size),
                    )
                    warped_bgr = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)

                    # Compute quality map
                    quality = self._compute_quality_map(warped_bgr, face_track.detection.confidence)

                    # Compute sharpness
                    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
                    sharpness = float(np.mean(np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))))
                    sharpness = np.clip(sharpness / 100.0, 0, 1)

                    # Store for bidirectional solver
                    canonical_faces[frame_idx] = warped_bgr
                    quality_maps[frame_idx] = quality

                    fq = FrameQuality(
                        frame_idx=frame_idx,
                        sharpness=sharpness,
                        motion_blur=0.0,
                        pose=(landmarks.yaw, landmarks.pitch, landmarks.roll),
                        detection_confidence=face_track.detection.confidence,
                    )
                    self.temporal_solver.collect_frame(
                        frame_idx, warped_bgr, quality,
                        sharpness=sharpness,
                        pose=(landmarks.yaw, landmarks.pitch, landmarks.roll),
                        detection_confidence=face_track.detection.confidence,
                    )

                    frame_data[frame_idx] = (source_frame, face_track, landmarks, crop_plan)

                except Exception:
                    frame_data[frame_idx] = (source_frame, face_track, landmarks, crop_plan)

            total_frames += 1

            if total_frames % 100 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"    {total_frames} frames ({total_frames / max(elapsed, 0.001):.0f} fps)")

        print(f"    Collected {len(canonical_faces)} canonical faces from {total_frames} frames")

        # === PASS 2: Bidirectional solve ===
        print("  Pass 2/3: Bidirectional temporal solve...")
        solved_faces = self.temporal_solver.solve()
        hq_count = self.temporal_solver.solver.get_hq_frame_count()
        print(f"    Solved {len(solved_faces)} frames, {hq_count} HQ frames")

        # Save forward-pass intrinsic belief for per-frame selection
        # D-06: Per-frame belief storage (not global mutable cache)
        self._frame_beliefs = {}
        self._forward_intrinsic_components = None
        self._forward_intrinsic_conf = None
        if self.identity_state is not None and self.identity_state._intrinsic_components is not None:
            self._forward_intrinsic_components = self.identity_state._intrinsic_components
            self._forward_intrinsic_conf = float(np.mean(self.identity_state.belief.get_confidence())) if self.identity_state.belief is not None else 0.0

        # Update identity state with solved faces (bidirectional refinement)
        for idx, (solved_face, solved_conf) in solved_faces.items():
            # D-04: Pass mesh_478 and warp_M for geometry-derived normals
            mesh_478 = None
            warp_M = None
            if idx in frame_data:
                _, face_track, landmarks, _ = frame_data[idx]
                if face_track is not None:
                    mesh_478 = getattr(face_track, 'mesh_478', None)
                if landmarks is not None:
                    try:
                        _, _, M = canonical_map.warp_to_canonical(
                            solved_face, landmarks,
                            canonical_size=tuple(cfg.canonical.atlas_size),
                        )
                        warp_M = M[:2] if M is not None else None
                    except Exception:
                        pass

            self.identity_state.update(
                solved_face, solved_conf, pose=None,
                mesh_478=mesh_478, warp_M=warp_M,
            )
            if idx in frame_data:
                _, _, landmarks, _ = frame_data[idx]
                if landmarks:
                    pose = (landmarks.yaw, landmarks.pitch, landmarks.roll)
                    self.patch_memory.update(solved_face, solved_conf, pose=pose, frame_idx=idx)

        # === PASS 3: Render ===
        print("  Pass 3/3: Rendering...")
        output_w = cfg.crop.output_size[0] if hasattr(cfg.crop, 'output_size') else 1080
        output_h = cfg.crop.output_size[1] if hasattr(cfg.crop, 'output_size') else 1920
        exporter = export_qc.VideoExporter(
            output_path, fps=cfg.export.fps,
            width=output_w, height=output_h,
            source_path=video_path,
        )

        all_frames = []
        face_detected_frames = 0

        # Re-read video for rendering
        for frame_idx, timestamp, source_frame in ingest.frame_reader(video_path):
            if max_frames and frame_idx >= max_frames:
                break

            if frame_idx in frame_data:
                _, face_track, landmarks, crop_plan = frame_data[frame_idx]

                # Get solved canonical face
                solved_face = None
                solved_conf = None
                if frame_idx in solved_faces:
                    solved_face, solved_conf = solved_faces[frame_idx]

                # D-06: Update renderer mode in orchestration layer (not in render)
                # This is the single source of truth for mode transitions.
                # Use the current solved-frame belief per frame, then compare against
                # the stored forward-pass snapshot if it is stronger.
                if USE_IDENTITY and self.identity_state is not None and self.identity_state.is_initialized():
                    quality_map = self._compute_quality_map(
                        solved_face,
                        face_track.detection.confidence if face_track and face_track.detection else 0.5,
                    )
                    current_intrinsic_components, current_intrinsic_conf = self.identity_state.query_intrinsic(quality_map)

                    selected_components = current_intrinsic_components
                    selected_conf = current_intrinsic_conf

                    if self._forward_intrinsic_components is not None and self._forward_intrinsic_conf is not None:
                        forward_conf_map = None
                        if current_intrinsic_conf is not None and current_intrinsic_conf.size > 0:
                            forward_conf_map = np.ones_like(current_intrinsic_conf, dtype=np.float32) * self._forward_intrinsic_conf
                        elif current_intrinsic_components is not None:
                            h, w = current_intrinsic_components.albedo.shape[:2]
                            forward_conf_map = np.ones((h, w), dtype=np.float32) * self._forward_intrinsic_conf

                        selected_components, selected_conf = self._merge_intrinsic_beliefs(
                            current_intrinsic_components,
                            current_intrinsic_conf,
                            self._forward_intrinsic_components,
                            forward_conf_map,
                        )

                    # Persist per-frame belief for debugging / telemetry inspection
                    self._frame_beliefs[frame_idx] = {
                        "intrinsic": selected_components,
                        "confidence": self._safe_mean_confidence(selected_conf),
                    }

                    # Commit renderer mode ONCE per frame from the selected belief
                    self._commit_renderer_mode(selected_components, selected_conf)

                    # Keep V3 telemetry/state evolution in sync
                    self._update_v3_modules(selected_components, selected_conf, frame_idx)

                # Render frame
                result = self._render_frame_v2(
                    source_frame, frame_idx, face_track, landmarks, crop_plan,
                    solved_face=solved_face, solved_conf=solved_conf,
                )

                if result is not None:
                    exporter.write_frame(result)
                    all_frames.append(result)
                    face_detected_frames += 1
            else:
                # No face detected — apply last known crop to maintain frame size
                if frame_idx in frame_data:
                    _, _, _, crop_plan = frame_data[frame_idx]
                elif self._last_good_crop_plan is not None:
                    crop_plan = self._last_good_crop_plan
                else:
                    crop_plan = self.crop.plan_crop(source_frame.shape[:2], None, None)
                cropped = crop_planner.apply_crop(source_frame, crop_plan)
                exporter.write_frame(cropped)
                all_frames.append(cropped)

        exporter.close()
        elapsed = time.perf_counter() - t_start

        self._post_process(output_path, video_path, all_frames, face_detected_frames, total_frames, elapsed)
        return output_path

    def _process_frame_v2(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
    ) -> Optional[np.ndarray]:
        """Process a single frame through the v2 pipeline.

        Flow:
          1. Detect & track (telemetry)
          2. Landmarks + pose (face telemetry)
          3. Canonical alignment (standard face space)
          4. Quality map computation
          5. Identity state update (belief update)
          6. Patch memory update
          7. Query identity (what does this region usually look like?)
          8. Crop planning
          9. Render (structure-preserving)
          10. Composite
        """
        self._frame_count = frame_idx

        # 1. Detect & track
        face_track = self.tracker.process_frame(frame, frame_idx)

        # 2. Landmarks + pose
        landmarks = None
        if face_track and face_track.smooth_bbox:
            landmarks = lm_module.extract_landmarks(frame, face_track.mesh_478)
            face_track.landmarks = landmarks

        # ═══════════════════════════════════════════════════════════════════
        # FACE LOCK STATE MACHINE
        # ═══════════════════════════════════════════════════════════════════
        face_detected = face_track is not None and landmarks is not None
        detection_conf = face_track.detection.confidence if face_track and face_track.detection else 0.0

        # Compute occupancy estimate
        occupancy = 0.0
        if face_detected and face_track.smooth_bbox:
            x, y, w, h = face_track.smooth_bbox
            bbox_area = w * h
            # Estimate face occupancy from landmarks spread
            if hasattr(landmarks, 'points') and landmarks.points is not None:
                pts = np.array(landmarks.points)
                hull_area = cv2.contourArea(cv2.convexHull(pts.astype(np.float32)))
                occupancy = hull_area / max(bbox_area, 1)

        # State transitions
        if face_detected and occupancy > 0.25 and detection_conf > 0.5:
            if self._face_state == "LOST_FACE":
                self._face_state = "RECOVERY"
                self._recovery_frame_count = 0
                print(f"  Frame {frame_idx}: RECOVERY — face returned (occ={occupancy:.2f}, conf={detection_conf:.2f})")
            elif self._face_state == "RECOVERY":
                self._recovery_frame_count += 1
                if self._recovery_frame_count > 5:
                    self._face_state = "FACE_LOCKED"
                    print(f"  Frame {frame_idx}: FACE_LOCKED — stable (occ={occupancy:.2f})")
            else:
                self._face_state = "FACE_LOCKED"
            self._lost_frame_count = 0
        else:
            self._lost_frame_count += 1
            if self._face_state != "LOST_FACE":
                self._face_state = "LOST_FACE"
                print(f"  Frame {frame_idx}: LOST_FACE — no valid detection (occ={occupancy:.2f}, conf={detection_conf:.2f})")

        # Log state periodically
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}: state={self._face_state}, occ={occupancy:.2f}, conf={detection_conf:.2f}")

        # ═══════════════════════════════════════════════════════════════════
        # GATE: Skip identity update if face is lost
        # ═══════════════════════════════════════════════════════════════════
        if self._face_state == "LOST_FACE":
            # H-02: Use predicted SIM(2) for short occlusion recovery (1-2 frames)
            if (self._predicted_SIM2 is not None
                    and self._last_SIM2 is not None
                    and self._lost_frame_count <= 2
                    and self.identity_state is not None
                    and self.identity_state.is_initialized()):
                try:
                    M_predicted = self._sim2_to_affine(self._predicted_SIM2)
                    crop_plan = self.crop.plan_crop(frame.shape[:2], None, None)
                    output = crop_planner.apply_crop(frame, crop_plan)
                    self._emit_frame_telemetry(
                        frame_idx, "face_lost_predicted", None, {}, 0, 0,
                        render_path="enhancement",
                        intrinsic_used=False,
                        geometry_source="predicted_sim2",
                        resample_count=0,
                        transform_det=float(self._predicted_SIM2.scale ** 2),
                    )
                    return output
                except Exception:
                    pass

            crop_plan = self.crop.plan_crop(frame.shape[:2], None, None)
            output = crop_planner.apply_crop(frame, crop_plan)
            self._emit_frame_telemetry(
                frame_idx, "face_lost", None, {}, 0, 0,
                render_path="enhancement",
                intrinsic_used=False,
                geometry_source="none",
                resample_count=0,
                transform_det=1.0,
            )
            return output

        # 3. Canonical alignment
        pose = None  # FIX: Initialize pose to prevent UnboundLocalError
        canonical_face = None
        quality_map = None
        canonical_face_mask = None  # Face mask in canonical space
        if landmarks and face_track.detection:
            try:
                warped_rgb, warped_lab, M = canonical_map.warp_to_canonical(
                    frame, landmarks,
                    canonical_size=tuple(cfg.canonical.atlas_size),
                )
                canonical_face = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)
                quality_map = self._compute_quality_map(canonical_face, face_track.detection.confidence)

                # Create face mask in canonical space
                # Use landmarks.points (not xy) to create convex hull, then warp to canonical
                if hasattr(landmarks, 'points') and landmarks.points is not None:
                    pts = np.array(landmarks.points, dtype=np.int32)
                    hull = cv2.convexHull(pts)
                    src_mask = np.zeros(frame.shape[:2], dtype=np.float32)
                    cv2.fillConvexPoly(src_mask, hull, 1.0)
                    src_mask = cv2.GaussianBlur(src_mask, (15, 15), 5)
                    # Warp to canonical space
                    canonical_face_mask = cv2.warpAffine(
                        src_mask, M[:2], (256, 256),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    canonical_face_mask = np.clip(canonical_face_mask, 0, 1)
            except Exception:
                pass

        # 4. Identity state update (skip if USE_IDENTITY=False)
        # geom_state is built by the shadow-update block below and REUSED by the
        # Phase 2 latent render path (one geometry truth per frame). Initialize
        # to None so it is always defined even when the update is skipped.
        geom_state = None
        if USE_IDENTITY and canonical_face is not None and quality_map is not None and face_track is not None:
            pose = (landmarks.yaw, landmarks.pitch, landmarks.roll) if landmarks else None
            # Mask quality_map to face region only — prevent background learning
            masked_quality = quality_map * canonical_face_mask if canonical_face_mask is not None else quality_map

            # GATE: Skip update if face mask is too small
            if canonical_face_mask is not None and canonical_face_mask.sum() < 100:
                # Face mask too small — skip update
                pass
            else:
                # Get verification parameters from track
                face_bbox = face_track.smooth_bbox
                
                # FIX: Robustly check for 478 (V4) or 468 (legacy) mesh attributes
                mesh = getattr(face_track, 'mesh_478', None)
                landmarks_pts = mesh[:, :2] if mesh is not None else (landmarks.points[:, :2] if landmarks and hasattr(landmarks, 'points') else None)
                
                embedding = face_track.detection.embedding if face_track.detection else None

                # Update with verification gate
                mesh_478 = getattr(face_track, 'mesh_478', None)
                identity_updated = self.identity_state.update(
                    canonical_face, masked_quality, pose=pose,
                    face_bbox=face_bbox,
                    landmarks_pts=landmarks_pts,
                    embedding=embedding,
                    mesh_478=mesh_478,
                    warp_M=M[:2] if M is not None else None,
                )

                # D-05 Task 2.6: SHADOW-MODE latent update.
                # The Geometry subsystem packages the geometry we ALREADY
                # extracted (no re-detection — one geometry truth per frame).
                # The Identity subsystem fuses this observation into its
                # lighting-invariant latent via uncertainty-weighted fusion.
                # This populates the latent and drives latent_confidence
                # telemetry, but DOES NOT drive the render path yet (Phase 2).
                #
                # Only fuse when identity_state.update() ACCEPTED the frame
                # (verification gate passed): a gate-rejected observation must
                # not enter the identity latent, AND its decomposition would be
                # stale. Reusing the decomposition update() just computed avoids
                # a redundant second decompose of the same canonical_face.
                # Shadow mode must never crash the pipeline.
                if (
                    self._identity_estimator is not None
                    and self._latent_shadow_enabled
                    and identity_updated
                    and self.identity_state._intrinsic_components is not None
                ):
                    try:
                        geom_state = self._geometry_estimator.assemble_state(
                            canonical_face=canonical_face,
                            canonical_transform=M,
                            mask=canonical_face_mask,
                            mesh=mesh_478,
                            landmarks=landmarks,
                            pose=pose if pose is not None else (0.0, 0.0, 0.0),
                            geometry_confidence=(
                                face_track.detection.confidence
                                if face_track.detection else 0.0
                            ),
                        )
                        latent = self._identity_estimator.update_latent(
                            canonical_face, geom_state, masked_quality,
                            intrinsic=self.identity_state._intrinsic_components,
                        )
                        self._last_latent_confidence = float(
                            latent.mean_confidence()
                        )
                    except Exception as exc:  # noqa: BLE001 — shadow never crashes
                        self._log_event("latent_shadow_update_failed", error=str(exc))

        # 5. Patch memory update (skip if USE_IDENTITY=False)
        if USE_IDENTITY and canonical_face is not None and quality_map is not None and landmarks:
            pose = (landmarks.yaw, landmarks.pitch, landmarks.roll)
            # Detect blink for eye freeze
            is_blink = self._detect_blink(landmarks) if landmarks else False
            self.patch_memory.update(canonical_face, quality_map, pose=pose, is_blink=is_blink, frame_idx=frame_idx)

        # 6. Query identity (skip if USE_IDENTITY=False)
        identity_face = None
        identity_confidence = None
        intrinsic_components = None
        intrinsic_conf = None
        if USE_IDENTITY and canonical_face is not None and quality_map is not None:
            identity_face, identity_confidence = self.identity_state.query(canonical_face, quality_map, pose=pose)
            # Mask confidence to face region only — prevent background reconstruction
            if canonical_face_mask is not None and identity_confidence is not None:
                identity_confidence = identity_confidence * canonical_face_mask

            # D-05: Query lighting-invariant albedo for forward path (via subsystem wrapper)
            albedo_face, albedo_conf = self._identity_estimator.query_albedo(quality_map)
            if self.render_source == 'legacy' and albedo_face is not None and albedo_conf is not None:
                albedo_weight = float(np.mean(albedo_conf)) * 0.4
                identity_face = (1 - albedo_weight) * identity_face + albedo_weight * albedo_face

            # V3: Query intrinsic components
            intrinsic_components, intrinsic_conf = self.identity_state.query_intrinsic(quality_map)

        # 7. Crop planning
        crop_plan = self.crop.plan_crop(frame.shape[:2], face_track, landmarks)
        cropped = crop_planner.apply_crop(frame, crop_plan)

        # 8. Get region masks
        face_mask = None
        region_masks = None
        if landmarks:
            adjusted_lm = self._adjust_landmarks_to_crop(landmarks, crop_plan)
            if adjusted_lm:
                region_masks = lm_module.create_region_masks(adjusted_lm, cropped.shape[:2])
                face_mask = region_masks.get("face")

        # V3: Commit renderer mode from the current frame belief (orchestration layer)
        self._commit_renderer_mode(intrinsic_components, intrinsic_conf)

        # V3: Update telemetry + temporal state (no renderer-mode mutation here)
        self._update_v3_modules(intrinsic_components, intrinsic_conf, frame_idx, landmarks=landmarks, cropped=cropped)

        # 9. Get identity eyes for structure-preserving rendering (skip if USE_IDENTITY=False)
        identity_eyes = None
        eye_confidence = 0.0
        if USE_IDENTITY and self.patch_memory and self.patch_memory._initialized and landmarks:
            pose = (landmarks.yaw, landmarks.pitch, landmarks.roll)
            left_eye, left_conf = self.patch_memory.query_region('left_eye', pose)
            right_eye, right_conf = self.patch_memory.query_region('right_eye', pose)
            if left_eye is not None and right_eye is not None:
                identity_eyes = left_eye
                eye_confidence = (left_conf + right_conf) / 2

        # 10. Render — shared rendering core (PhysicalRenderer → identity composite → enhance)
        output = self._render_core(
            cropped=cropped,
            source_frame=frame,
            intrinsic_components=intrinsic_components,
            intrinsic_conf=intrinsic_conf,
            identity_face=identity_face,
            landmarks=landmarks,
            crop_plan=crop_plan,
            region_masks=region_masks,
            face_mask=face_mask,
            frame_idx=frame_idx,
            identity_eyes=identity_eyes,
            eye_confidence=eye_confidence,
            geom_state=geom_state,
        )

        return output

    def _render_frame_v2(
        self,
        source_frame: np.ndarray,
        frame_idx: int,
        face_track: Optional[FaceTrack],
        landmarks: Optional[Landmarks],
        crop_plan: CropPlan,
        solved_face: Optional[np.ndarray] = None,
        solved_conf: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Render a frame.

        When USE_IDENTITY=True: identity reconstruction with anchor correction.
        When USE_IDENTITY=False: simple enhancement (crop + sharpen + denoise).
        """
        # Track total frames for telemetry
        self._telemetry["total_frames"] += 1

        # Apply crop
        cropped = crop_planner.apply_crop(source_frame, crop_plan)

        # Get region masks
        face_mask = None
        region_masks = None
        if landmarks:
            adjusted_lm = self._adjust_landmarks_to_crop(landmarks, crop_plan)
            if adjusted_lm:
                region_masks = lm_module.create_region_masks(adjusted_lm, cropped.shape[:2])
                face_mask = region_masks.get("face")

        # ─── SIMPLE ENHANCEMENT MODE (no identity) ───────────────────────
        # RULE C: All rendering must go through _render_core
        if not USE_IDENTITY:
            output = self._render_core(
                cropped=cropped,
                source_frame=source_frame,
                intrinsic_components=None,
                intrinsic_conf=None,
                identity_face=None,
                landmarks=landmarks,
                crop_plan=crop_plan,
                region_masks=region_masks,
                face_mask=face_mask,
                frame_idx=frame_idx,
            )
            return output

        # ─── IDENTITY RECONSTRUCTION MODE ────────────────────────────────
        # If we have a solved canonical face, warp it back to source space
        if solved_face is not None and landmarks is not None:
            try:
                # Module D: Query identity state for anchor-corrected appearance
                if self.identity_state.is_initialized():
                    # Compute quality map for current frame
                    quality_map = self._compute_quality_map(solved_face, face_track.detection.confidence if face_track and face_track.detection else 0.5)
                    
                    # V3: Query intrinsic components (via subsystem wrapper)
                    intrinsic_components, intrinsic_conf = self._identity_estimator.query_intrinsic(quality_map)
                    
                    # D-05: Query lighting-invariant albedo (via subsystem wrapper)
                    albedo_face, albedo_conf = self._identity_estimator.query_albedo(quality_map)
                    
                    # Render via shared _render_core
                    # NOTE: Mode update happens in orchestration layer, not here
                    # D-05: Use albedo as primary identity, fall back to RGB query
                    identity_face, identity_conf = self.identity_state.query_identity(quality_map)
                    # Blend albedo into identity face for lighting invariance (legacy only)
                    if self.render_source == 'legacy' and albedo_face is not None and albedo_conf is not None:
                        albedo_weight = float(np.mean(albedo_conf)) * 0.4
                        identity_face = (1 - albedo_weight) * identity_face + albedo_weight * albedo_face

                    # Use _render_core for PhysicalRenderer → identity composite → enhance
                    output = self._render_core(
                        cropped=cropped,
                        source_frame=source_frame,
                        intrinsic_components=intrinsic_components,
                        intrinsic_conf=intrinsic_conf,
                        identity_face=identity_face,
                        landmarks=landmarks,
                        crop_plan=crop_plan,
                        region_masks=region_masks,
                        face_mask=face_mask,
                        frame_idx=frame_idx,
                    )

                    return output

            except Exception as e:
                print(f"  Frame {frame_idx}: IDENTITY PATH FAILED: {e}")
                # D-01: Track identity path failures in telemetry
                self._telemetry["identity_path_failures"] = self._telemetry.get("identity_path_failures", 0) + 1
                pass

        # Fallback: route through _render_core (RULE C: single render core)
        output = self._render_core(
            cropped=cropped,
            source_frame=source_frame,
            intrinsic_components=None,
            intrinsic_conf=None,
            identity_face=None,
            landmarks=landmarks,
            crop_plan=crop_plan,
            region_masks=region_masks,
            face_mask=face_mask,
            frame_idx=frame_idx,
        )

        return output

    def _composite_identity_to_crop(
        self,
        cropped: np.ndarray,
        identity_face: np.ndarray,
        landmarks: Landmarks,
        crop_plan: CropPlan,
        frame_idx: int,
    ) -> np.ndarray:
        """Warp identity face from canonical space to crop space and composite.

        Shared rendering core — called by BOTH _process_frame_v2() and _render_frame_v2().
        Uses LieGroup SIM(2) geodesic interpolation for M_inv smoothing.
        Uses geometry-based canonical mask (brightness-invariant).
        """
        adjusted_lm = self._adjust_landmarks_to_crop(landmarks, crop_plan)
        M, _ = canonical_map.compute_alignment(
            adjusted_lm,
            canonical_size=tuple(cfg.canonical.atlas_size),
        )
        M_inv = np.linalg.inv(M)[:2]

        current_sim2 = self._affine_to_sim2(M_inv)
        if self._last_SIM2 is not None:
            interpolated = interpolate_sim2(self._last_SIM2, current_sim2, 0.6)
            M_inv = self._sim2_to_affine(interpolated)
        self._last_SIM2 = current_sim2
        self._last_transform_det = current_sim2.scale ** 2

        # D-01b: Single-resample — combine identity + mask into one warp
        h, w = cropped.shape[:2]
        canonical_mask = self._make_canonical_geometry_mask(identity_face.shape[:2])
        identity_with_mask = np.concatenate([
            identity_face,
            canonical_mask[:, :, np.newaxis]
        ], axis=2)  # (H, W, 4)

        warped = cv2.warpAffine(
            identity_with_mask, M_inv, (w, h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
        )
        identity_in_crop = warped[:, :, :3]
        aligned_mask = np.clip(warped[:, :, 3], 0, 1)

        blend_3d = aligned_mask[:, :, np.newaxis]
        # D-01: Linear-light compositing (physically correct gamma handling)
        # D-01c: Multi-band blending when configured
        blend_mode = self._resolve_blend_mode()
        if blend_mode == "laplacian":
            output = multiband_blend(cropped, identity_in_crop, aligned_mask)
        else:
            output = _blend_linear(cropped, identity_in_crop, aligned_mask)

        if frame_idx % 30 == 0:
            face_pixels = aligned_mask > 0.5
            face_blend = float(blend_3d[face_pixels].mean()) if face_pixels.sum() > 0 else 0.0
            print(f"  Frame {frame_idx}: identity composite face_mean={face_blend:.3f}")

        return output

    def _update_v3_modules(
        self,
        intrinsic_components: Optional['IntrinsicComponents'],
        intrinsic_conf: Optional[np.ndarray],
        frame_idx: int,
        landmarks: Optional[Landmarks] = None,
        cropped: Optional[np.ndarray] = None,
    ) -> None:
        """Shared V3 module updates — single source of truth.

        Called by BOTH _process_frame_v2() and _render_frame_v2().

        Eliminates duplicate telemetry tracking. Renderer mode is committed
        only by orchestration code, not here.

        RULE 1: No rendering logic outside _render_core.
        RULE 8: All telemetry tracked here.
        """
        # Track intrinsic decomposition telemetry
        if intrinsic_components is not None:
            self._telemetry["intrinsic_success_frames"] += 1
            avg_conf = float(np.mean(intrinsic_conf)) if intrinsic_conf is not None else 0.0
            self._telemetry["intrinsic_confidence_sum"] += avg_conf
            self._telemetry["intrinsic_confidence_count"] += 1
            self._telemetry["decomposition_error_sum"] += intrinsic_components.reconstruction_error
            self._telemetry["decomposition_error_count"] += 1
        else:
            self._telemetry["intrinsic_failure_frames"] += 1
            reason = "identity_not_initialized" if not self.identity_state.is_initialized() else "decomposition_failed"
            self._telemetry["intrinsic_failure_reasons"][reason] = (
                self._telemetry["intrinsic_failure_reasons"].get(reason, 0) + 1
            )

        # Normal source tracking moved to _render_with_physical_renderer (real runtime telemetry)

        # Renderer mode is committed only by orchestration code via _commit_renderer_mode().

        # Update state evolution model — D-06: Full Kalman predict-update cycle
        if USE_IDENTITY and self.state_evolution is not None and self._latent_state is not None:
            # Extract observations from current frame
            observation = self._latent_state.copy()  # Start with current state as default
            
            # Pose observations (yaw, pitch, roll) from landmarks
            if landmarks is not None:
                observation[0] = landmarks.yaw
                observation[1] = landmarks.pitch
                observation[2] = landmarks.roll
            
            # Brightness and contrast from cropped frame
            if cropped is not None:
                lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
                observation[9] = float(np.mean(lab[:, :, 0]))  # brightness_mean
                observation[10] = float(np.std(lab[:, :, 0]))   # contrast_mean
            
            # Identity uncertainty from intrinsic confidence
            if intrinsic_conf is not None:
                observation[3] = 1.0 - float(np.mean(intrinsic_conf))  # uncertainty = 1 - confidence
            
            # Temporal confidence from identity state
            if self.identity_state is not None and self.identity_state.belief is not None:
                observation[5] = float(np.mean(self.identity_state.belief.get_confidence()))
            
            # Full predict-update cycle (D-06: Bayesian temporal belief)
            # Observation matrix H: observe pose (0-2), uncertainty (3), temp_conf (5), brightness (9), contrast (10)
            H = np.zeros((7, self.state_evolution.state_dim))
            H[0, 0] = 1  # yaw
            H[1, 1] = 1  # pitch
            H[2, 2] = 1  # roll
            H[3, 3] = 1  # identity uncertainty
            H[4, 5] = 1  # temporal confidence
            H[5, 9] = 1  # brightness
            H[6, 10] = 1 # contrast
            
            # Observation vector (ALL observed dimensions)
            obs_vector = np.array([
                observation[0], observation[1], observation[2], 
                observation[3], observation[5], observation[9], observation[10]
            ])
            
            # Observation noise R (measurement uncertainty)
            R = np.diag([1.0, 1.0, 1.0, 5.0, 5.0, 10.0, 5.0])
            
            self._latent_state, self._latent_covariance = self.state_evolution.predict_update_full(
                self._latent_state, self._latent_covariance,
                obs_vector, H, R,
            )

            # D-06: SIM(2) velocity prediction for occlusion recovery (via subsystem wrapper)
            if self._last_SIM2 is not None and hasattr(self, '_prev_SIM2') and self._prev_SIM2 is not None:
                try:
                    temporal_state = self._temporal_estimator.predict(
                        current_sim2=self._last_SIM2,
                        observation=None,
                        H=None,
                        R=None,
                    )
                    if temporal_state.motion_field is not None:
                        self._predicted_SIM2 = temporal_state.motion_field
                except Exception:
                    pass
            self._prev_SIM2 = self._last_SIM2

    def _emit_frame_telemetry(
        self,
        frame_idx: int,
        fallback_reason: Optional[str],
        intrinsic_components: Optional['IntrinsicComponents'],
        energy_terms: dict,
        prev_physical: int,
        prev_alpha: int,
        render_path: Optional[str] = None,
        intrinsic_used: Optional[bool] = None,
        geometry_source: Optional[str] = None,
        resample_count: int = 0,
        transform_det: Optional[float] = None,
        contract_assertions_passed: bool = True,
        latent_primary: bool = False,
        source_pixel_fraction: float = 1.0,
    ) -> None:
        """D-08: Emit per-frame telemetry JSON.

        Called from ALL render paths (physical, identity, enhancement, lost-face)
        to ensure every frame is logged. Wrapped in try/except to guarantee emission.
        """
        try:
            sim2_det = self._last_transform_det if transform_det is None else float(transform_det)
            renderer_mode = "unknown"
            try:
                renderer_mode = self.renderer_mode_state.current_mode.value if self.renderer_mode_state else "unknown"
            except Exception:
                pass
            if geometry_source is None:
                geometry_source = self._last_geometry_source
            if intrinsic_used is None:
                intrinsic_used = intrinsic_components is not None
            if render_path is None:
                render_path = "physical" if self._telemetry["physical_render_frames"] > prev_physical else "alpha" if self._telemetry["alpha_fallback_frames"] > prev_alpha else "enhancement"
            record = {
                "frame_idx": frame_idx,
                "render_path": render_path,
                "renderer_mode": renderer_mode,
                "fallback_reason": fallback_reason,
                "intrinsic_used": bool(intrinsic_used),
                "geometry_source": geometry_source,
                "resample_count": int(resample_count),
                "energy_terms": energy_terms if energy_terms else {},
                "transform_det": sim2_det,
            }
            # D-05 Phase 0: attach a per-frame LatentRenderTelemetry sub-dict.
            # Additive only — legacy frames report latent_primary=False and
            # source_pixel_fraction=1.0 (face is source-derived in paste-then-relight).
            # Wrapped so a failure here never drops the frame telemetry record.
            try:
                # albedo_drift_from_anchor: read from the identity anchor when
                # available; never let a failure here drop the record.
                albedo_drift = 0.0
                if self.identity_state is not None and hasattr(self.identity_state, 'get_anchor_distance'):
                    try:
                        albedo_drift = float(self.identity_state.get_anchor_distance())
                    except Exception:
                        albedo_drift = 0.0
                # uncertainty_mean: current-frame mean of intrinsic albedo
                # uncertainty when available, else 1.0 (no carryover — built from
                # this frame's intrinsic_components only). Requirement 8.3.
                uncertainty_mean = 1.0
                if intrinsic_components is not None:
                    au = getattr(intrinsic_components, 'albedo_uncertainty', None)
                    if au is not None:
                        au_arr = np.asarray(au, dtype=np.float32)
                        if au_arr.size > 0:
                            uncertainty_mean = float(np.mean(au_arr))
                latent_render = LatentRenderTelemetry(
                    frame_idx=frame_idx,
                    render_path=render_path,
                    latent_primary=bool(latent_primary),
                    source_pixel_fraction=float(source_pixel_fraction),
                    latent_confidence=float(self._last_latent_confidence),
                    albedo_drift_from_anchor=albedo_drift,
                    uncertainty_mean=uncertainty_mean,
                    contract_assertions_passed=bool(contract_assertions_passed),
                    gate_state=str(self._last_gate_state),
                    hybrid_alpha_mean=float(self._last_hybrid_alpha_mean),
                    effective_blend_max=float(self._last_effective_blend_max),
                    appearance_uncertainty=float(self._last_appearance_uncertainty),
                    deform_max=float(self._last_deform_max),
                    deform_mean=float(self._last_deform_mean),
                )
                latent_dict = latent_render.to_dict()
                # Embed in the frame record AND append to the dedicated log.
                record["latent"] = latent_dict
                self._latent_telemetry_log.append(latent_dict)
            except Exception:
                # Keep the dedicated log aligned per-frame even on failure.
                fallback_latent = {
                    "frame_idx": frame_idx,
                    "render_path": render_path,
                    "latent_primary": False,
                    "source_pixel_fraction": 1.0,
                    "latent_confidence": 0.0,
                    "albedo_drift_from_anchor": 0.0,
                    "uncertainty_mean": 1.0,
                    "contract_assertions_passed": bool(contract_assertions_passed),
                    "gate_state": str(self._last_gate_state),
                    "hybrid_alpha_mean": float(self._last_hybrid_alpha_mean),
                    "effective_blend_max": float(self._last_effective_blend_max),
                    "appearance_uncertainty": float(self._last_appearance_uncertainty),
                    "deform_max": float(self._last_deform_max),
                    "deform_mean": float(self._last_deform_mean),
                }
                record["latent"] = fallback_latent
                self._latent_telemetry_log.append(fallback_latent)
            self._frame_telemetry_log.append(record)
            # JSONL visibility log
            self._append_visibility_record({
                "ts": time.time(),
                "run_id": self._run_id,
                "event": "frame_telemetry",
                **record,
                "totals": {
                    "physical_render_frames": self._telemetry["physical_render_frames"],
                    "alpha_fallback_frames": self._telemetry["alpha_fallback_frames"],
                    "intrinsic_success_frames": self._telemetry["intrinsic_success_frames"],
                    "intrinsic_failure_frames": self._telemetry["intrinsic_failure_frames"],
                    "renderer_mode_transitions": self._telemetry["renderer_mode_transitions"],
                    "render_time_count": self._telemetry["render_time_count"],
                },
            })
        except Exception:
            # Last resort: emit minimal telemetry so the frame is never lost
            error_latent = {
                "frame_idx": frame_idx,
                "render_path": "error",
                "latent_primary": False,
                "source_pixel_fraction": 1.0,
                "latent_confidence": 0.0,
                "albedo_drift_from_anchor": 0.0,
                "uncertainty_mean": 1.0,
                "contract_assertions_passed": bool(contract_assertions_passed),
                "gate_state": "disabled",
                "hybrid_alpha_mean": 1.0,
            }
            self._frame_telemetry_log.append({
                "frame_idx": frame_idx,
                "render_path": "error",
                "renderer_mode": "unknown",
                "fallback_reason": fallback_reason or "telemetry_error",
                "intrinsic_used": False,
                "geometry_source": "unknown",
                "resample_count": 0,
                "energy_terms": {},
                "transform_det": 1.0,
                "latent": error_latent,
            })
            self._latent_telemetry_log.append(error_latent)

    def _resolve_blend_mode(self) -> str:
        """Return an implemented compositor mode."""
        try:
            mode = cfg.compositor.get("blend_mode", "laplacian")
        except Exception:
            mode = "laplacian"
        return "alpha" if mode == "alpha" else "laplacian"

    def _postprocess_rendered_crop(
        self,
        output: np.ndarray,
        face_mask: Optional[np.ndarray],
        source_sharpness: Optional[float] = None,
        edge_protection: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply the common post-composite detail and photometric lock pass.

        Phase A (D-01): Adaptive dual-radius USM recovers HF lost through
        canonical warp + multiband blending. Amount scales with measured
        sharpness deficit — blurry frames get more, sharp frames get less
        (avoids over-sharpening ringing). Local contrast enhancement via
        CLAHE on the luminance channel follows.

        Phase A calibration: sharpness target is resolution-calibrated from
        the source crop's Laplacian variance. Target = max(source_hf * 1.3, 200),
        clamped to [200, 600]. Without source reference, falls back to 300.

        D-04: Edge protection mask from normal-variance analysis reduces
        sharpening at geometric edges to prevent halos at nose bridge,
        jaw contour, and eyebrow ridges.
        """
        if source_sharpness is not None:
            target = max(min(source_sharpness * 1.3, 600.0), 200.0)
        else:
            target = 300.0
        output = face_enhance.adaptive_sharpen(
            output, face_mask=face_mask, target_sharpness=target,
            edge_protection=edge_protection,
        )
        output = face_enhance.enhance_contrast(output, face_mask=face_mask)
        return photometric_lock(output, face_mask)

    def _compute_energy_terms(
        self,
        intrinsic_components: Optional['IntrinsicComponents'],
        identity_face: Optional[np.ndarray],
        landmarks: Optional[Landmarks],
        frame_idx: int,
    ) -> dict:
        """Compute and normalize energy terms.

        RULE 7: Energy terms must be normalized to unit variance.
        E_i_normalized = E_i / sigma_i^2

        Returns:
            Dict of energy term name -> normalized value
        """
        terms = {}

        # Geometry energy: landmark stability
        if landmarks is not None:
            pose_mag = abs(landmarks.yaw) + abs(landmarks.pitch) + abs(landmarks.roll)
            terms['E_geom'] = float(pose_mag / 180.0)

        # Identity energy: intrinsic decomposition quality
        if intrinsic_components is not None:
            terms['E_identity'] = float(intrinsic_components.reconstruction_error)

        # Photometric energy: reconstruction error
        if intrinsic_components is not None:
            terms['E_photometric'] = float(intrinsic_components.decomposition_quality)

        # Temporal energy: state evolution prediction error (innovation)
        # Use covariance trace as uncertainty measure, not raw state norm
        if self._latent_covariance is not None:
            terms['E_temporal'] = float(np.trace(self._latent_covariance))

        # Normalize all terms
        normalized_terms = {}
        for name, value in terms.items():
            normalized_terms[name] = self.energy_scaler.normalize(name, value)

        return normalized_terms

    def _render_core(
        self,
        cropped: np.ndarray,
        source_frame: np.ndarray,
        intrinsic_components: Optional['IntrinsicComponents'],
        intrinsic_conf: Optional[np.ndarray],
        identity_face: Optional[np.ndarray],
        landmarks: Optional[Landmarks],
        crop_plan: CropPlan,
        region_masks: Optional[dict],
        face_mask: Optional[np.ndarray],
        frame_idx: int,
        identity_eyes: Optional[np.ndarray] = None,
        eye_confidence: float = 0.0,
        geom_state: Optional['GeometryState'] = None,
    ) -> Optional[np.ndarray]:
        """Shared rendering core — single source of truth for all rendering logic.

        Called by BOTH _process_frame_v2() and _render_frame_v2().

        Flow:
          1. Try PhysicalRenderer if intrinsic available + mode allows
          2. Fallback: identity composite (warp canonical face to crop + blend)
          3. Last resort: enhancement-only (sharpen + denoise)
        """
        # D-05 Phase 0: reset per-frame contract result so this frame's
        # telemetry reflects only the current frame (Requirement 8.3/8.4).
        self._last_contract_passed = True
        # D-05 Phase 2B: per-frame reset so a stale gate label never leaks into
        # a legacy frame's telemetry. 'disabled' = render_source != 'latent' (or
        # no face this frame); the latent-path block below overwrites it with the
        # real gate decision.
        self._last_gate_state = "disabled"
        # D-05 Phase 2B: per-frame reset of the hybrid blend weight. 1.0 = pure
        # latent / no observation crossed (the truth on any non-hybrid frame);
        # the latent path overwrites it with the real mean alpha when it renders.
        self._last_hybrid_alpha_mean = 1.0
        self._last_effective_blend_max = self._hybrid_blend_max
        self._last_appearance_uncertainty = 0.0
        self._last_deform_max = 0.0
        self._last_deform_mean = 0.0

        # BHENCHOD SANITIZER: Kill 256-channel tensors before they reach the renderer
        if intrinsic_components is not None and getattr(intrinsic_components, 'shading', None) is not None:
            # D-05 Phase 0: warn-only contract check at the B->D boundary. In
            # 'warn' mode this only logs (no clamp, no raise) so behavior is
            # unchanged; in explicitly-configured 'fatal' mode it raises. Guard
            # so warn-only can never break the frame.
            try:
                albedo = getattr(intrinsic_components, 'albedo', None)
                if albedo is not None:
                    expect_hw = tuple(np.asarray(albedo).shape[:2])
                    passed = assert_intrinsic_contract(
                        intrinsic_components, expect_hw=expect_hw, mode=self._contract_mode
                    )
                    self._last_contract_passed = self._last_contract_passed and bool(passed)
            except Exception:
                if self._contract_mode == 'fatal':
                    raise
                self._last_contract_passed = False

            shd = intrinsic_components.shading
            if isinstance(shd, np.ndarray) and self.render_source == 'legacy':
                if shd.ndim == 3 and shd.shape[2] > 3:
                    intrinsic_components.shading = np.mean(shd, axis=2, keepdims=True).astype(np.float32)
                elif shd.ndim == 2:
                    intrinsic_components.shading = shd[:, :, np.newaxis].astype(np.float32)

        # Track why we skip PhysicalRenderer (for telemetry)
        fallback_reason = None
        prev_physical = self._telemetry["physical_render_frames"]
        prev_alpha = self._telemetry["alpha_fallback_frames"]

        # RULE 7: Compute and normalize energy terms
        energy_terms = self._compute_energy_terms(
            intrinsic_components, identity_face, landmarks, frame_idx
        )

        # D-04: Normal-variance edge protection mask (geometry-aware sharpening)
        normal_map = compositor.face_prior_normal_map(cropped.shape[0], cropped.shape[1])
        edge_protection_mask = compositor.compute_normal_variance_mask(normal_map)

        # RULE 8: Timing
        import time as _time
        _render_start = _time.perf_counter()

        # 1. PhysicalRenderer
        # D-02: Check render_mode_override for A/B validation
        if self.render_mode_override == 'alpha':
            physical_possible = False
            fallback_reason = "render_mode_override_alpha"
        else:
            physical_possible = (intrinsic_components is not None
                                and self.physical_renderer is not None
                                and self.renderer_mode_state is not None
                                and self.renderer_mode_state.current_mode in [RendererMode.PHYSICAL, RendererMode.HYBRID]
                                and landmarks is not None)

            # H-03 + A-8: physical-render quality gate (now a pure, tested
            # decision; see _evaluate_physical_gate). The magic z-score constants
            # 0.8/0.1 are named/justified there, and the latent's epistemic
            # uncertainty is read in as a first-class input — INITIALIZED-GUARDED
            # (None pre-enrollment, where query_uncertainty would be all-ones), so
            # legacy-only runs are byte-for-byte unchanged.
            if physical_possible:
                latent_unc_mean = None
                if self._identity_estimator is not None:
                    _lat_g = self._identity_estimator.latent()
                    if _lat_g.initialized:
                        latent_unc_mean = 1.0 - float(_lat_g.mean_confidence())
                allow_physical, gate_reason = self._evaluate_physical_gate(
                    energy_terms, latent_uncertainty_mean=latent_unc_mean,
                )
                if not allow_physical:
                    physical_possible = False
                    fallback_reason = gate_reason

        # D-05 Phase 2: LATENT render path (peer of the physical branch).
        # When render_source='latent' and a face is present, the Phase 2B
        # PRODUCTION GATE decides whether the latent has earned the right to
        # DRIVE the face interior this frame (relative-to-floor confidence +
        # spike check, see _evaluate_latent_gate). Only when the gate ENGAGES
        # does the identity latent synthesize the face — under lighting estimated
        # from the observation, NOT a decomposition of the source crop — on its
        # own path (skipping the legacy source-HF reinjection tail, retiring
        # A-2/A-3/A-5). On gate refusal OR any render failure it falls through to
        # the legacy path so a frame is never dropped, and gate_state records WHY.
        if (
            self.render_source == 'latent'
            and landmarks is not None
            and self._identity_estimator is not None
        ):
            _latent = self._identity_estimator.latent()
            if self._gate_policy == 'forced_latent':
                gate_engage, gate_state = self._evaluate_latent_gate_forced(
                    initialized=_latent.initialized,
                )
            else:
                gate_engage, gate_state = self._evaluate_latent_gate(
                    initialized=_latent.initialized,
                    confidence=self._last_latent_confidence,
                    confidence_prev=self._prev_latent_confidence,
                    confidence_floor=self._latent_confidence_floor,
                )
            self._last_gate_state = gate_state
            # Track this frame's confidence for the NEXT frame's spike check,
            # regardless of the decision, so the trajectory stays honest.
            self._prev_latent_confidence = self._last_latent_confidence
            latent_result = (
                self._render_with_latent(
                    cropped, landmarks, crop_plan, frame_idx, geom_state=geom_state,
                )
                if gate_engage else None
            )
            if latent_result is not None:
                latent_result = self._postprocess_rendered_crop(
                    latent_result, face_mask,
                    source_sharpness=face_enhance._measure_sharpness(cropped, face_mask),
                    edge_protection=edge_protection_mask,
                )
                self._telemetry["physical_render_frames"] += 1
                self._emit_frame_telemetry(
                    frame_idx, fallback_reason, intrinsic_components,
                    energy_terms, prev_physical, prev_alpha,
                    render_path="latent",
                    intrinsic_used=True,
                    geometry_source=self._last_geometry_source,
                    resample_count=1,
                    contract_assertions_passed=self._last_contract_passed,
                    latent_primary=True,
                    source_pixel_fraction=float(self._last_source_pixel_fraction),
                )
                render_time_ms = (_time.perf_counter() - _render_start) * 1000
                self._telemetry["render_time_sum_ms"] += render_time_ms
                self._telemetry["render_time_count"] += 1
                return latent_result
            # else: gate refused (gate_state != 'engaged') OR the latent render
            # was unavailable this frame -> legacy fallback below.

        if physical_possible:
            result = self._render_with_physical_renderer(
                source_frame, cropped, intrinsic_components, intrinsic_conf,
                landmarks, crop_plan, frame_idx, region_masks,
            )
            if result is not None:
                result = self._inject_detail_residual(
                    result,
                    intrinsic_components,
                    face_mask=face_mask,
                    strength=0.55,
                )
                # Source-HF re-injection: recover HF lost in canonical warp
                result = self._reinject_source_hf(result, cropped, face_mask, strength=0.80)
                self._telemetry["physical_render_frames"] += 1
                result = self._postprocess_rendered_crop(
                    result, face_mask,
                    source_sharpness=face_enhance._measure_sharpness(cropped, face_mask),
                    edge_protection=edge_protection_mask,
                )
                self._emit_frame_telemetry(
                    frame_idx, fallback_reason, intrinsic_components,
                    energy_terms, prev_physical, prev_alpha,
                    render_path="physical",
                    intrinsic_used=True,
                    geometry_source=self._last_geometry_source,
                    resample_count=1,
                    contract_assertions_passed=self._last_contract_passed,
                )
                self._logger.info(
                    "frame=%d | render=physical | fallback=%s | det=%.4f | energy=%s",
                    frame_idx,
                    fallback_reason,
                    float(self._last_transform_det),
                    json.dumps(energy_terms, default=self._json_default),
                )
                # PATCH 5: Render timing for physical path
                render_time_ms = (_time.perf_counter() - _render_start) * 1000
                self._telemetry["render_time_sum_ms"] += render_time_ms
                self._telemetry["render_time_count"] += 1
                return result
            else:
                fallback_reason = "physical_renderer_failed"
        else:
            # Determine why PhysicalRenderer wasn't attempted
            if intrinsic_components is None:
                fallback_reason = "intrinsic_unavailable"
            elif landmarks is None:
                fallback_reason = "no_landmarks"
            elif self.renderer_mode_state is None or self.renderer_mode_state.current_mode not in [RendererMode.PHYSICAL, RendererMode.HYBRID]:
                fallback_reason = "renderer_mode_alpha"

        # 2. Identity composite fallback
        if identity_face is not None and face_mask is not None and landmarks is not None:
            try:
                adjusted_lm = self._adjust_landmarks_to_crop(landmarks, crop_plan)
                if adjusted_lm:
                    output = self._composite_identity_to_crop(
                        cropped, identity_face, landmarks, crop_plan, frame_idx,
                    )
                    # Phase A (D-01): Energy conservation on alpha composite path
                    output = face_enhance.apply_energy_conservation(
                        output, cropped, face_mask=face_mask, energy_limit=0.95,
                    )
                    output = self._inject_detail_residual(
                        output,
                        intrinsic_components,
                        face_mask=face_mask,
                        strength=0.35,
                    )
                    # Source-HF re-injection on alpha path too
                    output = self._reinject_source_hf(output, cropped, face_mask, strength=0.75)
                    output = self._postprocess_rendered_crop(
                        output, face_mask,
                        source_sharpness=face_enhance._measure_sharpness(cropped, face_mask),
                        edge_protection=edge_protection_mask,
                    )
                    if fallback_reason:
                        fb_dist = self._telemetry["fallback_reason_distribution"]
                        fb_dist[fallback_reason] = fb_dist.get(fallback_reason, 0) + 1
                    # D-08: Per-frame telemetry (before early return)
                    self._emit_frame_telemetry(
                        frame_idx, fallback_reason, intrinsic_components,
                        energy_terms, prev_physical, prev_alpha,
                        render_path="alpha",
                        intrinsic_used=False,
                        geometry_source="canonical_identity",
                        resample_count=1,
                        contract_assertions_passed=self._last_contract_passed,
                    )
                    self._logger.info(
                        "frame=%d | render=alpha | fallback=%s | intrinsic=%s | geom=%s | det=%.4f",
                        frame_idx,
                        fallback_reason,
                        intrinsic_components is not None,
                        self._last_geometry_source,
                        float(self._last_transform_det),
                    )
                    # PATCH 5: Render timing for identity composite path
                    render_time_ms = (_time.perf_counter() - _render_start) * 1000
                    self._telemetry["render_time_sum_ms"] += render_time_ms
                    self._telemetry["render_time_count"] += 1
                    return output
            except Exception as e:
                print(f"  Frame {frame_idx}: COMPOSITOR FAILED: {e}")
                fallback_reason = "compositor_exception"

        # 3. Last resort: enhancement only
        enhancement_mask = None
        if region_masks:
            enhancement_mask = face_enhance._create_enhancement_mask(region_masks, cropped.shape)

        rendered = face_enhance.render_frame(
            cropped, enhancement_mask, region_masks,
            identity_eyes=identity_eyes,
            eye_confidence=eye_confidence,
        )
        rendered = self._inject_detail_residual(
            rendered,
            intrinsic_components,
            face_mask=face_mask,
            strength=0.30,
        )
        # Source-HF re-injection on enhancement path (was missing before)
        rendered = self._reinject_source_hf(rendered, cropped, face_mask, strength=0.65)
        rendered = self._postprocess_rendered_crop(
            rendered, face_mask,
            source_sharpness=face_enhance._measure_sharpness(cropped, face_mask),
            edge_protection=edge_protection_mask,
        )
        render_time_ms = (_time.perf_counter() - _render_start) * 1000
        self._telemetry["render_time_sum_ms"] += render_time_ms
        self._telemetry["render_time_count"] += 1

        self._logger.info(
            "frame=%d | render=enhancement | fallback=%s | intrinsic=%s | geom=%s | det=%.4f",
            frame_idx,
            fallback_reason,
            intrinsic_components is not None,
            self._last_geometry_source,
            float(self._last_transform_det),
        )

        # D-08: Per-frame telemetry JSON emission
        self._emit_frame_telemetry(
            frame_idx, fallback_reason, intrinsic_components,
            energy_terms, prev_physical, prev_alpha,
            render_path="enhancement",
            intrinsic_used=False,
            geometry_source="none",
            resample_count=0,
            contract_assertions_passed=self._last_contract_passed,
        )

        return rendered

    def _render_with_physical_renderer(
        self,
        source_frame: np.ndarray,
        cropped: np.ndarray,
        intrinsic_components: 'IntrinsicComponents',
        intrinsic_conf: np.ndarray,
        landmarks: Optional[Landmarks],
        crop_plan: CropPlan,
        frame_idx: int,
        region_masks: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """Render using V3 PhysicalRenderer with intrinsic decomposition.

        ARCHITECTURE CORRECTION: Decompose source frame directly in output space.
        NOT warp from canonical space. Canonical space is for alignment + latent indexing only.

        Flow:
          1. Decompose source crop directly (output-space decomposition)
          2. Render in output space using PhysicalRenderer
          3. Single composite
        """
        try:
            # D-01: Decompose source frame directly in output space
            # NOT from canonical space — that was the architectural drift
            source_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Use identity's anchor albedo to guide decomposition
            anchor_albedo = None
            if self.identity_state is not None and self.identity_state._anchor_albedo is not None:
                anchor_albedo = self.identity_state._anchor_albedo
            
            # Decompose source directly (output-space, not canonical-space)
            source_decomposer = self.identity_state._intrinsic_decomposer
            source_intrinsic = source_decomposer.decompose(source_rgb)

            # BHENCHOD SANITIZER: Kill 256-channel latent embeddings posing as shading
            if hasattr(source_intrinsic, 'shading') and isinstance(source_intrinsic.shading, np.ndarray):
                # D-05 Phase 0: warn-only contract check at the B->D boundary.
                # Warn mode only logs (no clamp, no raise); explicit fatal mode
                # raises. AND the result into the per-frame flag set in
                # _render_core so a frame fails the contract if either site does.
                try:
                    albedo = getattr(source_intrinsic, 'albedo', None)
                    if albedo is not None:
                        expect_hw = tuple(np.asarray(albedo).shape[:2])
                        passed = assert_intrinsic_contract(
                            source_intrinsic, expect_hw=expect_hw, mode=self._contract_mode
                        )
                        self._last_contract_passed = self._last_contract_passed and bool(passed)
                except Exception:
                    if self._contract_mode == 'fatal':
                        raise
                    self._last_contract_passed = False

                # Contract assertion is the only guard; no silent sanitizers.
            
            # Compute adjusted landmarks and M_inv once (reused for anchor + mask)
            adjusted_lm = self._adjust_landmarks_to_crop(landmarks, crop_plan)
            if adjusted_lm is None:
                return None

            M, _ = canonical_map.compute_alignment(
                adjusted_lm,
                canonical_size=tuple(cfg.canonical.atlas_size),
            )
            M_inv = np.linalg.inv(M)[:2]
            current_sim2 = self._affine_to_sim2(M_inv)
            if self._last_SIM2 is not None:
                interpolated = interpolate_sim2(self._last_SIM2, current_sim2, 0.6)
                M_inv = self._sim2_to_affine(interpolated)
            self._last_SIM2 = current_sim2
            self._last_transform_det = current_sim2.scale ** 2

            # Warp anchor albedo to source crop space for correction
            if anchor_albedo is not None:
                # Warp anchor to crop space
                anchor_crop = cv2.warpAffine(
                    anchor_albedo, M_inv, (cropped.shape[1], cropped.shape[0]),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT,
                )
                
                # Apply anchor correction to source albedo
                anchor_mean = np.mean(anchor_crop, axis=(0, 1))
                source_mean = np.mean(source_intrinsic.albedo, axis=(0, 1))
                drift = float(np.sqrt(np.sum((anchor_mean - source_mean) ** 2)))
                
                if drift > 0.05:
                    lambda_corr = min(0.3, drift * 2.0)
                    source_intrinsic.albedo = (
                        (1 - lambda_corr) * source_intrinsic.albedo
                        + lambda_corr * anchor_crop
                    )
                    source_intrinsic.albedo = np.clip(source_intrinsic.albedo, 0, 1).astype(np.float32)
            
            # Estimate lighting from source shading
            # NOTE: Do NOT scale lighting values by shading_mean — the shading map
            # already encodes spatial illumination; scaling the light intensity by its
            # mean causes double-attenuation (e.g. 0.03*0.3 = 0.009 ambient → black).
            # Energy conservation handles calibrating output to albedo×shading target.
            lighting = LightingModel(
                ambient=0.15,
                diffuse_intensity=0.85,
            )
            
            # ─────────────────────────────────────────────────
            # D-04: REAL dense geometry pipeline
            # landmarks → dense mesh → mesh normals → render_with_mesh()
            # ─────────────────────────────────────────────────
            dense_geometry = None
            try:
                if landmarks is not None and hasattr(landmarks, "points"):
                    dense_geometry = self._dense_geometry.estimate(
                        landmarks.points[:, :2]
                    )
            except Exception as e:
                print(f"  Frame {frame_idx}: dense geometry failed: {e}")

            if dense_geometry is not None:
                rendered_output = self._face_renderer.render_with_mesh(
                    albedo=source_intrinsic.albedo,
                    mesh_vertices=dense_geometry.vertices,
                    mesh_faces=dense_geometry.faces,
                    shading=source_intrinsic.shading,
                    lighting=lighting,
                    image_shape=self._normal_raster_shape(source_intrinsic.albedo.shape[:2]),
                )
                if rendered_output is None:
                    return None
                self._telemetry["mesh_normal_frames"] += 1
                self._last_geometry_source = "mesh"
            else:
                # Fallback: face-prior normals when geometry fails
                rendered_output = self._face_renderer.render(
                    albedo=source_intrinsic.albedo,
                    normal_map=source_intrinsic.normal_map,
                    shading=source_intrinsic.shading,
                    lighting=lighting,
                )
                if rendered_output is None:
                    return None
                self._telemetry["shading_normal_frames"] += 1
                self._last_geometry_source = "face_prior"
            
            # Convert from [0,1] to [0,255] uint8 with detail injection
            # Wrapper returns rendered image directly (not result object)
            if hasattr(rendered_output, "rendered"):
                rendered_face = np.clip(rendered_output.rendered, 0.0, 1.0).astype(np.float32)
                rendered_face = self._inject_detail_residual(
                    (rendered_face * 255.0).astype(np.uint8),
                    source_intrinsic,
                    face_mask=None,
                    strength=0.25,
                )
            else:
                rendered_face = np.clip(rendered_output, 0.0, 1.0).astype(np.float32)
                rendered_face = self._inject_detail_residual(
                    (rendered_face * 255.0).astype(np.uint8),
                    source_intrinsic,
                    face_mask=None,
                    strength=0.25,
                )
            
            # D-01b: Reuse M_inv for mask warp (same canonical transform)
            canonical_mask = self._make_canonical_geometry_mask(
                tuple(cfg.canonical.atlas_size)[::-1]
            )
            aligned_mask = cv2.warpAffine(
                canonical_mask, M_inv, (cropped.shape[1], cropped.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            aligned_mask = np.clip(aligned_mask, 0, 1)
            
            # Feather the mask
            feather_ksize = max(3, cfg.compositor.feather_pixels * 2 + 1)
            feathered_mask = cv2.GaussianBlur(
                aligned_mask, (feather_ksize, feather_ksize), cfg.compositor.feather_pixels / 2
            )
            
            # D-01: Single composite in output space (linear-light)
            # D-01c: Multi-band blending when configured
            blend_mode = self._resolve_blend_mode()
            if blend_mode == "laplacian":
                blended = multiband_blend(cropped, rendered_face, feathered_mask)
            else:
                blended = _blend_linear(cropped, rendered_face, feathered_mask)

            # D-01 FIX: Inject Identity HF, NOT Source HF (Source has lighting + noise)
            face_mask_bool = feathered_mask > 0.3
            if face_mask_bool.sum() > 100 and getattr(source_intrinsic, 'detail_residual', None) is not None:
                # Use identity/detail residual HF, not source noise
                detail_hf = source_intrinsic.detail_residual
                if detail_hf.shape[:2] != blended.shape[:2]:
                    detail_hf = cv2.resize(detail_hf, (blended.shape[1], blended.shape[0]))
                
                # Ensure 3 channels
                if detail_hf.ndim == 2:
                    detail_hf = np.stack([detail_hf]*3, axis=-1)
                elif detail_hf.shape[2] == 1:
                    detail_hf = np.repeat(detail_hf, 3, axis=2)
                    
                detail_strength = 0.6
                mask3 = (feathered_mask * detail_strength)[:, :, np.newaxis]
                blended_linear = _srgb_to_linear(blended)
                output_linear = blended_linear + detail_hf * mask3
                output = _linear_to_srgb(np.clip(output_linear, 0.0, 1.0))
            else:
                output = blended
            
            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx}: output-space render (albedo mean={np.mean(source_intrinsic.albedo):.3f})")
            
            return output

        except Exception as e:
            print(f"  Frame {frame_idx}: OUTPUT-SPACE RENDER FAILED: {e}")
            return None

    def estimate_lighting(self, cropped: np.ndarray, normal_map: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> 'LightingModel':
        """Estimate scene illumination from the OBSERVATION (never the latent).

        Derives a scalar shading field S = luminance(linear(cropped)) and fits it
        against the render ``normal_map`` via the closed-form inverse of the
        renderer's Lambertian term (``fit_lighting_from_shading_normals``):
        ``S = ambient + N·(diffuse·L)``. Lighting is read from the current frame
        so the latent albedo can be re-lit under the real scene — it is NEVER
        stored in or read from the identity latent (Requirement: lighting
        excluded from identity).

        Uses ONLY the source crop's luminance (a photometric observation), not a
        source albedo decomposition — so this introduces no paste-then-relight
        (A-2/A-3) coupling. Degenerate frames fall back to a safe floor inside
        the fit.
        """
        from face_os.physical_renderer import fit_lighting_from_shading_normals

        src = np.asarray(cropped, dtype=np.float32)
        if src.max() > 1.5:
            src = src / 255.0
        # BGR -> linear-light luminance (BT.709 on RGB). Crop is BGR.
        lin = _srgb_to_linear(np.clip(src, 0.0, 1.0))
        b, g, r = lin[..., 0], lin[..., 1], lin[..., 2]
        shading = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
        return fit_lighting_from_shading_normals(shading, normal_map, mask=mask)

    @staticmethod
    def _evaluate_latent_gate(
        initialized: bool,
        confidence: float,
        confidence_prev: float,
        confidence_floor: float,
        margin: float = 0.01,
        spike_drop: float = 0.05,
    ) -> tuple:
        """D-05 Phase 2B PRODUCTION GATE: should the latent DRIVE this frame?

        RELATIVE-TO-FLOOR by design (measured runtime truth). On real video the
        latent confidence (= 1 - mean(albedo_uncertainty)) lives in a tiny band
        — seed ~0.2335 at enrollment, rising ~0.006/frame for a few frames, then
        flat at the Kalman fixed point ~0.2567. An ABSOLUTE threshold (e.g. 0.5)
        would never fire, so the latent could never engage. The gate therefore
        measures confidence RELATIVE to the enrollment floor and watches the
        per-frame change for instability.

        Decision (first matching rule wins; refusals carry a specific reason so
        D-08 telemetry never has to infer branch truth):
          1. ``not initialized``                      -> (False, 'uninitialized')
          2. spike: ``confidence_prev - confidence >= spike_drop``
                                                       -> (False, 'confidence_spike')
             (instability THIS frame; checked before the floor so a sharp drop
             is labelled as a spike even when it also lands below the floor)
          3. floor: ``confidence < confidence_floor + margin``
                                                       -> (False, 'below_floor')
             (no evidence earned beyond the enrollment seed)
          4. otherwise                                 -> (True, 'engaged')

        The PLATEAU (dC/dt = 0, above floor) ENGAGES — it is the measured steady
        state, the whole point of the relative-to-floor formulation. ``dC/dt >=
        0`` is NOT required: only a *sharp* drop (>= spike_drop) refuses; normal
        jitter (real |delta| <= ~0.006) stays engaged.

        Args:
            initialized: latent has absorbed at least the enrollment observation.
            confidence: this frame's ``latent.mean_confidence()`` in [0, 1].
            confidence_prev: previous frame's confidence (spike detection).
            confidence_floor: enrollment-seed confidence (the relative baseline).
            margin: how far above the floor confidence must sit to count as
                real earned evidence (default 0.01).
            spike_drop: per-frame confidence drop that signals instability
                (default 0.05, ~8x a normal step).

        Returns:
            (engage: bool, gate_state: str) — gate_state is the telemetry label.
        """
        if not initialized:
            return False, "uninitialized"
        if (confidence_prev - confidence) >= spike_drop:
            return False, "confidence_spike"
        if confidence < (confidence_floor + margin):
            return False, "below_floor"
        return True, "engaged"

    @staticmethod
    def _evaluate_latent_gate_forced(
        initialized: bool,
        confidence: float = 0.0,
        confidence_prev: float = 0.0,
        confidence_floor: float = 0.0,
        margin: float = 0.01,
        spike_drop: float = 0.05,
    ) -> tuple:
        """D-05 Phase 2A OPTION 3: FORCED LATENT gate — A/B proving stage.

        Engage whenever the latent is initialized, unconditionally. Confidence,
        floor, and spike detection are accepted in the signature for drop-in
        substitution with _evaluate_latent_gate, but ignored — the gate's sole
        purpose is proving the latent path drives pixels end-to-end.

        Once the path is proven, the policy promotes to Option 1 (production
        relative-to-floor gate, _evaluate_latent_gate). Option 2 (per-pixel
        uncertainty blend) is a future refinement.

        Args:
            initialized: latent has absorbed at least the enrollment observation.
            confidence: ignored.
            confidence_prev: ignored.
            confidence_floor: ignored.
            margin: ignored.
            spike_drop: ignored.

        Returns:
            (engage: bool, gate_state: str) — gate_state is the telemetry label.
        """
        if not initialized:
            return False, "uninitialized"
        return True, "engaged"

    @staticmethod
    def _evaluate_physical_gate(
        energy_terms: dict,
        latent_uncertainty_mean: Optional[float] = None,
        geom_extreme_z: float = 0.8,
        photometric_low_z: float = 0.1,
        latent_uncertainty_max: float = 0.95,
    ) -> tuple:
        """H-03 / A-8 / A-9: may the PHYSICAL (legacy) renderer run this frame?

        Extracts the inline H-03 gate (formerly pipeline.py:2043-2052) into a pure,
        testable decision and promotes its two MAGIC constants into NAMED, justified
        parameters. It also closes A-8 by reading the latent's epistemic uncertainty
        as a first-class input (previously Kalman uncertainty was computed but unused
        by rendering).

        IMPORTANT — the energy terms are Z-SCORE normalized upstream (EnergyScaler
        default ``normalization_method='zscore'``), so these thresholds are compared
        against running z-scores, NOT raw values:

          - ``geom_extreme_z`` (0.8): ``E_geom`` is ``(|yaw|+|pitch|+|roll|)/180``
            z-scored. Above +0.8 sigma => this frame's pose is extreme relative to
            its running history => geometry too unreliable for physical render.
          - ``photometric_low_z`` (0.1): ``E_photometric`` is the intrinsic
            decomposition QUALITY z-scored (high = good). Below +0.1 sigma => the
            decomposition is degenerate => physical render would amplify garbage.
          - ``latent_uncertainty_max`` (0.95): NEW read input. ``latent_uncertainty_mean``
            is ``1 - latent.mean_confidence()`` (mean of the same ``albedo_uncertainty``
            field ``query_uncertainty`` exposes per pixel). At/above 0.95 the identity
            belief has collapsed toward maximal uncertainty; fall back rather than
            render from a worthless latent. The 0.95 floor sits ABOVE the measured
            real-video operating point (mean U: seed ~0.77, plateau ~0.74, spike
            ~0.8) so it is INERT in normal operation and only fires on near-total
            collapse (U->1).

        Caller MUST pass ``latent_uncertainty_mean=None`` when the latent is not
        initialized (pre-enrollment ``query_uncertainty`` is all-ones), so the new
        veto cannot fire and legacy-only runs stay byte-for-byte unchanged.

        Precedence (first match wins; energy vetoes keep their original order so the
        existing telemetry reason vocabulary is preserved):
          1. ``E_geom > geom_extreme_z``        -> (False, 'energy_geom_extreme')
          2. ``E_photometric < photometric_low_z`` -> (False, 'energy_photometric_low')
          3. ``latent_uncertainty_mean >= latent_uncertainty_max``
                                                 -> (False, 'latent_uncertainty_high')
          4. otherwise                          -> (True, None)

        Note the original ran its checks only under ``if energy_terms`` (truthy);
        an EMPTY dict skips the energy vetoes entirely. With a NON-empty dict the
        original ``energy_terms.get('E_photometric', 0.0)`` defaulted a missing key
        to 0.0, which is ``< 0.1`` and therefore vetoes — both behaviors preserved.

        Returns:
            (allow: bool, reason: Optional[str]) — reason is None when allowed, else
            the exact telemetry ``fallback_reason`` string.
        """
        if energy_terms:
            E_geom = energy_terms.get('E_geom', 0.0)
            E_photometric = energy_terms.get('E_photometric', 0.0)
            if E_geom > geom_extreme_z:
                return False, 'energy_geom_extreme'
            if E_photometric < photometric_low_z:
                return False, 'energy_photometric_low'
        if (
            latent_uncertainty_mean is not None
            and latent_uncertainty_mean >= latent_uncertainty_max
        ):
            return False, 'latent_uncertainty_high'
        return True, None

    @staticmethod
    def _hybrid_blend_alpha(
        uncertainty: np.ndarray,
        blend_max: float = 0.5,
    ) -> np.ndarray:
        """D-05 Phase 2B per-pixel HYBRID weight: the LATENT's authority per pixel.

        ``alpha = 1 - uncertainty * blend_max`` (uncertainty in [0,1]). High where
        the latent is confident (alpha→1, pure latent), lower where uncertain
        (alpha→1-blend_max, more observation crosses). ``blend_max`` CAPS how much
        the observation can ever take: at blend_max=0.5 the latent retains >=50%
        authority on EVERY pixel even at full uncertainty — so the synthesized
        identity is never fully overwritten and the source-leak metric stays
        bounded (measured worst-case leak 0.0089 < 0.02 on real video). Monotonic
        decreasing in uncertainty ("blend BY uncertainty", design.md:665 /
        requirements 10.4). blend_max=0 disables the hybrid (alpha≡1).

        Returns an (H, W) float32 map in [1-blend_max, 1].
        """
        u = np.clip(np.asarray(uncertainty, dtype=np.float32), 0.0, 1.0)
        alpha = 1.0 - u * float(blend_max)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _hybrid_face(
        latent_face: np.ndarray,
        observation: np.ndarray,
        uncertainty: np.ndarray,
        mask: np.ndarray,
        blend_max: float = 0.5,
    ) -> np.ndarray:
        """Blend the latent-rendered face TOWARD the observation, by uncertainty,
        WITHOUT leaking source detail (design.md:665, requirements.md 10.4).

        The observation IS the source crop, so blending toward it raw would
        reintroduce per-pixel source color and trip the no-source-leak contract
        (measured raw leak 0.33 ≫ 0.02). Two measured safeguards:
          1. blend toward ``LOWPASS(observation)`` only — the same anti-leak
             low-pass _observation_shading uses (sigma = max(4, min(H,W)/12)).
             Smooth illumination/chroma crosses; source HIGH FREQUENCY never
             returns per-pixel (the leak metric is HF/tol-6 sensitive). ~20×
             leak reduction.
          2. per-pixel ``alpha = _hybrid_blend_alpha(uncertainty, blend_max)``,
             so the latent keeps >=1-blend_max authority everywhere.

        Operates in uint8 sRGB (the same space as the composite + leak metric, so
        the measured leak predicts runtime). Only mask-interior pixels are
        touched; outside the mask the latent passes through untouched (the later
        composite owns the background). Returns uint8, same shape as input.

        ``out = alpha*latent + (1-alpha)*lowpass(observation)`` inside the mask.
        """
        L = np.asarray(latent_face, dtype=np.float32)
        s = np.asarray(observation, dtype=np.float32)
        if s.shape != L.shape:
            s = cv2.resize(s, (L.shape[1], L.shape[0])).astype(np.float32)
        u = np.asarray(uncertainty, dtype=np.float32)
        if u.shape[:2] != L.shape[:2]:
            u = cv2.resize(u, (L.shape[1], L.shape[0]))
        m = np.asarray(mask, dtype=np.float32)
        if m.shape[:2] != L.shape[:2]:
            m = cv2.resize(m, (L.shape[1], L.shape[0]))

        # Low-pass the observation: only smooth illumination/chroma may cross;
        # source high frequency (identity detail) is stripped (anti-leak).
        sigma = max(4.0, min(L.shape[:2]) / 12.0)
        s_lp = cv2.GaussianBlur(s, (0, 0), sigma)

        alpha = FaceOSPipeline._hybrid_blend_alpha(u, blend_max=blend_max)[:, :, np.newaxis]
        blended = alpha * L + (1.0 - alpha) * s_lp

        interior = (m > 0.5)[:, :, np.newaxis]
        out = np.where(interior, blended, L)
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _source_pixel_fraction(
        rendered: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
        tol: float = 6.0,
    ) -> float:
        """Spec leak metric (design.md:545, requirements.md:32).

        The fraction of pixels INSIDE the face mask whose rendered color still
        MATCHES the source crop within tolerance — i.e. how much of the
        synthesized face is a surviving paste of the source (paste-then-relight
        leak). A pixel "matches source" iff its max per-channel absolute
        difference from the source is <= ``tol`` intensity levels (0-255). High
        fraction => the latent is not truly driving the face; target < 0.02 on
        the latent path. Returns 0.0 for an empty mask interior (never NaN).

        This is the HONEST no-leak proof. The earlier proxy
        (``1 - mean(feathered_mask)``) measured the background fraction over the
        whole crop and said nothing about leak.
        """
        r = np.asarray(rendered, dtype=np.float32)
        s = np.asarray(source, dtype=np.float32)
        if r.shape != s.shape:
            s = cv2.resize(s, (r.shape[1], r.shape[0]))
        m = np.asarray(mask, dtype=np.float32)
        if m.shape[:2] != r.shape[:2]:
            m = cv2.resize(m, (r.shape[1], r.shape[0]))
        interior = m > 0.5
        n = int(interior.sum())
        if n == 0:
            return 0.0
        diff = np.abs(r - s)
        if diff.ndim == 3:
            diff = diff.max(axis=2)
        matches = (diff <= tol) & interior
        return float(matches.sum()) / float(n)

    @staticmethod
    def _observation_shading(
        observed: np.ndarray,
        albedo: np.ndarray,
        mask: Optional[np.ndarray] = None,
        eps: float = 1e-3,
    ) -> np.ndarray:
        """Scene-illumination shading field for the LATENT render path.

        The renderer derives ABSOLUTE brightness from the shading field, not the
        LightingModel: ``render()`` normalizes the lit base to unit mean
        (physical_renderer.py:374-379), multiplies by shading, then energy-
        conserves to ``mean(albedo * shading)``. A neutral unit shading therefore
        pins the output to the latent ALBEDO brightness (~0.84) — scene-
        independent and flat (the measured 2.1×-too-bright collapse).

        To render the latent albedo under the CURRENT scene exposure we need
        ``S = L / A`` so that ``A * S = L`` reconstructs the observed scene
        luminance ``L``. The latent supplies ``A`` (still lighting-invariant —
        no illumination is stored in identity); ``L`` is read from the current
        observation only. The field is LOW-PASSED so ONLY smooth illumination
        crosses into the render — source high-frequency detail (which would
        re-leak the source crop) is removed, while the blur preserves the local
        mean so scene exposure is unchanged. Returns a 2-D ``(H, W)`` float field.
        """
        obs = np.asarray(observed, dtype=np.float32)
        if obs.size and obs.max() > 1.5:
            obs = obs / 255.0
        obs = np.clip(obs, 0.0, 1.0)
        if obs.ndim == 3 and obs.shape[2] == 3:  # BGR -> BT.709 luminance
            lum = 0.2126 * obs[..., 2] + 0.7152 * obs[..., 1] + 0.0722 * obs[..., 0]
        else:
            lum = obs[..., 0] if obs.ndim == 3 else obs
        alb = np.asarray(albedo, dtype=np.float32)
        if alb.size and alb.max() > 1.5:
            alb = alb / 255.0
        alb_lum = np.mean(alb, axis=2) if alb.ndim == 3 else alb
        if alb_lum.shape != lum.shape:
            alb_lum = cv2.resize(alb_lum, (lum.shape[1], lum.shape[0]))

        # S = L / A (eps floor — a near-zero albedo must never yield NaN/inf).
        shading = lum / np.maximum(alb_lum, eps)

        # Fill outside the face mask with the interior median BEFORE blurring, so
        # background luminance cannot bleed into the masked face via the low-pass.
        if mask is not None:
            mi = np.asarray(mask, dtype=np.float32)
            if mi.shape != shading.shape:
                mi = cv2.resize(mi, (shading.shape[1], shading.shape[0]))
            mi = mi > 0.5
            if mi.any():
                shading = np.where(mi, shading, float(np.median(shading[mi])))

        # Low-pass: illumination is low-frequency; this strips source HF (anti-
        # leak) while preserving the local mean (scene exposure unchanged).
        sigma = max(4.0, min(shading.shape[:2]) / 12.0)
        shading = cv2.GaussianBlur(shading, (0, 0), sigma)
        return np.clip(shading, 0.0, None).astype(np.float32)

    def _render_with_latent(
        self,
        cropped: np.ndarray,
        landmarks: Optional[Landmarks],
        crop_plan: CropPlan,
        frame_idx: int,
        geom_state: Optional['GeometryState'] = None,
    ) -> Optional[np.ndarray]:
        """D-05 Phase 2: render the face from the identity LATENT (not the source).

        The latent — lighting-invariant reflectance + structure — is synthesized
        into the current geometry, shaded under lighting ESTIMATED from the
        observation, and composited into the crop. This is the architectural
        retirement of paste-then-relight (A-2/A-3) and source-HF reinjection
        (A-5): the face interior is produced from stored identity, never from a
        decomposition of the current source crop.

        Returns the rendered crop (BGR uint8) or ``None`` to fall back to legacy.
        """
        try:
            est = self._identity_estimator
            if est is None or not est.latent().initialized:
                return None  # nothing to render from yet -> legacy fallback

            crop_h, crop_w = cropped.shape[:2]

            # ── Crop-space render geometry ─────────────────────────────────────
            # Derive the canonical<->crop transform from the SAME landmarks the
            # frame already extracted (no re-detection). M: crop->canonical;
            # M_inv: canonical->crop (used to warp the latent albedo into crop
            # space). This is the render projection of the one geometry truth.
            adjusted_lm = self._adjust_landmarks_to_crop(landmarks, crop_plan)
            if adjusted_lm is None:
                return None
            M, _ = canonical_map.compute_alignment(
                adjusted_lm, canonical_size=tuple(cfg.canonical.atlas_size),
            )
            M_inv_2x3 = np.linalg.inv(M)[:2]
            inverse_transform = np.eye(3, dtype=np.float32)
            inverse_transform[:2, :] = M_inv_2x3

            # Crop-sized geometry mask (canonical elliptical mask warped to crop).
            canonical_mask = self._make_canonical_geometry_mask(
                tuple(cfg.canonical.atlas_size)[::-1]
            )
            crop_mask = cv2.warpAffine(
                canonical_mask, M_inv_2x3, (crop_w, crop_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            crop_mask = np.clip(crop_mask, 0.0, 1.0).astype(np.float32)

            mesh_478 = getattr(geom_state, 'mesh', None) if geom_state is not None else None
            render_geom = GeometryState(
                landmarks=landmarks,
                pose=(getattr(geom_state, 'pose', None) or (0.0, 0.0, 0.0)),
                canonical_transform=np.asarray(M, dtype=np.float32),
                inverse_transform=inverse_transform,
                mesh=mesh_478,
                mask=crop_mask,
                canonical_face=cropped,  # sets render size = crop (H, W)
            )

            # ── Synthesize identity (latent albedo + microdetail + normals) ────
            components = est.synthesize_identity(render_geom)

            # ── Scene exposure via the SHADING field (brightness fix) ──────────
            # The renderer derives ABSOLUTE brightness from shading, not the
            # LightingModel (it normalizes the model's amplitude away and energy-
            # conserves to mean(albedo*shading) — physical_renderer.py:374-386).
            # synthesize_identity emits NEUTRAL unit shading, so the latent would
            # render at its own albedo brightness (~0.84): scene-independent and
            # flat (the measured 2.1× collapse). Replace it with scene
            # illumination S = observed_luminance / latent_albedo (low-passed:
            # only smooth illumination crosses, no source-HF leak) so the latent
            # albedo is rendered under the CURRENT scene exposure. Lighting is
            # read from the observation only; the identity latent stays
            # lighting-invariant (it supplies albedo, never illumination).
            scene_shading = self._observation_shading(
                cropped, components.albedo, crop_mask,
            )
            import dataclasses as _dc
            components = _dc.replace(
                components,
                shading=scene_shading[:, :, np.newaxis].astype(np.float32),
            )

            # ── Lighting from the OBSERVATION, applied to the latent at render ─
            lighting = self.estimate_lighting(
                cropped, components.normal_map, mask=crop_mask > 0.3
            )

            # ── Render: latent albedo shaded under estimated lighting (no crop) ─
            rendered = self._face_renderer.render_from_latent(
                components, render_geom, lighting,
            )
            if rendered is None:
                return None
            rendered_face = np.clip(np.asarray(rendered, np.float32), 0.0, 1.0)

            # ── Composite the rendered face into the crop (linear-light) ───────
            # Compositing the face region into the frame is legitimate; only
            # SOURCE albedo/HF reinjection is forbidden — and we do neither.
            feather_ksize = max(3, cfg.compositor.feather_pixels * 2 + 1)
            feathered_mask = cv2.GaussianBlur(
                crop_mask, (feather_ksize, feather_ksize), cfg.compositor.feather_pixels / 2
            )
            # D-05 Phase 3: inject identity-sourced detail (latent microdetail).
            # The legacy path does this via _inject_detail_residual with the
            # source crop's intrinsic decomposition; the latent path uses its
            # OWN stored microdetail (warped into crop geometry by
            # synthesize_identity). Same 0.55 strength as the physical path.
            detail_res = getattr(components, 'detail_residual', None)
            if detail_res is not None:
                d = np.asarray(detail_res, dtype=np.float32)
                if d.shape[:2] != rendered_face.shape[:2]:
                    d = cv2.resize(d, (rendered_face.shape[1], rendered_face.shape[0]))
                fm = feathered_mask.astype(np.float32)
                if fm.ndim == 2:
                    fm = fm[:, :, np.newaxis]
                d = d * fm
                rendered_face = np.clip(rendered_face + 0.55 * d, 0.0, 1.0)
            rendered_u8 = (rendered_face * 255.0).astype(np.uint8)
            if rendered_u8.shape[:2] != (crop_h, crop_w):
                rendered_u8 = cv2.resize(rendered_u8, (crop_w, crop_h))

            # Keep the PURE latent face for the synthesis-quality guards
            # (exposure, structure): they must keep measuring the latent's own
            # render, not the hybrid, so a brightness/flatness regression can
            # never hide behind the observation that the hybrid mixes in.
            pure_rendered_u8 = rendered_u8.copy()

            # ── D-05 Phase 2B per-pixel uncertainty HYBRID ─────────────────────
            # WHERE the latent is uncertain (query_uncertainty high), blend the
            # rendered face TOWARD the low-frequency observation, per pixel and
            # capped so the latent keeps >=(1-blend_max) authority. Confident
            # pixels stay pure latent. Only smooth illumination/chroma crosses
            # (low-pass) — source HF never leaks (design.md:665, requirements
            # 10.4). RESTRICTED to the SOLID mask interior (feathered_mask>0.99):
            # measurement PROVED 100% of hybrid-induced source-leak lived in the
            # feather transition band, where the multiband composite ALREADY
            # mixes the source crop — blending there pushed those pixels within
            # tol of source. In the solid interior |latent-source| stays ≫ tol,
            # so restricted leak == pure-latent (<0.01) even at blend_max=0.5.
            # The uncertainty map is warped into the SAME crop geometry as the
            # render (render_geom.canonical_face=cropped), so it is pixel-aligned.
            solid_interior = (feathered_mask > 0.99).astype(np.float32)
            latent_uncertainty = est.query_uncertainty(render_geom)

            # ── D-05 Task 2.5: expression-aware hybrid blend ────────────────
            # Scale blend_max by appearance divergence from enrollment. At
            # neutral expression the latent albedo faithfully represents the
            # face; at extreme expression the static albedo is less reliable,
            # so we allow more source observation crossing.
            appear_unc = float(getattr(est.latent(), "appearance_uncertainty", 0.0) or 0.0)
            effective_blend_max = self._hybrid_blend_max + (1.0 - self._hybrid_blend_max) * appear_unc
            effective_blend_max = float(np.clip(effective_blend_max, self._hybrid_blend_max, 1.0))
            self._last_effective_blend_max = effective_blend_max
            self._last_appearance_uncertainty = appear_unc
            self._last_deform_max = float(getattr(est, "_last_deform_max", 0.0) or 0.0)
            self._last_deform_mean = float(getattr(est, "_last_deform_mean", 0.0) or 0.0)

            rendered_u8 = self._hybrid_face(
                rendered_u8, cropped, latent_uncertainty, solid_interior,
                blend_max=effective_blend_max,
            )
            alpha_map = self._hybrid_blend_alpha(latent_uncertainty, effective_blend_max)
            _zone = solid_interior > 0.5
            self._last_hybrid_alpha_mean = (
                float(np.mean(alpha_map[_zone])) if bool(_zone.any()) else 1.0
            )

            blend_mode = self._resolve_blend_mode()
            if blend_mode == "laplacian":
                output = multiband_blend(cropped, rendered_u8, feathered_mask)
            else:
                output = _blend_linear(cropped, rendered_u8, feathered_mask)

            self._last_geometry_source = est._last_normal_source
            # SPEC leak metric (design.md:545): fraction of FACE-INTERIOR pixels
            # whose composited color still matches the SOURCE crop within
            # tolerance — the honest paste-then-relight leak. Measured over the
            # geometry mask interior (NOT the whole crop), comparing the final
            # composited output against the source. Target < 0.02 on the latent
            # path; a high value means the latent is not truly driving the face.
            self._last_source_pixel_fraction = self._source_pixel_fraction(
                output, cropped, crop_mask,
            )
            # DIAGNOSTIC capture (opt-in, default off): stash the pre-composite
            # rendered face, the actual crop_mask, source crop, and composited
            # output so an external report can measure the TRUE mask-interior
            # latent-vs-legacy-vs-source signal (no landmark-bbox dilution).
            if self._capture_latent_debug:
                self._last_latent_debug = {
                    "frame_idx": frame_idx,
                    "rendered_face": pure_rendered_u8.copy(),  # PURE latent (pre-hybrid) — quality ref
                    "hybrid_face": rendered_u8.copy(),         # post-hybrid, pre-composite
                    "hybrid_alpha_mean": float(self._last_hybrid_alpha_mean),
                    "crop_mask": crop_mask.copy(),         # real geometry mask (0..1)
                    "source_crop": np.asarray(cropped).copy(),
                    "composited": output.copy(),
                    "normal_source": est._last_normal_source,
                    "light_ambient": float(getattr(lighting, "ambient", float("nan"))),
                    "light_diffuse": float(getattr(lighting, "diffuse_intensity", float("nan"))),
                    "light_dir": np.asarray(getattr(lighting, "diffuse_direction", [0, 0, 0]), float).tolist(),
                    "albedo_mean": float(np.mean(getattr(components, "albedo", np.float32(0)))),
                    "shading_mean": float(np.mean(getattr(components, "shading", np.float32(0)))),
                    "shading_std": float(np.std(getattr(components, "shading", np.float32(0)))),
                }
            return output

        except Exception as exc:  # noqa: BLE001 — a latent-render failure must fall back, never crash
            self._log_event("latent_render_failed", error=str(exc), frame_idx=frame_idx)
            return None

    @staticmethod
    def _normal_raster_shape(image_shape: tuple, max_side: int = 384) -> tuple:
        """Bound mesh-normal raster resolution for real-time physical rendering."""
        h, w = image_shape
        largest = max(h, w)
        if largest <= max_side:
            return (h, w)
        scale = max_side / float(largest)
        return (max(1, int(round(h * scale))), max(1, int(round(w * scale))))

    @staticmethod
    def _make_canonical_geometry_mask(
        canonical_size: tuple,
    ) -> np.ndarray:
        """Create a brightness-invariant geometry-based face mask for canonical space.

        Uses a fixed elliptical mask based on canonical face geometry, NOT
        intensity thresholding. This ensures the mask is stable across frames
        regardless of lighting changes.

        The canonical face occupies the central ~70% of the atlas.
        This mask defines the expected face region as a smooth elliptical area.

        Args:
            canonical_size: (height, width) of canonical space, typically (256, 256)

        Returns:
            Mask (H, W) float32, values [0, 1] with feathered edges
        """
        h, w = canonical_size
        # Face region in canonical space: centered oval occupying ~60% of the area
        cy, cx = h / 2, w / 2
        ry, rx = h * 0.50, w * 0.45  # semi-axes

        Y, X = np.ogrid[:h, :w]
        d = ((X - cx) / max(rx, 1)) ** 2 + ((Y - cy) / max(ry, 1)) ** 2
        mask = np.clip(1.0 - d, 0, 1)

        # Light feathering only
        k = 11  # fixed small kernel
        mask = cv2.GaussianBlur(mask, (k, k), k / 5.0)
        mask = np.clip(mask, 0, 1).astype(np.float32)
        return mask

    @staticmethod
    def validate_frame_contract(frame: np.ndarray, expected_h: int, expected_w: int,
                                expected_dtype=np.uint8, expected_channels: int = 3) -> bool:
        """Validate that a frame meets the output contract.

        The contract guarantees:
          - Correct spatial dimensions
          - Correct dtype
          - Correct number of channels
          - No NaN or Inf
          - Values in uint8 range [0, 255]

        Args:
            frame: Output frame to validate
            expected_h: Expected height
            expected_w: Expected width
            expected_dtype: Expected dtype (default np.uint8)
            expected_channels: Expected channels (default 3)

        Returns:
            True if valid, False otherwise
        """
        if frame is None:
            return False
        if frame.shape != (expected_h, expected_w, expected_channels):
            return False
        if frame.dtype != expected_dtype:
            return False
        if np.any(np.isnan(frame)) or np.any(np.isinf(frame)):
            return False
        return True

    def _compute_quality_map(
        self,
        canonical_face: np.ndarray,
        detection_confidence: float,
    ) -> np.ndarray:
        """Compute per-pixel quality map for a canonical face.

        Quality = sharpness × brightness × detection_confidence
        """
        h, w = canonical_face.shape[:2]

        # Sharpness
        gray = cv2.cvtColor(canonical_face, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
        sharpness = np.clip(lap / 50.0, 0, 1)

        # Brightness (prefer well-lit)
        brightness = gray.astype(np.float32) / 255.0
        brightness_weight = 1.0 - np.abs(brightness - 0.5) * 2
        brightness_weight = np.clip(brightness_weight, 0.1, 1.0)

        return sharpness * brightness_weight * detection_confidence

    def _detect_blink(self, landmarks: Landmarks) -> bool:
        """Detect if eyes are blinking using MediaPipe 478-point EAR."""
        if landmarks is None:
            return False

        pts = landmarks.points
        if len(pts) < 468:
            return False

        # V4: MediaPipe 478-point eye indices
        # Left eye: inner(33), top(159), top-mid(158), outer(133), bottom-mid(153), bottom(145)
        left_eye = pts[[33, 159, 158, 133, 153, 145]]
        left_h = abs(left_eye[1][1] - left_eye[5][1]) + abs(left_eye[2][1] - left_eye[4][1])
        left_w = abs(left_eye[0][0] - left_eye[3][0])
        left_ear = left_h / (2.0 * left_w + 1e-6)

        # Right eye: inner(362), top(386), top-mid(385), outer(263), bottom-mid(380), bottom(374)
        right_eye = pts[[362, 386, 385, 263, 380, 374]]
        right_h = abs(right_eye[1][1] - right_eye[5][1]) + abs(right_eye[2][1] - right_eye[4][1])
        right_w = abs(right_eye[0][0] - right_eye[3][0])
        right_ear = right_h / (2.0 * right_w + 1e-6)

        avg_ear = (left_ear + right_ear) / 2
        return avg_ear < 0.15

    def _adjust_landmarks_to_crop(
        self,
        landmarks: Landmarks,
        crop_plan: CropPlan,
    ) -> Optional[Landmarks]:
        """Adjust landmark coordinates from source space to cropped space."""
        if crop_plan.src_w <= 0 or crop_plan.src_h <= 0:
            return None

        sx = crop_plan.dst_w / crop_plan.src_w
        sy = crop_plan.dst_h / crop_plan.src_h
        ox = crop_plan.src_x
        oy = crop_plan.src_y

        new_points = landmarks.points.copy()
        new_points[:, 0] = (landmarks.points[:, 0] - ox) * sx
        new_points[:, 1] = (landmarks.points[:, 1] - oy) * sy

        return Landmarks(
            points=new_points,
            yaw=landmarks.yaw,
            pitch=landmarks.pitch,
            roll=landmarks.roll,
            left_eye_center=(
                (landmarks.left_eye_center[0] - ox) * sx,
                (landmarks.left_eye_center[1] - oy) * sy,
            ),
            right_eye_center=(
                (landmarks.right_eye_center[0] - ox) * sx,
                (landmarks.right_eye_center[1] - oy) * sy,
            ),
            nose_tip=(
                (landmarks.nose_tip[0] - ox) * sx,
                (landmarks.nose_tip[1] - oy) * sy,
            ),
            mouth_center=(
                (landmarks.mouth_center[0] - ox) * sx,
                (landmarks.mouth_center[1] - oy) * sy,
            ),
            landmark_confidence=landmarks.landmark_confidence,
        )

    def _post_process(
        self,
        output_path: str,
        video_path: str,
        all_frames: list,
        face_detected_frames: int,
        total_frames: int,
        elapsed: float,
    ) -> None:
        """Apply fades, run QC, save report."""
        # Apply fades
        if cfg.export.fade_in > 0 or cfg.export.fade_out > 0:
            print("  Applying fades...")
            export_qc.apply_fades(output_path, fade_in=cfg.export.fade_in, fade_out=cfg.export.fade_out)

        # Run QC
        print("  Running QC checks...")
        ref_lab = None
        if self.identity and self.identity.appearance.atlas_lab is not None:
            ref_lab = tuple(np.mean(self.identity.appearance.atlas_lab, axis=(0, 1)).tolist())

        qc_report = export_qc.compute_quality_metrics(all_frames, ref_lab)
        qc_report.face_detection_rate = face_detected_frames / max(total_frames, 1)

        if not qc_report.check():
            print(f"  QC WARNINGS:")
            for f in qc_report.failures:
                print(f"    - {f}")
        else:
            print("  QC: All checks passed")

        # Save QC report
        report_path = Path(output_path).with_suffix(".qc.json")
        with open(report_path, "w") as f:
            json.dump(qc_report.to_dict(), f, indent=2)

        print(f"\n  DONE: {total_frames} frames in {elapsed:.1f}s ({total_frames / max(elapsed, 0.001):.0f} fps)")
        print(f"  Output: {output_path}")

        # Module D: Report anchor distance
        if self.identity_state and self.identity_state.is_initialized():
            anchor_dist = self.identity_state.get_anchor_distance()
            print(f"  Anchor distance: {anchor_dist:.1f} LAB (threshold: {self.identity_state._anchor_threshold})")
            if anchor_dist > self.identity_state._anchor_threshold:
                print(f"  WARNING: Identity drift detected! Output may not match reference.")
            else:
                print(f"  Identity anchored to reference.")

    def _reset_state(self) -> None:
        """Reset per-clip state.

        NOTE: Identity state is NOT reset — it preserves the anchor
        and accumulated observations from enrollment.
        """
        self._start_visibility_run()
        self._log_event("reset_state", reason="new_clip", use_identity=USE_IDENTITY)
        if self.crop:
            self.crop.reset()
        # DON'T reset identity state — it preserves the anchor
        # and accumulated observations from enrollment
        if self.patch_memory:
            self.patch_memory.reset()
        if self.compositor:
            self.compositor.reset()
        self._frame_count = 0
        self._last_M_inv = None
        self._last_good_crop_plan = None
        self._last_geometry_source = "none"
        self._last_transform_det = 1.0
        self._frame_beliefs = {}
        self._forward_intrinsic_components = None
        self._forward_intrinsic_conf = None
        # Reset V3 telemetry for fresh per-clip stats
        self._telemetry = {
            "total_frames": 0,
            "physical_render_frames": 0,
            "alpha_fallback_frames": 0,
            "intrinsic_success_frames": 0,
            "intrinsic_failure_frames": 0,
            "renderer_mode_transitions": 0,
            "intrinsic_failure_reasons": {},
            "fallback_reason_distribution": {},
            # D-01: Identity path failure tracking
            "identity_path_failures": 0,
            "renderer_mode_distribution": {"physical": 0, "hybrid": 0, "alpha": 0},
            "intrinsic_confidence_sum": 0.0,
            "intrinsic_confidence_count": 0,
            "decomposition_error_sum": 0.0,
            "decomposition_error_count": 0,
            "mesh_normal_frames": 0,
            "shading_normal_frames": 0,
            # RULE 8: Timing telemetry
            "render_time_sum_ms": 0.0,
            "render_time_count": 0,
        }
        # RULE 7: Reset energy scaler for fresh per-clip stats
        self.energy_scaler.reset()
        # D-08: Reset per-frame telemetry log
        self._frame_telemetry_log = []
        # D-05 Phase 0: Reset per-frame latent telemetry log
        self._latent_telemetry_log = []
        # D-06: Reset SIM2 prediction state
        self._prev_SIM2 = None
        # D-10: Reset temporal estimator wrapper
        if self._temporal_estimator:
            self._temporal_estimator.reset()
        self._predicted_SIM2 = None
        # D-01: Reset photometric lock temporal state
        reset_photometric_lock()


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Face OS v2 — Identity Belief State Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--reference", default="expectation.png", help="Reference image")
    parser.add_argument("--photos", default="photos/", help="Reference photos directory")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no-bidirectional", action="store_true", help="Disable bidirectional solve")
    parser.add_argument("--no-identity", action="store_true", help="Disable identity memory (simple enhancement mode)")

    args = parser.parse_args()

    # Apply feature flags from CLI
    global USE_IDENTITY
    if args.no_identity:
        USE_IDENTITY = False

    output = args.output or "output/face_os/output.mp4"

    pipeline = FaceOSPipeline(use_bidirectional=not args.no_bidirectional)
    if not pipeline.enroll(args.reference, args.photos):
        return

    result = pipeline.process(args.video, output, max_frames=args.max_frames)
    if result:
        print(f"\nSuccess: {result}")
    else:
        print("\nFailed.")


if __name__ == "__main__":
    main()
