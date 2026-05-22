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
import time
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
    IdentityProfile,
    Landmarks,
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
from face_os.compositor import photometric_lock, reset_photometric_lock
from face_os import export_qc

# NEW modules
from face_os.identity_state import IdentityState
from face_os.patch_memory import PatchMemory
from face_os.temporal_solve import TemporalRepairEngine, FrameQuality

# V3 modules
from face_os.physical_renderer import PhysicalRenderer, LightingModel
from face_os.intrinsic_decomposition import IntrinsicComponents
from face_os.lie_group import SIM2Transform, interpolate_sim2
from face_os.renderer_mode import RendererMode, RendererModeState
from face_os.state_evolution import StateEvolution
from face_os.energy_scaling import EnergyScaler


# ─── D-01: Linear-light conversion helpers ──────────────────────────────────

def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 image to linear-light float32 [0,1].

    D-01: Gamma-space compositing is physically incorrect.
    Blending must happen in linear-light space.
    """
    f = img.astype(np.float32) / 255.0
    return np.power(f, 2.2)


def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Convert linear-light float32 [0,1] back to sRGB uint8."""
    g = np.power(np.clip(img, 0, 1), 1.0 / 2.2)
    return (g * 255).astype(np.uint8)


def _blend_linear(bg: np.ndarray, fg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend two sRGB images in linear-light space.

    D-01: Fixes gamma-space compositing.
    bg, fg: (H,W,3) uint8 BGR
    mask: (H,W) float32 [0,1]
    Returns: (H,W,3) uint8 BGR
    """
    bg_lin = _srgb_to_linear(bg)
    fg_lin = _srgb_to_linear(fg)
    m3 = mask[:, :, np.newaxis] if mask.ndim == 2 else mask
    blended = bg_lin * (1 - m3) + fg_lin * m3
    return _linear_to_srgb(blended)


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

        # V3: Initialize renderer mode state
        self.renderer_mode_state = RendererModeState()

        # V3: Initialize state evolution model
        self.state_evolution = StateEvolution()
        self._latent_state = np.zeros(11)  # Initial latent state
        self._latent_covariance = np.eye(11)  # Initial covariance

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
        else:
            self.identity_state = None
            self.patch_memory = None
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

        print(f"\n=== FACE OS PROCESSING ===")
        print(f"  Input: {video_path}")
        print(f"  Output: {output_path}")
        print(f"  Bidirectional: {self.use_bidirectional}")

        meta = ingest.load_video_meta(video_path)
        print(f"  Video: {meta.width}x{meta.height} @ {meta.fps:.1f}fps ({meta.total_frames} frames)")

        # Reset per-clip state
        self._reset_state()

        # Simple enhancement mode: skip bidirectional solve (needs identity_state)
        if not USE_IDENTITY:
            print("  Mode: SIMPLE ENHANCEMENT (no identity, no bidirectional)")
            return self._process_forward(video_path, output_path, max_frames, meta)

        if self.use_bidirectional:
            return self._process_bidirectional(video_path, output_path, max_frames, meta)
        else:
            return self._process_forward(video_path, output_path, max_frames, meta)

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
            # FIX: Apply last good crop instead of returning full frame (prevents dimension jump)
            crop_plan = self.crop.plan_crop(frame.shape[:2], None, None)
            return crop_planner.apply_crop(frame, crop_plan)

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
                self.identity_state.update(
                    canonical_face, masked_quality, pose=pose,
                    face_bbox=face_bbox,
                    landmarks_pts=landmarks_pts,
                    embedding=embedding,
                    mesh_478=mesh_478,
                    warp_M=M[:2] if M is not None else None,
                )

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
                    
                    # V3: Query intrinsic components
                    intrinsic_components, intrinsic_conf = self.identity_state.query_intrinsic(quality_map)
                    
                    # D-05: Query lighting-invariant albedo (decouple identity from lighting)
                    albedo_face, albedo_conf = self.identity_state.query_albedo(quality_map)
                    
                    # Render via shared _render_core
                    # NOTE: Mode update happens in orchestration layer, not here
                    # D-05: Use albedo as primary identity, fall back to RGB query
                    identity_face, identity_conf = self.identity_state.query_identity(quality_map)
                    # Blend albedo into identity face for lighting invariance
                    if albedo_face is not None and albedo_conf is not None:
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

                    # Post-sharpen to recover detail from low-res source
                    if output is not None:
                        output = face_enhance._sharpen(output, amount=0.8, radius=0.8)
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
        _, _, M = canonical_map.warp_to_canonical(
            cropped, adjusted_lm,
            canonical_size=tuple(cfg.canonical.atlas_size),
        )
        M_inv = np.linalg.inv(M)[:2]

        current_sim2 = self._affine_to_sim2(M_inv)
        if self._last_SIM2 is not None:
            interpolated = interpolate_sim2(self._last_SIM2, current_sim2, 0.6)
            M_inv = self._sim2_to_affine(interpolated)
        self._last_SIM2 = current_sim2

        identity_in_crop = cv2.warpAffine(
            identity_face, M_inv, (cropped.shape[1], cropped.shape[0]),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
        )

        canonical_mask = self._make_canonical_geometry_mask(identity_face.shape[:2])
        aligned_mask = cv2.warpAffine(
            canonical_mask, M_inv, (cropped.shape[1], cropped.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        aligned_mask = np.clip(aligned_mask, 0, 1)

        blend_3d = aligned_mask[:, :, np.newaxis]
        # D-01: Linear-light compositing (physically correct gamma handling)
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

        # Track normal source
        normal_source = self.identity_state.get_normal_source()
        if normal_source == "mesh":
            self._telemetry["mesh_normal_frames"] += 1
        else:
            self._telemetry["shading_normal_frames"] += 1

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
            # Observation matrix H: observe pose (0-2), brightness (9), contrast (10)
            H = np.zeros((5, self.state_evolution.state_dim))
            H[0, 0] = 1  # yaw
            H[1, 1] = 1  # pitch
            H[2, 2] = 1  # roll
            H[3, 9] = 1  # brightness
            H[4, 10] = 1  # contrast
            
            # Observation vector (only observed dimensions)
            obs_vector = np.array([observation[0], observation[1], observation[2], observation[9], observation[10]])
            
            # Observation noise R (measurement uncertainty)
            R = np.diag([1.0, 1.0, 1.0, 10.0, 5.0])  # pose is reliable, brightness/contrast less so
            
            self._latent_state, self._latent_covariance = self.state_evolution.predict_update_full(
                self._latent_state, self._latent_covariance,
                obs_vector, H, R,
            )

            # D-06: SIM(2) velocity prediction for occlusion recovery
            if self._last_SIM2 is not None and hasattr(self, '_prev_SIM2') and self._prev_SIM2 is not None:
                try:
                    predicted_sim2 = self.state_evolution.predict_with_velocity(
                        self._prev_SIM2, self._last_SIM2
                    )
                    # Store prediction for use during face-loss recovery
                    self._predicted_SIM2 = predicted_sim2
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
    ) -> None:
        """D-08: Emit per-frame telemetry JSON.

        Called from ALL render paths (physical, identity, enhancement)
        to ensure every frame is logged.
        """
        sim2_det = 1.0
        if self._last_SIM2 is not None:
            try:
                sim2_det = self._last_SIM2.scale ** 2
            except Exception:
                pass
        self._frame_telemetry_log.append({
            "frame_idx": frame_idx,
            "render_path": "physical" if self._telemetry["physical_render_frames"] > prev_physical else "alpha" if self._telemetry["alpha_fallback_frames"] > prev_alpha else "enhancement",
            "renderer_mode": self.renderer_mode_state.current_mode.value if self.renderer_mode_state else "unknown",
            "fallback_reason": fallback_reason,
            "intrinsic_used": intrinsic_components is not None,
            "geometry_source": self.identity_state.get_normal_source() if self.identity_state else "unknown",
            "resample_count": 2,
            "energy_terms": energy_terms,
            "transform_det": sim2_det,
        })

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
    ) -> Optional[np.ndarray]:
        """Shared rendering core — single source of truth for all rendering logic.

        Called by BOTH _process_frame_v2() and _render_frame_v2().

        Flow:
          1. Try PhysicalRenderer if intrinsic available + mode allows
          2. Fallback: identity composite (warp canonical face to crop + blend)
          3. Last resort: enhancement-only (sharpen + denoise)
        """
        # Track why we skip PhysicalRenderer (for telemetry)
        fallback_reason = None
        prev_physical = self._telemetry["physical_render_frames"]
        prev_alpha = self._telemetry["alpha_fallback_frames"]

        # RULE 7: Compute and normalize energy terms
        energy_terms = self._compute_energy_terms(
            intrinsic_components, identity_face, landmarks, frame_idx
        )

        # RULE 8: Timing
        import time as _time
        _render_start = _time.perf_counter()

        # 1. PhysicalRenderer
        physical_possible = (intrinsic_components is not None
                            and self.physical_renderer is not None
                            and self.renderer_mode_state is not None
                            and self.renderer_mode_state.current_mode in [RendererMode.PHYSICAL, RendererMode.HYBRID]
                            and landmarks is not None)

        if physical_possible:
            result = self._render_with_physical_renderer(
                source_frame, cropped, intrinsic_components, intrinsic_conf,
                landmarks, crop_plan, frame_idx, region_masks,
            )
            if result is not None:
                self._telemetry["physical_render_frames"] += 1
                # D-01: Temporal photometric locking
                result = photometric_lock(result, face_mask)
                # D-08: Per-frame telemetry (before early return)
                self._emit_frame_telemetry(
                    frame_idx, fallback_reason, intrinsic_components,
                    energy_terms, prev_physical, prev_alpha,
                )
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
                    # D-01: Post-sharpen identity composite path
                    output = face_enhance._sharpen(output, amount=0.8, radius=0.8)
                    # D-01: Temporal photometric locking
                    output = photometric_lock(output, face_mask)
                    self._telemetry["alpha_fallback_frames"] += 1
                    if fallback_reason:
                        fb_dist = self._telemetry["fallback_reason_distribution"]
                        fb_dist[fallback_reason] = fb_dist.get(fallback_reason, 0) + 1
                    # D-08: Per-frame telemetry (before early return)
                    self._emit_frame_telemetry(
                        frame_idx, fallback_reason, intrinsic_components,
                        energy_terms, prev_physical, prev_alpha,
                    )
                    return output
            except Exception as e:
                print(f"  Frame {frame_idx}: COMPOSITOR FAILED: {e}")

        # 3. Last resort: enhancement only
        enhancement_mask = None
        if region_masks:
            enhancement_mask = face_enhance._create_enhancement_mask(region_masks, cropped.shape)

        rendered = face_enhance.render_frame(
            cropped, enhancement_mask, region_masks,
            identity_eyes=identity_eyes,
            eye_confidence=eye_confidence,
        )
        # D-01: Post-sharpen last resort path
        rendered = face_enhance._sharpen(rendered, amount=0.8, radius=0.8)
        # D-01: Temporal photometric locking
        rendered = photometric_lock(rendered, face_mask)
        # RULE 8: Track render timing
        render_time_ms = (_time.perf_counter() - _render_start) * 1000
        self._telemetry["render_time_sum_ms"] += render_time_ms
        self._telemetry["render_time_count"] += 1

        # D-08: Per-frame telemetry JSON emission
        self._emit_frame_telemetry(
            frame_idx, fallback_reason, intrinsic_components,
            energy_terms, prev_physical, prev_alpha,
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
            
            # Warp anchor albedo to source crop space for correction
            if anchor_albedo is not None:
                _, _, M = canonical_map.warp_to_canonical(
                    cropped, self._adjust_landmarks_to_crop(landmarks, crop_plan) or landmarks,
                    canonical_size=tuple(cfg.canonical.atlas_size),
                )
                M_inv = np.linalg.inv(M)[:2]
                
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
            lighting = LightingModel(
                ambient=float(np.mean(source_intrinsic.shading)) * 0.3,
                diffuse_intensity=float(np.mean(source_intrinsic.shading)) * 0.8,
            )
            
            # Render in output space using PhysicalRenderer
            rendered_output = self.physical_renderer.render(
                albedo=source_intrinsic.albedo,
                normal_map=source_intrinsic.normal_map,
                shading=source_intrinsic.shading,
                lighting=lighting,
            )
            
            # Convert from [0,1] to [0,255] uint8
            rendered_face = (rendered_output.rendered * 255).astype(np.uint8)
            
            # Create face mask from landmarks
            adjusted_lm = self._adjust_landmarks_to_crop(landmarks, crop_plan)
            if adjusted_lm is None:
                return None
            
            _, _, M_mask = canonical_map.warp_to_canonical(
                cropped, adjusted_lm,
                canonical_size=tuple(cfg.canonical.atlas_size),
            )
            M_inv_mask = np.linalg.inv(M_mask)[:2]
            
            canonical_mask = self._make_canonical_geometry_mask(
                tuple(cfg.canonical.atlas_size)[::-1]
            )
            aligned_mask = cv2.warpAffine(
                canonical_mask, M_inv_mask, (cropped.shape[1], cropped.shape[0]),
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
            blended = _blend_linear(cropped, rendered_face, feathered_mask)

            # D-01: Detail residual injection — preserve source HF detail
            # The decomposition produces blurry albedo. Inject HF from source.
            face_mask_bool = feathered_mask > 0.3
            if face_mask_bool.sum() > 100:
                # Extract HF from source (sharp detail)
                source_hf = cropped.astype(np.float32) - cv2.GaussianBlur(cropped, (0, 0), 2.0).astype(np.float32)
                
                # Extract LF from rendered (appearance)
                rendered_lf = cv2.GaussianBlur(blended, (0, 0), 2.0).astype(np.float32)
                
                # Combine: LF from rendered + HF from source
                detail_strength = 0.9
                mask3 = (feathered_mask * detail_strength)[:, :, np.newaxis]
                output = rendered_lf + source_hf * mask3
                output = np.clip(output, 0, 255).astype(np.uint8)
            else:
                output = blended
            
            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx}: output-space render (albedo mean={np.mean(source_intrinsic.albedo):.3f})")
            
            return output

        except Exception as e:
            print(f"  Frame {frame_idx}: OUTPUT-SPACE RENDER FAILED: {e}")
            return None

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
        # D-06: Reset SIM2 prediction state
        self._prev_SIM2 = None
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
