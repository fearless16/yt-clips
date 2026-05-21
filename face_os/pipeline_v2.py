"""
pipeline_v2.py — Face OS V2 Pipeline Orchestrator.

THE MENTAL SHIFT:
  OLD: "how do I enhance this frame?"
  NEW: "what does this person's face usually look like?"

ARCHITECTURE:
  Face OS V2 decomposes into 4 isolated subsystems:
  1. Geometry Estimator - estimates all spatial structure
  2. Identity Estimator - estimates stable identity representation
  3. Temporal Estimator - maintains temporal consistency
  4. Renderer - generates physically consistent output

CORE EQUATION:
  Y = M ⊙ Y_face + (1 - M) ⊙ Y_bg
  
  Where:
  - M is geometry-derived semantic mask
  - Y_face is latent-rendered face  
  - Y_bg is untouched background
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
    GeometryState,
    IdentityState,
    TemporalState,
)

# Module imports
from face_os import ingest
from face_os import detect_track
from face_os import landmarks as lm_module
from face_os import canonical_map
from face_os import crop_planner
from face_os import face_enhance
from face_os import compositor
from face_os import export_qc

# NEW V2 subsystems
from face_os.subsystems.geometry_estimator import GeometryEstimator
from face_os.subsystems.identity_estimator import IdentityEstimator
from face_os.subsystems.temporal_estimator import TemporalEstimator
from face_os.subsystems.renderer import Renderer


cfg = get_config()

# ─── Feature flags ──────────────────────────────────────────────────────────
USE_IDENTITY = True


class FaceOSPipelineV2:
    """Face OS V2 Pipeline - subsystem-based architecture.
    
    Philosophy:
      - Source video is TELEMETRY, not ground truth
      - Each frame is a noisy photon observation
      - Maintain IDENTITY BELIEF STATE
      - Query memory, don't enhance pixels
      - Frequency decomposition: low freq smooth, high freq best-only
      - Per-region independent dynamics
      - Bidirectional temporal solve (offline superpower)
    """

    def __init__(self, use_bidirectional: bool = True):
        # Core modules
        self.tracker: Optional[detect_track.FaceTracker] = None
        self.appearance_builder: Optional[canonical_map.AppearanceFieldBuilder] = None
        self.crop: Optional[crop_planner.CropPlanner] = None
        self.compositor: Optional[compositor.Compositor] = None

        # V2 Subsystems
        self.geometry_estimator: Optional[GeometryEstimator] = None
        self.identity_estimator: Optional[IdentityEstimator] = None
        self.temporal_estimator: Optional[TemporalEstimator] = None
        self.renderer: Optional[Renderer] = None

        # Identity profile
        self.identity: Optional[IdentityProfile] = None

        # State
        self._enrolled = False
        self._frame_count = 0

        # Face lock state machine
        self._face_state = "LOST_FACE"  # FACE_LOCKED, LOST_FACE, RECOVERY
        self._lost_frame_count = 0
        self._recovery_frame_count = 0

        # Last good crop plan for fallback paths (prevents frame size change)
        self._last_good_crop_plan: Optional[CropPlan] = None
        
        # Bidirectional solver
        self.use_bidirectional = use_bidirectional

    def enroll(
        self,
        reference_image: str = "expectation.png",
        reference_dir: str = "photos/",
    ) -> bool:
        """Enroll the target identity from reference images."""
        print("=== FACE OS V2 ENROLLMENT ===")

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

        # Initialize core modules
        self.tracker = detect_track.FaceTracker(self.identity.embeddings)
        self.appearance_builder = canonical_map.AppearanceFieldBuilder()
        self.crop = crop_planner.CropPlanner(reference_image=reference_image)
        self.compositor = compositor.Compositor()

        # Extract reference mesh for quality gates
        ref_mesh = detect_track.extract_face_mesh(primary)
        if ref_mesh is not None:
            self.tracker.set_reference_mesh(ref_mesh)
            print(f"  Reference mesh: {ref_mesh.shape[0]} landmarks")
        else:
            print("  WARNING: Could not extract reference mesh — quality gates will be relaxed")

        # Initialize V2 subsystems
        self.geometry_estimator = GeometryEstimator(self.crop)
        self.identity_estimator = IdentityEstimator()
        self.temporal_estimator = TemporalEstimator()
        self.renderer = Renderer()

        # Set identity anchor from reference
        if USE_IDENTITY and self.identity.embeddings:
            if self.identity.enrolled and self.identity.appearance.atlas_rgb is not None:
                self.appearance_builder.atlas = self.identity.appearance
                ref_rgb = self.identity.appearance.atlas_rgb
                if ref_rgb is not None:
                    ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)
                    self.identity_estimator.set_anchor(ref_bgr)
                    print(f"  Identity anchor set from reference")

        self._enrolled = True
        print("  Enrollment complete.")
        return True

    def process(
        self,
        video_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
    ) -> Optional[str]:
        """Process a video through the V2 pipeline."""
        if not self._enrolled:
            print("ERROR: Must enroll before processing.")
            return None

        print(f"\n=== FACE OS V2 PROCESSING ===")
        print(f"  Input: {video_path}")
        print(f"  Output: {output_path}")
        print(f"  Bidirectional: {self.use_bidirectional}")

        meta = ingest.load_video_meta(video_path)
        print(f"  Video: {meta.width}x{meta.height} @ {meta.fps:.1f}fps ({meta.total_frames} frames)")

        # Reset per-clip state
        self._reset_state()

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
        """Standard forward-only processing using V2 subsystems."""
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
        
        previous_geometry_state = None
        previous_identity_state = None
        previous_temporal_state = None

        try:
            for frame_idx, timestamp, source_frame in ingest.frame_reader(video_path):
                if max_frames and total_frames >= max_frames:
                    break

                result = self._process_frame_v2(
                    source_frame, frame_idx, timestamp,
                    previous_geometry_state, previous_identity_state, previous_temporal_state
                )

                if result is not None:
                    exporter.write_frame(result)
                    all_frames.append(result)
                    if self.tracker and self.tracker.tracks:
                        face_detected_frames += 1

                # Update previous states for next frame
                if result is not None and self.geometry_estimator:
                    # These would be set in _process_frame_v2
                    pass

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
        """Bidirectional processing using V2 subsystems."""
        # === PASS 1: Forward collection ===
        print("  Pass 1/3: Forward collection...")
        canonical_faces = {}
        quality_maps = {}
        frame_data = {}
        total_frames = 0
        t_start = time.perf_counter()

        for frame_idx, timestamp, source_frame in ingest.frame_reader(video_path):
            if max_frames and total_frames >= max_frames:
                break

            # Use geometry estimator
            face_track = self.tracker.process_frame(source_frame, frame_idx)
            geometry_state = self.geometry_estimator.estimate(source_frame, face_track)
            
            crop_plan = geometry_state.crop_transform
            if crop_plan is not None:
                self._last_good_crop_plan = crop_plan

            if geometry_state.landmarks and face_track.detection:
                try:
                    canonical_face = geometry_state.canonical_face
                    if canonical_face is not None:
                        # Compute quality map
                        quality_map = self._compute_quality_map(canonical_face, face_track.detection.confidence)
                        
                        # Compute sharpness
                        gray = cv2.cvtColor(canonical_face, cv2.COLOR_BGR2GRAY)
                        sharpness = float(np.mean(np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))))
                        sharpness = np.clip(sharpness / 100.0, 0, 1)

                        # Store for bidirectional solver
                        canonical_faces[frame_idx] = canonical_face
                        quality_maps[frame_idx] = quality_map

                        # Collect for temporal solver
                        self.temporal_estimator.collect_frame_for_bidirectional_solve(
                            frame_idx, canonical_face, quality_map,
                            sharpness=sharpness,
                            pose=geometry_state.pose,
                            detection_confidence=face_track.detection.confidence,
                        )

                        frame_data[frame_idx] = (source_frame, face_track, geometry_state, crop_plan)

                except Exception:
                    frame_data[frame_idx] = (source_frame, face_track, geometry_state, crop_plan)

            total_frames += 1

            if total_frames % 100 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"    {total_frames} frames ({total_frames / max(elapsed, 0.001):.0f} fps)")

        print(f"    Collected {len(canonical_faces)} canonical faces from {total_frames} frames")

        # === PASS 2: Bidirectional solve ===
        print("  Pass 2/3: Bidirectional temporal solve...")
        solved_faces = self.temporal_estimator.solve_bidirectional()
        hq_count = self.temporal_estimator.temporal_solver.solver.get_hq_frame_count()
        print(f"    Solved {len(solved_faces)} frames, {hq_count} HQ frames")

        # Update identity state with solved faces
        for idx, (solved_face, solved_conf) in solved_faces.items():
            if self.identity_estimator:
                # Create temporary geometry state for identity update
                temp_geometry = GeometryState(
                    canonical_face=solved_face,
                    pose=(0.0, 0.0, 0.0)  # Unknown pose for solved faces
                )
                self.identity_estimator.estimate(temp_geometry, solved_conf)

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

        for frame_idx, timestamp, source_frame in ingest.frame_reader(video_path):
            if max_frames and frame_idx >= max_frames:
                break

            if frame_idx in frame_data:
                _, face_track, geometry_state, crop_plan = frame_data[frame_idx]

                # Get solved canonical face
                solved_face = None
                solved_conf = None
                if frame_idx in solved_faces:
                    solved_face, solved_conf = solved_faces[frame_idx]

                # Render frame using V2 renderer
                result = self._render_frame_v2(
                    source_frame, frame_idx, geometry_state, crop_plan,
                    solved_face=solved_face, solved_conf=solved_conf,
                )

                if result is not None:
                    exporter.write_frame(result)
                    all_frames.append(result)
                    face_detected_frames += 1
            else:
                # No face detected — apply last known crop
                crop_plan = self._last_good_crop_plan or self.crop.plan_crop(source_frame.shape[:2], None, None)
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
        previous_geometry_state = None,
        previous_identity_state = None,
        previous_temporal_state = None,
    ) -> Optional[np.ndarray]:
        """Process a single frame through the V2 subsystem pipeline."""
        self._frame_count = frame_idx

        # 1. Detect & track
        face_track = self.tracker.process_frame(frame, frame_idx)

        # ═══════════════════════════════════════════════════════════════════
        # FACE LOCK STATE MACHINE
        # ═══════════════════════════════════════════════════════════════════
        face_detected = face_track is not None and face_track.mesh_478 is not None
        detection_conf = face_track.detection.confidence if face_track and face_track.detection else 0.0

        # Compute occupancy estimate
        occupancy = 0.0
        if face_detected and face_track.smooth_bbox:
            x, y, w, h = face_track.smooth_bbox
            bbox_area = w * h
            if hasattr(face_track, 'landmarks') and face_track.landmarks and face_track.landmarks.points is not None:
                pts = np.array(face_track.landmarks.points)
                hull_area = cv2.contourArea(cv2.convexHull(pts.astype(np.float32)))
                occupancy = hull_area / max(bbox_area, 1)

        # State transitions
        if face_detected and occupancy > 0.25 and detection_conf > 0.5:
            if self._face_state == "LOST_FACE":
                self._face_state = "RECOVERY"
                self._recovery_frame_count = 0
            elif self._face_state == "RECOVERY":
                self._recovery_frame_count += 1
                if self._recovery_frame_count > 5:
                    self._face_state = "FACE_LOCKED"
            else:
                self._face_state = "FACE_LOCKED"
            self._lost_frame_count = 0
        else:
            self._lost_frame_count += 1
            if self._face_state != "LOST_FACE":
                self._face_state = "LOST_FACE"

        # GATE: Skip identity update if face is lost
        if self._face_state == "LOST_FACE":
            crop_plan = self.crop.plan_crop(frame.shape[:2], None, None)
            return crop_planner.apply_crop(frame, crop_plan)

        # 2. Geometry Estimation (Subsystem A)
        geometry_state = self.geometry_estimator.estimate(
            frame, face_track, previous_geometry_state
        )

        # 3. Identity Estimation (Subsystem B)
        quality_map = None
        if geometry_state.canonical_face is not None:
            quality_map = self._compute_quality_map(
                geometry_state.canonical_face, detection_conf
            )
            
        identity_state = self.identity_estimator.estimate(
            geometry_state, quality_map, face_track
        )

        # 4. Temporal Estimation (Subsystem C)
        temporal_state = self.temporal_estimator.estimate(
            geometry_state, identity_state, previous_temporal_state
        )

        # 5. Rendering (Subsystem D)
        crop_plan = geometry_state.crop_transform
        output = self.renderer.render(
            frame, geometry_state, identity_state, temporal_state, crop_plan
        )

        return output

    def _render_frame_v2(
        self,
        source_frame: np.ndarray,
        frame_idx: int,
        geometry_state,
        crop_plan: CropPlan,
        solved_face: Optional[np.ndarray] = None,
        solved_conf: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Render a frame using V2 renderer."""
        # Apply crop
        cropped = crop_planner.apply_crop(source_frame, crop_plan)

        # Get region masks
        region_masks = geometry_state.semantic_regions
        face_mask = region_masks.get("face") if region_masks else None

        # ─── SIMPLE ENHANCEMENT MODE (no identity) ───────────────────────
        if not USE_IDENTITY:
            enhancement_mask = face_enhance._create_enhancement_mask(region_masks, cropped.shape) if region_masks else None
            rendered = face_enhance.render_frame(
                cropped, enhancement_mask, region_masks,
                identity_eyes=None, eye_confidence=0.0,
            )
            return rendered

        # ─── IDENTITY RECONSTRUCTION MODE ────────────────────────────────
        if solved_face is not None:
            try:
                # Query identity state for anchor-corrected appearance
                if self.identity_estimator.is_initialized():
                    quality_map = self._compute_quality_map(solved_face, 0.5)
                    identity_face, identity_conf = self.identity_estimator.identity_belief.query_identity(quality_map)
                    solved_face = identity_face

                # Warp solved face back to source crop space
                if geometry_state.inverse_transform is not None:
                    M_inv = geometry_state.inverse_transform
                    solved_in_crop = cv2.warpAffine(
                        solved_face, M_inv, (cropped.shape[1], cropped.shape[0]),
                        flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_REFLECT,
                    )

                    # Use geometry-based mask
                    if geometry_state.mask is not None:
                        canonical_mask = geometry_state.mask
                        aligned_mask = cv2.warpAffine(
                            canonical_mask, M_inv, (cropped.shape[1], cropped.shape[0]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        aligned_mask = np.clip(aligned_mask, 0, 1)
                        
                        # Feather the mask
                        feather_ksize = max(3, cfg.compositor.feather_pixels * 2 + 1)
                        feathered_mask = cv2.GaussianBlur(aligned_mask, (feather_ksize, feather_ksize), cfg.compositor.feather_pixels / 2)

                        # Blend
                        conf_3d = feathered_mask[:, :, np.newaxis]
                        blended = cropped.astype(np.float32) * (1 - conf_3d) + solved_in_crop.astype(np.float32) * conf_3d
                        blended = np.clip(blended, 0, 255).astype(np.uint8)

                        # Apply structure-preserving rendering
                        enhancement_mask = face_enhance._create_enhancement_mask(region_masks, blended.shape) if region_masks else None
                        rendered = face_enhance.render_frame(
                            blended, enhancement_mask, region_masks,
                            identity_eyes=None, eye_confidence=0.0,
                        )

                        # Post-sharpen
                        output = face_enhance._sharpen(rendered, amount=0.3, radius=0.8)
                        return output

            except Exception as e:
                print(f"  Frame {frame_idx}: IDENTITY PATH FAILED: {e}")

        # Fallback: structure-preserving rendering only
        enhancement_mask = face_enhance._create_enhancement_mask(region_masks, cropped.shape) if region_masks else None
        rendered = face_enhance.render_frame(
            cropped, enhancement_mask, region_masks,
            identity_eyes=None, eye_confidence=0.0,
        )
        return rendered

    @staticmethod
    def _compute_quality_map(
        canonical_face: np.ndarray,
        detection_confidence: float,
    ) -> np.ndarray:
        """Compute per-pixel quality map for a canonical face."""
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

    def _reset_state(self) -> None:
        """Reset per-clip state."""
        if self.crop:
            self.crop.reset()
        if self.compositor:
            self.compositor.reset()
        self._frame_count = 0
        self._last_good_crop_plan = None


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Face OS V2 — Subsystem Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--reference", default="expectation.png", help="Reference image")
    parser.add_argument("--photos", default="photos/", help="Reference photos directory")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no-bidirectional", action="store_true", help="Disable bidirectional solve")
    parser.add_argument("--no-identity", action="store_true", help="Disable identity memory")

    args = parser.parse_args()

    global USE_IDENTITY
    if args.no_identity:
        USE_IDENTITY = False

    output = args.output or "output/face_os/output.mp4"

    pipeline = FaceOSPipelineV2(use_bidirectional=not args.no_bidirectional)
    if not pipeline.enroll(args.reference, args.photos):
        return

    result = pipeline.process(args.video, output, max_frames=args.max_frames)
    if result:
        print(f"\nSuccess: {result}")
    else:
        print("\nFailed.")


if __name__ == "__main__":
    main()