"""
pipeline.py — Face OS Pipeline Orchestrator.

Ties all 10 modules together into a single processing pipeline.

Usage:
    from face_os.pipeline import FaceOSPipeline

    pipeline = FaceOSPipeline()
    pipeline.enroll("expectation.png", reference_dir="photos/")
    pipeline.process("input/video.mp4", "output/shorts/")

CLI:
    python -m face_os.pipeline --video input/video.mp4 --reference expectation.png
"""

import argparse
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
from face_os import temporal_stabilize
from face_os import face_enhance
from face_os import identity_memory
from face_os import compositor
from face_os import export_qc


cfg = get_config()


class FaceOSPipeline:
    """The main Face OS processing pipeline.

    Philosophy:
      - Overfit is the feature. One face, one environment, one camera.
      - Face is a dynamic appearance function, not a static render.
      - Pixels are noisy photon observations — accumulate confidence.
      - Identity inertia: source fluctuates, identity stays stable.
      - Eyes dominate perception — always highest fidelity.

    Pipeline flow per frame:
      1. Detect & track face
      2. Extract landmarks + pose
      3. Map to canonical space + update appearance field
      4. Plan 9:16 crop with headroom
      5. Apply temporal stabilization
      6. Enhance face regions (eye-dominant)
      7. Update identity memory
      8. Composite using confidence weights
      9. Write to output video
      10. Run QC checks
    """

    def __init__(self):
        # Initialize modules
        self.tracker: Optional[detect_track.FaceTracker] = None
        self.appearance_builder: Optional[canonical_map.AppearanceFieldBuilder] = None
        self.memory_atlas: Optional[identity_memory.IdentityMemoryAtlas] = None
        self.crop: Optional[crop_planner.CropPlanner] = None
        self.temporal: Optional[temporal_stabilize.TemporalStabilizer] = None
        self.compositor: Optional[compositor.Compositor] = None

        # Identity profile
        self.identity: Optional[IdentityProfile] = None

        # State
        self._enrolled = False
        self._frame_count = 0

    def enroll(
        self,
        reference_image: str = "expectation.png",
        reference_dir: str = "photos/",
    ) -> bool:
        """Enroll the target identity from reference images.

        This is the "Apple Face ID" style enrollment:
        1. Load reference images
        2. Extract face embeddings for identity matching
        3. Build initial canonical appearance atlas

        Args:
            reference_image: Path to primary reference image
            reference_dir: Directory with additional reference photos

        Returns:
            True if enrollment succeeded
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
        self.memory_atlas = identity_memory.IdentityMemoryAtlas()
        self.crop = crop_planner.CropPlanner()
        self.temporal = temporal_stabilize.TemporalStabilizer()
        self.compositor = compositor.Compositor()

        # Pre-populate appearance builder from reference
        if self.identity.enrolled and self.identity.appearance.atlas_rgb is not None:
            self.appearance_builder.atlas = self.identity.appearance

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

        Args:
            video_path: Input video path
            output_path: Output video path
            max_frames: Maximum frames to process (None = all)

        Returns:
            Output path on success, None on failure
        """
        if not self._enrolled:
            print("ERROR: Must enroll before processing. Call enroll() first.")
            return None

        print(f"\n=== FACE OS PROCESSING ===")
        print(f"  Input: {video_path}")
        print(f"  Output: {output_path}")

        # Load video metadata
        meta = ingest.load_video_meta(video_path)
        print(f"  Video: {meta.width}x{meta.height} @ {meta.fps:.1f}fps ({meta.total_frames} frames)")

        # Reset per-clip state
        self._reset_state()

        # Open exporter
        exporter = export_qc.VideoExporter(
            output_path,
            fps=cfg.export.fps,
            width=cfg.export.output_size[0] if hasattr(cfg.export, 'output_size') else cfg.crop.output_size[0],
            height=cfg.export.output_size[1] if hasattr(cfg.export, 'output_size') else cfg.crop.output_size[1],
            source_path=video_path,
        )

        # QC tracking
        face_detected_frames = 0
        total_frames = 0
        all_frames = []

        t_start = time.perf_counter()

        try:
            # Process each frame
            for frame_idx, timestamp, source_frame in ingest.frame_reader(video_path):
                if max_frames and total_frames >= max_frames:
                    break

                # Process frame through pipeline
                result = self._process_frame(source_frame, frame_idx, timestamp)

                if result is not None:
                    # Write to output
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
            # Close exporter
            exporter.close()

        elapsed = time.perf_counter() - t_start

        # Apply fades
        if cfg.export.fade_in > 0 or cfg.export.fade_out > 0:
            print("  Applying fades...")
            export_qc.apply_fades(
                output_path,
                fade_in=cfg.export.fade_in,
                fade_out=cfg.export.fade_out,
            )

        # Run QC
        print("  Running QC checks...")
        ref_lab = None
        if self.identity and self.identity.appearance.atlas_lab is not None:
            ref_lab = tuple(np.mean(
                self.identity.appearance.atlas_lab,
                axis=(0, 1)
            ).tolist())

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

        return output_path

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
    ) -> Optional[np.ndarray]:
        """Process a single frame through the pipeline.

        Flow:
          1. Detect & track
          2. Landmarks + pose
          3. Canonical mapping
          4. Crop planning
          5. Temporal stabilization
          6. Face enhancement
          7. Identity memory update
          8. Compositing
        """
        self._frame_count = frame_idx

        # 1. Detect & track
        face_track = self.tracker.process_frame(frame, frame_idx)

        # 2. Landmarks + pose
        landmarks = None
        if face_track and face_track.smooth_bbox:
            landmarks = lm_module.extract_landmarks(frame, face_track.smooth_bbox)
            face_track.landmarks = landmarks

        # 3. Canonical mapping + appearance field update
        if landmarks and face_track.detection:
            self.appearance_builder.update(
                frame, landmarks,
                detection_confidence=face_track.detection.confidence,
            )

        # 4. Crop planning
        crop_plan = self.crop.plan_crop(
            frame.shape[:2], face_track, landmarks,
        )

        # Apply crop
        cropped = crop_planner.apply_crop(frame, crop_plan)

        # 5. Temporal stabilization
        # Create face mask for region-specific stabilization
        face_mask = None
        region_masks = None
        if landmarks:
            # Adjust landmarks to cropped space
            adjusted_lm = self._adjust_landmarks_to_crop(landmarks, crop_plan)
            if adjusted_lm:
                region_masks = lm_module.create_region_masks(adjusted_lm, cropped.shape[:2])
                face_mask = region_masks.get("face")

        stabilized = self.temporal.stabilize_face_region(
            cropped, face_mask, face_track,
        )

        # 6. Face enhancement
        enhancement_mask = None
        if region_masks:
            enhancement_mask = face_enhance.create_enhancement_mask(
                region_masks, stabilized.shape,
            )

        enhanced = face_enhance.enhance_frame(
            stabilized, enhancement_mask, region_masks,
        )

        # 7. Identity memory update
        confidence = None
        if face_track and face_track.smooth_bbox and landmarks:
            # Extract face region for memory
            face_region = self._extract_face_region(cropped, landmarks, crop_plan)
            if face_region is not None:
                pose = (landmarks.yaw, landmarks.pitch, landmarks.roll)
                conf = face_track.detection.confidence if face_track.detection else 0.5
                confidence = self.memory_atlas.update(face_region, conf, pose)

        # 8. Compositing
        memory_face = self.memory_atlas.get_stable_face()

        if memory_face is not None and face_mask is not None:
            output = self.compositor.composite_with_memory(
                stabilized, memory_face, confidence or ConfidenceMap(), face_mask,
            )
            # Blend enhanced features on top
            if enhancement_mask:
                # Use enhanced version for eye/brow regions
                eye_blend = enhancement_mask.eye_mask[:, :, np.newaxis] * 0.5
                brow_blend = enhancement_mask.brow_mask[:, :, np.newaxis] * 0.3
                feature_blend = np.clip(eye_blend + brow_blend, 0, 1)
                output = output.astype(np.float32) * (1 - feature_blend) + enhanced.astype(np.float32) * feature_blend
                output = np.clip(output, 0, 255).astype(np.uint8)
        else:
            output = enhanced

        return output

    def _adjust_landmarks_to_crop(
        self,
        landmarks: Landmarks,
        crop_plan: CropPlan,
    ) -> Optional[Landmarks]:
        """Adjust landmark coordinates from source space to cropped space."""
        if crop_plan.src_w <= 0 or crop_plan.src_h <= 0:
            return None

        # Scale factors
        sx = crop_plan.dst_w / crop_plan.src_w
        sy = crop_plan.dst_h / crop_plan.src_h

        # Offset
        ox = crop_plan.src_x
        oy = crop_plan.src_y

        # Transform points
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

    def _extract_face_region(
        self,
        cropped: np.ndarray,
        landmarks: Landmarks,
        crop_plan: CropPlan,
    ) -> Optional[np.ndarray]:
        """Extract aligned face region for memory atlas."""
        adjusted = self._adjust_landmarks_to_crop(landmarks, crop_plan)
        if adjusted is None:
            return None

        try:
            # Warp to canonical space
            warped, _, _ = canonical_map.warp_to_canonical(cropped, adjusted)
            return cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _reset_state(self) -> None:
        """Reset per-clip state."""
        if self.crop:
            self.crop.reset()
        if self.temporal:
            self.temporal.reset()
        if self.memory_atlas:
            self.memory_atlas.reset()
        if self.compositor:
            self.compositor.reset()
        self._frame_count = 0


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Face OS — Personal Face Operating System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--reference", default="expectation.png", help="Reference image")
    parser.add_argument("--photos", default="photos/", help="Reference photos directory")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")

    args = parser.parse_args()

    output = args.output or "output/face_os/output.mp4"

    pipeline = FaceOSPipeline()
    if not pipeline.enroll(args.reference, args.photos):
        return

    result = pipeline.process(args.video, output, max_frames=args.max_frames)
    if result:
        print(f"\nSuccess: {result}")
    else:
        print("\nFailed.")


if __name__ == "__main__":
    main()
