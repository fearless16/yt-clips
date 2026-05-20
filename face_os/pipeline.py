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
from face_os import export_qc

# NEW modules
from face_os.identity_state import IdentityState
from face_os.patch_memory import PatchMemory
from face_os.temporal_solve import TemporalRepairEngine, FrameQuality


cfg = get_config()


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

        # NEW: Initialize identity belief state
        self.identity_state = IdentityState()
        self.patch_memory = PatchMemory()

        # Pre-populate from reference
        if self.identity.enrolled and self.identity.appearance.atlas_rgb is not None:
            self.appearance_builder.atlas = self.identity.appearance

            # Initialize identity state from reference
            ref_rgb = self.identity.appearance.atlas_rgb
            if ref_rgb is not None:
                h, w = ref_rgb.shape[:2]
                quality = np.ones((h, w), dtype=np.float32) * 0.9
                ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)

                # Module D: Set identity anchor from reference
                # This prevents identity drift — all reconstructions must stay close to anchor
                self.identity_state.set_anchor(ref_bgr)
                print(f"  Anchor set from reference (LAB distance threshold: {self.identity_state._anchor_threshold})")

                # Pre-populate identity state with MULTIPLE reference observations
                # This gives the identity state a strong starting point
                # Like a Bayesian prior — strong belief from reference
                for _ in range(50):
                    self.identity_state.update(ref_bgr, quality, pose=(0, 0, 0))

                print(f"  Identity pre-populated with 50 reference observations")

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
                landmarks = lm_module.extract_landmarks(source_frame, face_track.smooth_bbox)
                face_track.landmarks = landmarks

            crop_plan = self.crop.plan_crop(source_frame.shape[:2], face_track, landmarks)

            if landmarks and face_track.detection:
                # Warp to canonical space
                try:
                    warped_rgb, warped_lab, M = canonical_map.warp_to_canonical(
                        source_frame, landmarks
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

        # Update identity state with solved faces
        for idx, (solved_face, solved_conf) in solved_faces.items():
            self.identity_state.update(solved_face, solved_conf, pose=None)
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
                # No face detected — write original
                cropped = source_frame
                if frame_idx in frame_data:
                    _, _, _, crop_plan = frame_data[frame_idx]
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
            landmarks = lm_module.extract_landmarks(frame, face_track.smooth_bbox)
            face_track.landmarks = landmarks

        # 3. Canonical alignment
        canonical_face = None
        quality_map = None
        if landmarks and face_track.detection:
            try:
                warped_rgb, warped_lab, M = canonical_map.warp_to_canonical(frame, landmarks)
                canonical_face = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)
                quality_map = self._compute_quality_map(canonical_face, face_track.detection.confidence)
            except Exception:
                pass

        # 4. Identity state update
        if canonical_face is not None and quality_map is not None:
            pose = (landmarks.yaw, landmarks.pitch, landmarks.roll) if landmarks else None
            self.identity_state.update(canonical_face, quality_map, pose=pose)

        # 5. Patch memory update
        if canonical_face is not None and quality_map is not None and landmarks:
            pose = (landmarks.yaw, landmarks.pitch, landmarks.roll)
            # Detect blink for eye freeze
            is_blink = self._detect_blink(landmarks) if landmarks else False
            self.patch_memory.update(canonical_face, quality_map, pose=pose, is_blink=is_blink, frame_idx=frame_idx)

        # 6. Query identity (THE MENTAL SHIFT)
        identity_face = None
        identity_confidence = None
        if canonical_face is not None and quality_map is not None:
            identity_face, identity_confidence = self.identity_state.query(canonical_face, quality_map)

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

        # 9. Get identity eyes for structure-preserving rendering
        identity_eyes = None
        eye_confidence = 0.0
        if self.patch_memory and self.patch_memory._initialized and landmarks:
            pose = (landmarks.yaw, landmarks.pitch, landmarks.roll)
            left_eye, left_conf = self.patch_memory.query_region('left_eye', pose)
            right_eye, right_conf = self.patch_memory.query_region('right_eye', pose)
            if left_eye is not None and right_eye is not None:
                # Combine eye patches (simplified — in canonical space)
                identity_eyes = left_eye  # Use left as reference
                eye_confidence = (left_conf + right_conf) / 2

        # 10. Render (structure-preserving)
        enhancement_mask = None
        if region_masks:
            enhancement_mask = face_enhance._create_enhancement_mask(region_masks, cropped.shape)

        rendered = face_enhance.render_frame(
            cropped, enhancement_mask, region_masks,
            identity_eyes=identity_eyes,
            eye_confidence=eye_confidence,
        )

        # 11. Composite
        if identity_face is not None and face_mask is not None:
            # Warp identity face back to cropped space
            # (simplified — use confidence-weighted blend)
            conf = identity_confidence if identity_confidence is not None else np.ones(cropped.shape[:2], dtype=np.float32) * 0.3

            if conf.shape[:2] != cropped.shape[:2]:
                conf = cv2.resize(conf, (cropped.shape[1], cropped.shape[0]))

            # The compositor handles the final blend
            output = self.compositor.composite(
                cropped, rendered,
                confidence=ConfidenceMap(combined=conf),
                face_mask=face_mask,
            )
        else:
            output = rendered

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
        """Render a frame using solved identity data.

        THE KEY INSIGHT: The solved canonical face IS the identity belief.
        Warp it back to source space and composite with confidence.

        MODULE D: Apply anchor correction AFTER blending with source.
        This ensures the output identity stays close to reference,
        even when confidence is low.
        """
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

        # If we have a solved canonical face, warp it back to source space
        if solved_face is not None and landmarks is not None:
            try:
                # Module D: Query identity state for anchor-corrected appearance
                if self.identity_state.is_initialized():
                    # Compute quality map for current frame
                    quality_map = self._compute_quality_map(solved_face, face_track.detection.confidence if face_track and face_track.detection else 0.5)
                    identity_face, identity_conf = self.identity_state.query(solved_face, quality_map)
                    # Use anchor-corrected identity instead of raw solved face
                    solved_face = identity_face
                    solved_conf = identity_conf

                # Warp solved face back to source crop space
                _, _, M = canonical_map.warp_to_canonical(cropped, self._adjust_landmarks_to_crop(landmarks, crop_plan) or landmarks)
                M_inv = np.linalg.inv(M)[:2]
                solved_in_crop = cv2.warpAffine(
                    solved_face, M_inv, (cropped.shape[1], cropped.shape[0]),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT,
                )

                # Blend solved identity with source using confidence
                if solved_conf is not None:
                    conf = cv2.resize(solved_conf, (cropped.shape[1], cropped.shape[0]))
                else:
                    conf = np.ones(cropped.shape[:2], dtype=np.float32) * 0.5

                # Higher confidence = more identity (solved), less source
                conf_3d = conf[:, :, np.newaxis]
                blended = cropped.astype(np.float32) * (1 - conf_3d) + solved_in_crop.astype(np.float32) * conf_3d
                blended = np.clip(blended, 0, 255).astype(np.uint8)

                # Module D: Apply anchor correction AFTER blending
                # This pulls the blended result toward reference, ensuring
                # identity stays close to anchor even with low confidence
                if self.identity_state.is_initialized() and face_mask is not None:
                    blended = self._apply_anchor_to_frame(blended, face_mask)

                # Apply structure-preserving rendering on top
                enhancement_mask = None
                if region_masks:
                    enhancement_mask = face_enhance._create_enhancement_mask(region_masks, blended.shape)

                rendered = face_enhance.render_frame(
                    blended, enhancement_mask, region_masks,
                    identity_eyes=None, eye_confidence=0.0,
                )

                # DON'T composite back with original — that would undo
                # the anchor correction. The anchor-corrected + enhanced
                # frame IS the final result.
                output = rendered

                # Post-sharpen to recover detail from low-res source
                output = face_enhance._sharpen(output, amount=0.3, radius=0.8)
                return output

            except Exception:
                pass

        # Fallback: structure-preserving rendering only
        enhancement_mask = None
        if region_masks:
            enhancement_mask = face_enhance._create_enhancement_mask(region_masks, cropped.shape)

        rendered = face_enhance.render_frame(
            cropped, enhancement_mask, region_masks,
            identity_eyes=None, eye_confidence=0.0,
        )

        return rendered

    def _apply_anchor_to_frame(
        self,
        frame: np.ndarray,
        face_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply anchor correction to the rendered frame.

        Module D: Pull face region toward reference brightness/tone.
        This is applied AFTER blending with source, ensuring the output
        identity stays close to reference even when confidence is low.

        Math:
          For face pixels:
            result = frame + (anchor_mean - face_mean) * pull_strength * face_mask

        Args:
            frame: Rendered frame (BGR)
            face_mask: Face region mask (H, W) float [0, 1]

        Returns:
            Anchor-corrected frame (BGR)
        """
        if not self.identity_state.is_initialized():
            return frame

        # Get anchor LAB values
        anchor_lab = self.identity_state._anchor_lab
        if anchor_lab is None:
            return frame

        # Get anchor mean (reference face)
        anchor_mean = np.mean(anchor_lab, axis=(0, 1))  # [L, a, b]

        # Get current face mean
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        face_mask_bool = face_mask > 0.5
        if face_mask_bool.sum() < 100:
            return frame

        face_mean = np.mean(frame_lab[face_mask_bool], axis=0)  # [L, a, b]

        # Calculate correction needed
        diff = anchor_mean - face_mean  # [ΔL, Δa, Δb]

        # Apply correction with distance-proportional strength
        distance = np.sqrt(np.sum(diff ** 2))
        if distance < 5.0:
            return frame  # Already close enough

        # Pull strength: more aggressive for larger drift
        # Like SLAM loop closure — strong correction when far from anchor
        if distance > 30:
            pull = 0.90  # Very strong pull for large drift
        elif distance > 15:
            pull = 0.80  # Strong pull
        else:
            pull = 0.60  # Moderate pull (even when close)

        # Apply correction to face region only
        correction = np.zeros_like(frame_lab)
        correction[:, :, 0] = diff[0] * pull  # L channel
        correction[:, :, 1] = diff[1] * pull * 0.5  # a channel (less aggressive)
        correction[:, :, 2] = diff[2] * pull * 0.5  # b channel (less aggressive)

        # Mask to face region only
        face_mask_3d = face_mask[:, :, np.newaxis]
        correction *= face_mask_3d

        # Apply correction
        corrected_lab = frame_lab + correction
        corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)

        return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

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
        """Detect if eyes are blinking."""
        if landmarks is None:
            return False

        pts = landmarks.points
        if len(pts) < 48:
            return False

        # Eye aspect ratio for left eye (points 36-41)
        left_eye = pts[36:42]
        left_h = abs(left_eye[1][1] - left_eye[5][1]) + abs(left_eye[2][1] - left_eye[4][1])
        left_w = abs(left_eye[0][0] - left_eye[3][0])
        left_ear = left_h / (2.0 * left_w + 1e-6)

        # Eye aspect ratio for right eye (points 42-47)
        right_eye = pts[42:48]
        right_h = abs(right_eye[1][1] - right_eye[5][1]) + abs(right_eye[2][1] - right_eye[4][1])
        right_w = abs(right_eye[0][0] - right_eye[3][0])
        right_ear = right_h / (2.0 * right_w + 1e-6)

        avg_ear = (left_ear + right_ear) / 2
        return avg_ear < 0.15  # Threshold for blink

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

    args = parser.parse_args()

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
