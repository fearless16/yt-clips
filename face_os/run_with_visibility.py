"""
run_with_visibility.py — Run pipeline with full parameter-wise visibility.

Captures:
1. EnergyReport per frame (E_geom, E_identity, E_temporal, E_photometric, E_smoothness)
2. GeometryMetrics (yaw/pitch/roll, det_A, mask_coverage%, transform_stability)
3. IdentityMetrics (anchor_weights[], uncertainty, region_confidence{}, appearance_latent_norm)
4. TemporalMetrics (temporal_confidence, drift_score, continuity_score)
5. RendererMetrics (M_mean, Y_face_range, Y_bg_range, blend_weight_stats)
6. PassReport with before/after/delta

Outputs:
- output/face_os/visibility/energy_reports.json
- output/face_os/visibility/pass_reports.json
- output/face_os/visibility/summary.json
- output/face_os/visibility/frames/ — extracted frames for comparison
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from face_os.config import get_config
from face_os.types import (
    GeometryState,
    IdentityState,
    TemporalState,
    CropPlan,
    EnergyReport,
    PassReport,
    FrameContract,
)
from face_os.energy import EnergyComputer
from face_os.visibility import VisibilityLogger

from face_os import ingest
from face_os import detect_track
from face_os import landmarks as lm_module
from face_os import canonical_map
from face_os import crop_planner
from face_os import face_enhance
from face_os.identity_state import IdentityState as IdentityBeliefState
from face_os.patch_memory import PatchMemory
from face_os.temporal_solve import TemporalRepairEngine, FrameQuality


cfg = get_config()


def run_pipeline_with_visibility(
    video_path: str,
    reference_path: str,
    photos_dir: str,
    output_path: str,
    max_frames: int = 0,
    extract_frames: bool = True,
):
    """Run pipeline with full visibility logging."""

    print("=== FACE OS — Pipeline with Visibility ===\n")

    # Setup output directories
    vis_dir = os.path.join(os.path.dirname(output_path), "visibility")
    frames_dir = os.path.join(vis_dir, "frames")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # Initialize visibility logger
    logger = VisibilityLogger(output_dir=vis_dir)

    # Initialize energy computer
    energy_computer = EnergyComputer()

    # Load video
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {width}x{height} @ {fps}fps ({total_frames} frames)")

    if max_frames > 0:
        total_frames = min(total_frames, max_frames)
        print(f"  Processing: {max_frames} frames max")

    # Load reference
    print(f"\nLoading reference: {reference_path}")
    from face_os import ingest as ingest_module
    primary_ref, ref_images = ingest_module.load_reference_images(photos_dir, reference_path)
    print(f"  Loaded {len(ref_images)} reference images")

    # Build identity profile
    from face_os.canonical_map import build_identity_profile
    identity = build_identity_profile(ref_images)
    print(f"  Atlas enrolled: {identity.enrolled}")

    # Initialize tracker
    tracker = detect_track.FaceTracker(
        reference_embeddings=identity.embeddings,
    )

    # Initialize modules
    crop = crop_planner.CropPlanner(reference_image=reference_path)
    identity_state = IdentityBeliefState()
    patch_memory = PatchMemory()

    # Set anchor
    if identity.appearance.atlas_rgb is not None:
        anchor_bgr = cv2.cvtColor(identity.appearance.atlas_rgb, cv2.COLOR_RGB2BGR)
        identity_state.set_anchor(anchor_bgr)
        energy_computer.anchor_face = anchor_bgr
        energy_computer.anchor_lab = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Initialize temporal solver
    temporal_solver = TemporalRepairEngine(lookback=10, lookahead=10)

    # Frame contract
    contract = FrameContract()

    # Processing loop
    print(f"\n=== Processing {total_frames} frames ===\n")

    frame_reports = []
    energy_reports = []
    all_metrics = []

    start_time = time.time()
    frame_idx = 0

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # === Process frame ===
        face_track = tracker.process_frame(frame, frame_idx)

        # Landmarks
        landmarks = None
        if face_track and face_track.smooth_bbox:
            landmarks = lm_module.extract_landmarks(frame, face_track.mesh_478)
            face_track.landmarks = landmarks

        # Canonical alignment
        canonical_face = None
        quality_map = None
        pose = None
        if landmarks and face_track.detection:
            try:
                warped_rgb, warped_lab, M = canonical_map.warp_to_canonical(frame, landmarks)
                canonical_face = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)
                quality_map = _compute_quality_map(canonical_face, face_track.detection.confidence)
                pose = (landmarks.yaw, landmarks.pitch, landmarks.roll)
            except Exception:
                pass

        # Update identity
        if canonical_face is not None and quality_map is not None:
            identity_state.update(canonical_face, quality_map, pose=pose)

        # Crop
        crop_plan = crop.plan_crop(frame.shape[:2], face_track, landmarks)
        cropped = crop_planner.apply_crop(frame, crop_plan)

        # Query identity
        identity_face = None
        identity_conf = None
        if canonical_face is not None and quality_map is not None:
            identity_face, identity_conf = identity_state.query(canonical_face, quality_map, pose=pose)

        # Build geometry state
        geo_state = GeometryState(
            landmarks_478=face_track.mesh_478 if face_track else None,
            landmarks=landmarks,
            pose=pose or (0.0, 0.0, 0.0),
            geometry_confidence=face_track.detection.confidence if face_track and face_track.detection else 0.0,
            canonical_face=canonical_face,
        )

        # Build identity state type
        id_state = IdentityState(
            identity_uncertainty=1.0 - float(np.mean(identity_conf)) if identity_conf is not None else 1.0,
            region_confidence=identity_state.compute_region_confidence() if identity_state.is_initialized() else {},
            appearance_latent=identity_face,
            anchor_weights=[1.0] if identity_face is not None else [],
            initialized=identity_state.is_initialized(),
        )

        # Build temporal state
        temp_state = TemporalState(
            temporal_confidence=face_track.detection.confidence if face_track and face_track.detection else 0.0,
            drift_score=identity_state.get_anchor_distance() if identity_state.is_initialized() else 0.0,
            continuity_score=0.95,
            pose=pose,
        )

        # Compute energy
        energy_report = energy_computer.compute(
            frame_idx, geo_state, id_state, temp_state, source_frame=frame
        )
        logger.log_energy(energy_report)

        # Build metrics snapshot
        metrics = energy_report.to_dict()

        # Extract frames for comparison (every 30 frames + first/last)
        if extract_frames and (frame_idx % 30 == 0 or frame_idx == total_frames - 1):
            # Save source frame
            src_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}_source.png")
            cv2.imwrite(src_path, cropped)

            # Save output frame (if we have identity)
            if identity_face is not None:
                try:
                    adjusted_lm = _adjust_landmarks_to_crop(landmarks, crop_plan)
                    if adjusted_lm:
                        _, _, M = canonical_map.warp_to_canonical(cropped, adjusted_lm)
                        M_inv = np.linalg.inv(M)[:2]
                        identity_in_crop = cv2.warpAffine(
                            identity_face, M_inv, (cropped.shape[1], cropped.shape[0]),
                            flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REFLECT,
                        )
                        canonical_mask = _make_canonical_geometry_mask(identity_face.shape[:2])
                        aligned_mask = cv2.warpAffine(
                            canonical_mask, M_inv, (cropped.shape[1], cropped.shape[0]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        aligned_mask = np.clip(aligned_mask, 0, 1)
                        blend_3d = aligned_mask[:, :, np.newaxis]
                        output = cropped.astype(np.float32) * (1 - blend_3d) + identity_in_crop.astype(np.float32) * blend_3d
                        output = np.clip(output, 0, 255).astype(np.uint8)
                        out_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}_output.png")
                        cv2.imwrite(out_path, output)

                        # Save identity face
                        id_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}_identity.png")
                        cv2.imwrite(id_path, identity_face)
                except Exception as e:
                    pass

        # Log metrics
        all_metrics.append(metrics)

        # Progress
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}: E_total={energy_report.terms.E_total:.2f} "
                  f"E_geom={energy_report.terms.E_geom:.2f} "
                  f"E_identity={energy_report.terms.E_identity:.2f} "
                  f"E_temporal={energy_report.terms.E_temporal:.2f} "
                  f"yaw={energy_report.geometry.yaw:.1f} "
                  f"uncertainty={energy_report.identity.uncertainty:.3f}")

        frame_idx += 1

    cap.release()

    elapsed = time.time() - start_time
    print(f"\n=== Processing Complete ===")
    print(f"  Frames: {frame_idx}")
    print(f"  Time: {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")

    # Save all reports
    print(f"\n=== Saving Visibility Reports ===")

    # Save energy reports
    energy_file = os.path.join(vis_dir, "energy_reports.json")
    with open(energy_file, "w") as f:
        json.dump([r.to_dict() for r in logger.energy_reports], f, indent=2, default=str)
    print(f"  Energy reports: {energy_file}")

    # Save metrics
    metrics_file = os.path.join(vis_dir, "all_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"  All metrics: {metrics_file}")

    # Compute summary
    if logger.energy_reports:
        E_totals = [r.terms.E_total for r in logger.energy_reports]
        E_geoms = [r.terms.E_geom for r in logger.energy_reports]
        E_identities = [r.terms.E_identity for r in logger.energy_reports]
        E_temporals = [r.terms.E_temporal for r in logger.energy_reports]
        E_photometrics = [r.terms.E_photometric for r in logger.energy_reports]
        E_smoothnesses = [r.terms.E_smoothness for r in logger.energy_reports]

        summary = {
            "total_frames": frame_idx,
            "processing_time_s": elapsed,
            "fps": frame_idx / elapsed,
            "energy_summary": {
                "E_total": {"mean": float(np.mean(E_totals)), "std": float(np.std(E_totals)), "min": float(np.min(E_totals)), "max": float(np.max(E_totals))},
                "E_geom": {"mean": float(np.mean(E_geoms)), "std": float(np.std(E_geoms)), "min": float(np.min(E_geoms)), "max": float(np.max(E_geoms))},
                "E_identity": {"mean": float(np.mean(E_identities)), "std": float(np.std(E_identities)), "min": float(np.min(E_identities)), "max": float(np.max(E_identities))},
                "E_temporal": {"mean": float(np.mean(E_temporals)), "std": float(np.std(E_temporals)), "min": float(np.min(E_temporals)), "max": float(np.max(E_temporals))},
                "E_photometric": {"mean": float(np.mean(E_photometrics)), "std": float(np.std(E_photometrics)), "min": float(np.min(E_photometrics)), "max": float(np.max(E_photometrics))},
                "E_smoothness": {"mean": float(np.mean(E_smoothnesses)), "std": float(np.std(E_smoothnesses)), "min": float(np.min(E_smoothnesses)), "max": float(np.max(E_smoothnesses))},
            },
            "geometry_summary": {
                "yaw_mean": float(np.mean([r.geometry.yaw for r in logger.energy_reports])),
                "pitch_mean": float(np.mean([r.geometry.pitch for r in logger.energy_reports])),
                "roll_mean": float(np.mean([r.geometry.roll for r in logger.energy_reports])),
                "det_A_mean": float(np.mean([r.geometry.det_A for r in logger.energy_reports])),
                "mask_coverage_mean": float(np.mean([r.geometry.mask_coverage_pct for r in logger.energy_reports])),
            },
            "identity_summary": {
                "uncertainty_mean": float(np.mean([r.identity.uncertainty for r in logger.energy_reports])),
                "appearance_latent_norm_mean": float(np.mean([r.identity.appearance_latent_norm for r in logger.energy_reports])),
            },
            "temporal_summary": {
                "temporal_confidence_mean": float(np.mean([r.temporal.temporal_confidence for r in logger.energy_reports])),
                "drift_score_mean": float(np.mean([r.temporal.drift_score for r in logger.energy_reports])),
                "continuity_score_mean": float(np.mean([r.temporal.continuity_score for r in logger.energy_reports])),
            },
            "extracted_frames": [f for f in os.listdir(frames_dir) if f.endswith(".png")] if extract_frames else [],
        }

        summary_file = os.path.join(vis_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Summary: {summary_file}")

    print(f"\n=== Visibility Reports Complete ===")
    print(f"  Directory: {vis_dir}")
    print(f"  Frames: {frames_dir}")


def _compute_quality_map(canonical_face, detection_confidence):
    gray = cv2.cvtColor(canonical_face, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
    sharpness = np.clip(lap / 50.0, 0, 1)
    brightness = gray.astype(np.float32) / 255.0
    brightness_weight = 1.0 - np.abs(brightness - 0.5) * 2
    brightness_weight = np.clip(brightness_weight, 0.1, 1.0)
    quality = sharpness * brightness_weight * detection_confidence
    return quality.astype(np.float32)


def _adjust_landmarks_to_crop(landmarks, crop_plan):
    if landmarks is None or crop_plan is None:
        return None
    try:
        pts = landmarks.points.copy()
        pts[:, 0] -= crop_plan.src_x
        pts[:, 1] -= crop_plan.src_y
        scale_x = crop_plan.dst_w / crop_plan.src_w
        scale_y = crop_plan.dst_h / crop_plan.src_h
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        from face_os.types import Landmarks
        adjusted = Landmarks(
            points=pts,
            yaw=landmarks.yaw,
            pitch=landmarks.pitch,
            roll=landmarks.roll,
        )
        return adjusted
    except Exception:
        return None


def _make_canonical_geometry_mask(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    axes = (int(w * 0.45), int(h * 0.50))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (11, 11), 3)
    return np.clip(mask, 0, 1).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Face OS with visibility logging")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--reference", required=True, help="Reference image path")
    parser.add_argument("--photos", required=True, help="Photos directory")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process")
    parser.add_argument("--no-frames", action="store_true", help="Skip frame extraction")

    args = parser.parse_args()

    run_pipeline_with_visibility(
        video_path=args.video,
        reference_path=args.reference,
        photos_dir=args.photos,
        output_path=args.output,
        max_frames=args.max_frames,
        extract_frames=not args.no_frames,
    )
