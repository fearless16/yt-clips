"""Parallel real video test — 3 clips with frame-by-frame comparison report."""
import sys
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from dataclasses import dataclass
from typing import List
import json

sys.path.insert(0, str(Path(__file__).parent))

from face_os.pipeline import FaceOSPipeline


@dataclass
class FrameMetrics:
    frame_idx: int
    render_path: str
    accept: bool
    accept_reason: str
    latent_primary: bool
    source_pixel_fraction: float
    latent_confidence: float
    c_recon: float
    observation_residual_mean: float
    observation_noise_mean: float
    observation_confidence: float
    hybrid_alpha_mean: float
    coverage_pose: float
    coverage_light: float
    mean_visibility: float


def process_clip(args):
    clip_path, photos_dir, num_frames = args
    print(f"[{clip_path}] Starting...")

    pipeline = FaceOSPipeline(use_bidirectional=False)
    pipeline.enroll(reference_dir=photos_dir)

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"[{clip_path}] ERROR: Cannot open")
        return clip_path, []

    frames: List[FrameMetrics] = []
    frame_idx = 0
    processed = 0

    while processed < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame, frame_idx=frame_idx)

        if result and result.get('frame') is not None:
            telem = pipeline._frame_telemetry_log[-1] if pipeline._frame_telemetry_log else {}
            latent = telem.get('latent', {})

            frames.append(FrameMetrics(
                frame_idx=frame_idx,
                render_path=latent.get('render_path', 'unknown'),
                accept=pipeline._last_accept_decision.accept,
                accept_reason=pipeline._last_accept_decision.reason or "accepted",
                latent_primary=latent.get('latent_primary', False),
                source_pixel_fraction=latent.get('source_pixel_fraction', 1.0),
                latent_confidence=latent.get('latent_confidence', 0.0),
                c_recon=latent.get('c_recon', 0.0),
                observation_residual_mean=latent.get('observation_residual_mean', 0.0),
                observation_noise_mean=latent.get('observation_noise_mean', 0.0),
                observation_confidence=latent.get('observation_confidence', 0.0),
                hybrid_alpha_mean=latent.get('hybrid_alpha_mean', 1.0),
                coverage_pose=latent.get('coverage_pose', 0.0),
                coverage_light=latent.get('coverage_light', 0.0),
                mean_visibility=latent.get('mean_visibility', 1.0),
            ))
            processed += 1

        frame_idx += 1

    cap.release()
    print(f"[{clip_path}] Processed {processed} frames")
    return clip_path, frames


def generate_report(results, output_path):
    report = []
    report.append("=" * 120)
    report.append("FACE OS — REAL VIDEO TEST REPORT (3 clips × 50 frames)")
    report.append("=" * 120)
    report.append("")

    for clip_path, frames in results:
        report.append(f"CLIP: {clip_path}")
        report.append(f"Frames processed: {len(frames)}")
        report.append("")

        header = f"{'Frame':>5} | {'Path':>8} | {'Accept':>6} | {'Reason':>25} | {'Latent':>7} | {'SrcFrac':>7} | {'Conf':>6} | {'C_recon':>7} | {'ObsRes':>7} | {'ObsNoise':>8} | {'ObsConf':>7} | {'Hybrid':>6} | {'CovPose':>7} | {'CovLight':>8} | {'Visib':>5}"
        report.append(header)
        report.append("-" * len(header))

        for f in frames:
            line = f"{f.frame_idx:>5} | {f.render_path:>8} | {str(f.accept):>6} | {f.accept_reason:>25} | {str(f.latent_primary):>7} | {f.source_pixel_fraction:>7.4f} | {f.latent_confidence:>6.3f} | {f.c_recon:>7.4f} | {f.observation_residual_mean:>7.2f} | {f.observation_noise_mean:>8.2f} | {f.observation_confidence:>7.4f} | {f.hybrid_alpha_mean:>6.3f} | {f.coverage_pose:>7.4f} | {f.coverage_light:>8.4f} | {f.mean_visibility:>5.3f}"
            report.append(line)

        report.append("")

        if frames:
            latent_frames = [f for f in frames if f.latent_primary]
            alpha_frames = [f for f in frames if not f.latent_primary]

            report.append("SUMMARY:")
            report.append(f"  Total frames: {len(frames)}")
            report.append(f"  Latent-driven: {len(latent_frames)} ({100*len(latent_frames)/len(frames):.1f}%)")
            report.append(f"  Alpha fallback: {len(alpha_frames)} ({100*len(alpha_frames)/len(frames):.1f}%)")

            if latent_frames:
                report.append("")
                report.append("  LATENT PATH METRICS:")
                report.append(f"    Source pixel fraction: mean={np.mean([f.source_pixel_fraction for f in latent_frames]):.4f}")
                report.append(f"    Latent confidence: mean={np.mean([f.latent_confidence for f in latent_frames]):.4f}")
                report.append(f"    C_recon: mean={np.mean([f.c_recon for f in latent_frames]):.4f}")
                report.append(f"    Observation residual: mean={np.mean([f.observation_residual_mean for f in latent_frames]):.2f}")
                report.append(f"    Observation confidence: mean={np.mean([f.observation_confidence for f in latent_frames]):.4f}")
                report.append(f"    Hybrid alpha: mean={np.mean([f.hybrid_alpha_mean for f in latent_frames]):.4f}")

            report.append("")
            report.append("  ACCEPT GATE METRICS:")
            accepted = [f for f in frames if f.accept]
            report.append(f"    Accepted: {len(accepted)} ({100*len(accepted)/len(frames):.1f}%)")
            report.append(f"    Rejected: {len(frames) - len(accepted)} ({100*(len(frames) - len(accepted))/len(frames):.1f}%)")

            reasons = {}
            for f in frames:
                reasons[f.accept_reason] = reasons.get(f.accept_reason, 0) + 1
            report.append(f"    Rejection reasons: {reasons}")

        report.append("")
        report.append("=" * 120)
        report.append("")

    report_text = "\n".join(report)
    print(report_text)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to: {output_path}")


def main():
    clips = [
        ("clips_test/test_clip_5s_1.mp4", "photos/", 50),
        ("clips_test/test_clip_5s_2.mp4", "photos/", 50),
        ("clips_test/test_clip_5s_3.mp4", "photos/", 50),
    ]

    print("=" * 120)
    print("FACE OS — PARALLEL REAL VIDEO TEST")
    print("=" * 120)
    print(f"Processing {len(clips)} clips in parallel...")
    print()

    with Pool(processes=3) as pool:
        results = pool.map(process_clip, clips)

    print()
    print("All clips processed. Generating report...")
    print()

    generate_report(results, "output/real_video_report.txt")


if __name__ == "__main__":
    main()
