"""Real video test — 50 frames with observation model metrics."""
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from face_os.pipeline import FaceOSPipeline

CLIP_PATH = "clips_test/test_clip_5s.mp4"
PHOTOS_DIR = "photos/"
NUM_FRAMES = 50

def main():
    print("=" * 80)
    print("FACE OS — REAL VIDEO TEST (50 frames)")
    print("=" * 80)
    print(f"Clip: {CLIP_PATH}")
    print(f"Reference: {PHOTOS_DIR}")
    print()

    pipeline = FaceOSPipeline(use_bidirectional=False)

    print("Enrolling reference images...")
    pipeline.enroll(reference_dir=PHOTOS_DIR)
    print(f"  Enrollment complete: {pipeline.identity_state is not None}")
    print()

    cap = cv2.VideoCapture(CLIP_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {CLIP_PATH}")
        return

    print(f"Processing {NUM_FRAMES} frames...")
    print()

    header = f"{'Frame':>5} | {'Path':>8} | {'Accept':>6} | {'Reason':>25} | {'Latent':>7} | {'SrcFrac':>7} | {'Conf':>6} | {'C_recon':>7} | {'ObsRes':>7} | {'ObsNoise':>8} | {'ObsConf':>7}"
    print(header)
    print("-" * len(header))

    frame_idx = 0
    processed = 0

    while processed < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame, frame_idx=frame_idx)

        if result and result.get('frame') is not None:
            telem = pipeline._frame_telemetry_log[-1] if pipeline._frame_telemetry_log else {}
            latent = telem.get('latent', {})

            accept = pipeline._last_accept_decision.accept
            reason = pipeline._last_accept_decision.reason or "accepted"
            render_path = latent.get('render_path', 'unknown')
            latent_primary = latent.get('latent_primary', False)
            source_frac = latent.get('source_pixel_fraction', 1.0)
            latent_conf = latent.get('latent_confidence', 0.0)
            c_recon = latent.get('c_recon', 0.0)
            obs_res = latent.get('observation_residual_mean', 0.0)
            obs_noise = latent.get('observation_noise_mean', 0.0)
            obs_conf = latent.get('observation_confidence', 0.0)

            print(f"{frame_idx:>5} | {render_path:>8} | {str(accept):>6} | {reason:>25} | {str(latent_primary):>7} | {source_frac:>7.4f} | {latent_conf:>6.3f} | {c_recon:>7.4f} | {obs_res:>7.2f} | {obs_noise:>8.2f} | {obs_conf:>7.4f}")

            processed += 1

        frame_idx += 1

    cap.release()

    print()
    print("=" * 80)
    print(f"Processed {processed} frames from {frame_idx} total frames")
    print("=" * 80)

    if pipeline._latent_telemetry_log:
        obs_residuals = [t.get('observation_residual_mean', 0.0) for t in pipeline._latent_telemetry_log]
        obs_confidences = [t.get('observation_confidence', 0.0) for t in pipeline._latent_telemetry_log]
        c_recons = [t.get('c_recon', 0.0) for t in pipeline._latent_telemetry_log]

        print()
        print("OBSERVATION MODEL SUMMARY:")
        print(f"  Residual mean: {np.mean(obs_residuals):.2f} (min={np.min(obs_residuals):.2f}, max={np.max(obs_residuals):.2f})")
        print(f"  Confidence mean: {np.mean(obs_confidences):.4f} (min={np.min(obs_confidences):.4f}, max={np.max(obs_confidences):.4f})")
        print(f"  C_recon mean: {np.mean(c_recons):.4f} (min={np.min(c_recons):.4f}, max={np.max(c_recons):.4f})")
        print()

if __name__ == "__main__":
    main()
