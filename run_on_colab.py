#!/usr/bin/env python3
"""run_on_colab.py — Run Face OS pipeline on Colab GPU from local machine.

Usage:
    python run_on_colab.py <tunnel_url> [options]

Example:
    python run_on_colab.py https://abc123.serveo.net
    python run_on_colab.py https://abc123.serveo.net --video clips_test/test_clip.mp4 --frames 10
    python run_on_colab.py https://abc123.serveo.net --enroll-only
    python run_on_colab.py https://abc123.serveo.net --telemetry-only
"""

import sys
import argparse
import json
import time
from pathlib import Path

from face_os.colab_client import ColabClient


def main():
    parser = argparse.ArgumentParser(description="Run Face OS on Colab GPU")
    parser.add_argument("tunnel_url", help="Colab tunnel URL (from serveo/ngrok)")
    parser.add_argument("--video", default="clips_test/test_clip.mp4", help="Input video path")
    parser.add_argument("--reference", default="expectation.png", help="Reference image path")
    parser.add_argument("--photos", default="photos/", help="Photos directory")
    parser.add_argument("--frames", type=int, default=30, help="Max frames to process")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout (seconds)")
    parser.add_argument("--enroll-only", action="store_true", help="Only enroll, don't process")
    parser.add_argument("--process-only", action="store_true", help="Only process (skip enroll)")
    parser.add_argument("--telemetry-only", action="store_true", help="Only fetch telemetry")
    parser.add_argument("--gpu", action="store_true", help="Show GPU info")
    args = parser.parse_args()

    client = ColabClient(args.tunnel_url, timeout=args.timeout)

    # ── Health check ──────────────────────────────────────────────────────
    print(f"Connecting to {args.tunnel_url} ...")
    health = client.health()
    if "error" in health:
        print(f"FAIL: {health['error']}")
        sys.exit(1)
    print(f"OK: {health['status']} (engine={health.get('engine')})")

    # ── GPU info ──────────────────────────────────────────────────────────
    if args.gpu or args.telemetry_only is False:
        gpu = client.gpu_info()
        if gpu.get("available"):
            print(f"GPU: {gpu['name']} ({gpu['memory_total']:.1f}GB total, {gpu['memory_free']:.1f}GB free)")
        else:
            print("GPU: not available (CPU only)")

    if args.telemetry_only:
        result = client.get_telemetry()
        print(json.dumps(result, indent=2))
        frames = client.get_frame_telemetry(start=0, limit=20)
        for f in frames.get("frames", []):
            print(f"  F{f['frame_idx']:3d}: path={f['render_path']:20s} mode={f['renderer_mode']:12s} mesh={f['geometry_source']}")
        return

    # ── Enroll ────────────────────────────────────────────────────────────
    if not args.process_only:
        ref_path = Path(args.reference)
        photos_path = Path(args.photos)
        if not ref_path.exists():
            print(f"Reference not found: {ref_path}")
            sys.exit(1)

        print(f"Enrolling: ref={ref_path}, photos={photos_path} ...")
        t0 = time.time()
        result = client.enroll(str(ref_path), str(photos_path) if photos_path.is_dir() else None)
        print(f"Enroll: {result} ({time.time()-t0:.1f}s)")

        if "error" in result:
            sys.exit(1)

        if args.enroll_only:
            print("Enrollment complete. Use --process-only to process video.")
            return

    # ── Process ───────────────────────────────────────────────────────────
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    file_size_mb = video_path.stat().st_size / 1e6
    print(f"Processing: {video_path} ({file_size_mb:.1f}MB, max_frames={args.frames}) ...")
    print(f"Uploading + processing (may take a while) ...")

    t0 = time.time()
    result = client.process(str(video_path), max_frames=args.frames)
    elapsed = time.time() - t0

    if "error" in result:
        print(f"FAIL: {result['error']}")
        sys.exit(1)

    # ── Results ───────────────────────────────────────────────────────────
    print()
    print(f"{'='*50}")
    print(f"FRAMES PROCESSED : {result['frames_processed']}")
    print(f"WALL TIME        : {result['wall_time_s']}s")
    print(f"FPS              : {result['fps']}")
    print(f"OUTPUT SIZE      : {result['output_size_bytes']/1e6:.1f}MB")
    print(f"{'='*50}")

    # Per-frame detail
    frames = client.get_frame_telemetry(start=0, limit=10)
    if frames.get("frames"):
        print(f"\nFirst {len(frames['frames'])} frames:")
        for f in frames["frames"]:
            print(f"  F{f['frame_idx']:3d}: path={f['render_path']:20s} mode={f['renderer_mode']:12s} mesh={f['geometry_source']}")

    # Aggregate telemetry
    tel = result.get("telemetry", {})
    if tel:
        print(f"\nTelemetry: {json.dumps(tel, indent=2)}")

    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
