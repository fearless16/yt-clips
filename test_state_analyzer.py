"""
test_state_analyzer.py — Pull a short test clip and run state analysis.

Uses yt-dlp to download a short clip from YouTube for testing.
Falls back to generating synthetic test frames if no video available.

Usage:
    python test_state_analyzer.py --url "https://youtube.com/watch?v=..."
    python test_state_analyzer.py --url "https://youtube.com/watch?v=..." --duration 10
    python test_state_analyzer.py --synthetic  # generate test frames without video
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from state_analyzer import analyze_clip

CLIPS_DIR = Path("clips_test")


def download_short_clip(url: str, duration: int = 10, output_dir: Path = CLIPS_DIR) -> str:
    """Download a short clip from YouTube using yt-dlp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_clip.mp4"
    
    print(f"Downloading {duration}s clip from: {url}")
    
    cmd = [
        "yt-dlp",
        "-f", "bv*[height<=480]+ba/b[height<=480]",  # Low res for fast testing
        "--downloader", "aria2c",
        "--downloader-args", "aria2c:-x 16 -s 16 -k 1M",
        "--external-downloader", "ffmpeg",
        "--external-downloader-args", f"ffmpeg:-ss 00:00:00 -t {duration}",
        "-o", str(output_path),
        url,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0 or not output_path.exists():
        # Try without aria2c
        cmd = [
            "yt-dlp",
            "-f", "bv*[height<=480]+ba/b[height<=480]",
            "--external-downloader", "ffmpeg",
            "--external-downloader-args", f"ffmpeg:-ss 00:00:00 -t {duration}",
            "-o", str(output_path),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0 or not output_path.exists():
        print(f"Download failed: {result.stderr[-500:]}")
        return None
    
    size_mb = output_path.stat().st_size / 1e6
    print(f"Downloaded: {output_path} ({size_mb:.1f} MB)")
    return str(output_path)


def generate_synthetic_frames(output_dir: Path = CLIPS_DIR, num_frames: int = 60) -> str:
    """Generate synthetic test frames with varying face/mouth/eye states."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_test.mp4"
    
    print(f"Generating {num_frames} synthetic test frames...")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    w, h = 640, 480
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background gradient (simulating lighting changes)
        brightness = 100 + int(30 * np.sin(i * 0.1))
        frame[:, :, 0] = brightness - 20  # B
        frame[:, :, 1] = brightness       # G
        frame[:, :, 2] = brightness + 10  # R
        
        # Simulated face (ellipse)
        face_cx = w // 2 + int(20 * np.sin(i * 0.05))  # Slight head movement
        face_cy = h // 2
        face_w = 120
        face_h = 150
        
        # Face color (skin tone)
        face_color = (80, 130, 200)  # BGR
        cv2.ellipse(frame, (face_cx, face_cy), (face_w // 2, face_h // 2), 0, 0, 360, face_color, -1)
        
        # Eyes
        eye_y = face_cy - 30
        eye_open = max(0, int(8 * np.sin(i * 0.3)))  # Blinking pattern
        if eye_open > 1:
            cv2.ellipse(frame, (face_cx - 25, eye_y), (8, eye_open), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(frame, (face_cx + 25, eye_y), (8, eye_open), 0, 0, 360, (255, 255, 255), -1)
            cv2.circle(frame, (face_cx - 25, eye_y), 3, (0, 0, 0), -1)
            cv2.circle(frame, (face_cx + 25, eye_y), 3, (0, 0, 0), -1)
        else:
            # Closed eyes (lines)
            cv2.line(frame, (face_cx - 33, eye_y), (face_cx - 17, eye_y), (200, 200, 200), 2)
            cv2.line(frame, (face_cx + 17, eye_y), (face_cx + 33, eye_y), (200, 200, 200), 2)
        
        # Mouth (open/close pattern)
        mouth_y = face_cy + 35
        mouth_open = max(0, int(15 * np.sin(i * 0.15)))  # Speaking pattern
        if mouth_open > 2:
            cv2.ellipse(frame, (face_cx, mouth_y), (20, mouth_open), 0, 0, 360, (0, 0, 100), -1)
        else:
            cv2.line(frame, (face_cx - 20, mouth_y), (face_cx + 20, mouth_y), (100, 50, 50), 3)
        
        # Add motion blur on some frames
        if i % 7 == 0:
            frame = cv2.GaussianBlur(frame, (5, 5), 2)
        
        # Add compression-like artifacts on some frames
        if i % 11 == 0:
            # Blocky compression simulation
            for y in range(0, h, 16):
                for x in range(0, w, 16):
                    block = frame[y:y+16, x:x+16]
                    if block.size > 0:
                        block[:, :] = np.mean(block, axis=(0, 1))
        
        writer.write(frame)
    
    writer.release()
    
    size_mb = output_path.stat().st_size / 1e6
    print(f"Generated: {output_path} ({size_mb:.1f} MB)")
    return str(output_path)


def run_analysis(video_path: str, output_path: str = None):
    """Run state analysis on video."""
    print("\n" + "=" * 60)
    print("Running State Analysis...")
    print("=" * 60 + "\n")
    
    result = analyze_clip(
        video_path=video_path,
        sample_rate=2,
        segment_size_sec=2.0,
        output_path=output_path,
    )
    
    if "error" in result:
        print(f"Analysis failed: {result['error']}")
        return
    
    summary = result["summary"]
    print(f"\nAnalysis complete in {summary['analysis_time_sec']:.1f}s")
    print(f"Frames analyzed: {summary['total_frames_analyzed']}")
    print(f"Face detection rate: {summary['face_detection_rate']}%")
    print(f"Enhancement distribution:")
    dist = summary["enhancement_distribution"]
    print(f"  Heavy: {dist['heavy']} ({dist['heavy_pct']}%)")
    print(f"  Light: {dist['light']} ({dist['light_pct']}%)")
    print(f"  Skip:  {dist['skip']} ({dist['skip_pct']}%)")
    
    if result["segments"]:
        print(f"\nSegments ({len(result['segments'])}):")
        for seg in result["segments"][:5]:
            print(f"  [{seg['segment_id']}] {seg['start_sec']:.1f}s-{seg['end_sec']:.1f}s → {seg['recommended_strategy']}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Test state analyzer with real or synthetic video")
    parser.add_argument("--url", default=None, help="YouTube URL to download clip from")
    parser.add_argument("--duration", type=int, default=10, help="Clip duration in seconds")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test frames")
    parser.add_argument("--video", default=None, help="Path to local video file")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    args = parser.parse_args()
    
    video_path = None
    
    if args.video:
        video_path = args.video
    elif args.url:
        video_path = download_short_clip(args.url, args.duration)
    elif args.synthetic:
        video_path = generate_synthetic_frames()
    else:
        # Default: try synthetic
        print("No video specified, generating synthetic test frames...")
        video_path = generate_synthetic_frames()
    
    if not video_path:
        print("No video available for testing")
        sys.exit(1)
    
    run_analysis(video_path, args.output)


if __name__ == "__main__":
    main()
