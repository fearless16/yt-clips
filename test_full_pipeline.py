"""
test_full_pipeline.py — Test Pass 1 + Pass 2 + Pass 3 end-to-end.

Runs state analysis → selective enhancement → temporal consistency on a test clip.

Usage:
    python test_full_pipeline.py --video clips_test/test_clip.mp4
    python test_full_pipeline.py --video clips_test/test_clip.mp4 --esrgan
"""

import argparse
import time
from pathlib import Path

from state_analyzer import analyze_clip
from selective_enhancer import enhance_clip
from temporal_consistency import apply_temporal_consistency


def run_pipeline(video_path: str, use_esrgan: bool = False):
    """Run Pass 1 + Pass 2 + Pass 3 end-to-end."""
    t_start = time.perf_counter()
    
    print("\n" + "=" * 60)
    print("  FULL PIPELINE TEST: Pass 1 + Pass 2 + Pass 3")
    print("=" * 60 + "\n")
    
    # Pass 1: Analysis
    print("─── Pass 1: State Analysis ──────────────────────────────")
    analysis_path = "temp/test_full_analysis.json"
    
    result = analyze_clip(
        video_path=video_path,
        sample_rate=2,
        segment_size_sec=2.0,
        output_path=analysis_path,
    )
    
    if "error" in result:
        print(f"Pass 1 failed: {result['error']}")
        return
    
    summary = result["summary"]
    dist = summary["enhancement_distribution"]
    print(f"\nPass 1 complete in {summary['analysis_time_sec']:.1f}s")
    print(f"  Heavy: {dist['heavy']} ({dist['heavy_pct']}%)")
    print(f"  Light: {dist['light']} ({dist['light_pct']}%)")
    print(f"  Skip:  {dist['skip']} ({dist['skip_pct']}%)")
    
    # Pass 2: Enhancement
    print("\n─── Pass 2: Selective Enhancement ───────────────────────")
    pass2_output = "temp/test_pass2_enhanced.mp4"
    
    enhanced_path = enhance_clip(
        video_path=video_path,
        analysis_path=analysis_path,
        output_path=pass2_output,
        use_esrgan=use_esrgan,
    )
    
    # Pass 3: Temporal Consistency
    print("\n─── Pass 3: Temporal Consistency ────────────────────────")
    pass3_output = "temp/test_pass3_consistent.mp4"
    
    consistent_path = apply_temporal_consistency(
        video_path=enhanced_path,
        analysis_path=analysis_path,
        output_path=pass3_output,
        face_alpha=0.7,
        global_alpha=0.85,
    )
    
    t_total = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE in {t_total:.1f}s")
    print(f"  Final output: {consistent_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test full Pass 1 + Pass 2 + Pass 3 pipeline")
    parser.add_argument("--video", default="clips_test/test_clip.mp4", help="Input video")
    parser.add_argument("--esrgan", action="store_true", help="Enable super-resolution")
    args = parser.parse_args()
    
    run_pipeline(args.video, args.esrgan)
