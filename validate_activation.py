"""Runtime Activation Validation Script.

Runs the pipeline and measures:
- PhysicalRenderer activation rate
- IntrinsicDecomposer success rate
- RendererMode distribution
- StateEvolution usage
- EnergyScaling usage
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from face_os.pipeline import FaceOSPipeline


def main():
    """Run pipeline and measure runtime activation."""
    print("=== FACE OS RUNTIME ACTIVATION VALIDATION ===")
    print()

    # Initialize pipeline
    pipeline = FaceOSPipeline(use_bidirectional=False)

    # Enroll identity
    print("1. Enrolling identity...")
    success = pipeline.enroll(
        reference_image="expectation.png",
        reference_dir="photos/",
    )
    if not success:
        print("ERROR: Enrollment failed")
        return

    print()

    # Process test video
    print("2. Processing test video...")
    try:
        pipeline.process(
            video_path="clips_test/test_clip.mp4",
            output_path="output/face_os/validation_test.mp4",
            max_frames=100,  # Limit frames for quick test
        )
    except Exception as e:
        print(f"WARNING: Video processing failed: {e}")
        print("Continuing with telemetry analysis...")

    print()

    # Get telemetry report
    print("3. Runtime Telemetry Report:")
    print("=" * 60)
    report = pipeline.get_telemetry_report()
    print(json.dumps(report, indent=2, default=str))
    print("=" * 60)

    print()

    # Analysis
    print("4. Activation Analysis:")
    print("-" * 60)

    total = report.get("total_frames", 0)
    if total > 0:
        print(f"Total frames processed: {total}")
        print()
        
        # PhysicalRenderer activation
        phys_rate = report.get("physical_render_rate", 0)
        print(f"PhysicalRenderer activation rate: {phys_rate:.1%}")
        if phys_rate > 0.5:
            print("  ✅ PhysicalRenderer is ACTIVE (>50% of frames)")
        elif phys_rate > 0.1:
            print("  ⚠️ PhysicalRenderer is PARTIALLY ACTIVE (10-50% of frames)")
        else:
            print("  ❌ PhysicalRenderer is INACTIVE (<10% of frames)")
        
        print()
        
        # IntrinsicDecomposer success
        intrinsic_rate = report.get("intrinsic_success_rate", 0)
        print(f"IntrinsicDecomposer success rate: {intrinsic_rate:.1%}")
        if intrinsic_rate > 0.5:
            print("  ✅ IntrinsicDecomposer is WORKING (>50% success)")
        elif intrinsic_rate > 0.1:
            print("  ⚠️ IntrinsicDecomposer is PARTIALLY WORKING (10-50% success)")
        else:
            print("  ❌ IntrinsicDecomposer is FAILING (<10% success)")
        
        print()
        
        # RendererMode distribution
        mode_dist = report.get("renderer_mode_distribution", {})
        print("RendererMode distribution:")
        for mode, count in mode_dist.items():
            rate = count / total
            print(f"  {mode}: {count} frames ({rate:.1%})")
        
        print()
        
        # Confidence statistics
        avg_conf = report.get("avg_intrinsic_confidence", 0)
        avg_error = report.get("avg_decomposition_error", 0)
        print(f"Average intrinsic confidence: {avg_conf:.3f}")
        print(f"Average decomposition error: {avg_error:.3f}")
        
        print()
        
        # Failure reasons
        failure_reasons = report.get("intrinsic_failure_reasons", {})
        if failure_reasons:
            print("Intrinsic failure reasons:")
            for reason, count in failure_reasons.items():
                print(f"  {reason}: {count}")
    else:
        print("No frames processed - cannot analyze activation")

    print()
    print("5. Validation Summary:")
    print("-" * 60)
    
    # Check if V3 modules are actually contributing
    if total > 0:
        phys_rate = report.get("physical_render_rate", 0)
        intrinsic_rate = report.get("intrinsic_success_rate", 0)
        
        if phys_rate > 0.1 and intrinsic_rate > 0.1:
            print("✅ V3 modules are ACTIVELY CONTRIBUTING to production")
        elif phys_rate > 0 or intrinsic_rate > 0:
            print("⚠️ V3 modules are PARTIALLY ACTIVE - some frames using V3 path")
        else:
            print("❌ V3 modules are NOT ACTIVE - all frames using legacy path")
            print("   Possible reasons:")
            print("   - IntrinsicDecomposer failing")
            print("   - RendererMode staying in ALPHA_FALLBACK")
            print("   - Confidence thresholds too high")


if __name__ == "__main__":
    main()
