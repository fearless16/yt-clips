"""Comprehensive Face OS metrics validation on real video.

Verifies ALL claims from AGENTS.md and AGAINST.md with pass/fail per metric.
No output photos/videos saved — clean terminal report only.
"""

import sys
import json
import time
import tempfile
import os
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from face_os.pipeline import FaceOSPipeline


def check_frame_contract(frame: np.ndarray, frame_idx: int, expected_h=1920, expected_w=1080) -> dict:
    """Validate a single output frame against the frame contract."""
    issues = []
    if frame.shape != (expected_h, expected_w, 3):
        issues.append(f"shape={frame.shape}, expected ({expected_h},{expected_w},3)")
    if frame.dtype != np.uint8:
        issues.append(f"dtype={frame.dtype}, expected uint8")
    if np.any(np.isnan(frame)):
        issues.append("contains NaN")
    if np.any(np.isinf(frame)):
        issues.append("contains Inf")
    if frame.ndim != 3:
        issues.append(f"ndim={frame.ndim}, expected 3")
    if frame.shape[2] != 3:
        issues.append(f"channels={frame.shape[2]}, expected 3")
    return {
        "frame": frame_idx,
        "pass": len(issues) == 0,
        "issues": issues,
    }


def validate_output_video(path: str, max_check: int = 100) -> dict:
    """Validate every frame of an output video against frame contract."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"pass": False, "error": "Cannot open output video", "frames_checked": 0, "frames_passing": 0, "total_frames": 0}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_checked = 0
    frames_passing = 0
    all_issues = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = check_frame_contract(frame, frame_idx)
        frames_checked += 1
        if result["pass"]:
            frames_passing += 1
        else:
            all_issues.append(result)
        frame_idx += 1
        if max_check and frame_idx >= max_check:
            break

    cap.release()
    return {
        "pass": frames_passing == frames_checked,
        "frames_checked": frames_checked,
        "frames_passing": frames_passing,
        "total_frames": total,
        "all_issues": all_issues[:5],  # First 5 issues only
    }


def main():
    print("=" * 72)
    print("  FACE OS — REAL VIDEO METRICS VALIDATION")
    print("=" * 72)
    print()
    ts_start = time.perf_counter()

    # ── Step 1: Enroll identity ──
    print("1. ENROLLING IDENTITY")
    print("-" * 60)
    pipeline = FaceOSPipeline(use_bidirectional=False)
    enroll_ok = pipeline.enroll(
        reference_image="expectation.png",
        reference_dir="photos/",
    )
    print(f"   Enrollment: {'✅' if enroll_ok else '❌'}")
    print()

    if not enroll_ok:
        print("FATAL: Enrollment failed")
        return

    # ── Step 2: Process test video to temp ──
    print("2. PROCESSING TEST VIDEO")
    print("-" * 60)
    video_path = "clips_test/test_clip.mp4"
    if not os.path.exists(video_path):
        print(f"   ❌ Video not found: {video_path}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output.mp4")
        print(f"   Input:  {video_path}")
        print(f"   Output: (temp — cleaned up)")
        print(f"   Max frames: 100")
        print()

        try:
            result_path = pipeline.process(
                video_path=video_path,
                output_path=output_path,
                max_frames=100,
            )
        except Exception as e:
            print(f"   ❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            result_path = None

        if result_path and not os.path.exists(result_path):
            result_path = None

        print()

        # ── Step 3: Telemetry report ──
        print("3. TELEMETRY REPORT")
        print("-" * 60)
        report = pipeline.get_telemetry_report()
        total = report.get("total_frames", 0)
        print(f"   Total frames processed: {total}")
        print()

        # ── Step 4: Frame contract validation ──
        print("4. FRAME CONTRACT VALIDATION")
        print("-" * 60)
        if result_path and os.path.exists(result_path):
            contract = validate_output_video(result_path, max_check=50)
            print(f"   Frames checked: {contract['frames_checked']}")
            print(f"   Frames passing: {contract['frames_passing']}")
            print(f"   Frame contract: {'✅ ALL PASS' if contract['pass'] else f'❌ {contract["frames_checked"] - contract["frames_passing"]} FAILED'}")
            if contract.get("all_issues"):
                for issue in contract["all_issues"][:3]:
                    print(f"     Frame {issue['frame']}: {', '.join(issue['issues'])}")
        else:
            print("   ⚠️ No output video to validate")
            contract = {"frames_checked": 0, "frames_passing": 0, "pass": False, "total_frames": 0}
        print()

        # ── Step 5: Validation Dashboard ──
        print("5. CLAIM VALIDATION DASHBOARD")
        print("-" * 60)
        print()

        results = []

        # Claim 1: V3 PhysicalRenderer active (>50%)
        phys_rate = report.get("physical_render_rate", 0)
        p1 = phys_rate > 0.5
        results.append(("V3-01 PhysicalRenderer active", f"PhysicalRender rate: {phys_rate:.1%}", p1))

        # Claim 2: IntrinsicDecomposer success (>50%)
        intr_rate = report.get("intrinsic_success_rate", 0)
        p2 = intr_rate > 0.5
        results.append(("V3-02 IntrinsicDecomposer active", f"Intrinsic success rate: {intr_rate:.1%}", p2))

        # Claim 3: Frame contract upheld (all frames pass)
        p3 = contract["pass"] if total > 0 else False
        results.append(("V3-03 Frame contract (1920x1080x3, uint8, no NaN/Inf)",
                       f"{contract['frames_passing']}/{contract['frames_checked']} frames pass", p3))

        # Claim 4: RendererMode transitions stable (< 10 on 100 frames)
        transitions = report.get("renderer_mode_transitions", 0)
        p4 = transitions < 10
        results.append(("V3-04 RendererMode stable", f"{transitions} transitions in {total} frames", p4))

        # Claim 5: Avg intrinsic confidence > 0.5
        avg_conf = report.get("avg_intrinsic_confidence", 0)
        p5 = avg_conf > 0.5
        results.append(("V3-05 Intrinsic confidence adequate", f"Avg confidence: {avg_conf:.3f}", p5))

        # Claim 6: Avg decomposition error < 0.1
        avg_err = report.get("avg_decomposition_error", 0)
        p6 = avg_err < 0.1
        results.append(("V3-06 Decomposition error low", f"Avg error: {avg_err:.3f}", p6))

        # Claim 7: Fallback reasons tracked
        fb_dist = report.get("fallback_reason_distribution", {})
        p7 = len(fb_dist) > 0
        fb_str = ", ".join(f"{k}={v}" for k, v in sorted(fb_dist.items())) if fb_dist else "none"
        results.append(("V3-07 Fallback reason telemetry", f"Reasons: {fb_str}", p7))

        # Claim 8: No NaN/Inf in output frames
        p8 = contract["pass"] if total > 0 else False
        results.append(("V3-08 No NaN/Inf in output",
                       f"{contract['frames_passing']}/{contract['frames_checked']} clean", p8))

        # Claim 9: Telemetry coverage — all expected keys present
        expected_keys = [
            "total_frames", "physical_render_frames", "alpha_fallback_frames",
            "intrinsic_success_frames", "intrinsic_failure_frames",
            "renderer_mode_transitions", "fallback_reason_distribution",
            "renderer_mode_distribution", "avg_intrinsic_confidence",
            "avg_decomposition_error", "physical_render_rate", "alpha_fallback_rate",
            "intrinsic_success_rate", "intrinsic_failure_rate",
        ]
        missing = [k for k in expected_keys if k not in report]
        p9 = len(missing) == 0
        results.append(("V3-09 Telemetry key coverage",
                       f"{len(expected_keys) - len(missing)}/{len(expected_keys)} keys present" + (f" — missing: {missing}" if missing else ""), p9))

        # Claim 10: Alpha fallback rate < 50% (PhysicalRenderer dominant)
        alpha_rate = report.get("alpha_fallback_rate", 0)
        p10 = alpha_rate < 0.5
        results.append(("V3-10 PhysicalRenderer dominant", f"Alpha fallback rate: {alpha_rate:.1%}", p10))

        # Print dashboard
        for claim_id, description, passed in results:
            status = "✅" if passed else "❌"
            print(f"   {status}  {claim_id}")
            print(f"        {description}")
            print()

        # ── Summary ──
        print("6. SUMMARY")
        print("-" * 60)
        total_claims = len(results)
        passed_claims = sum(1 for _, _, p in results if p)
        elapsed = time.perf_counter() - ts_start
        print(f"   Claims passed: {passed_claims}/{total_claims}")
        print(f"   Time elapsed:  {elapsed:.1f}s")
        print(f"   Overall:       {'✅ ALL CLAIMS VERIFIED' if passed_claims == total_claims else f'❌ {total_claims - passed_claims} claim(s) failed'}")
        print()
        print("=" * 72)

        # Final verdict for AGENTS.md
        print()
        verdict = "PASS" if passed_claims == total_claims else "REGRESSION"
        print(f"   VERDICT: {verdict}")
        print()

        # Print raw telemetry for reference
        if passed_claims != total_claims:
            print("   Raw telemetry for debugging:")
            print(json.dumps({k: v for k, v in report.items() if not k.endswith("_sum") and not k.endswith("_count")}, indent=4, default=str))


if __name__ == "__main__":
    main()
