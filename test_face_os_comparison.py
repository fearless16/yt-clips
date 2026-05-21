"""
test_face_os_comparison.py — Comprehensive comparison test.

Runs Face OS pipeline on test_clip.mp4 and compares with:
  1. Original clip (baseline)
  2. ref_grade output (existing Phase 4.25)
  3. face_mapper output (existing Phase 4.25)

Extracts per-frame metrics and generates comparison tables.

Usage:
    python test_face_os_comparison.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Metrics extraction ─────────────────────────────────────────────────────

def extract_frame_metrics(frame: np.ndarray) -> Dict:
    """Extract all metrics from a single frame."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # LAB statistics
    l_mean = float(np.mean(lab[:, :, 0]))
    a_mean = float(np.mean(lab[:, :, 1]))
    b_mean = float(np.mean(lab[:, :, 2]))
    l_std = float(np.std(lab[:, :, 0]))

    # Sharpness (Laplacian variance)
    lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))

    # Saturation
    sat_mean = float(np.mean(hsv[:, :, 1]))

    # Brightness distribution
    dark_pct = float(np.mean(gray < 51)) * 100
    mid_pct = float(np.mean((gray >= 51) & (gray < 204))) * 100
    bright_pct = float(np.mean(gray >= 204)) * 100

    # Contrast (Michelson)
    l_max = float(np.max(gray))
    l_min = float(np.min(gray))
    contrast = (l_max - l_min) / max(l_max + l_min, 1)

    return {
        "L": l_mean,
        "a": a_mean,
        "b": b_mean,
        "L_std": l_std,
        "sharpness": lap_var,
        "saturation": sat_mean,
        "dark_pct": dark_pct,
        "mid_pct": mid_pct,
        "bright_pct": bright_pct,
        "contrast": contrast,
        "brightness": l_mean / 255.0,
    }


def extract_face_metrics(
    frame: np.ndarray,
    cascade: cv2.CascadeClassifier,
) -> Optional[Dict]:
    """Extract metrics from face region only."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = frame[y:y+h, x:x+w]
    face_lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    return {
        "face_L": float(np.mean(face_lab[:, :, 0])),
        "face_a": float(np.mean(face_lab[:, :, 1])),
        "face_b": float(np.mean(face_lab[:, :, 2])),
        "face_sharpness": float(np.var(cv2.Laplacian(face_gray, cv2.CV_64F))),
        "face_saturation": float(np.mean(face_hsv[:, :, 1])),
        "face_contrast": float(np.std(face_gray)),
        "face_size": w * h,
        "face_bbox": (int(x), int(y), int(w), int(h)),
    }


def extract_video_metrics(
    video_path: str,
    sample_interval: float = 0.5,
    max_frames: int = 200,
) -> Dict:
    """Extract metrics from all frames of a video.

    Args:
        video_path: Path to video file
        sample_interval: Seconds between sampled frames
        max_frames: Maximum frames to sample

    Returns:
        Dict with per-frame metrics and summary statistics
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0

    # Sample frames
    sample_frames = int(sample_interval * fps)
    if sample_frames < 1:
        sample_frames = 1

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    frame_metrics = []
    face_metrics = []
    frame_idx = 0
    sampled = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_frames == 0 and sampled < max_frames:
            fm = extract_frame_metrics(frame)
            fm["frame_idx"] = frame_idx
            fm["timestamp"] = frame_idx / fps
            frame_metrics.append(fm)

            # Face metrics
            face_m = extract_face_metrics(frame, cascade)
            if face_m:
                face_m["frame_idx"] = frame_idx
                face_metrics.append(face_m)

            sampled += 1

        frame_idx += 1

    cap.release()

    if not frame_metrics:
        return {"error": "No frames extracted"}

    # Compute summary statistics
    def stats(key, data):
        values = [m[key] for m in data if key in m]
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    # Frame-to-frame flicker
    l_values = [m["L"] for m in frame_metrics]
    flicker = 0.0
    if len(l_values) > 1:
        diffs = [abs(l_values[i] - l_values[i-1]) for i in range(1, len(l_values))]
        flicker = float(np.mean(diffs))

    # LAB distance between consecutive frames
    lab_distances = []
    for i in range(1, len(frame_metrics)):
        dL = frame_metrics[i]["L"] - frame_metrics[i-1]["L"]
        da = frame_metrics[i]["a"] - frame_metrics[i-1]["a"]
        db = frame_metrics[i]["b"] - frame_metrics[i-1]["b"]
        lab_distances.append(float(np.sqrt(dL**2 + da**2 + db**2)))

    summary = {
        "video": video_path,
        "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}" if cap.isOpened() else "unknown",
        "fps": fps,
        "duration": duration,
        "total_frames": total,
        "sampled_frames": sampled,
        "face_detection_rate": len(face_metrics) / max(sampled, 1),
        "L": stats("L", frame_metrics),
        "a": stats("a", frame_metrics),
        "b": stats("b", frame_metrics),
        "L_std": stats("L_std", frame_metrics),
        "sharpness": stats("sharpness", frame_metrics),
        "saturation": stats("saturation", frame_metrics),
        "contrast": stats("contrast", frame_metrics),
        "flicker_L": flicker,
        "flicker_LAB": float(np.mean(lab_distances)) if lab_distances else 0,
        "flicker_LAB_max": float(np.max(lab_distances)) if lab_distances else 0,
    }

    # Face-specific summary
    if face_metrics:
        summary["face_L"] = stats("face_L", face_metrics)
        summary["face_a"] = stats("face_a", face_metrics)
        summary["face_b"] = stats("face_b", face_metrics)
        summary["face_sharpness"] = stats("face_sharpness", face_metrics)
        summary["face_saturation"] = stats("face_saturation", face_metrics)
        summary["face_contrast"] = stats("face_contrast", face_metrics)

    return {
        "summary": summary,
        "frame_metrics": frame_metrics,
        "face_metrics": face_metrics,
    }


# ─── Reference analysis ─────────────────────────────────────────────────────

def analyze_reference(image_path: str) -> Dict:
    """Extract metrics from reference image (expectation.png)."""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Cannot read {image_path}"}

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    frame_m = extract_frame_metrics(img)
    face_m = extract_face_metrics(img, cascade)

    result = {"frame": frame_m}
    if face_m:
        result["face"] = face_m

    return result


# ─── Comparison table generation ────────────────────────────────────────────

def print_comparison_table(
    original: Dict,
    ref_graded: Optional[Dict],
    face_mapper_out: Optional[Dict],
    face_os_out: Optional[Dict],
    reference: Dict,
) -> str:
    """Generate a formatted comparison table."""
    lines = []
    lines.append("=" * 120)
    lines.append("FACE OS PIPELINE — PARAMETER COMPARISON")
    lines.append("=" * 120)

    # Header
    header = f"{'Metric':<25} {'Reference':>12} {'Original':>12}"
    if ref_graded:
        header += f" {'ref_grade':>12}"
    if face_mapper_out:
        header += f" {'face_mapper':>12}"
    if face_os_out:
        header += f" {'Face OS':>12}"
    lines.append(header)
    lines.append("-" * 120)

    # Reference face metrics
    ref_face = reference.get("face", reference.get("frame", {}))

    def get_val(data, key, subkey="mean"):
        if subkey in data.get(key, {}):
            return data[key][subkey]
        return data.get(key, 0)

    # Metrics to compare
    metrics = [
        ("L (brightness)", "L", "face_L"),
        ("a (red-green)", "a", "face_a"),
        ("b (yellow-blue)", "b", "face_b"),
        ("L std (contrast)", "L_std", "face_contrast"),
        ("Sharpness", "sharpness", "face_sharpness"),
        ("Saturation", "saturation", "face_saturation"),
    ]

    for label, frame_key, face_key in metrics:
        # Reference value
        ref_val = ref_face.get(face_key, ref_face.get(frame_key, 0))

        # Original
        orig_m = original.get("summary", {})
        orig_val = get_val(orig_m, face_key) or get_val(orig_m, frame_key)

        line = f"{label:<25} {ref_val:>12.1f} {orig_val:>12.1f}"

        # ref_grade
        if ref_graded:
            rg_m = ref_graded.get("summary", {})
            rg_val = get_val(rg_m, face_key) or get_val(rg_m, frame_key)
            delta = rg_val - orig_val
            line += f" {rg_val:>7.1f} ({delta:>+5.1f})"

        # face_mapper
        if face_mapper_out:
            fm_m = face_mapper_out.get("summary", {})
            fm_val = get_val(fm_m, face_key) or get_val(fm_m, frame_key)
            delta = fm_val - orig_val
            line += f" {fm_val:>7.1f} ({delta:>+5.1f})"

        # Face OS
        if face_os_out:
            fo_m = face_os_out.get("summary", {})
            fo_val = get_val(fo_m, face_key) or get_val(fo_m, frame_key)
            delta = fo_val - orig_val
            line += f" {fo_val:>7.1f} ({delta:>+5.1f})"

        lines.append(line)

    # Flicker metrics
    lines.append("-" * 120)
    lines.append(f"{'Flicker (L mean diff)':<25} {'N/A':>12} {original.get('summary', {}).get('flicker_L', 0):>12.2f}", )
    if ref_graded:
        lines[-1] += f" {ref_graded.get('summary', {}).get('flicker_L', 0):>12.2f}"
    if face_mapper_out:
        lines[-1] += f" {face_mapper_out.get('summary', {}).get('flicker_L', 0):>12.2f}"
    if face_os_out:
        lines[-1] += f" {face_os_out.get('summary', {}).get('flicker_L', 0):>12.2f}"

    lines.append(f"{'Flicker (LAB dist)':<25} {'N/A':>12} {original.get('summary', {}).get('flicker_LAB', 0):>12.2f}")
    if ref_graded:
        lines[-1] += f" {ref_graded.get('summary', {}).get('flicker_LAB', 0):>12.2f}"
    if face_mapper_out:
        lines[-1] += f" {face_mapper_out.get('summary', {}).get('flicker_LAB', 0):>12.2f}"
    if face_os_out:
        lines[-1] += f" {face_os_out.get('summary', {}).get('flicker_LAB', 0):>12.2f}"

    # Face detection rate
    lines.append("-" * 120)
    lines.append(f"{'Face detection rate':<25} {'N/A':>12} {original.get('summary', {}).get('face_detection_rate', 0):>12.1%}")
    if ref_graded:
        lines[-1] += f" {ref_graded.get('summary', {}).get('face_detection_rate', 0):>12.1%}"
    if face_mapper_out:
        lines[-1] += f" {face_mapper_out.get('summary', {}).get('face_detection_rate', 0):>12.1%}"
    if face_os_out:
        lines[-1] += f" {face_os_out.get('summary', {}).get('face_detection_rate', 0):>12.1%}"

    # LAB distance from reference
    lines.append("-" * 120)
    ref_L = ref_face.get("face_L", ref_face.get("L", 0))
    ref_a = ref_face.get("face_a", ref_face.get("a", 0))
    ref_b = ref_face.get("face_b", ref_face.get("b", 0))

    def lab_distance(data, face_key_l="face_L", face_key_a="face_a", face_key_b="face_b"):
        s = data.get("summary", {})
        L = get_val(s, face_key_l) or get_val(s, "L")
        a = get_val(s, face_key_a) or get_val(s, "a")
        b = get_val(s, face_key_b) or get_val(s, "b")
        return float(np.sqrt((L - ref_L)**2 + (a - ref_a)**2 + (b - ref_b)**2))

    orig_dist = lab_distance(original)
    line = f"{'LAB dist from ref':<25} {0:>12.1f} {orig_dist:>12.1f}"
    if ref_graded:
        rg_dist = lab_distance(ref_graded)
        line += f" {rg_dist:>12.1f}"
    if face_mapper_out:
        fm_dist = lab_distance(face_mapper_out)
        line += f" {fm_dist:>12.1f}"
    if face_os_out:
        fo_dist = lab_distance(face_os_out)
        line += f" {fo_dist:>12.1f}"
    lines.append(line)

    # L distance from reference
    def l_distance(data):
        s = data.get("summary", {})
        L = get_val(s, "face_L") or get_val(s, "L")
        return abs(L - ref_L)

    orig_l_dist = l_distance(original)
    line = f"{'L distance from ref':<25} {0:>12.1f} {orig_l_dist:>12.1f}"
    if ref_graded:
        line += f" {l_distance(ref_graded):>12.1f}"
    if face_mapper_out:
        line += f" {l_distance(face_mapper_out):>12.1f}"
    if face_os_out:
        line += f" {l_distance(face_os_out):>12.1f}"
    lines.append(line)

    lines.append("=" * 120)

    # Summary verdict
    lines.append("")
    lines.append("VERDICT:")

    distances = {"Original": orig_dist}
    if ref_graded:
        distances["ref_grade"] = lab_distance(ref_graded)
    if face_mapper_out:
        distances["face_mapper"] = lab_distance(face_mapper_out)
    if face_os_out:
        distances["Face OS"] = lab_distance(face_os_out)

    best = min(distances, key=distances.get)
    lines.append(f"  Closest to reference: {best} (LAB distance: {distances[best]:.1f})")

    # Flicker comparison
    flickers = {"Original": original.get("summary", {}).get("flicker_LAB", 0)}
    if ref_graded:
        flickers["ref_grade"] = ref_graded.get("summary", {}).get("flicker_LAB", 0)
    if face_mapper_out:
        flickers["face_mapper"] = face_mapper_out.get("summary", {}).get("flicker_LAB", 0)
    if face_os_out:
        flickers["Face OS"] = face_os_out.get("summary", {}).get("flicker_LAB", 0)

    most_stable = min(flickers, key=flickers.get)
    lines.append(f"  Most stable (lowest flicker): {most_stable} (LAB flicker: {flickers[most_stable]:.2f})")

    return "\n".join(lines)


def print_detailed_frame_table(
    face_os_metrics: List[Dict],
    original_metrics: List[Dict],
    max_rows: int = 20,
) -> str:
    """Print per-frame detailed comparison."""
    lines = []
    lines.append("")
    lines.append("=" * 100)
    lines.append("PER-FRAME DETAIL: Face OS vs Original (sampled)")
    lines.append("=" * 100)
    lines.append(f"{'Frame':>6} {'Time':>6} {'Orig_L':>8} {'OS_L':>8} {'ΔL':>6} {'Orig_a':>8} {'OS_a':>8} {'Δa':>6} {'Orig_b':>8} {'OS_b':>8} {'Δb':>6} {'Orig_sharp':>10} {'OS_sharp':>10}")
    lines.append("-" * 100)

    for i in range(min(len(face_os_metrics), len(original_metrics), max_rows)):
        om = original_metrics[i]
        fm = face_os_metrics[i]

        dL = fm["L"] - om["L"]
        da = fm["a"] - om["a"]
        db = fm["b"] - om["b"]

        lines.append(
            f"{om.get('frame_idx', i):>6} "
            f"{om.get('timestamp', 0):>6.1f} "
            f"{om['L']:>8.1f} {fm['L']:>8.1f} {dL:>+6.1f} "
            f"{om['a']:>8.1f} {fm['a']:>8.1f} {da:>+6.1f} "
            f"{om['b']:>8.1f} {fm['b']:>8.1f} {db:>+6.1f} "
            f"{om['sharpness']:>10.1f} {fm['sharpness']:>10.1f}"
        )

    return "\n".join(lines)


# ─── Main test runner ────────────────────────────────────────────────────────

def main():
    """Run the full comparison test."""
    print("=" * 80)
    print("FACE OS COMPARISON TEST")
    print("=" * 80)

    # Paths
    test_clip = "clips_test/test_clip.mp4"
    reference = "expectation.png"
    output_dir = Path("output/face_os_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(test_clip).exists():
        print(f"ERROR: Test clip not found: {test_clip}")
        return

    if not Path(reference).exists():
        print(f"ERROR: Reference not found: {reference}")
        return

    # ── 1. Analyze reference ─────────────────────────────────────────────
    print("\n[1/6] Analyzing reference image...")
    ref_metrics = analyze_reference(reference)
    print(f"  Reference face: L={ref_metrics.get('face', {}).get('face_L', 0):.1f}, "
          f"a={ref_metrics.get('face', {}).get('face_a', 0):.1f}, "
          f"b={ref_metrics.get('face', {}).get('face_b', 0):.1f}")

    # ── 2. Analyze original clip ─────────────────────────────────────────
    print("\n[2/6] Analyzing original clip...")
    t0 = time.perf_counter()
    original_metrics = extract_video_metrics(test_clip, sample_interval=0.5)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    print(f"  Frames sampled: {original_metrics['summary']['sampled_frames']}")
    print(f"  Face detection rate: {original_metrics['summary']['face_detection_rate']:.1%}")
    print(f"  Mean L: {original_metrics['summary']['L']['mean']:.1f}, "
          f"a: {original_metrics['summary']['a']['mean']:.1f}, "
          f"b: {original_metrics['summary']['b']['mean']:.1f}")
    print(f"  Flicker (LAB): {original_metrics['summary']['flicker_LAB']:.2f}")

    # ── 3. Run ref_grade ─────────────────────────────────────────────────
    ref_graded_path = str(output_dir / "ref_graded.mp4")
    print("\n[3/6] Running ref_grade...")
    t0 = time.perf_counter()
    try:
        from ref_grade import grade_video
        result = grade_video(test_clip, reference, ref_graded_path)
        if result == ref_graded_path and Path(ref_graded_path).exists():
            ref_graded_metrics = extract_video_metrics(ref_graded_path, sample_interval=0.5)
            print(f"  Done in {time.perf_counter() - t0:.1f}s")
            print(f"  Mean L: {ref_graded_metrics['summary']['L']['mean']:.1f}")
        else:
            ref_graded_metrics = None
            print("  ref_grade failed — skipping")
    except Exception as e:
        ref_graded_metrics = None
        print(f"  ref_grade error: {e}")

    # ── 4. Run face_mapper ───────────────────────────────────────────────
    face_mapper_path = str(output_dir / "face_mapped.mp4")
    print("\n[4/6] Running face_mapper...")
    t0 = time.perf_counter()
    try:
        from face_mapper import enhance_video
        result = enhance_video(test_clip, reference, face_mapper_path, use_region_grading=True)
        if result == face_mapper_path and Path(face_mapper_path).exists():
            face_mapper_metrics = extract_video_metrics(face_mapper_path, sample_interval=0.5)
            print(f"  Done in {time.perf_counter() - t0:.1f}s")
            print(f"  Mean L: {face_mapper_metrics['summary']['L']['mean']:.1f}")
        else:
            face_mapper_metrics = None
            print("  face_mapper failed — skipping")
    except Exception as e:
        face_mapper_metrics = None
        print(f"  face_mapper error: {e}")

    # ── 5. Run Face OS ───────────────────────────────────────────────────
    face_os_path = str(output_dir / "face_os_output.mp4")
    print("\n[5/6] Running Face OS pipeline...")
    t0 = time.perf_counter()
    try:
        from face_os.pipeline import FaceOSPipeline
        pipeline = FaceOSPipeline()
        if pipeline.enroll(reference, "photos/"):
            result = pipeline.process(test_clip, face_os_path)
            if result and Path(face_os_path).exists():
                face_os_metrics = extract_video_metrics(face_os_path, sample_interval=0.5)
                print(f"  Done in {time.perf_counter() - t0:.1f}s")
                print(f"  Mean L: {face_os_metrics['summary']['L']['mean']:.1f}")
            else:
                face_os_metrics = None
                print("  Face OS processing failed — skipping")
        else:
            face_os_metrics = None
            print("  Face OS enrollment failed — skipping")
    except Exception as e:
        face_os_metrics = None
        print(f"  Face OS error: {e}")
        import traceback
        traceback.print_exc()

    # ── 6. Generate comparison tables ────────────────────────────────────
    print("\n[6/6] Generating comparison tables...")

    table = print_comparison_table(
        original_metrics,
        ref_graded_metrics,
        face_mapper_metrics,
        face_os_metrics,
        ref_metrics,
    )
    print(table)

    # Per-frame detail
    if face_os_metrics and original_metrics:
        detail = print_detailed_frame_table(
            face_os_metrics.get("frame_metrics", []),
            original_metrics.get("frame_metrics", []),
        )
        print(detail)

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "reference": ref_metrics,
        "original": original_metrics.get("summary", {}),
        "ref_graded": ref_graded_metrics.get("summary", {}) if ref_graded_metrics else None,
        "face_mapper": face_mapper_metrics.get("summary", {}) if face_mapper_metrics else None,
        "face_os": face_os_metrics.get("summary", {}) if face_os_metrics else None,
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {results_path}")

    # Save table
    table_path = output_dir / "comparison_table.txt"
    with open(table_path, "w") as f:
        f.write(table)
    print(f"Table saved to: {table_path}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
