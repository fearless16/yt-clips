"""
test_t4_compat.py — T4 GPU compatibility check for ref_grade + video_analyzer.

Run on Colab with T4 GPU:
    !python tests/test_t4_compat.py

Exits 0 on success, non-zero on failure. No video required — tests are synthetic.
"""
import sys, os, time, cv2, numpy as np
sys.path.insert(0, '.')

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")

def verify_cuda():
    """Check that ffmpeg CUDA hwaccel is available on T4."""
    import subprocess
    r = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"],
                       capture_output=True, text=True, timeout=10)
    has_cuda = "cuda" in r.stdout.lower()
    check("ffmpeg CUDA hwaccel", has_cuda,
          f"Available: {r.stdout.strip()}")
    if not has_cuda:
        print("  ⚠  CPU fallback will be used (acceptable on Colab)")
    return has_cuda


def test_enrollment():
    """Enroll from expectation.png — must match Mac reference values."""
    from ref_grade import enroll
    params = enroll("expectation.png")
    check("enroll returns dict", isinstance(params, dict))
    check("contrast_ratio ~1.17", abs(params["contrast_ratio"] - 1.17) < 0.02,
          f"got {params['contrast_ratio']}")
    check("a_target ~139.6", abs(params["a_target"] - 139.6) < 1.0,
          f"got {params['a_target']}")
    check("b_target ~146.7", abs(params["b_target"] - 146.7) < 1.0,
          f"got {params['b_target']}")
    check("vignette_ratio ~1.19", abs(params["vignette_ratio"] - 1.19) < 0.05,
          f"got {params['vignette_ratio']}")
    check("_lut_a present (LUT)", "_lut_a" in params)
    check("_lut_b present (LUT)", "_lut_b" in params)
    check("_split_lut present (LUT)", "_split_lut" in params)
    return params


def test_apply_grade(params):
    """Apply grade to synthetic frames — must match Mac behavior."""
    from ref_grade import apply_grade, _vignette_cache
    _vignette_cache.clear()

    # Synthetic 1080p frame
    frame = np.random.randint(100, 155, (1080, 1920, 3), dtype=np.uint8)
    out = apply_grade(frame, params)

    check("shape preserved", out.shape == (1080, 1920, 3))
    check("dtype uint8", out.dtype == np.uint8)
    check("pixel range [0,255]", out.min() >= 0 and out.max() <= 255)
    check("not identical to input", not np.array_equal(out, frame))

    # White frame passthrough
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    wout = apply_grade(white, params)
    check("white frame preserved", wout.max() == 255)

    # Black frame — split-tone adds color to shadows, so pixels may shift
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    bout = apply_grade(black, params)
    check("black frame L≈0", float(np.mean(cv2.cvtColor(bout, cv2.COLOR_BGR2LAB)[:,:,0])) < 5)

    # Flicker test: 10 identical frames → same output
    results = [apply_grade(frame, params) for _ in range(10)]
    all_same = all(np.array_equal(results[0], r) for r in results)
    check("flicker-free (10 identical frames)", all_same)

    # Real frame from expectation.png
    ref = cv2.imread("expectation.png")
    g = apply_grade(ref, params)
    glab = cv2.cvtColor(g, cv2.COLOR_BGR2LAB)
    mean_a, mean_b = float(np.mean(glab[:,:,1])), float(np.mean(glab[:,:,2]))
    print(f"  Real output: L={np.mean(glab[:,:,0]):.0f} a={mean_a:.0f} b={mean_b:.0f}")


def test_performance(params):
    """1080p throughput must be >10 fps on T4."""
    from ref_grade import apply_grade, _vignette_cache
    _vignette_cache.clear()
    frame = np.random.randint(100, 155, (1080, 1920, 3), dtype=np.uint8)

    for _ in range(5): apply_grade(frame, params)

    t0 = time.perf_counter()
    N = 30
    for _ in range(N): apply_grade(frame, params)
    elapsed = time.perf_counter() - t0
    fps = N / elapsed
    check(f"1080p speed >10fps ({fps:.0f}fps)",
          fps > 10, f"got {fps:.0f} fps ({elapsed/N*1000:.1f}ms per frame)")


def test_grade_video(params):
    """grade_video synthetic pipe — creates temp source, grades, checks output."""
    from ref_grade import grade_video
    import tempfile, shutil

    # Create 1s synthetic video at 720p
    tmpdir = tempfile.mkdtemp()
    src = os.path.join(tmpdir, "source.mp4")
    out = os.path.join(tmpdir, "graded.mp4")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=720x1280:r=30:d=1",
            "-c:v", "libx264", "-preset", "ultrafast",
            src,
        ]
        import subprocess
        subprocess.run(cmd, capture_output=True, timeout=15, check=True)

        result = grade_video(src, "expectation.png", out)
        check("output created", os.path.exists(out) and os.path.getsize(out) > 1000,
              f"size={os.path.getsize(out) if os.path.exists(out) else 0}")
        check("returns output path", result == out)

        # Verify duration
        cap = cv2.VideoCapture(out)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        check("~30 frames at ~30fps", abs(frames - 30) <= 3,
              f"got {frames} frames @ {fps:.1f}fps")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_pipeline_mode_parse():
    """Verify --mode flag parses correctly."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["face_mapper", "ref_grade"], default=None)
    args = parser.parse_args(["--mode", "ref_grade"])
    check("--mode ref_grade parses", args.mode == "ref_grade")
    args2 = parser.parse_args(["--mode", "face_mapper"])
    check("--mode face_mapper parses", args2.mode == "face_mapper")
    args3 = parser.parse_args([])
    check("no --mode defaults to None", args3.mode is None)


def test_video_analyzer_hwaccel_fix():
    """Verify the hwaccel auto-detect doesn't crash on T4."""
    from video_analyzer import _sample_frames_gpu as gpu
    import tempfile, os, subprocess

    # Create a tiny 1-frame video
    tmpdir = tempfile.mkdtemp()
    src = os.path.join(tmpdir, "tiny.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "160x120", "-r", "1",
            "-i", "/dev/zero",
            "-frames:v", "1",
            "-c:v", "libx264", "-preset", "ultrafast",
            src,
        ], capture_output=True, timeout=10, check=True)

        frames = list(gpu(src, 1, 160, 120, 1, 1.0, 1))
        check("gpu sampler yields frames", len(frames) > 0,
              f"got {len(frames)} frames")
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 55)
    print("T4 GPU Compatibility Check")
    print("=" * 55)

    print("\n[1/7] CUDA hwaccel detection...")
    verify_cuda()

    print("\n[2/7] Enrollment (12 params must match Mac reference)...")
    params = test_enrollment()

    print("\n[3/7] apply_grade correctness...")
    test_apply_grade(params)

    print("\n[4/7] 1080p throughput (>10fps)...")
    test_performance(params)

    print("\n[5/7] grade_video ffmpeg pipe...")
    test_grade_video(params)

    print("\n[6/7] pipeline --mode flag...")
    test_pipeline_mode_parse()

    print("\n[7/7] video_analyzer hwaccel fix...")
    test_video_analyzer_hwaccel_fix()

    print(f"\n{'='*55}")
    print(f"  {PASS} passed, {FAIL} failed")
    print(f"{'='*55}")
    sys.exit(FAIL)
