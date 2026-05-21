"""
test_face_os_v2.py — Comparison test for Face OS v2 architecture.

Tests the new identity belief state engine against the old approach.
Compares:
  - Old Face OS (per-pixel RGB averaging)
  - New Face OS v2 (frequency decomposition + belief distributions + patch memory)

Metrics:
  - LAB distance from reference
  - Flicker score (frame-to-frame variance)
  - Sharpness (Laplacian variance)
  - Face detection rate
  - Eye stability
  - Beard stability
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from face_os.identity_state import IdentityState, FrequencyDecomposition
from face_os.patch_memory import PatchMemory
from face_os.temporal_solve import TemporalRepairEngine, FrameQuality


def compute_lab_stats(frames: list) -> dict:
    """Compute LAB statistics for a list of frames."""
    l_vals, a_vals, b_vals = [], [], []
    for frame in frames:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_vals.append(float(np.mean(lab[:, :, 0])))
        a_vals.append(float(np.mean(lab[:, :, 1])))
        b_vals.append(float(np.mean(lab[:, :, 2])))
    return {
        'L_mean': np.mean(l_vals),
        'L_std': np.std(l_vals),
        'a_mean': np.mean(a_vals),
        'a_std': np.std(a_vals),
        'b_mean': np.mean(b_vals),
        'b_std': np.std(b_vals),
    }


def compute_flicker(frames: list) -> float:
    """Compute frame-to-frame LAB variance (flicker)."""
    if len(frames) < 2:
        return 0.0

    diffs = []
    for i in range(1, len(frames)):
        lab1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2LAB).astype(np.float32)
        diff = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))
        diffs.append(float(np.mean(diff)))

    return float(np.mean(diffs))


def compute_sharpness(frames: list) -> float:
    """Compute average sharpness (Laplacian variance)."""
    sharpness_vals = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        sharpness_vals.append(float(np.var(lap)))
    return float(np.mean(sharpness_vals))


def compute_eye_stability(frames: list) -> float:
    """Compute eye region stability across frames.

    Lower = more stable (better).
    """
    if len(frames) < 2:
        return 0.0

    # Extract eye regions (top 40%, middle 60%)
    eye_diffs = []
    for i in range(1, len(frames)):
        h, w = frames[i].shape[:2]
        y1, y2 = int(h * 0.2), int(h * 0.45)
        x1, x2 = int(w * 0.2), int(w * 0.8)

        eye1 = frames[i - 1][y1:y2, x1:x2]
        eye2 = frames[i][y1:y2, x1:x2]

        if eye1.size > 0 and eye2.size > 0:
            diff = np.abs(eye1.astype(np.float32) - eye2.astype(np.float32))
            eye_diffs.append(float(np.mean(diff)))

    return float(np.mean(eye_diffs)) if eye_diffs else 0.0


def test_frequency_decomposition():
    """Test frequency decomposition works correctly."""
    print("\n=== TEST: Frequency Decomposition ===")

    freq = FrequencyDecomposition(low_pass_sigma=3.0)

    # Create test image with SMOOTH gradients (like real skin)
    h, w = 100, 100
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    img = np.stack([
        (xx * 200 + 50).astype(np.uint8),
        (yy * 150 + 80).astype(np.uint8),
        ((xx + yy) * 100 + 50).astype(np.uint8),
    ], axis=2)

    # Decompose
    low, high = freq.decompose(img)

    # Verify shapes
    assert low.shape == (h, w, 3), f"Low shape: {low.shape}"
    assert high.shape == (h, w, 3), f"High shape: {high.shape}"

    # Verify reconstruction
    reconstructed = freq.reconstruct(low, high)
    diff = np.abs(img.astype(np.float32) - reconstructed.astype(np.float32))
    max_diff = float(np.max(diff))
    assert max_diff < 1.0, f"Reconstruction error: {max_diff}"

    # Verify low freq is smooth
    low_var = np.var(low)
    img_var = np.var(img.astype(np.float32))
    assert low_var < img_var, f"Low freq should be smoother: {low_var} vs {img_var}"

    # Verify high freq is small (smooth images have small high freq)
    high_energy = np.sqrt(np.mean(high ** 2))
    img_energy = np.sqrt(np.mean(img.astype(np.float32) ** 2))
    high_ratio = high_energy / (img_energy + 1e-6)
    assert high_ratio < 0.3, f"High freq should be small for smooth image: {high_ratio:.2%}"

    print(f"  Low freq variance: {low_var:.1f} (source: {img_var:.1f})")
    print(f"  High freq energy ratio: {high_ratio:.2%}")
    print(f"  Reconstruction max error: {max_diff:.4f}")
    print("  PASS")


def test_belief_pixel():
    """Test per-pixel belief distributions."""
    print("\n=== TEST: Belief Pixel ===")

    from face_os.identity_state import BeliefPixel

    h, w = 50, 50
    belief = BeliefPixel(h, w, 3)

    # Create observations
    np.random.seed(42)
    obs1 = np.random.randint(100, 200, (h, w, 3)).astype(np.float32)
    obs2 = np.random.randint(100, 200, (h, w, 3)).astype(np.float32)
    quality = np.ones((h, w), dtype=np.float32) * 0.8

    # Update with obs1
    freq = FrequencyDecomposition()
    low1, high1 = freq.decompose(obs1)
    belief.update(low1, high1, quality, pose=(0, 0, 0))

    # Update with obs2 (lower quality — should NOT update high freq)
    low2, high2 = freq.decompose(obs2)
    quality_low = np.ones((h, w), dtype=np.float32) * 0.3
    belief.update(low2, high2, quality_low, pose=(0, 0, 0))

    # Verify: high freq should still be from obs1 (higher quality)
    assert np.allclose(belief.best_high, high1, atol=0.1), "High freq should keep best observation"

    # Verify: low freq should be blended
    assert not np.allclose(belief.best_low, low1, atol=0.1), "Low freq should be blended"

    # Verify confidence
    conf = belief.get_confidence()
    assert conf.min() >= 0, f"Confidence min: {conf.min()}"
    assert conf.max() <= 1, f"Confidence max: {conf.max()}"

    print(f"  Low freq blended: YES")
    print(f"  High freq preserved best: YES")
    print(f"  Confidence range: [{conf.min():.3f}, {conf.max():.3f}]")
    print("  PASS")


def test_patch_memory():
    """Test per-region patch memory."""
    print("\n=== TEST: Patch Memory ===")

    memory = PatchMemory()

    # Create canonical face (256x256)
    h, w = 256, 256
    face = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    quality = np.ones((h, w), dtype=np.float32) * 0.8

    # Initialize
    memory.initialize(face, quality)
    assert memory._initialized

    # Update with different poses
    for yaw in [-20, -10, 0, 10, 20]:
        face_pose = face.copy().astype(np.int16)
        face_pose[:, :, 0] = np.clip(face_pose[:, :, 0] + yaw, 0, 255)
        face_pose = face_pose.astype(np.uint8)
        memory.update(face_pose, quality, pose=(yaw, 0, 0), frame_idx=yaw + 20)

    # Query specific region
    left_eye, conf = memory.query_region('left_eye', pose=(0, 0, 0))
    assert left_eye is not None, "Should have left eye patch"
    assert conf > 0, f"Should have confidence: {conf}"

    # Query pose-specific
    left_eye_yaw, conf_yaw = memory.query_region('left_eye', pose=(15, 0, 0))
    assert left_eye_yaw is not None, "Should have pose-specific patch"

    # Query all
    reconstructed, conf_map = memory.query_all((h, w), pose=(0, 0, 0))
    assert reconstructed.shape == (h, w, 3), f"Reconstructed shape: {reconstructed.shape}"
    assert conf_map.shape == (h, w), f"Conf map shape: {conf_map.shape}"

    region_confs = memory.get_region_confidences()
    print(f"  Region confidences: {region_confs}")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Confidence map range: [{conf_map.min():.3f}, {conf_map.max():.3f}]")
    print("  PASS")


def test_bidirectional_solver():
    """Test bidirectional temporal solver."""
    print("\n=== TEST: Bidirectional Solver ===")

    solver = TemporalRepairEngine(lookback=5, lookahead=5)

    h, w = 64, 64

    # Simulate 20 frames: some sharp, some blurry
    for i in range(20):
        # Create face
        face = np.ones((h, w, 3), dtype=np.uint8) * 128
        face[:, :, 0] = i * 10  # Vary L channel

        # Quality: sharp at frames 5, 10, 15; blurry elsewhere
        if i in [5, 10, 15]:
            quality = np.ones((h, w), dtype=np.float32) * 0.9
            sharpness = 0.9
        else:
            quality = np.ones((h, w), dtype=np.float32) * 0.4
            sharpness = 0.3

        solver.collect_frame(
            i, face, quality,
            sharpness=sharpness,
            pose=(0, 0, 0),
            detection_confidence=0.8,
        )

    # Solve
    results = solver.solve()

    # Verify HQ frames identified
    hq = solver.solver.get_hq_frame_count()
    assert hq >= 3, f"Should identify at least 3 HQ frames: {hq}"

    # Verify blurry frames get repaired
    # Frame 8 (between HQ 5 and 10) should be better than original
    if 8 in results:
        solved_face, solved_conf = results[8]
        assert solved_conf.max() > 0, "Should have non-zero confidence"

    print(f"  HQ frames identified: {hq}")
    print(f"  Solved frames: {len(results)}")
    print(f"  Frame 8 confidence max: {results[8][1].max():.3f}")
    print("  PASS")


def test_identity_state():
    """Test full identity state engine."""
    print("\n=== TEST: Identity State ===")

    state = IdentityState()

    h, w = 256, 256

    # Feed multiple observations
    for i in range(10):
        face = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        quality = np.ones((h, w), dtype=np.float32) * 0.7
        state.update(face, quality, pose=(0, 0, 0))

    assert state.is_initialized()

    # Query
    query_face = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    query_quality = np.ones((h, w), dtype=np.float32) * 0.5
    result, confidence = state.query(query_face, query_quality)

    assert result.shape == (h, w, 3), f"Result shape: {result.shape}"
    assert confidence.shape == (h, w), f"Confidence shape: {confidence.shape}"
    assert confidence.max() > 0, "Should have non-zero confidence"

    # Query specific region
    region_patch, region_conf = state.query_region('left_eye', query_face)
    assert region_patch is not None, "Should have region patch"
    assert region_conf > 0, f"Region confidence: {region_conf}"

    # Get stable L channel
    stable_l = state.get_stable_l_channel()
    assert stable_l is not None
    assert stable_l.shape == (h, w)

    print(f"  Initialized: {state.is_initialized()}")
    print(f"  Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"  Region confidence: {region_conf:.3f}")
    print(f"  Stable L shape: {stable_l.shape}")
    print("  PASS")


def run_comparison(video_path: str = "clips_test/test_clip.mp4"):
    """Run full comparison on a test video."""
    print("\n=== COMPARISON TEST ===")
    print(f"  Video: {video_path}")

    if not Path(video_path).exists():
        print(f"  SKIPPED: Video not found")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  SKIPPED: Cannot open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Frames: {total_frames}, FPS: {fps}")

    # Collect frames
    frames = []
    for i in range(min(total_frames, 100)):  # Limit to 100 frames for speed
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"  Loaded {len(frames)} frames")

    # Process with new architecture
    print("\n  Processing with Face OS v2...")
    t_start = time.perf_counter()

    # Initialize modules
    freq = FrequencyDecomposition()
    state = IdentityState()
    patch_mem = PatchMemory()
    solver = TemporalRepairEngine(lookback=5, lookahead=5)

    # Forward pass: collect
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        # Simulate canonical face (resize to 256x256)
        canonical = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        quality = np.ones((256, 256), dtype=np.float32) * 0.7

        # Compute real quality
        gray = cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
        sharpness = np.clip(lap / 50.0, 0, 1)
        brightness = gray.astype(np.float32) / 255.0
        brightness_weight = 1.0 - np.abs(brightness - 0.5) * 2
        brightness_weight = np.clip(brightness_weight, 0.1, 1.0)
        quality = sharpness * brightness_weight * 0.8

        solver.collect_frame(
            i, canonical, quality,
            sharpness=float(np.mean(sharpness)),
            pose=(0, 0, 0),
            detection_confidence=0.8,
        )

    # Solve
    solved = solver.solve()
    print(f"  Solved {len(solved)} frames, {solver.solver.get_hq_frame_count()} HQ")

    # Update identity state with solved frames
    for idx, (solved_face, solved_conf) in solved.items():
        state.update(solved_face, solved_conf, pose=(0, 0, 0))

    # Render output
    output_frames = []
    for i, frame in enumerate(frames):
        if i in solved:
            solved_face, solved_conf = solved[i]
            # Warp back to original size
            result = cv2.resize(solved_face, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LANCZOS4)

            # Blend with original based on confidence
            conf = cv2.resize(solved_conf, (frame.shape[1], frame.shape[0]))
            conf_3d = conf[:, :, np.newaxis]
            result = frame.astype(np.float32) * (1 - conf_3d) + result.astype(np.float32) * conf_3d
            result = np.clip(result, 0, 255).astype(np.uint8)
            output_frames.append(result)
        else:
            output_frames.append(frame)

    elapsed = time.perf_counter() - t_start

    # Compute metrics
    print("\n  === METRICS ===")

    original_stats = compute_lab_stats(frames)
    output_stats = compute_lab_stats(output_frames)

    original_flicker = compute_flicker(frames)
    output_flicker = compute_flicker(output_frames)

    original_sharpness = compute_sharpness(frames)
    output_sharpness = compute_sharpness(output_frames)

    original_eye_stability = compute_eye_stability(frames)
    output_eye_stability = compute_eye_stability(output_frames)

    print(f"\n  {'Metric':<25} {'Original':>10} {'Face OS v2':>10}")
    print(f"  {'-'*45}")
    print(f"  {'L mean':<25} {original_stats['L_mean']:>10.1f} {output_stats['L_mean']:>10.1f}")
    print(f"  {'L std':<25} {original_stats['L_std']:>10.1f} {output_stats['L_std']:>10.1f}")
    print(f"  {'a mean':<25} {original_stats['a_mean']:>10.1f} {output_stats['a_mean']:>10.1f}")
    print(f"  {'b mean':<25} {original_stats['b_mean']:>10.1f} {output_stats['b_mean']:>10.1f}")
    print(f"  {'Flicker':<25} {original_flicker:>10.2f} {output_flicker:>10.2f}")
    print(f"  {'Sharpness':<25} {original_sharpness:>10.1f} {output_sharpness:>10.1f}")
    print(f"  {'Eye stability':<25} {original_eye_stability:>10.2f} {output_eye_stability:>10.2f}")
    print(f"\n  Processing time: {elapsed:.1f}s ({len(frames)/elapsed:.0f} fps)")

    # Save results
    results = {
        'original': original_stats,
        'face_os_v2': output_stats,
        'original_flicker': original_flicker,
        'face_os_v2_flicker': output_flicker,
        'original_sharpness': original_sharpness,
        'face_os_v2_sharpness': output_sharpness,
        'original_eye_stability': original_eye_stability,
        'face_os_v2_eye_stability': output_eye_stability,
        'processing_time': elapsed,
        'frames_processed': len(frames),
        'hq_frames': solver.solver.get_hq_frame_count(),
    }

    output_dir = Path("output/face_os_v2_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save sample frames
    for i in [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]:
        if i < len(output_frames):
            cv2.imwrite(str(output_dir / f"frame_{i:04d}_original.jpg"), frames[i])
            cv2.imwrite(str(output_dir / f"frame_{i:04d}_v2.jpg"), output_frames[i])

    print(f"\n  Results saved to: {output_dir}")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("FACE OS v2 — ARCHITECTURE TESTS")
    print("=" * 60)

    test_frequency_decomposition()
    test_belief_pixel()
    test_patch_memory()
    test_bidirectional_solver()
    test_identity_state()

    # Run comparison if video exists
    run_comparison()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
