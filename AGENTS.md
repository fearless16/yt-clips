# AGENTS.md — Source of Truth

Last updated: 2026-05-21 (Face OS V3.1 — Architectural Consolidation + Benchmark Validation)

---

## LOCKED ARCHITECTURE — READ ONLY

**`LOCKED_ARCHITECTURE.md` is the frozen architectural truth.**

- File permissions: `444` (read-only)
- `.gitattributes`: marked as binary/unmergeable
- Unlock: `./unlock_architecture.sh` (only after PROVEN changes)
- Re-lock: `./lock_architecture.sh`

**DO NOT modify LOCKED_ARCHITECTURE.md unless:**
1. The change is PROVEN with tests and runtime validation
2. The change addresses a specific drift item (D-01 through D-10)
3. The change has A/B evidence showing improvement

**This file defines the remaining work. Follow it.**

---

## Current State Summary

Three parallel systems in the codebase:

1. **Legacy pipeline** (download → transcribe → highlight → export → SEO → upload) — working
2. **Face OS V0.5 pipeline** (identity reconstruction via MediaPipe V4) — **220 tests passing, 0 failures**
3. **Face OS V3.1 pipeline** (subsystem-based + V3 modules + architectural consolidation) — **830 tests passing, 0 failures**

### Face OS Test Suite (830 tests)

| File | Tests | Status | Purpose |
|---|---|---|---|---|
| `test_strict_regression.py` | 31 | ✅ | Frame contract, mask stability, NaN/Inf, bidirectional frame size, EMA convergence, render core path coverage |
| `test_v31_consolidation.py` | 37 | ✅ | V3.1 rules: render core consolidation, normal circularity fix, Lie algebra prediction, energy normalization, telemetry, benchmark/AB modules |
| `test_math_hardening.py` | 37 | ✅ | 10 invariant classes: UV roundtrip, transform det, temporal drift, flow shimmer, reprojection, lighting/pose invariance, mask topology, subpixel drift |
| `test_v2_subsystems.py` | 20 | ✅ | V2 subsystem isolation, coordinate systems, mathematical invariants |
| `test_phase1_hardening.py` | 37 | ✅ | Long-horizon drift (500 frames), system identifiability, renderer equation, VerificationGate, BeliefPixel properties |
| `test_detection.py` | 14 | ✅ | MediaPipe detection, poster rejection, identity matching, no-fallback |
| `test_identity_state.py` | 17 | ✅ | Identity state, frequency decomposition, anchor correction |
| `test_identity_state_fixes.py` | 5 | ✅ | LastUpdateFrame, region confidence |
| `test_patch_memory.py` | 18 | ✅ | Region patches, pose-conditioned, freeze-on-blink |
| `test_temporal_solve.py` | 10 | ✅ | Bidirectional solver, HQ frame identification |
| `test_face_enhance.py` | 18 | ✅ | Blink detection, eye freeze, cinematic noise, temporal noise field |
| `test_quality_gates.py` | 13 | ✅ | Procrustes, jitter, occupancy, poster rejection |
| `test_appearance_field.py` | 14 | ✅ | Appearance field |
| `test_neural_codec.py` | 12 | ✅ | Neural codec, identity score |
| `test_hypothesis_matching.py` | 4 | ✅ | Hypothesis space |
| `test_region_confidence.py` | 4 | ✅ | Region confidence |
| `test_renderer_mode.py` | 21 | ✅ | RendererMode state machine, hysteresis, transitions |
| `test_adversarial.py` | 31 | ✅ | Pathological lighting, landmark corruption, transform singularities |
| `test_visibility_calibration.py` | 16 | ✅ | VisibilityCalibrator, metric-truth correlation, drift detection |
| `test_identity_manifold.py` | 26 | ✅ | IdentityManifold (Riemannian, d=16), exp/log maps, geodesic |
| `test_mathematical_foundation.py` | 25 | ✅ | StateEvolution, EnergyScaler, OptimizationEngine |
| `test_long_horizon.py` | 9 | ✅ | 1000-frame identity drift, transform stability, covariance bounded |
| `test_architectural_completeness.py` | 10 | ✅ | Completeness levels, critical gaps identification |
| `test_phase0_contract.py` | 28 | ✅ | FrameContract, EnergyReport, RendererReport, PhaseState |
| `test_intrinsic_decomposition.py` | 26 | ✅ | IntrinsicDecomposer, Retinex decomposition, uncertainty |
| `test_physical_renderer.py` | 26 | ✅ | PhysicallyInspiredRenderer, Lambertian + Blinn-Phong |
| `test_dense_geometry.py` | 23 | ✅ | DenseGeometryEstimator (icosphere + RBF, de-scoped for V3) |
| `test_lie_group.py` | 23 | ✅ | SE2Transform, SIM2Transform, geodesic interpolation |
| `test_state_space.py` | 39 | ✅ | LatentState (11D), StateTransitionModel, StateSpaceEstimator |
| `test_optimizer_architecture.py` | 32 | ✅ | GaussNewtonOptimizer, LevenbergMarquardtOptimizer |
| `test_observability.py` | 28 | ✅ | ObservabilityAnalyzer, DegeneracyReport |
| `test_state_separation.py` | 34 | ✅ | PhysicalState, BeliefState, MetaState, StateSeparator |
| `test_map_estimation.py` | 19 | ✅ | MAPOptimizer, LocalMAPApproximation, MAPReport |
| `test_energy_normalization.py` | 6 | ✅ | normalize_energy flag, z-score normalization |
| `test_recovery_dynamics.py` | 38 | ✅ | RecoveryTransitionMatrix (5x5), Bayesian update |
| **Total** | **773** | **0 failures** | **All green** |

### Phase 1 Hardening Tests

| Test Class | Tests | What They Verify |
|---|---|---|
| TestLongHorizonIdentityDrift | 5 | Identity stays within 10 LAB of anchor over 500 frames, resists slow brightness/color drift |
| TestSystemIdentifiability | 4 | Two different faces produce distinguishable identity states (>20 LAB apart), same face converges (<5 LAB) |
| TestRendererBlendingEquation | 5 | `Y = M * Y_face + (1-M) * Y_bg` verified with known inputs, output contract preserved |
| TestVerificationGate | 10 | All 3 gate checks tested: face pixels, embedding distance, liveness jitter |
| TestRendererWithIdentity | 4 | Identity path exercised with actual identity data, low confidence handled |
| TestTemporalStateProperties | 3 | Confidence/drift/continuity bounds verified |
| TestFrequencyDecompositionProperties | 3 | Lossless reconstruction, low-freq smoother, high-freq mean near zero |
| TestBeliefPixelProperties | 3 | Observation count grows, variance decreases, confidence bounded |

### System Identifiability Analysis (V2.1.0)

Key findings from architecture review:

| Issue | Current State | Correct State |
|---|---|---|
| Identity representation | `appearance_latent` = RGB image (256x256x3) | Intrinsic albedo + geometric micro-detail |
| Temporal state | Scalar confidence (float) | Bayesian belief (mean + covariance) |
| Rendering | Alpha-blend compositing | Physical rendering `Y = R(G, A, L, V)` |
| Transforms | Linear EMA (`0.4*last + 0.6*new`) | Lie-group geodesic (SE(2)/SIM(2)) |
| Identity anchors | Single discrete anchor | Continuous latent manifold |
| Geometry | 478 sparse landmarks | Dense mesh / neural implicit |

See `face_os/FULL_REFERENCE.md` Sections 12-13 for full analysis and Phase 1 roadmap.

### V2 Architecture (NEW)

Face OS V2 decomposes the pipeline into 4 isolated subsystems:

1. **Geometry Estimator** (`subsystems/geometry_estimator.py`)
   - Estimates all spatial structure
   - Outputs: `GeometryState` (landmarks, pose, transforms, masks, confidence)
   - Forbidden: identity logic, lighting logic, RGB blending

2. **Identity Estimator** (`subsystems/identity_estimator.py`)
   - Estimates stable identity representation
   - Outputs: `IdentityState` (anchor basis, appearance latent, region confidence)
   - Forbidden: RGB EMA blending, raw frame accumulation

3. **Temporal Estimator** (`subsystems/temporal_estimator.py`)
   - Maintains temporal consistency
   - Outputs: `TemporalState` (motion field, confidence, drift score)
   - Forbidden: backward texture injection, frame averaging

4. **Renderer** (`subsystems/renderer.py`)
   - Generates physically consistent output
   - Equation: `Y = M ⊙ Y_face + (1 - M) ⊙ Y_bg`
   - Forbidden: RGB-space rescue compositing, heuristic blending

### V4 Migration (Complete)
- Config: `model: mediapipe_478`, no dlib references
- `types.py`: `FaceTrack.mesh_478`, landmarks `Landmarks.points` (478, 2)
- `detect_track.py`: MediaPipe FaceDetector + FaceLandmarker
- `landmarks.py`: 100% MediaPipe 478-point, PnP from 6 key points
- `face_enhance.py`: Eye indices `[33,159,158,133,153,145]`
- `canonical_map.py`: Handles 478-point + 68-point dynamically

### What Works (Face OS)
- MediaPipe Face Detection + FaceLandmarker (tasks API, 478-point mesh)
- Face tracking with identity matching (face_recognition embeddings)
- Occupancy gate (rejects face_area/bbox_area < 0.25)
- No fallback to non-target tracks in `_get_target_track()`
- Identity state with frequency decomposition, anchor correction, hypothesis space
- Patch memory with pose-conditioned retrieval
- Bidirectional temporal solver
- Geometry-based canonical face mask (brightness-invariant, deterministic)
- Frame contract validation helper
- Frame size invariance across ALL pipeline paths
- V2 subsystem isolation with explicit state types
- **V3.1: _render_core consolidation** — single rendering path, no duplicate V3 module updates
- **V3.1: Normal circularity broken** — face-prior normals replace shading gradient fallback
- **V3.1: Lie algebra velocity prediction** — SIM(2) constant-velocity extrapolation
- **V3.1: Energy normalization** — EnergyScaler wired into pipeline, z-score normalization
- **V3.1: Identity lighting decoupling** — white balance + exposure normalization in query path
- **V3.1: Comprehensive telemetry** — timing, fallback reasons, energy stats
- **V3.1: Benchmark suite** — easy/medium/hard/adversarial clip categories
- **V3.1: A/B validation** — photometric, geometric, perceptual metrics

---

## Known Issues & Fixed Bugs

### ✅ FIXED — Frame Size Invariance (Bug Class B)

**Root Cause:** `pipeline.py:_process_bidirectional()` render pass assigned `cropped = source_frame` when `frame_idx not in frame_data`. The original 16:9 frame was written directly, breaking the 9:16 output contract. Only the face-found path applied `crop_planner.apply_crop()`.

**Fix:**
- Added `self._last_good_crop_plan` to persist the last valid crop plan across pipeline passes
- Bidirectional pass 3 now always applies `crop_planner.apply_crop()` using last known plan
- `_reset_state()` clears the saved crop plan

**Tests:**
- `test_bidirectional_path_frame_size.py`: asserts fallback and face-locked paths produce identical dimensions
- `test_apply_crop_*`: verifies center, last_known, face_locked, and degenerate paths all match contract
- `test_repeated_crop_planner_calls_same_output_size`
- `validate_frame_contract()` — centralised helper, tested via `test_compositor_*`

### ✅ FIXED — Mask Stability / Intensity Threshold (Bug Class A)

**Root Cause:** Both `_process_frame_v2` and `_render_frame_v2` used `gray_canon < 5` as an intensity threshold to define the canonical face mask. Any pixel darker than gray=5 was erased from the mask. This caused:
- Beards, eyebrows, eye sockets, shadows, and dark skin to be cut out
- Mask area to shrink drastically on darker frames
- Identity blend weight to collapse to near-zero on dark frames
- Frame-to-frame flicker as lighting changed which pixels fell below the threshold

**Fix:**
- Replaced `np.ones(...); gray < 5 = 0.0` with `_make_canonical_geometry_mask()` — a fixed elliptical mask based on canonical face geometry
- The geometry mask is **brightness-invariant**: identical on every frame regardless of lighting
- Centered on canonical atlas, semi-axes 45% × 50%, feathered with 11x11 GaussianBlur
- Typically covers ~60% of canonical area

**Tests:**
- `test_canonical_geometry_mask_has_minimum_coverage` (> 30%)
- `test_canonical_geometry_mask_has_maximum_coverage` (< 90%)
- `test_canonical_geometry_mask_brightness_invariant` (deterministic across repeated calls)
- `test_canonical_geometry_mask_has_smooth_edges` (transition zone > 5%)

### ✅ FIXED — M_inv EMA Too Aggressive

**Root Cause:** `M_inv = 0.7 * self._last_M_inv + 0.3 * M_inv` — the EMA required ~10 frames to reach 97% of the target transform, causing visible lag/ghosting when the face moved.

**Fix:** Changed to `0.4 * self._last_M_inv + 0.6 * M_inv` — convergences to 95% within 5 frames.

**Tests:**
- `test_M_inv_ema_not_too_aggressive` (n < 15 frames to 95% at alpha=0.3)
- `test_alpha_can_be_increased` (alpha >= 0.5 converges in < 7 frames)

### ✅ FIXED — Mask Values Outside [0, 1]

**Root Cause:** `GaussianBlur` in `create_region_masks()` and `_elliptical_mask()` could produce values > 1.0 (floating point overshoot at blur edges).

**Fix:** Added `np.clip(mask, 0, 1)` after every GaussianBlur in `landmarks.py`.

### ⚠️ Face Flicker (Expected — NOT a bug)
- User has a side screen that plays videos; coloured light reflects onto face
- This is not pipeline instability; it's real-world lighting variation
- Tests have variance tolerance for this

### ⚠️ Black Fade In/Out — FIXED
- Export.py has fade support (`config.yaml: fade_in=0.5s, fade_out=0.5s`)

### ⚠️ Headroom Cropping — FIXED
- `frame_analyzer._apply_top_padding()` positions face at ~30% from top

### ✅ FIXED — V3 Modules Not Active in Forward-Only Path (Bug Class C)

**Root Cause:** V3 module integration (IntrinsicDecomposer, PhysicalRenderer, RendererMode, StateEvolution) and their telemetry were ONLY implemented in `_render_frame_v2()`, which is only called during bidirectional mode's render pass. The forward-only path (`_process_frame_v2()`) handled rendering inline and completely bypassed all V3 modules.

**Impact:** Despite being implemented, tested (768 tests), and "integrated", V3 modules had ZERO runtime activation in the primary processing path.

**Fix:**
- Added V3 intrinsic query (`identity_state.query_intrinsic()`) after identity query in `_process_frame_v2()`
- Added RendererMode state machine update based on intrinsic availability
- Added StateEvolution predict step each frame
- Added PhysicalRenderer path (tried before legacy alpha compositing)
- Added V3 telemetry tracking to `_process_frame_v2()`
- Added `total_frames` telemetry to forward-only processing loop
- Reset telemetry in `_reset_state()` for fresh per-clip stats
- Fixed cv2.warpAffine collapsing (H, W, 1) shading to (H, W) by restoring channel dim

**Runtime Validation Results (100 frames):**

| Metric | Value | Status |
|---|---|---|
| IntrinsicDecomposer success rate | 100% | ✅ |
| PhysicalRenderer activation rate | 96% | ✅ |
| RendererMode: physical | 96% | ✅ |
| RendererMode: hybrid | 0% | — |
| RendererMode: alpha | 4% | — |
| Avg intrinsic confidence | 0.758 | ✅ |
| Avg decomposition error | 0.053 | ✅ |
| RendererMode transitions | 1 | ✅ (stable) |

**Tests:** All 773 tests pass (0 failures, 0 regressions)

**IMPORTANT:** Any new V3 module integration must be added to BOTH `_process_frame_v2()` (forward-only path) and `_render_frame_v2()` (bidirectional render pass).

### ℹ️ Identity Face Not Used — ALREADY FIXED (prior session)
- `_process_frame_v2` now does direct blend: `cropped * (1-mask) + identity_in_crop * mask`
- `_render_frame_v2` does the same: `cropped * (1-conf_3d) + solved_in_crop * conf_3d`
- AGENTS.md from prior session documented an older version of the code

---

## Strict Regression Tests (test_strict_regression.py)

31 tests enforcing deterministic numeric assertions across 5 bug classes:

| Class | Tests | What They Guard |
|---|---|---|
| **Frame Contract** | 7 | Output shape must be (1920, 1080, 3) on every path, dtype uint8, no NaN/Inf |
| **Mask Stability** | 7 | Geometry mask coverage, brightness invariance, determinism, smooth edges, centroid drift < 2px, IoU > 0.9 |
| **Numeric Stability** | 6 | No NaN/Inf in compositor, all-black/white edge cases, frequency decomposition, identity query |
| **Bidirectional Size** | 2 | Fallback and face-locked paths produce same dimensions |
| **No-Identity Path** | 2 | `render_frame` preserves shape/dtype with and without masks |
| **Landmark Scaling** | 1 | `_adjust_landmarks_to_crop` coordinate contract |
| **EMA Convergence** | 2 | EMA alpha must converge in < 15 frames (now < 5) |
| **Render Core** | 5 | Both runtime paths use `_render_core()`, no inline rendering outside `_render_core()`, telemetry tracked in core |

Run with: `.venv/bin/python -m pytest tests/face_os/test_strict_regression.py -v`

---

## Architecture Decisions

### Why Geometry-Based Mask (Not Intensity Threshold)
- Old: `mask[gray < 5] = 0.0` → beard, shadows, dark skin get erased; mask area varies per frame
- New: `_make_canonical_geometry_mask()` → fixed elliptical mask, brightness-invariant, deterministic
- Result: identity blend weight is consistent regardless of per-frame lighting

### Why Direct Blend (Not Compositor.composite())
- `_process_frame_v2` and `_render_frame_v2` use direct `src * (1-mask) + identity * mask` instead of `self.compositor.composite()`. This is correct because the identity face is already anchor-corrected in canonical space and warped back to crop space. Re-introducing compositor blending would de-correct the anchor.

### Why EMA Smoothing at 0.4/0.6
- Old 0.7/0.3 caused 10-frame lag (~300ms at 30fps)
- New 0.4/0.6 converges in 5 frames (~150ms)
- Still smooths out detection jitter without visible ghosting

### Why Last Good Crop Plan
- When face is lost mid-clip, the pipeline must not switch to full-frame 16:9 output
- `_last_good_crop_plan` preserves the crop position and size from the last face-found frame
- Prevents jarring size/position jumps when face is temporarily lost

### V3.1 Architectural Consolidation

#### Why _render_core Consolidation (RULE 1)
- Old: `_process_frame_v2()` and `_render_frame_v2()` duplicated V3 module updates (intrinsic tracking, RendererMode, StateEvolution)
- This duplication caused the original V3 bypass bug where modules were never activated
- New: `_update_v3_modules()` is the single source of truth for all V3 module updates
- `_render_core()` is the single source of truth for all rendering logic
- No rendering logic may exist outside `_render_core()`

#### Why Face-Prior Normals (RULE 4)
- Old: shading gradient normals `[-dS/dx, -dS/dy, 1]` created circular dependency: shading → normals → shading
- New: face-prior ellipsoidal normals when mesh unavailable, mesh-derived normals when available
- Face-prior is deterministic and brightness-invariant
- Breaks the mathematical circularity

#### Why Lie Algebra Velocity Prediction (RULE 6)
- Old: StateEvolution only predicted via diagonal damping `x = A * x` (effectively a no-op)
- New: `predict_with_velocity()` implements `T_hat(t+1) = T(t) * exp(v_t)` where `v_t = log(T_t) - log(T_t-1)`
- Enables constant-velocity extrapolation on SIM(2) for occlusion recovery
- Full predict-update cycle via `predict_update_full()`

#### Why Energy Normalization (RULE 7)
- Old: EnergyScaler existed but was not wired into pipeline
- New: `_compute_energy_terms()` computes and normalizes energy terms per frame
- Z-score normalization ensures unit variance for stable optimizer convergence

#### Why Identity Lighting Decoupling (RULE 5)
- Old: `appearance_latent` = RGB image, leaking lighting into identity
- New: `_normalize_white_balance()` and `_normalize_exposure()` in query path
- Gray-world white balance removes color cast from lighting
- Exposure normalization targets standard luminance (L=128)

#### Why Comprehensive Telemetry (RULE 8)
- Old: No timing data, no fallback reason tracking
- New: `render_time_sum_ms`, `render_time_count`, `fallback_reason_distribution`
- `get_telemetry_report()` includes `avg_render_time_ms` and `energy_scaler_stats`
- No hidden state, no silent fallback

---

## Project Structure (Face OS)

```
face_os/
├── pipeline.py              # Orchestrator V0.5 (forward/ bidirectional), contract validation
├── pipeline_v2.py           # Orchestrator V2 (subsystem-based architecture)
├── detect_track.py          # MediaPipe detection + tracking
├── identity_state.py        # Frequency decomposition, anchor correction, hypotheses
├── patch_memory.py          # Per-region memory, pose-conditioned retrieval
├── temporal_solve.py        # Bidirectional temporal solver
├── face_enhance.py          # Structure-preserving rendering, blink detection
├── crop_planner.py          # Reference-based crop planning
├── compositor.py            # Confidence-weighted compositing
├── canonical_map.py         # Canonical UV alignment
├── landmarks.py             # 478-point MediaPipe landmarks + PnP head pose
├── appearance_field.py      # AppearanceField + DynamicAppearanceField
├── neural_codec.py          # PersonalizedSpace + NeuralCodec
├── types.py                 # Core data structures (includes GeometryState, IdentityState, TemporalState)
├── config.py                # YAML config loader
├── face_detector.tflite     # MediaPipe face detection model
├── face_os_config.yaml      # All tuning parameters
├── benchmark_suite.py       # V3.1: Clip categories + per-clip metrics
├── ab_validation.py         # V3.1: A/B comparison (photometric, geometric, perceptual)
└── subsystems/              # V2 Architecture
    ├── __init__.py
    ├── geometry_estimator.py    # Subsystem A — spatial structure estimation
    ├── identity_estimator.py    # Subsystem B — stable identity representation
    ├── temporal_estimator.py    # Subsystem C — temporal consistency
    └── renderer.py              # Subsystem D — physically consistent rendering

tests/face_os/
├── test_strict_regression.py  # 26 tests — frame contract, mask stability, NaN/Inf
├── test_v2_subsystems.py      # 20 tests — V2 subsystem isolation, invariants
├── test_detection.py          # 14 tests
├── test_identity_state.py     # 17 tests
├── test_identity_state_fixes.py
├── test_patch_memory.py       # 18 tests
├── test_temporal_solve.py     # 10 tests
├── test_face_enhance.py       # 18 tests
├── test_quality_gates.py      # 13 tests
├── test_appearance_field.py   # 14 tests
├── test_neural_codec.py       # 12 tests
├── test_hypothesis_matching.py
├── test_region_confidence.py
└── conftest.py
```

---

## Next Steps (Priority Order — from AGAINST.md)

### P0 — Build Benchmark Suite (AGAINST.md I-02)
- Categorised clips: easy / medium / hard / adversarial
- Per-clip: physical_render_rate, drift, flicker, fallback, geometric consistency
- A/B test: PhysicalRenderer output vs alpha compositing (I-03)

### P1 — Geometry Normals (AGAINST.md I-04)
- Break circularity: landmarks → geometry normals → renderer
- Currently: shading → normals → shading (circular)

### P1 — Identity Anchor Decoupling (AGAINST.md I-05)
- Split anchor: albedo + appearance + white-balance normalize
- Currently RGB-entangled — lighting leaks into identity

### P1 — Geometric Consistency Metric (AGAINST.md I-07)
- SIM(2) vs linear EMA A/B on high-rotation clips
- Mesh distortion, determinant stability, landmark coherence

### P2 — State Prediction (AGAINST.md I-09)
- Constant velocity model on SIM(2): T_hat(t+1) = T(t) * exp(v_t)
- Needed for occlusion recovery, missed detections

### P3 — Stranded Modules (AGAINST.md I-10)
- IdentityManifold, VisibilityCalibration, OptimizationEngine, DenseGeometry
- Each: integrate, schedule, isolate, or delete

### Short-term
- Anchor correction verification — output LAB ~108, std < 1.5
- Face map comparison — output L within 5 of reference
- Prototype lasso cut — MediaPipe Selfie Segmentation
- Multi-anchor system — 7+ (frontal, smile, yaw left/right, etc.)
- Per-face exposure normalisation — L=16→155 swings

---

## How to Run Tests

```bash
# Full Face OS test suite (773 tests)
.venv/bin/python -m pytest tests/face_os/ -v

# Strict regression tests only (31 tests)
.venv/bin/python -m pytest tests/face_os/test_strict_regression.py -v

# V2 subsystem tests only (20 tests)
.venv/bin/python -m pytest tests/face_os/test_v2_subsystems.py -v

# Single file
.venv/bin/python -m pytest tests/face_os/test_patch_memory.py -v

# Real-video metrics validation (10 claims pass/fail)
.venv/bin/python validate_metrics.py
```

## API — Key Validation Entry Points

```python
# Frame contract — every output frame must pass this
from face_os.pipeline import FaceOSPipeline
assert FaceOSPipeline.validate_frame_contract(frame, expected_h=1920, expected_w=1080)

# Geometry-based canonical mask (brightness-invariant)
mask = FaceOSPipeline._make_canonical_geometry_mask((256, 256))
# Shape: (256, 256), dtype: float32, range: [0, 1], identical across frames
```

## User Context

- **Content**: Portrait-mode studio videos
- **Reference**: `expectation.png` — enhanced portrait in studio
- **Side screen**: User has a side screen; coloured light reflects onto face (expected flicker)
- **Background**: Never changes — good candidate for lasso cut approach
- **Logo**: Preserved on left side
- **Fade**: First/last frame black with smooth transition (configured in export.py)
- **Test video**: `clips_test/test_clip.mp4` (640x360, 30fps, 15s, 450 frames)
