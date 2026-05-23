# Architecture

> Face OS status note: use `face_os/STATE.md` as the current source of truth.
> Runtime percentages and test counts below are historical snapshots unless
> repeated in `face_os/STATE.md`.

Two parallel systems co-exist in this codebase:

1. **Face OS** (primary) — Identity-reconstruction pipeline for portrait-mode studio video
2. **Legacy cricket pipeline** — 16:9 live stream → 9:16 shorts (Haar/YOLO + GFPGAN)

---

## Face OS Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  FACE OS — Identity Belief State Engine                             │
│  (pipeline.py, face_os/*)                                           │
│                                                                     │
│  Core equation:  OUTPUT = source * (1 - conf) + identity * conf    │
│  Frequency-aware: low-freq trust identity, high-freq trust source   │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 1: ENROLL                                                    │
│  expectation.png + photos/* → identity embeddings + canonical atlas │
│  MediaPipe FaceLandmarker (478-point mesh)                          │
│  PnP head pose from 6 key landmarks                                 │
│  Verification gate: embedding distance + face pixels + liveness     │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 2: PER-FRAME PROCESSING (forward path)                       │
│  ┌─ Detect & track ───────────────────────────────────────────────┐ │
│  │  MediaPipe FaceDetector + FaceLandmarker                        │ │
│  │  Identity matching (face_recognition embeddings)                │ │
│  │  Occupancy gate (face_area/bbox_area < 0.25 → reject)          │ │
│  │  No fallback to non-target tracks                               │ │
│  ├─ Geometry ──────────────────────────────────────────────────────┤ │
│  │  478-point landmarks + PnP head pose → SE(2)/SIM(2) transform  │ │
│  │  Canonical warp via LieGroup interpolation                      │ │
│  │  Geometry-based elliptical mask (brightness-invariant)          │ │
│  ├─ Identity ──────────────────────────────────────────────────────┤ │
│  │  Query identity belief state (frequency decomposition)          │ │
│  │  Query intrinsic (albedo/shading/specular) from IntrinsicDecomp │ │
│  │  Query patch memory (pose-conditioned retrieval)                │ │
│  ├─ Render ────────────────────────────────────────────────────────┤ │
│  │  _render_core() — SINGLE source of truth for ALL rendering      │ │
│  │    1. PhysicalRenderer (96%): albedo + shading + specular       │ │
│  │    2. Identity composite fallback: warp anchor face + blend     │ │
│  │    3. Enhancement last resort: sharpen + denoise                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 3: BIDIRECTIONAL SOLVE (offline, optional)                   │
│  Forward pass: collect all frames + quality metrics                 │
│  Temporal solve: future frames repair past frames                   │
│  Render pass: query solved identity for each frame                  │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 4: EXPORT + QC                                               │
│  VideoExporter (1080x1920, H.264, audio muxing)                     │
│  Fade in/out transitions (configurable duration)                    │
│  QC checks: identity drift, sharpness, flicker, face detection rate │
└─────────────────────────────────────────────────────────────────────┘
```

### V3 Module Integration Status

```
Module                Integrated   Active    Validated    Default
──────────────────────────────────────────────────────────────────
IntrinsicDecomposer   ✅ Yes       ✅ 100%   ❌ No        ✅ Yes
PhysicalRenderer      ✅ Yes       ✅ 96%    ❌ No        ✅ Yes
LieGroup SIM(2)       ✅ Yes       ✅ Yes    ⚠️ Partial   ✅ Yes
RendererMode          ✅ Yes       ✅ Yes    ❌ No        ✅ Yes
StateEvolution        ✅ Yes       ✅ Yes    ❌ No        ✅ Yes
EnergyScaler          ✅ Yes       ⚠️ Opt-in ❌ No        ❌ No
OptimizationEngine    ❌ No        ❌ No     ❌ No        ❌ No
DenseGeometry         ❌ No        ❌ No     ❌ No        ❌ No
IdentityManifold      ❌ No        ❌ No     ❌ No        ❌ No
VisibilityCalibration ❌ No        ❌ No     ❌ No        ❌ No
```

**Key:** ACTIVE ≠ VALIDATED. PhysicalRenderer runs 96% of frames but no proof yet that output quality improved over alpha compositing. See `AGAINST.md`.

### V2 Subsystem Architecture

Face OS decomposes into 4 isolated subsystems (face_os/subsystems/):

1. **Geometry Estimator** — all spatial structure, no identity/lighting logic
2. **Identity Estimator** — stable identity, no RGB blending
3. **Temporal Estimator** — temporal consistency, no texture injection
4. **Renderer** — physically consistent output, no heuristic compositing

---

## Legacy Cricket Pipeline

```text
URL → Download (yt-dlp + aria2c)
    → Transcribe (faster-whisper, Hindi/English)
    → Video Analysis (face/lighting map)
    → Highlight Detection (audio RMS + transcript scoring + Gemini AI)
    → Frame Analysis (cheap=Haar / premium=YOLO+ByteTrack)
    → Export (crop + enhance + interpolate + encode)
    → Selective Enhancement (3-pass: state→enhance→temporal)
    → SEO + Thumbnails
    → Upload to YouTube
```

Two analysis paths:
- **Cheap** (`frame_analyzer.py`): Haar Cascade + heuristics, no GPU
- **Premium** (`premium_analyzer.py` + `premium_render.py`): YOLOv8-face + ByteTrack + Kalman + RIFE + GFPGAN, GPU required

---

## Key Design Decisions (Face OS)

### Why Geometry-Based Mask (Not Intensity Threshold)
Old: `mask[gray < 5] = 0.0` → beard, shadows, dark skin erased → flicker
New: fixed elliptical geometry mask → brightness-invariant, deterministic

### Why Direct Blend
Both frames use `src * (1-mask) + identity * mask` (not compositor.composite()).
Identity face is already anchor-corrected in canonical space and warped to crop space.
Re-introducing compositor would de-correct the anchor.

### Why EMA at 0.4/0.6
Old 0.7/0.3 caused 10-frame lag (~300ms at 30fps, visible ghosting).
New converges in 5 frames (~150ms), smooths jitter without visible lag.

### Why Last Good Crop Plan
When face is lost mid-clip, `_last_good_crop_plan` preserves the last valid crop position.
Prevents jarring 16:9 full-frame output when face temporarily disappears.

### Why _render_core()
Both `_process_frame_v2()` (forward) and `_render_frame_v2()` (bidirectional) had duplicated rendering logic. This caused the V3 modules to be bypassed in the forward path. `_render_core()` is now the single source of truth for all rendering: PhysicalRenderer → identity composite → enhancement fallback.

---

## Test Suite (773 Face OS tests)

```
tests/face_os/
├── test_strict_regression.py       # 31 — Frame contract, mask stability, render core
├── test_math_hardening.py          # 37 — Invariant classes
├── test_v2_subsystems.py           # 20 — Subsystem isolation
├── test_phase1_hardening.py        # 37 — Long-horizon drift, system identifiability
├── test_detection.py               # 14 — MediaPipe detection
├── test_identity_state.py          # 17 — Frequency decomposition
├── test_identity_state_fixes.py    #  5 — LastUpdateFrame
├── test_patch_memory.py            # 18 — Region patches
├── test_temporal_solve.py          # 10 — Bidirectional solver
├── test_face_enhance.py            # 18 — Blink detection, eye freeze
├── test_quality_gates.py           # 13 — Procrustes, jitter, occupancy
├── test_appearance_field.py        # 14 — Appearance field
├── test_neural_codec.py            # 12 — Neural codec
├── test_hypothesis_matching.py     #  4 — Hypothesis space
├── test_region_confidence.py       #  4 — Region confidence
├── test_renderer_mode.py           # 21 — RendererMode state machine
├── test_adversarial.py             # 31 — Pathological inputs
├── test_visibility_calibration.py  # 16 — VisibilityCalibrator
├── test_identity_manifold.py       # 26 — Riemannian manifold
├── test_mathematical_foundation.py # 25 — StateEvolution, EnergyScaler
├── test_long_horizon.py            #  9 — 1000-frame drift
├── test_architectural_completeness.py # 10 — Completeness levels
├── test_phase0_contract.py         # 28 — FrameContract, EnergyReport
├── test_intrinsic_decomposition.py # 26 — IntrinsicDecomposer
├── test_physical_renderer.py       # 26 — PhysicalRenderer
├── test_dense_geometry.py          # 23 — DenseGeometry (de-scoped)
├── test_lie_group.py               # 23 — SE2/SIM2 transforms
├── test_state_space.py             # 39 — LatentState
├── test_optimizer_architecture.py  # 32 — GaussNewton, LM
├── test_observability.py           # 28 — ObservabilityAnalyzer
├── test_state_separation.py        # 34 — PhysicalState, BeliefState
├── test_map_estimation.py          # 19 — MAPOptimizer
├── test_energy_normalization.py    #  6 — Normalize energy
├── test_recovery_dynamics.py       # 38 — RecoveryTransitionMatrix
└── conftest.py
```

---

## Runtime Validation

Run with `.venv/bin/python validate_metrics.py`

### Latest Dashboard (100 frames, test_clip.mp4)

| Claim | Value | Status |
|---|---|---|
| PhysicalRenderer active | 96.0% | ✅ |
| IntrinsicDecomposer active | 100.0% | ✅ |
| Frame contract (1920x1080x3, uint8) | 50/50 frames | ✅ |
| RendererMode stable | 1 transition | ✅ |
| Avg intrinsic confidence | 0.758 | ✅ |
| Avg decomposition error | 0.053 | ✅ |
| Fallback reason telemetry | renderer_mode_alpha=4 | ✅ |
| No NaN/Inf in output | 50/50 clean | ✅ |
| Telemetry key coverage | 14/14 keys | ✅ |
| PhysicalRenderer dominant | 4% alpha fallback | ✅ |

---

## Configuration

Two config files:

| File | Purpose |
|---|---|
| `face_os_config.yaml` | Face OS tuning (identity, renderer, crop, export, enhancement) |
| `config.yaml` | Legacy pipeline config (download, transcription, premium toggle) |

---

## Stale/Unresolved (Face OS)

| Issue | Status |
|---|---|
| I-01 Duplicate render paths | ✅ FIXED (_render_core()) |
| I-02 Benchmark suite | ❌ PENDING |
| I-03 Normals circular (shading→normals→shading) | ❌ PENDING |
| I-05 Identity anchor RGB-entangled | ❌ PENDING |
| I-07 SIM(2) benefit unmeasured | ❌ PENDING |
| I-09 State prediction (constant velocity) | ❌ PENDING |
| I-10 Stranded modules | ❌ PENDING |
| ARCHITECTURE.md stale | ✅ UPDATED |

See `AGAINST.md` for full analysis.
