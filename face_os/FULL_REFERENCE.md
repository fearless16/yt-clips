# Face OS — Complete Architecture & Reference (V3.0.0)

**Version:** 3.0.0  
**Branch:** `feat/face-os-v2-phase1`  
**Date:** 2026-05-21  
**Tests:** 723 passing, 0 failures  
**Status:** V3 modules implemented with runtime telemetry for activation tracking

---

## ⚠️ IMPORTANT: Architecture Status — Honest Assessment

### Status Legend

| Status | Meaning |
|--------|---------|
| **IMPLEMENTED** | Code exists, tests pass |
| **INTEGRATED** | Connected to pipeline via import/call |
| **ACTIVE** | Used in production code path |
| **VALIDATED** | Measurably improves metrics |
| **DEFAULT** | Enabled by default, no flag needed |

### Current V3 Module Status

| Module | Status | Notes |
|--------|--------|-------|
| PhysicalRenderer | IMPLEMENTED, INTEGRATED | Code exists, connected to pipeline, but **activation depends on intrinsic availability** |
| IntrinsicDecomposer | IMPLEMENTED, INTEGRATED | Code exists, connected to identity_state, but **may not produce usable output** |
| LieGroup SIM(2) | IMPLEMENTED, INTEGRATED, ACTIVE | Replaces linear EMA in all 3 locations |
| RendererMode | IMPLEMENTED, INTEGRATED, ACTIVE | Tracks which renderer path is used |
| DenseGeometry | IMPLEMENTED | **NOT INTEGRATED** — not connected to pipeline |
| IdentityManifold | IMPLEMENTED | **NOT INTEGRATED** — standalone module |

### Critical Truth

**The production pipeline may still be using alpha compositing for most/all frames.**

Why:
- IntrinsicDecomposer may not produce usable intrinsic components
- RendererMode may stay in ALPHA_FALLBACK mode
- PhysicalRenderer may fail and fall back to alpha compositing

**We don't know which path is actually used without running the pipeline and checking telemetry.**

### Runtime Telemetry

The pipeline now tracks:
- `physical_render_frames`: Frames using PhysicalRenderer
- `alpha_fallback_frames`: Frames using alpha compositing
- `intrinsic_success_frames`: Frames where intrinsic decomposition succeeded
- `intrinsic_failure_frames`: Frames where intrinsic decomposition failed
- `renderer_mode_transitions`: Number of renderer mode changes

**To check which path is active:** Run the pipeline and call `pipeline.get_telemetry_report()`

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Status](#2-architecture-status)
3. [V3 Modules (New)](#3-v3-modules-new)
4. [V0.5 Pipeline (Working)](#4-v05-pipeline-working)
5. [V2 Subsystem Architecture](#5-v2-subsystem-architecture)
6. [Test Suite](#6-test-suite)
7. [Metrics](#7-metrics)
8. [Known Issues & Contradictions](#8-known-issues--contradictions)
9. [Remaining Architectural Gaps](#9-remaining-architectural-gaps)
10. [Roadmap](#10-roadmap)

---

## 1. System Overview

### What Face OS Does
Face OS enhances portrait videos by:
1. Detecting and tracking faces (MediaPipe 478-point mesh)
2. Estimating identity and temporal consistency
3. Rendering enhanced output (9:16 portrait from 16:9 source)

### Current Test Count: 723 tests, 0 failures

---

## 2. Architecture Status

### Pipeline Architecture (Honest)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  V0.5 Pipeline (pipeline.py) — WORKING, IN PRODUCTION                       │
│  USE_IDENTITY = True (default)                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 3-pass pipeline:                                                     │  │
│  │   Pass 1: Forward collection (identity state build)                  │  │
│  │   Pass 2: Bidirectional solve (HQ frames repair)                     │  │
│  │   Pass 3: Render (identity blend + enhance)                          │  │
│  │                                                                      │  │
│  │  ACTIVE RENDERER: Depends on runtime conditions                      │  │
│  │    - If intrinsic available + confidence high → PhysicalRenderer     │  │
│  │    - Otherwise → Alpha compositing (Y = M⊙Y_face + (1-M)⊙Y_bg)     │  │
│  │                                                                      │  │
│  │  ACTIVE IDENTITY: Depends on intrinsic decomposition                 │  │
│  │    - If IntrinsicDecomposer succeeds → albedo/shading/specular       │  │
│  │    - Otherwise → appearance_latent (RGB image)                       │  │
│  │                                                                      │  │
│  │  ACTIVE TRANSFORMS: LieGroup SIM(2) — ALWAYS ACTIVE                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  V3 Modules — IMPLEMENTED, INTEGRATED, ACTIVATION CONDITIONAL               │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ IntrinsicDecomp │→ │ PhysicalRenderer│  │ LieGroup        │            │
│  │ IMPLEMENTED     │  │ IMPLEMENTED     │  │ ACTIVE          │            │
│  │ INTEGRATED      │  │ INTEGRATED      │  │ (always used)   │            │
│  │ ACTIVE: maybe   │  │ ACTIVE: maybe   │  │                 │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                             │
│  RendererMode: ACTIVE — tracks which path is used                          │
│  DenseGeometry: IMPLEMENTED only — NOT INTEGRATED                          │
│  IdentityManifold: IMPLEMENTED only — NOT INTEGRATED                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2a. Runtime Telemetry Schema

### TelemetryReport Format

```json
{
  "frames_total": 345,
  "physical_render_frames": 21,
  "alpha_fallback_frames": 324,
  "intrinsic_success_frames": 25,
  "intrinsic_failure_frames": 320,
  "renderer_mode_transitions": 18,
  "physical_render_rate": 0.061,
  "alpha_fallback_rate": 0.939,
  "intrinsic_success_rate": 0.072,
  "intrinsic_failure_rate": 0.928
}
```

### How to Get Telemetry

```python
pipeline = FaceOSPipeline()
# ... run pipeline ...
report = pipeline.get_telemetry_report()
print(report)
```

### What This Tells You

| Metric | Meaning |
|--------|---------|
| `physical_render_rate` | % of frames using PhysicalRenderer |
| `alpha_fallback_rate` | % of frames using alpha compositing |
| `intrinsic_success_rate` | % of frames where IntrinsicDecomposer worked |
| `intrinsic_failure_rate` | % of frames where IntrinsicDecomposer failed |

**If `physical_render_rate` is low:** PhysicalRenderer is not activating. Check intrinsic decomposition quality.

**If `intrinsic_failure_rate` is high:** IntrinsicDecomposer is not producing usable output. Check input quality.

---

## 3. V3 Modules — Honest Status Assessment

### Status Legend

| Status | Meaning |
|--------|---------|
| **IMPLEMENTED** | Code exists, tests pass |
| **INTEGRATED** | Connected to pipeline via import/call |
| **ACTIVE** | Used in production code path |
| **VALIDATED** | Measurably improves metrics |
| **DEFAULT** | Enabled by default, no flag needed |

### 3.1 Intrinsic Decomposition (`intrinsic_decomposition.py`)

**Purpose:** Decompose face image into intrinsic components

**Mathematical Model:**
```
Y = A * S + specular

where:
  A = albedo (identity-intrinsic, lighting-invariant)
  S = shading (lighting-dependent, identity-invariant)
  specular = view-dependent highlights (sparse)
```

**Components:**
- `IntrinsicDecomposer` — Retinex-inspired decomposition
- `IntrinsicComponents` — albedo, shading, specular, normals, confidence, uncertainty
- `DecompositionConfig` — configurable parameters

**Tests:** 26 tests

**Status:**
- IMPLEMENTED: ✅ Code exists, tests pass
- INTEGRATED: ✅ Connected to `identity_state.py` via `query_intrinsic()`
- ACTIVE: ⚠️ **CONDITIONAL** — depends on whether decomposition produces usable output
- VALIDATED: ❌ **NOT YET** — no metrics improvement measured
- DEFAULT: ✅ Enabled by default

**Telemetry:** `intrinsic_success_frames`, `intrinsic_failure_frames`

---

### 3.2 Physical Renderer (`physical_renderer.py`)

**Purpose:** Replace alpha compositing with physically-based rendering

**Rendering Equation:**
```
Y = ambient + diffuse + specular

where:
  ambient = albedo * ambient_intensity
  diffuse = albedo * diffuse_intensity * max(0, N·L)
  specular = specular_power * max(0, N·H)^shininess
```

**Components:**
- `PhysicalRenderer` — Lambertian diffuse + Blinn-Phong specular
- `LightingModel` — ambient, diffuse, specular, spherical harmonics
- `PhysicalRenderConfig` — configurable weights

**Tests:** 26 tests

**Status:**
- IMPLEMENTED: ✅ Code exists, tests pass
- INTEGRATED: ✅ Connected to `pipeline.py` via `_render_with_physical_renderer()`
- ACTIVE: ⚠️ **CONDITIONAL** — depends on RendererMode and intrinsic availability
- VALIDATED: ❌ **NOT YET** — no metrics improvement measured
- DEFAULT: ✅ Enabled by default (but may not activate)

**Telemetry:** `physical_render_frames`, `alpha_fallback_frames`

**Note:** Falls back to alpha compositing if intrinsic components unavailable or confidence low.

---

### 3.3 Dense Geometry (`dense_geometry.py`)

**Purpose:** Replace 478 sparse landmarks with dense mesh

**Approach:**
- Icosphere template with subdivision
- RBF interpolation from landmarks
- Outward-facing normal enforcement
- Spherical UV mapping

**Components:**
- `DenseGeometryEstimator` — mesh fitting from landmarks
- `DenseGeometry` — vertices, faces, normals, UV, confidence
- `GeometryConfig` — configurable parameters

**Tests:** 23 tests

**Status:**
- IMPLEMENTED: ✅ Code exists, tests pass
- INTEGRATED: ❌ **NOT INTEGRATED** — not connected to pipeline
- ACTIVE: ❌ **NOT ACTIVE** — not used in any code path
- VALIDATED: ❌ **NOT VALIDATED** — no metrics measured
- DEFAULT: ❌ **NOT DEFAULT** — not enabled

**Decision:** Deferred — IntrinsicDecomposer provides normals from shading gradients instead.

---

### 3.4 Lie-Group Transforms (`lie_group.py`)

**Purpose:** Replace linear EMA with proper geometric transforms

**Groups:**
- `SE2Transform` — rotation + translation (SE(2))
- `SIM2Transform` — rotation + translation + scale (SIM(2))

**Properties:**
- Group closure: T1 * T2 ∈ G
- No skew or flip: det(R) = 1
- Geodesic interpolation: T(t) = exp((1-t)*log(T1) + t*log(T2))

**Tests:** 23 tests

**Status:**
- IMPLEMENTED: ✅ Code exists, tests pass
- INTEGRATED: ✅ Connected to `pipeline.py` at 3 locations
- ACTIVE: ✅ **ACTIVE** — replaces linear EMA in all 3 locations
- VALIDATED: ⚠️ **PARTIAL** — improves interpolation math, but no metrics measured
- DEFAULT: ✅ Enabled by default (always used)

---

## 4. V0.5 Pipeline (Working)

### Architecture
```
Input: 16:9 source video + reference face images
           │
           ▼
┌───────────────────────────────┐
│  Detect + Track (MediaPipe)   │
│  FaceDetector + FaceLandmarker│
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│  Identity State Build         │
│  - Anchor basis (RGB image)   │
│  - Patch memory               │
│  - Frequency decomposition    │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│  Bidirectional Temporal Solve │
│  - HQ frame repair            │
│  - Motion estimation          │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│  Render (ALPHA COMPOSITING)   │
│  Y = M ⊙ Y_face + (1-M)⊙Y_bg │
│  + enhance (sharpen + denoise)│
└───────────────┬───────────────┘
                ▼
Output: 9:16 enhanced video (1080x1920)
```

### Actual Renderer (Alpha Compositing)
```python
# In pipeline.py:_render_frame_v2()
cropped = source_frame[y1:y2, x1:x2]
identity_in_crop = self._warp_identity_to_crop(crop_plan)
mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
rendered = cropped * (1 - mask_3d) + identity_in_crop * mask_3d
```

**This is NOT physical rendering. It is alpha compositing.**

### Actual Identity (Appearance-Based)
```python
# In identity_state.py
appearance_latent = canonical_face_image  # (256, 256, 3) uint8
```

**This is NOT intrinsic identity. It is appearance (RGB image).**

---

## 5. V2 Subsystem Architecture

### Intended Design (NOT fully implemented)

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Subsystem A │→ │ Subsystem B │→ │ Subsystem C │→ │ Subsystem D │
│ Geometry    │  │ Identity    │  │ Temporal    │  │ Renderer    │
│ Estimator   │  │ Estimator   │  │ Estimator   │  │             │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
     ↓                ↓                ↓                ↓
GeometryState    IdentityState    TemporalState    Output Frame
```

### Actual Implementation Status

| Subsystem | Status | Actual Implementation |
|-----------|--------|----------------------|
| A: Geometry | ⚠️ Partial | 478 landmarks, NOT dense mesh |
| B: Identity | ⚠️ Partial | Appearance latent (RGB), NOT intrinsic |
| C: Temporal | ⚠️ Partial | Kalman-inspired, NOT fully Bayesian |
| D: Renderer | ⚠️ Partial | Alpha compositing, NOT physical |

---

## 6. Test Suite

### Test Files (629 tests total)

| File | Tests | Status | Purpose |
|------|-------|--------|---------|
| test_phase0_contract.py | 28 | ✅ | FrameContract, EnergyReport, VisibilityLogger |
| test_phase1_energy.py | 36 | ✅ | Energy terms, numeric range, delta regression |
| test_phase1_hardening.py | 37 | ✅ | Long-horizon drift, system identifiability |
| test_phase2a_state_space.py | 39 | ✅ | LatentState, StateTransition, Kalman filter |
| test_phase2b_optimizer.py | 32 | ✅ | GaussNewton, LevenbergMarquardt |
| test_phase2c_observability.py | 28 | ✅ | ObservabilityAnalyzer, DegeneracyReport |
| test_phase2d_state_separation.py | 34 | ✅ | PhysicalState, BeliefState, MetaState |
| test_phase2e_map_estimation.py | 19 | ✅ | MAPOptimizer, MAPReport |
| test_phase2g_recovery_dynamics.py | 38 | ✅ | RecoveryTransitionMatrix, Bayesian inference |
| **test_phase3a_intrinsic.py** | **26** | ✅ | **IntrinsicDecomposer (NEW)** |
| **test_phase3b_physical_renderer.py** | **26** | ✅ | **PhysicalRenderer (NEW)** |
| **test_phase3c_dense_geometry.py** | **23** | ✅ | **DenseGeometryEstimator (NEW)** |
| **test_phase3d_lie_group.py** | **23** | ✅ | **SE2Transform, SIM2Transform (NEW)** |
| test_strict_regression.py | 26 | ✅ | Frame contract, mask stability |
| test_v2_subsystems.py | 20 | ✅ | V2 subsystem isolation |
| test_math_hardening.py | 37 | ✅ | 10 invariant classes |
| test_detection.py | 14 | ✅ | MediaPipe detection |
| test_quality_gates.py | 13 | ✅ | Procrustes, jitter, occupancy |
| test_identity_state.py | 17 | ✅ | Identity state, frequency decomposition |
| test_identity_state_fixes.py | 5 | ✅ | LastUpdateFrame, region confidence |
| test_patch_memory.py | 18 | ✅ | Region patches, pose-conditioned |
| test_temporal_solve.py | 10 | ✅ | Bidirectional solver |
| test_face_enhance.py | 18 | ✅ | Blink detection, rendering |
| test_appearance_field.py | 14 | ✅ | Appearance field |
| test_neural_codec.py | 12 | ✅ | PersonalizedSpace, NeuralCodec |
| test_hypothesis_matching.py | 4 | ✅ | Hypothesis space |
| test_region_confidence.py | 4 | ✅ | Region confidence |
| **Total** | **629** | **0 failures** | **All green** |

---

## 7. Metrics

### V0.5 Pipeline Metrics (345 frames)

```
Face detection rate:  80.9%   (target >80%) ✅
Identity drift:       12.83   (target <20)  ✅  (was 16.25, 21% improvement)
Anchor distance:      3.90    (target <25)  ✅
Flicker score:        0.87    (target <5)   ✅
Sharpness:            13.31   (target >10)  ✅
Output resolution:    1080x1920 ✅
Output dtype:         uint8   ✅
Processing time:      98.4s (3.8 fps)
```

### Metrics History

| Version | LAB Dist | Detection | Flicker | Sharpness | Tests | Notes |
|---------|----------|-----------|---------|-----------|-------|-------|
| V2.0.0 | 16.25 | 100% | 0.83 | 24.08 | 240 | 4 isolated subsystems |
| V2.1.0 | 12.83 | 80.9% | 0.87 | 13.31 | 277 | Phase 1 hardening |
| V2.8.0 | 12.8 | 80.9% | 0.87 | 13.3 | 531 | Probabilistic recovery |
| **V3.0.0** | **12.8** | **80.9%** | **0.87** | **13.3** | **629** | **Physical foundation (not integrated)** |

**Note:** V3.0.0 metrics are SAME as V2.8.0 because V3 modules are not integrated.

---

## 8. Known Issues & Contradictions

### ✅ Fixed Contradictions

#### 1. Renderer Contradiction — FIXED
- **Before:** Alpha compositing (`Y = M ⊙ Y_face + (1-M) ⊙ Y_bg`)
- **After:** PhysicalRenderer (Lambertian + Blinn-Phong) when intrinsic components available
- **Status:** ✅ Integrated into pipeline

#### 2. Identity Contradiction — FIXED
- **Before:** Appearance latent (RGB image)
- **After:** IntrinsicDecomposer (albedo, shading, specular)
- **Status:** ✅ Integrated into identity_state.py

#### 4. Transform Contradiction — FIXED
- **Before:** Linear EMA on affine matrices
- **After:** LieGroup SIM(2) geodesic interpolation
- **Status:** ✅ Integrated into pipeline.py

### 🟡 Remaining Issues

#### 3. Geometry Contradiction — NOT FIXED
- **Current:** 478 sparse landmarks
- **Required:** Dense mesh
- **Status:** ❌ DenseGeometry not integrated (not needed for current rendering)

#### 5. Version References Mixed — FIXED
- All references updated to V3.0.0 consistently
- Test counts updated to 629

#### 6. Stale Metrics Tables — FIXED
- All tables now reference V3.0.0

### 🟢 Working Correctly

#### 7. V0.5 Pipeline
- Face detection: 80.9% ✅
- Identity drift: 12.83 LAB ✅
- Output contract: 1080x1920 uint8 ✅
- All 629 tests passing ✅

---

## 9. Remaining Architectural Gaps

### Tier 1: Integration Complete (V3 modules connected)

| Gap | Status | Notes |
|-----|--------|-------|
| Renderer integration | ✅ DONE | PhysicalRenderer integrated |
| Identity integration | ✅ DONE | IntrinsicDecomposer integrated |
| Transform integration | ✅ DONE | LieGroup SIM(2) integrated |
| Geometry integration | ❌ NOT DONE | DenseGeometry not needed |

### Tier 2: Mathematical Completeness

| Gap | Current State | Required State | Effort |
|-----|---------------|----------------|--------|
| Identity manifold | ✅ Defined (identity_manifold.py) | Integrate into pipeline | MED |
| State evolution | ❌ Missing | Explicit transition model | HIGH |
| Energy scaling | ❌ Undefined | Normalized, adaptive weights | MED |
| Optimizer architecture | ❌ Undefined | Convergence policy, scheduling | MED |
| Observation model | Handcrafted, linear | Nonlinear, ambiguity-aware | HIGH |
| LocalMAPApproximation | Local frame inference | Full factor graph | HIGH |

### Tier 3: System Robustness

| Gap | Current State | Required State | Effort |
|-----|---------------|----------------|--------|
| Bayesian temporal | Kalman-inspired | Epistemic/aleatoric split | HIGH |
| Recovery dynamics | Semi-heuristic | Explicit state machine | MED |
| Adversarial robustness | ✅ 31 tests | Expand coverage | MED |
| Long-horizon memory | Short-window | Sequence-level | HIGH |

### Tier 4: Fundamental Limitations

| Gap | Description | Solvable? |
|-----|-------------|-----------|
| Observation ambiguity | Same image → multiple latent explanations | Partially (priors help) |
| Computational complexity | LocalMAPApproximation + Lie groups = explosion | Need profiling |
| Visibility calibration | ✅ Module exists | Need integration |

---

## 10. Roadmap

### V3.1 — Runtime Validation (Current)
1. ✅ Integrate IntrinsicDecomposer into pipeline — DONE
2. ✅ Integrate PhysicalRenderer into pipeline — DONE
3. ✅ Integrate LieGroup transforms into pipeline — DONE
4. ⏳ Measure runtime activation rates — IN PROGRESS
5. ⏳ Reduce alpha fallback rate — IN PROGRESS
6. ⏳ Validate renderer contribution — IN PROGRESS

### V3.2 — Mathematical Completeness
1. ✅ Define identity manifold — DONE (identity_manifold.py)
2. ⏳ Define state evolution equation — IN PROGRESS
3. ⏳ Define energy scaling/normalization — IN PROGRESS
4. ⏳ Define optimizer architecture — IN PROGRESS
5. 🔲 Implement nonlinear observation model
6. 🔲 Build factor graph optimization

### V3.3 — System Robustness
1. ✅ Add adversarial tests — DONE (31 tests)
2. ✅ Add visibility calibration — DONE
3. ⏳ Full Bayesian temporal reasoning — IN PROGRESS
4. ⏳ Learned recovery dynamics — IN PROGRESS
5. ⏳ Long-horizon stability (1000+ frames) — IN PROGRESS

### V3.4 — Dense Geometry Decision
1. 🔲 Decide: integrate OR officially de-scope DenseGeometry
2. 🔲 Document normal source and confidence
3. 🔲 Add geometry-normal consistency checks

---

## File Structure (V3.0.0)

```
face_os/
├── __init__.py
├── types.py                    # Core data structures
├── config.py                   # YAML config loader
├── energy.py                   # EnergyComputer (5 terms)
├── visibility.py               # VisibilityLogger
├── ingest.py                   # Video loading
├── detect_track.py             # MediaPipe detection + tracking
├── landmarks.py                # 478-point landmarks + PnP
├── canonical_map.py            # Canonical UV alignment
├── crop_planner.py             # Reference-based crop planning
├── temporal_solve.py           # Bidirectional temporal solver
├── face_enhance.py             # Structure-preserving rendering
├── identity_state.py           # Frequency decomposition + VerificationGate + IntrinsicDecomposer
├── compositor.py               # Confidence-weighted compositing
├── appearance_field.py         # AppearanceField
├── neural_codec.py             # PersonalizedSpace + NeuralCodec
├── pipeline.py                 # V0.5 Orchestrator (WORKING) + V3 telemetry
├── pipeline_v2.py              # V2 Orchestrator (PARTIAL)
├── face_detector.tflite        # MediaPipe model
├── face_os_config.yaml         # Configuration
├── intrinsic_decomposition.py  # IMPLEMENTED, INTEGRATED, CONDITIONAL ACTIVE
├── physical_renderer.py        # IMPLEMENTED, INTEGRATED, CONDITIONAL ACTIVE
├── dense_geometry.py           # IMPLEMENTED, NOT INTEGRATED
├── lie_group.py                # IMPLEMENTED, INTEGRATED, ACTIVE
├── renderer_mode.py            # IMPLEMENTED, INTEGRATED, ACTIVE
├── visibility_calibration.py   # IMPLEMENTED, NOT INTEGRATED
├── identity_manifold.py        # IMPLEMENTED, NOT INTEGRATED
├── state_space.py              # Phase 2A — LatentState, Kalman filter
├── optimizer.py                # Phase 2B — GaussNewton, LevenbergMarquardt
├── observability.py            # Phase 2C — ObservabilityAnalyzer
├── state_separation.py        # Phase 2D — PhysicalState, BeliefState, MetaState
├── map_estimation.py           # Phase 2E — MAPOptimizer (LocalMAPApproximation)
├── recovery_dynamics.py        # Phase 2G — RecoveryTransitionMatrix
└── subsystems/                 # V2 Architecture (PARTIAL)
    ├── __init__.py
    ├── geometry_estimator.py
    ├── identity_estimator.py
    ├── temporal_estimator.py
    └── renderer.py
```
face_os/
├── __init__.py
├── types.py                    # Core data structures
├── config.py                   # YAML config loader
├── energy.py                   # EnergyComputer (5 terms)
├── visibility.py               # VisibilityLogger
├── ingest.py                   # Video loading
├── detect_track.py             # MediaPipe detection + tracking
├── landmarks.py                # 478-point landmarks + PnP
├── canonical_map.py            # Canonical UV alignment
├── crop_planner.py             # Reference-based crop planning
├── temporal_solve.py           # Bidirectional temporal solver
├── face_enhance.py             # Structure-preserving rendering
├── identity_state.py           # Frequency decomposition + VerificationGate
├── compositor.py               # Confidence-weighted compositing
├── appearance_field.py         # AppearanceField
├── neural_codec.py             # PersonalizedSpace + NeuralCodec
├── pipeline.py                 # V0.5 Orchestrator (WORKING)
├── pipeline_v2.py              # V2 Orchestrator (PARTIAL)
├── face_detector.tflite        # MediaPipe model
├── face_os_config.yaml         # Configuration
├── intrinsic_decomposition.py  # V3 — IntrinsicDecomposer (NOT INTEGRATED)
├── physical_renderer.py        # V3 — PhysicalRenderer (NOT INTEGRATED)
├── dense_geometry.py           # V3 — DenseGeometryEstimator (NOT INTEGRATED)
├── lie_group.py                # V3 — SE2Transform, SIM2Transform (NOT INTEGRATED)
├── state_space.py              # Phase 2A — LatentState, Kalman filter
├── optimizer.py                # Phase 2B — GaussNewton, LevenbergMarquardt
├── observability.py            # Phase 2C — ObservabilityAnalyzer
├── state_separation.py        # Phase 2D — PhysicalState, BeliefState, MetaState
├── map_estimation.py           # Phase 2E — MAPOptimizer
├── recovery_dynamics.py        # Phase 2G — RecoveryTransitionMatrix
└── subsystems/                 # V2 Architecture (PARTIAL)
    ├── __init__.py
    ├── geometry_estimator.py
    ├── identity_estimator.py
    ├── temporal_estimator.py
    └── renderer.py

tests/face_os/
├── test_phase0_contract.py     # 28 tests
├── test_phase1_energy.py       # 36 tests
├── test_phase1_hardening.py    # 37 tests
├── test_phase2a_state_space.py # 39 tests
├── test_phase2b_optimizer.py   # 32 tests
├── test_phase2c_observability.py # 28 tests
├── test_phase2d_state_separation.py # 34 tests
├── test_phase2e_map_estimation.py # 19 tests
├── test_phase2g_recovery_dynamics.py # 38 tests
├── test_phase3a_intrinsic.py   # 26 tests (NEW)
├── test_phase3b_physical_renderer.py # 26 tests (NEW)
├── test_phase3c_dense_geometry.py # 23 tests (NEW)
├── test_phase3d_lie_group.py   # 23 tests (NEW)
├── test_strict_regression.py   # 26 tests
├── test_v2_subsystems.py       # 20 tests
├── test_math_hardening.py      # 37 tests
├── test_detection.py           # 14 tests
├── test_quality_gates.py       # 13 tests
├── test_identity_state.py      # 17 tests
├── test_identity_state_fixes.py # 5 tests
├── test_patch_memory.py        # 18 tests
├── test_temporal_solve.py      # 10 tests
├── test_face_enhance.py        # 18 tests
├── test_appearance_field.py    # 14 tests
├── test_neural_codec.py        # 12 tests
├── test_hypothesis_matching.py # 4 tests
├── test_region_confidence.py   # 4 tests
└── conftest.py

output/face_os/
├── v05_phase1_test.mp4         # V0.5 pipeline output
├── v05_phase1_test.qc.json     # QC report
└── visibility/                 # Pass/energy/renderer reports
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| OpenCV (cv2) | ≥4.5 | Image processing |
| NumPy | ≥1.20 | Array operations |
| SciPy | ≥1.7 | Optimization, sparse matrices |
| mediapipe | ≥0.10.35 | Face detection + landmarks |
| FFmpeg | ≥5.0 | Video encoding |
| PyYAML | ≥5.0 | Config parsing |

---

## How to Run Tests

```bash
# Full test suite (629 tests)
.venv/bin/python -m pytest tests/face_os/ -v

# V3 modules only (98 tests)
.venv/bin/python -m pytest tests/face_os/test_phase3*.py -v

# Phase 3A: Intrinsic Decomposition (26 tests)
.venv/bin/python -m pytest tests/face_os/test_phase3a_intrinsic.py -v

# Phase 3B: Physical Renderer (26 tests)
.venv/bin/python -m pytest tests/face_os/test_phase3b_physical_renderer.py -v

# Phase 3C: Dense Geometry (23 tests)
.venv/bin/python -m pytest tests/face_os/test_phase3c_dense_geometry.py -v

# Phase 3D: Lie-Group Transforms (23 tests)
.venv/bin/python -m pytest tests/face_os/test_phase3d_lie_group.py -v
```

---

**Last Updated:** 2026-05-21  
**Document Status:** Honest assessment of current implementation vs intended architecture
