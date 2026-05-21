# Face OS — Complete Architecture & Reference (V3.0.0)

**Version:** 3.0.0  
**Branch:** `feat/face-os-v2-phase1`  
**Date:** 2026-05-21  
**Status:** Phase 0-3D COMPLETE | **629 tests passing** | Physical foundation modules added | Architecture contradictions documented

---

## ⚠️ IMPORTANT: Architecture Status

**Current Reality:**
- Face OS has **two parallel systems**: V0.5 (working pipeline) and V2 (subsystem architecture)
- V3 modules (PhysicalRenderer, IntrinsicDecomposition, DenseGeometry, LieGroup) are **implemented but NOT yet integrated**
- The **production pipeline still uses alpha compositing**, not physical rendering
- The **identity system still uses appearance-based representation**, not intrinsic decomposition
- This document honestly documents both the **intended architecture** and the **actual implementation**

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

### Current Test Count: 629 tests, 0 failures

---

## 2. Architecture Status

### Two Parallel Systems

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
│  │  ACTUAL RENDERER: Y = M ⊙ Y_face + (1-M) ⊙ Y_bg (alpha compositing)│  │
│  │  ACTUAL IDENTITY: appearance_latent (RGB image, NOT intrinsic)       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  V3 Modules (New) — IMPLEMENTED BUT NOT INTEGRATED                          │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ IntrinsicDecomp │  │ PhysicalRenderer│  │ DenseGeometry   │            │
│  │ (albedo,shade)  │  │ (Lambert+Phong) │  │ (icosphere mesh)│            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                             │
│  ┌─────────────────┐                                                       │
│  │ LieGroup        │                                                       │
│  │ (SE(2), SIM(2)) │                                                       │
│  └─────────────────┘                                                       │
│                                                                             │
│  These modules EXIST and are TESTED but are NOT connected to the pipeline.  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. V3 Modules (New)

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
- `IntrinsicComponents` — albedo, shading, specular, normals, confidence
- `DecompositionConfig` — configurable parameters

**Tests:** 26 tests

**Status:** ✅ Implemented and tested, ❌ NOT integrated into pipeline

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

**Status:** ✅ Implemented and tested, ❌ NOT integrated into pipeline

**⚠️ CONTRADICTION:** The production pipeline (`pipeline.py`) still uses:
```
Y = M ⊙ Y_face + (1-M) ⊙ Y_bg  (alpha compositing)
```
NOT the physical renderer.

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

**Status:** ✅ Implemented and tested, ❌ NOT integrated into pipeline

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

**Status:** ✅ Implemented and tested, ❌ NOT integrated into pipeline

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

### 🔴 Critical Contradictions

#### 1. Renderer Contradiction
- **Claimed:** Physical rendering (Lambertian + Blinn-Phong)
- **Actual:** Alpha compositing (`Y = M ⊙ Y_face + (1-M) ⊙ Y_bg`)
- **Impact:** Not physically grounded, lighting mismatch, uncanny blending
- **Fix Required:** Integrate PhysicalRenderer into pipeline

#### 2. Identity Contradiction
- **Claimed:** Intrinsic decomposition (albedo, shading, specular)
- **Actual:** Appearance latent (RGB image)
- **Impact:** Lighting leaks into identity, shadow ambiguity
- **Fix Required:** Integrate IntrinsicDecomposer into pipeline

#### 3. Geometry Contradiction
- **Claimed:** Dense mesh (icosphere + RBF fitting)
- **Actual:** 478 sparse landmarks
- **Impact:** Weak curvature understanding, poor micro-geometry
- **Fix Required:** Integrate DenseGeometryEstimator into pipeline

#### 4. Transform Contradiction
- **Claimed:** Lie-group transforms (SE(2), SIM(2))
- **Actual:** Linear EMA on affine matrices
- **Impact:** Skew drift, covariance inconsistency
- **Fix Required:** Integrate LieGroup transforms into pipeline

### 🟡 Documentation Issues

#### 5. Version References Mixed
- V2, V3, V4 references scattered throughout
- Test counts inconsistent (240, 277, 531, 629)
- Duplicate file structure sections

#### 6. Stale Metrics Tables
- Some tables still reference V2.1.0 or V2.8.0
- V3.0.0 metrics not differentiated from V2.8.0

### 🟢 Working Correctly

#### 7. V0.5 Pipeline
- Face detection: 80.9% ✅
- Identity drift: 12.83 LAB ✅
- Output contract: 1080x1920 uint8 ✅
- All 629 tests passing ✅

---

## 9. Remaining Architectural Gaps

### Tier 1: Integration Required (V3 modules exist but not connected)

| Gap | Current State | Required State | Effort |
|-----|---------------|----------------|--------|
| Renderer integration | Alpha compositing | Physical rendering | HIGH |
| Identity integration | Appearance RGB | Intrinsic albedo | HIGH |
| Geometry integration | 478 landmarks | Dense mesh | MED |
| Transform integration | Linear EMA | Lie-group | MED |

### Tier 2: Mathematical Completeness

| Gap | Current State | Required State | Effort |
|-----|---------------|----------------|--------|
| Identity manifold | Conceptual only | Topology, parameterization, geodesics | HIGH |
| Observation model | Handcrafted, linear | Nonlinear, ambiguity-aware | HIGH |
| Energy function | Engineered terms | Learned priors | HIGH |
| MAP optimization | Local frame inference | Full factor graph | HIGH |

### Tier 3: System Robustness

| Gap | Current State | Required State | Effort |
|-----|---------------|----------------|--------|
| Bayesian temporal | Kalman-inspired | Full particle filtering | HIGH |
| Recovery dynamics | Semi-heuristic | Learned transitions | MED |
| Adversarial robustness | Partial | Full coverage | HIGH |
| Long-horizon memory | Short-window | Sequence-level | HIGH |

### Tier 4: Fundamental Limitations

| Gap | Description | Solvable? |
|-----|-------------|-----------|
| Observation ambiguity | Same image → multiple latent explanations | Partially (priors help) |
| Computational complexity | MAP + Lie groups + dense geometry = explosion | Need profiling |
| Visibility calibration | Visible ≠ correct | Need validation |

---

## 10. Roadmap

### V3.1 — Integration (Next)
1. Integrate IntrinsicDecomposer into pipeline
2. Integrate PhysicalRenderer into pipeline
3. Integrate DenseGeometryEstimator into pipeline
4. Integrate LieGroup transforms into pipeline
5. Validate metrics improvement

### V3.2 — Mathematical Completeness
1. Define identity manifold properly
2. Implement nonlinear observation model
3. Add learned energy priors
4. Build factor graph optimization

### V3.3 — System Robustness
1. Full Bayesian temporal reasoning
2. Learned recovery dynamics
3. Adversarial robustness suite
4. Long-horizon memory

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
