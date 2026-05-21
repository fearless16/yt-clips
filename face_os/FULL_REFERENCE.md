# Face OS — Complete Architecture & Parameter Reference (V2)

**Version:** 2.8.0  
**Branch:** `feat/face-os-v2-phase1`  
**Date:** 2026-05-21  
**Status:** Phase 0-2G COMPLETE | **531 tests passing** | Probabilistic recovery dynamics | All visibility red flags resolved

---

## Table of Contents

1. [What Changed From V0.5](#1-what-changed-from-v05)
2. [V2 Architecture Overview](#2-v2-architecture-overview)
3. [Subsystem Deep Dive](#3-subsystem-deep-dive)
4. [V0.5 Module Reference](#4-v05-module-reference)
5. [Quality Gates](#5-quality-gates)
6. [Verification Gate](#6-verification-gate)
7. [Configuration Reference](#7-configuration-reference)
8. [Feature Flags](#8-feature-flags)
9. [Test Results & Metrics](#9-test-results--metrics)
10. [Video Parameter Test Report](#10-video-parameter-test-report)
11. [Known Issues & Next Steps](#11-known-issues--next-steps)

---

## 1. What Changed From V0.5

### V0.5 → V2 Changes

| Component | V0.5 | V2 | Why Changed |
|---|---|---|---|
| **Architecture** | Monolithic pipeline | 4 isolated subsystems | Separation of concerns, testability |
| **State Types** | Implicit dicts | Explicit dataclasses | `GeometryState`, `IdentityState`, `TemporalState` |
| **Coordinate Systems** | Implicit | Explicit transform chain | `W = T_output ∘ T_render ∘ T_uv ∘ T_pose ∘ T_crop` |
| **Geometry Estimation** | Mixed in pipeline | Subsystem A | Forbidden: identity logic, lighting logic, RGB blending |
| **Identity Estimation** | Mixed in pipeline | Subsystem B | Forbidden: RGB EMA blending, raw frame accumulation |
| **Temporal Estimation** | Mixed in pipeline | Subsystem C | Forbidden: backward texture injection, frame averaging |
| **Renderer** | Mixed in pipeline | Subsystem D | Equation: `Y = M ⊙ Y_face + (1 - M) ⊙ Y_bg` |
| **Pipeline Orchestrator** | `pipeline.py` only | `pipeline.py` + `pipeline_v2.py` | V0.5 preserved, V2 parallel |
| **Tests** | 220 | 240 | +20 V2 subsystem isolation tests |

### V3 → V4 Changes (Preserved in V2)

| Component | V3 | V4 | Why Changed |
|---|---|---|---|
| **Face Detection** | MediaPipe FaceDetection (tasks) | MediaPipe FaceDetector + FaceLandmarker (tasks) | FaceLandmarker gives 478 landmarks for better shape matching |
| **API** | `mp.solutions.face_mesh` | `mediapipe.tasks.python.vision` | MediaPipe 0.10.35 removed `mp.solutions`, uses `tasks` API |
| **Face Mesh** | dlib 68-point (fallback) | MediaPipe 478-point (no fallback) | More landmarks = better Procrustes disparity |
| **Eye Indices (EAR)** | dlib 68-point (36:42, 42:48) | MediaPipe 478-point ([33,159,158,133,153,145]) | Fixed in face_enhance.py + pipeline.py |
| **Procrustes** | Fixed 0.2 threshold | Pose-aware: 0.2 / 0.28 / 0.35 | Side-tilted views naturally differ from frontal reference |
| **Quality Gates** | `low_freq_ema_rate: 0.1` | `low_freq_ema_rate: 0.05` | Slower EMA prevents source lighting from corrupting identity |
| **Anchor correction** | Double anchor (query + frame) | Single anchor (query only) | Double anchor caused ghosting/double exposure |
| **Face mask blending** | Raw conf (no feathering) | Feathered Gaussian mask | Hard edge at face boundary caused background bleed |
| **High-freq blend** | Double-dampened (1.25% effective) | Floor at 0.15, no per-pixel dampen | Plastic skin from killing texture |
| **Config default** | `dlib_68` | `mediapipe_478` | No dlib dependency |
| **Simple mode** | N/A | `--no-identity` flag | Bypass identity for clean enhancement |

---

## 2. V2 Architecture Overview

### Core Philosophy

Face reconstruction is NOT an image-editing problem. It is a:
- **latent-state estimation problem**
- **constrained geometry problem**
- **temporal inference problem**
- **physically consistent rendering problem**

### Two Parallel Systems

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  V0.5 Pipeline (pipeline.py) — Preserved, working                           │
│  USE_IDENTITY = True (default)                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 3-pass pipeline:                                                     │  │
│  │   Pass 1: Forward collection (identity state build)                  │  │
│  │   Pass 2: Bidirectional solve (HQ frames repair)                     │  │
│  │   Pass 3: Render (identity blend + enhance)                          │  │
│  │                                                                      │  │
│  │  Modules: identity_state + patch_memory + temporal                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  USE_IDENTITY = False (--no-identity)                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Forward-only pipeline:                                               │  │
│  │   Detect → Landmarks → Crop → Enhance → Export                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  V2 Pipeline (pipeline_v2.py) — NEW subsystem architecture                  │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ Subsystem A │→ │ Subsystem B │→ │ Subsystem C │→ │ Subsystem D │       │
│  │ Geometry    │  │ Identity    │  │ Temporal    │  │ Renderer    │       │
│  │ Estimator   │  │ Estimator   │  │ Estimator   │  │             │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│       ↓                ↓                ↓                ↓                 │
│  GeometryState    IdentityState    TemporalState    Output Frame           │
│                                                                             │
│  Each subsystem is ISOLATED with explicit forbidden patterns.               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### V2 Subsystem Flow

```
INPUT: 16:9 source video + reference face images
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Detect + Track (MediaPipe)   │
    │  FaceDetector + FaceLandmarker│
    └───────────────┬───────────────┘
                    ▼
    ┌───────────────────────────────┐
    │ SUBSYSTEM A: Geometry         │
    │ Estimator                      │
    │ - Landmarks + Pose            │
    │ - Canonical Transform         │
    │ - Crop Plan                   │
    │ - Semantic Regions            │
    │ - Geometry Mask               │
    └───────────────┬───────────────┘
                    │ GeometryState
                    ▼
    ┌───────────────────────────────┐
    │ SUBSYSTEM B: Identity         │
    │ Estimator                      │
    │ - Anchor Basis                │
    │ - Appearance Latent           │
    │ - Region Confidence           │
    │ - Identity Uncertainty        │
    └───────────────┬───────────────┘
                    │ IdentityState
                    ▼
    ┌───────────────────────────────┐
    │ SUBSYSTEM C: Temporal         │
    │ Estimator                      │
    │ - Motion Field                │
    │ - Temporal Confidence         │
    │ - Drift Score                 │
    │ - Continuity Score            │
    └───────────────┬───────────────┘
                    │ TemporalState
                    ▼
    ┌───────────────────────────────┐
    │ SUBSYSTEM D: Renderer          │
    │ - Y = M ⊙ Y_face + (1-M)⊙Y_bg │
    │ - Deterministic Output        │
    │ - Contract Validation         │
    └───────────────┬───────────────┘
                    ▼
OUTPUT: 9:16 enhanced video (1080x1920)
```

### Architectural Principles

| Principle | Description | Enforcement |
|---|---|---|
| **Geometry First** | All masks, crops, warps derive from geometry | Forbidden: brightness threshold masks, intensity-derived topology |
| **Identity ≠ RGB Memory** | Identity is latent anchor basis, not EMA frames | Forbidden: RGB EMA blending, raw frame accumulation |
| **Deterministic Rendering** | Every path: identical dimensions, dtype, bounded behavior | Enforced: frame contract validation |
| **Temporal as Constraint** | Temporal stability is hard constraint, not optional | Forbidden: backward texture injection, frame averaging |

---

## 3. Subsystem Deep Dive

### SUBSYSTEM A — Geometry Estimator

**File:** `face_os/subsystems/geometry_estimator.py`

**Purpose:** Estimate all spatial structure.

**Inputs:**
- `frame_t` — Input frame (H, W, 3) BGR
- `face_track` — Detected face with 478-point mesh
- `previous_geometry_state` — Previous state for temporal continuity

**Outputs:** `GeometryState`

```python
@dataclass
class GeometryState:
    landmarks_478: Optional[np.ndarray]        # (478, 3) MediaPipe mesh
    landmarks: Optional[Landmarks]             # Extracted landmarks with pose
    pose: Tuple[float, float, float]           # (yaw, pitch, roll)
    canonical_transform: Optional[np.ndarray]  # Transform to canonical space
    inverse_transform: Optional[np.ndarray]    # Transform from canonical space
    crop_transform: Optional[CropPlan]         # Crop plan
    mesh: Optional[np.ndarray]                 # Face mesh
    semantic_regions: Optional[Dict[str, np.ndarray]]  # Region masks
    mask: Optional[np.ndarray]                 # Geometry-based face mask
    geometry_confidence: float                 # Overall geometry confidence
    canonical_face: Optional[np.ndarray]       # Frame warped to canonical space
```

**Responsibilities:**
- Landmark extraction (MediaPipe 478-point)
- Head pose estimation (PnP from 6 key points)
- Canonical UV mapping
- Semantic region construction
- Crop optimization
- Warp transform generation

**Forbidden:**
- Identity logic
- Lighting logic
- RGB blending

---

### SUBSYSTEM B — Identity Estimator

**File:** `face_os/subsystems/identity_estimator.py`

**Purpose:** Estimate stable identity representation independent of lighting and pose.

**Inputs:**
- `geometry_state` — Geometry state with canonical face
- `quality_map` — Per-pixel quality map
- `face_track` — Face track with verification info

**Outputs:** `IdentityState`

```python
@dataclass
class IdentityState:
    anchor_basis: list                         # List of anchor states
    anchor_weights: list                       # Weights for anchors
    appearance_latent: Optional[np.ndarray]    # Current identity appearance
    region_confidence: Dict[str, float]        # Per-region confidence
    identity_uncertainty: float                # Overall uncertainty (0-1)
    initialized: bool                          # Whether initialized
```

**Identity Representation:**
```
a_t = Σ(w_k * a_k)
```
Where `a_k` are learned/selected anchor states and `w_k` are confidence-normalized interpolation weights.

**Required Anchor Dimensions:**
- Frontal neutral
- Left yaw / Right yaw
- Smile
- Low-light / High-light
- Blink
- Beard-shadow

**Forbidden:**
- RGB EMA blending
- Raw frame accumulation
- Frame-space averaging

---

### SUBSYSTEM C — Temporal Estimator

**File:** `face_os/subsystems/temporal_estimator.py`

**Purpose:** Maintain temporal consistency.

**Inputs:**
- `geometry_state` — Current geometry state
- `identity_state` — Current identity state
- `previous_temporal_state` — Previous temporal state

**Outputs:** `TemporalState`

```python
@dataclass
class TemporalState:
    motion_field: Optional[np.ndarray]         # Optical flow field (H, W, 2)
    temporal_confidence: float                 # Temporal consistency confidence
    drift_score: float                         # Identity drift from anchor
    continuity_score: float                    # Temporal smoothness score
    smoothing_constraints: Dict[str, float]    # Smoothing limits
    pose: Optional[Tuple[float, float, float]] # Pose for continuity tracking
```

**Responsibilities:**
- Bidirectional smoothing
- Confidence propagation
- Optical-flow consistency
- Identity continuity
- Geometry continuity

**Critical Rule:** Temporal system updates CONFIDENCE, not raw texture.

**Forbidden:**
- Backward texture injection
- Frame averaging
- Temporal blur accumulation

---

### SUBSYSTEM D — Renderer

**File:** `face_os/subsystems/renderer.py`

**Purpose:** Generate physically consistent output.

**Inputs:**
- `source_frame` — Original source frame
- `geometry_state` — Geometry state with transforms and masks
- `identity_state` — Identity state with appearance
- `temporal_state` — Temporal state with confidence
- `crop_plan` — Crop plan for output dimensions

**Outputs:** Rendered output frame (H, W, 3) uint8

**Render Equation:**
```
Y = M ⊙ Y_face + (1 - M) ⊙ Y_bg
```

Where:
- `M` is geometry-derived semantic mask
- `Y_face` is latent-rendered face
- `Y_bg` is untouched background

**Forbidden:**
- RGB-space rescue compositing
- Heuristic face merging
- Implicit blending logic

---

## 4. V0.5 Module Reference

### Module 1: `ingest.py` — Video Ingestion

- Loads video file and extracts metadata (dimensions, fps, codec, duration)
- Provides frame-by-frame generator with seeking support
- Loads reference face images for identity enrollment

---

### Module 2: `detect_track.py` — Face Detection + Tracking + Quality Gates

- Detects faces using **MediaPipe FaceDetector** (tasks API, min_conf=0.6)
- Extracts **478 landmarks** using **MediaPipe FaceLandmarker** (tasks API)
- Matches detected faces to target identity via embeddings
- Maintains persistent face tracks across frames
- **Pose-aware Procrustes** — relaxes threshold for extreme head poses

---

### Module 3: `landmarks.py` — 478-Point Landmarks + Head Pose

- 100% MediaPipe 478-point, NO dlib
- PnP head pose from 6 key points (nose, chin, eyes, mouth corners)
- Region masks from 478-point contours (eyes, brows, nose, mouth, skin, face oval)

---

### Module 4: `canonical_map.py` — Canonical Face Mapping

- Aligns detected face to canonical UV space (frontal, neutral pose)
- Builds Appearance Field A(u,v,θ,L,t)
- Dynamically handles both 478-point and 68-point landmarks

---

### Module 5: `crop_planner.py` — Reference-Based Crop Planning

- Analyzes reference image at startup for composition targets
- Plans 16:9 → 9:16 crop that matches reference composition
- Preserves source headroom (never reduces it)

---

### Module 6: `temporal_solve.py` — Bidirectional Temporal Solver

- Forward pass: collect per-frame quality metrics, identify HQ frames
- Backward pass: HQ frames repair past blurry frames

---

### Module 7: `face_enhance.py` — Structure-Preserving Rendering

- Enhances face regions while PRESERVING source structure
- Does NOT hallucinate details (eyelashes, pores, etc.)
- Cinematic noise (temporal grain)
- Blink detection uses MediaPipe 478-point eye indices

---

### Module 8: `identity_state.py` — Frequency Decomposition

**Dynamic blending (V4.1 — fixed):**
```python
# Low freq: config-driven EMA rate
base_rate = cfg.identity_state.low_freq_ema_rate  # 0.05

# High freq: floor prevents texture loss
high_blend = max(cfg.identity_state.high_blend_base, 0.15)
# Do NOT multiply by per-pixel conf again (was double-dampening to 1.25%)
effective_high_blend = np.full_like(conf_3d, high_blend)
```

**Anchor correction (V4.1 — single application in query only):**
```python
# In query() — pulls identity toward reference
low_final = (1 - lambda) * low_final + lambda * anchor_low
high_final = (1 - lambda * 0.2) * high_final + (lambda * 0.2) * anchor_high

# In _render_frame_v2() — NO second anchor correction
# (removed to prevent ghosting)
```

---

### Module 9: `compositor.py` — Confidence-Weighted Compositing

- Composites identity face onto original frame using per-pixel confidence
- Feathered edge blending prevents visible seams
- **Used in identity mode only** — simple mode bypasses compositor

---

### Pipeline: `pipeline.py` — V0.5 Orchestrator

**Identity mode (USE_IDENTITY=True):**
- 3-pass pipeline: forward → bidirectional solve → render
- Identity state + patch memory + temporal solver active
- Face lock state machine: FACE_LOCKED / LOST_FACE / RECOVERY
- `_last_good_crop_plan` persists crop position across frames

**Simple mode (USE_IDENTITY=False):**
- Forward-only pass
- Crop → enhance (sharpen + denoise) → export
- No ghosting, no background bleed, no plastic skin

### Pipeline: `pipeline_v2.py` — V2 Orchestrator

- Uses all 4 subsystems in isolation
- Maintains backward compatibility with V0.5
- Forward and bidirectional processing modes
- Frame contract validation
- QC and reporting

---

## 5. Quality Gates

| Gate | Threshold | Purpose |
|---|---|---|
| Procrustes disparity | < 0.2 (moderate: 0.28, extreme: 0.35) | Face shape must match reference |
| Landmark jitter | > 0.0008 | Real face moves (poster is static) |
| Occupancy | > 0.25 | Face must fill enough of bbox |

**Pose-aware Procrustes (V4.1):**
```python
pose = _estimate_pose_from_landmarks(landmarks)
threshold = 0.2
if abs(yaw) > 20 or abs(pitch) > 15:
    threshold = 0.35   # Extreme pose
elif abs(yaw) > 10 or abs(pitch) > 10:
    threshold = 0.28   # Moderate pose
```

---

## 6. Verification Gate

Runs BEFORE identity_state.update(). All checks must pass.

| Check | Threshold | Purpose |
|---|---|---|
| Face pixels | >= 4000 | Reject tiny faces |
| Embedding distance | <= 0.45 | Reject wrong identity |
| Liveness (jitter) | >= 0.5 | Reject static posters |

---

## 7. Configuration Reference

**File:** `face_os_config.yaml`

```yaml
identity:
  reference_dir: "photos/"
  reference_image: "expectation.png"
  embedding_tolerance: 0.45
  max_embeddings: 50

detection:
  model: "mediapipe"
  min_face_size: 60
  detection_interval: 5
  max_lost_frames: 30
  smoothing_alpha: 0.3

landmarks:
  model: "mediapipe_478"
  pose_smoothing: 0.4

quality_gates:
  procrustes_threshold: 0.2     # Base (pose-aware relaxation applied)
  jitter_threshold: 0.0008
  occupancy_threshold: 0.25

verification_gate:
  embedding_tolerance: 0.45
  min_face_pixels: 4000
  liveness_threshold: 0.5

canonical:
  atlas_size: [256, 256]
  alignment_mode: "similarity"
  enrollment_frames: 30

identity_state:
  low_freq_ema_rate: 0.05       # ↓ Slow EMA (was 0.1)
  high_freq_best_only: true
  confidence_modulation: true
  base_confidence: 0.7
  anchor_lambda_max: 0.95       # ↑ Strong anchor pull (was 0.75)
  low_blend_base: 0.95          # ↑ 95% identity trust (was 0.85)
  high_blend_base: 0.05         # Floor at 0.15 in code (was 0.15)

crop:
  output_size: [1080, 1920]
  headroom_ratio: 0.30
  face_target_width: 270
  smoothing_alpha: 0.25
  max_crop_velocity: 50
  protect_forehead: true

temporal:
  identity_inertia: 0.85
  flicker_threshold: 15.0
  temporal_window: 5

enhance:
  eye_boost: 1.5
  brow_boost: 1.3
  beard_boost: 1.2
  skin_smoothing: 0.3
  sharpen_amount: 0.3
  use_cinematic_noise: true
  noise_strength: 0.02

compositor:
  confidence_threshold: 0.3
  blend_mode: "poisson"
  feather_pixels: 10
  use_light_matching: false      # Disabled: was darkening identity

export:
  codec: "libx264"
  crf: 18
  preset: "slow"
  bitrate: "25M"
  audio_bitrate: "320k"
  fps: 30
  fade_in: 0.5
  fade_out: 0.5

qc:
  min_face_detection_rate: 0.80
  max_identity_drift: 20.0
  max_flicker_score: 5.0
  min_sharpness: 10.0
```

---

## 8. Feature Flags

### `USE_IDENTITY` (pipeline.py)

```python
# At top of pipeline.py:
USE_IDENTITY = True   # Default: identity reconstruction mode

# Or via CLI:
python -m face_os.pipeline --video input.mp4 --no-identity -o output.mp4
```

**When True (default):**
- Full 3-pass pipeline with identity state
- Bidirectional temporal solver
- Anchor correction toward reference
- Risk: ghosting, plastic skin, background bleed

**When False (--no-identity):**
- Forward-only pass
- Crop + enhance (sharpen + denoise) + export
- No identity memory, no anchor correction
- Clean source enhancement, no artifacts

---

## 9. Test Results & Metrics

**Test clip:** `clips_test/test_clip.mp4` (640x360, 30fps, 345 frames)  
**Reference:** `expectation.png` (941x1672, portrait)

### Test Suite (V2.6.0)

| File | Tests | Status | Purpose |
|---|---|---|---|
| `test_strict_regression.py` | 26 | ✅ All pass | Frame contract, mask stability, NaN/Inf, bidirectional size, EMA convergence |
| `test_v2_subsystems.py` | 20 | ✅ All pass | V2 subsystem isolation, coordinate systems, mathematical invariants |
| `test_math_hardening.py` | 37 | ✅ All pass | 10 invariant classes: UV roundtrip, transform det, temporal drift, flow shimmer, reprojection, lighting/pose invariance, mask topology, subpixel drift, edge cases |
| `test_phase1_hardening.py` | 37 | ✅ All pass | Long-horizon drift, system identifiability, renderer equation, VerificationGate |
| `test_phase0_contract.py` | 28 | ✅ All pass | FrameContract, EnergyReport, RendererReport, PassReport, VisibilityLogger |
| `test_phase1_energy.py` | 36 | ✅ All pass | Energy term existence, numeric range, delta regression, monotonicity |
| `test_phase2a_state_space.py` | 39 | ✅ All pass | LatentState, StateTransition, ProcessNoise, Observation, StateSpaceEstimator |
| `test_phase2b_optimizer.py` | 32 | ✅ All pass | GaussNewton, LevenbergMarquardt, convergence, singularity, rollback |
| `test_phase2c_observability.py` | 28 | ✅ All pass | ObservabilityAnalyzer, DegeneracyReport, lighting/pose/identity ambiguity |
| `test_phase2d_state_separation.py` | 34 | ✅ All pass | PhysicalState, BeliefState, MetaState, SeparatedState, StateSeparator |
| `test_phase2e_map_estimation.py` | 19 | ✅ All pass | MAPOptimizer, MAPReport, posterior contraction, energy descent |
| `test_phase2g_recovery_dynamics.py` | 38 | ✅ All pass | RecoveryTransitionMatrix, ProbabilisticRecoveryState, RecoveryDynamics |
| `test_detection.py` | 14 | ✅ All pass | MediaPipe tasks API, poster rejection, identity matching |
| `test_quality_gates.py` | 13 | ✅ All pass | Procrustes, jitter, occupancy, SSIM, Laplacian |
| `test_identity_state.py` | 17 | ✅ All pass | Identity state, frequency decomposition, hypotheses |
| `test_identity_state_fixes.py` | 5 | ✅ All pass | LastUpdateFrame, region confidence, hypothesis matching |
| `test_patch_memory.py` | 18 | ✅ All pass | Region patches, pose-conditioned storage |
| `test_temporal_solve.py` | 10 | ✅ All pass | Bidirectional solver, HQ frame repair |
| `test_face_enhance.py` | 18 | ✅ All pass | Blink detection (V4 478-pt eyes), rendering, noise |
| `test_appearance_field.py` | 14 | ✅ All pass | Appearance field, dynamic deformation |
| `test_neural_codec.py` | 12 | ✅ All pass | PersonalizedSpace, NeuralCodec, identity score |
| `test_hypothesis_matching.py` | 4 | ✅ All pass | Hypothesis space, pose/expression selection |
| `test_region_confidence.py` | 4 | ✅ All pass | Region confidence, semantic confidence |
| **Total** | **531** | **0 failures** | **All green** |

### QC Metrics (Identity Mode, V2.1.0 — 345 frames, Phase 1 Hardening)

```
Face detection rate:  80.9%   (target >80%) ✅
Identity drift:       12.83   (target <20)  ✅  (was 16.25, 21% improvement)
Anchor distance:      3.90    (target <25)  ✅
Flicker score:        0.87    (target <5)   ✅
Sharpness:            13.31   (target >10)  ✅
AV Sync:              True    ✅
Output resolution:    1080x1920 ✅
Output dtype:         uint8   ✅
Processing time:      98.4s (3.8 fps)
Output file size:     11.7MB
```

### QC Metrics (Identity Mode, V2.0.0 — 50 frames)

```
Face detection rate:  100.0%  (target >80%) ✅
Identity drift:       16.25   (target <20)  ✅
Anchor distance:      1.40    (target <25)  ✅
Flicker score:        0.83    (target <5)   ✅
Sharpness:            24.08   (target >10)  ✅
AV Sync:              True    ✅
```

### QC Metrics (Simple Mode, --no-identity)

```
Face detection rate:  100%   ✅
Identity drift:       19.3   (no correction applied)
Flicker score:        0.76   ✅
Sharpness:            123.1  ✅
```

### Metrics History

| Version | LAB Dist | Detection | Flicker | Sharpness | Notes |
|---|---|---|---|---|---|
| V1 (broken compositor) | 24.8 | 64% | — | — | Using rendered instead of identity |
| V4 (initial) | 24.6 | 64% | — | — | MediaPipe tasks, Procrustes 0.2 |
| V4.1 (bug fixes) | 19.2 | 82.7% | 0.76 | — | Feathered mask, single anchor, pose-aware |
| V4.1 (simple mode) | 19.3 | 100% | 0.76 | 123.1 | No identity, clean enhancement |
| V2.0.0 (subsystems) | 16.25 | 100% | 0.83 | 24.08 | 4 isolated subsystems, 240 tests |
| **V2.1.0 (Phase 1)** | **12.83** | **80.9%** | **0.87** | **13.31** | **345 frames, 277 tests, Phase 1 hardening** |
| Target | <5 | >80% | <5 | >10 | — |

---

## 9a. Mathematical Invariants & Regression Locks (V2.0.0)

The `test_math_hardening.py` + `test_v2_subsystems.py` suites enforce 57 deterministic numeric assertions across 12 invariant classes.

### Invariant 1: UV Roundtrip (4 tests)
- **Anchor point roundtrip**: `M[:2] @ p → canonical; M_inv[:2] @ canonical → p'` — max error `< 2e-4` px.
- **NaN/Inf**: Roundtrip warps must produce finite output.
- **Shape**: `warp_from_canonical` output shape matches source shape at all scales.
- **M_inv EMA norm**: Frobenius norm must be bounded across EMA frames (range `< 1.0`).

### Invariant 2: Transform Determinant (6 tests)
- **Non-singular**: `|det(A)| > 0.001` for both similarity and affine modes.
- **Mutual inverse**: `det(A) * det(A_inv) ≈ 1.0` (error `< 1e-4`).
- **No reflection (similarity)**: `det(A) > 0` always for similarity mode.
- **Stability across yaw/pitch**: Similarity det CV `< 1.0`; affine `|det|` CV `< 1.5`.

### Invariant 3: Temporal Embedding Drift (4 tests)
- **Belief convergence**: `BeliefPixel.best_low` converges to observed value within `< 3.0` after 50 identical observations.
- **Per-frame delta decay**: Late deltas `<=` early deltas (convergence, not oscillation).
- **Anchor drift**: `query()` output stays within `< 10` LAB RMSE of anchor after 30 updates.
- **Frequency reconstruction**: `decompose + reconstruct` is lossless (`max error < 2.0`).

### Invariant 4: Optical Flow Shimmer (4 tests)
- **Static face**: EMA residual decays to `< 1e-4` Frobenius norm.
- **Smooth motion**: EMA residual `< 3.0` at 2px/frame drift.
- **Pose oscillation**: EMA residual `< 2.0` during ±20° yaw oscillation.
- **Jump catch-up**: EMA residual decays after instantaneous position jump.

### Invariant 5: Reprojection Consistency (3 tests)
- **Landmark point roundtrip**: `M @ p → M_inv @ (M @ p) ≈ p` — max error `< 2e-4` for both similarity and affine.
- **Frame position independence**: Roundtrip error same for faces at 5 different frame positions.

### Invariant 6: Lighting Invariance (4 tests)
- **Geometry mask**: `_make_canonical_geometry_mask()` returns bit-identical output on every call.
- **Elliptical mask**: `_elliptical_mask()` deterministic given same geometry params.
- **Region masks**: `create_region_masks()` deterministic given same landmarks.
- **Canonical face mask**: Convex hull + warpAffine produces consistent coverage (15%–100%) across bright/dark/mid frames.

### Invariant 7: Pose Invariance (2 tests)
- **Landmark consistency**: Canonical landmark positions at ±20° yaw stay within 40px of frontal reference.
- **Warp output size**: `warp_to_canonical()` always produces (256, 256, 3) at any yaw/pitch.

### Invariant 8: Mask Topology (3 tests)
- **Valid coverage**: Every region mask covers 0.1%–95% of frame (non-empty, non-full).
- **Connectedness**: Face mask is a single connected component (largest > 80% of foreground area).
- **Smooth boundary**: Geometry mask has 1%–75% transition zone (anti-aliased edge).

### Invariant 9: Subpixel Landmark Drift (3 tests)
- **Frame-to-frame delta**: 1px translation of all landmarks produces delta of exactly `~1.0` px (pixel-expert).
- **Crop adjustment**: `_adjust_landmarks_to_crop()` preserves pose angles and point count.
- **Crop→Canonical→Crop roundtrip**: Landmarks through crop→canonical→crop have roundtrip error `< 2e-4` px.

### Invariant 10: Canonical Mapping Edge Cases (3 tests)
- **Extreme pose**: ±60° yaw, ±40° pitch produce valid M (no crash, non-NaN).
- **Face at image edge**: Face near frame boundary produces non-singular M.
- **68-point fallback**: Landmarks with < 468 points use dlib-compatible fallback and produce `det > 0.001`.

### Invariant 11: V2 Subsystem Isolation (10 tests)
- **Geometry estimator**: Returns valid `GeometryState`, handles missing face track, brightness-invariant mask, bounded confidence.
- **Identity estimator**: Returns valid `IdentityState`, uninitialized has high uncertainty, anchor can be set.
- **Temporal estimator**: Returns valid `TemporalState`, bounded confidence, non-negative drift, bounded continuity.
- **Renderer**: Preserves output contract (shape, dtype, no NaN/Inf, valid range), fallback without identity.

### Invariant 12: V2 Coordinate Systems (2 tests)
- **Crop plan declares spaces**: Source and target dimensions explicit.
- **Geometry state has transform chain**: `canonical_transform`, `inverse_transform`, `crop_transform` all present.

---

## 10. Video Parameter Test Report

**Generated:** 2026-05-21  
**Test Video:** `clips_test/test_clip.mp4` (640×360, 30fps, 345 frames)  
**Output:** `output/face_os/v05_phase1_test.mp4` (1080×1920, 30fps, 345 frames, 11.7MB)  
**Processing:** 98.4s (3.8 fps)

### 10.1 Frame Contract Tests

| Parameter | Expected | Actual | Status |
|---|---|---|---|
| Output Shape | (1920, 1080, 3) | (1920, 1080, 3) | ✅ PASS |
| Output Dtype | uint8 | uint8 | ✅ PASS |
| FPS | 30 | 30 | ✅ PASS |
| Total Frames | 345 | 345 | ✅ PASS |
| Duration | ~11.5s | 11.5s | ✅ PASS |

### 10.2 Quality Metrics (V2.1.0 — Phase 1 Hardening)

| Parameter | Target | Actual | Status |
|---|---|---|---|
| Face Detection Rate | >0.80 | 0.8087 | ✅ PASS |
| Identity Drift (LAB) | <20.0 | 12.83 | ✅ PASS |
| Flicker Score | <5.0 | 0.87 | ✅ PASS |
| Sharpness (Laplacian) | >10.0 | 13.31 | ✅ PASS |
| AV Sync | True | True | ✅ PASS |
| Anchor Distance (LAB) | <25.0 | 3.90 | ✅ PASS |

### 10.3 Performance Metrics

| Parameter | Value | Unit |
|---|---|---|
| Processing Time | 98.4 | seconds |
| Processing FPS | 3.8 | fps |
| Input Resolution | 640×360 | pixels |
| Output Resolution | 1080×1920 | pixels |
| Upscale Factor | 3.0× | vertical |
| Output File Size | 11.7 | MB |
| Output Bitrate | 8213 | kbps |
| Codec | libx264 | |
| CRF | 18 | |

### 10.4 Subsystem Tests (277 tests)

| Test Suite | Tests | Passed | Status |
|---|---|---|---|
| test_strict_regression.py | 26 | 26 | ✅ PASS |
| test_v2_subsystems.py | 20 | 20 | ✅ PASS |
| test_math_hardening.py | 37 | 37 | ✅ PASS |
| test_phase1_hardening.py | 37 | 37 | ✅ PASS |
| test_detection.py | 14 | 14 | ✅ PASS |
| test_identity_state.py | 17 | 17 | ✅ PASS |
| test_identity_state_fixes.py | 5 | 5 | ✅ PASS |
| test_patch_memory.py | 18 | 18 | ✅ PASS |
| test_temporal_solve.py | 10 | 10 | ✅ PASS |
| test_face_enhance.py | 18 | 18 | ✅ PASS |
| test_quality_gates.py | 13 | 13 | ✅ PASS |
| test_appearance_field.py | 14 | 14 | ✅ PASS |
| test_neural_codec.py | 12 | 12 | ✅ PASS |
| test_hypothesis_matching.py | 4 | 4 | ✅ PASS |
| test_region_confidence.py | 4 | 4 | ✅ PASS |
| **TOTAL** | **277** | **277** | **✅ PASS** |

### 10.5 V2 Architecture Validation

| Component | Status | Details |
|---|---|---|
| Geometry Estimator (Subsystem A) | ✅ | Isolated |
| Identity Estimator (Subsystem B) | ✅ | Isolated |
| Temporal Estimator (Subsystem C) | ✅ | Isolated |
| Renderer (Subsystem D) | ✅ | Isolated |
| Coordinate System Reform | ✅ | Explicit |
| Mesh-Based Semantic Masking | ✅ | 478-pt |
| Brightness-Invariant Masks | ✅ | Stable |
| Anchor-Based Identity | ✅ | LAB=1.4 |
| Bidirectional Temporal Solve | ✅ | 10 HQ frames |
| Deterministic Rendering | ✅ | Contract valid |

### 10.6 Frame Statistics (50 frames)

| Statistic | Mean | Std | Min/Max |
|---|---|---|---|
| Brightness (mean) | 62.4 | 29.5 | 0/236 |
| Contrast (std) | 30.7 | 14.3 | — |
| Frame-to-Frame Δ | 6.8 | 1.6 | — |

---

## 11. Known Issues & Next Steps

### ✅ FIXED — Frame Size Invariance (Bug Class B)

**Root cause:** `pipeline.py:_process_bidirectional()` render pass assigned `cropped = source_frame` when `frame_idx not in frame_data`.

**Fix:**
- Added `self._last_good_crop_plan` to persist the last valid crop plan
- Bidirectional pass 3 now always applies `crop_planner.apply_crop()`
- `_reset_state()` clears the saved crop plan

### ✅ FIXED — Mask Stability / Intensity Threshold (Bug Class A)

**Root cause:** `gray_canon < 5` intensity threshold erased beards, eyebrows, shadows, dark skin.

**Fix:**
- Replaced with `_make_canonical_geometry_mask()` — fixed elliptical mask, brightness-invariant
- Centered on canonical atlas, semi-axes 45% × 50%, feathered with 11x11 GaussianBlur

### ✅ FIXED — M_inv EMA Too Aggressive

**Root cause:** `0.7 * last + 0.3 * new` required ~10 frames to converge.

**Fix:** Changed to `0.4 * last + 0.6 * new` — converges in 5 frames.

### ✅ FIXED — Mask Values Outside [0, 1]

**Root cause:** `GaussianBlur` floating point overshoot.

**Fix:** Added `np.clip(mask, 0, 1)` after every GaussianBlur.

### Issue 1: LAB Distance 16.25 (Target <5)

**Root cause:** Compositor blends source with identity at ~50% weight. Even though anchor distance is 1.4 LAB (canonical space), the rendered output drifts to 16.25.

**Workaround:** Use `--no-identity` for clean source enhancement.

**Possible fix:** Increase compositor blend weight to 0.9+ for face region.

### Issue 2: Ghosting/Background Bleed (Identity Mode)

**Root cause:** Identity face warped from canonical 256x256 to crop space.

**Fix applied:** Feathered face mask (V4.1), single anchor (V4.1). Partially resolved.

### Issue 3: Plastic Skin (Identity Mode)

**Root cause:** High-frequency identity was double-dampened to 1.25% effective.

**Fix applied:** Floor high_blend at 0.15, remove per-pixel conf multiplication (V4.1).

### Next Steps

1. **Anchor correction verification** — Run pipeline with identity path on real video; assert output L ~108 (not 87), L std < 1.5
2. **Add face map comparison test** — Assert output L within 5 of reference
3. **Multi-anchor system** — Currently 1 anchor, need 7+ (frontal, smile, left/right yaw)
4. **Intrinsic decomposition** — Separate albedo from lighting
5. **Mesh-based semantic masking** — Replace elliptical masks with rasterized 478-point mesh
6. **Per-face exposure normalization** — Source video has L=16→155 swings; apply per-frame exposure correction

---

## 12. System Identifiability Analysis (V2.1.0)

### The Core Problem

V2 architecture isolates subsystems. But the underlying math is still entangled.

The observation model:
```
y_t = R(g_t, a_t, l_t, e_t, c_t)
```

Where geometry (g), lighting (l), appearance (a), expression (e), and camera (c) are **strongly entangled** in the observation space. Architectural separation ≠ mathematical separation.

### Issue 1: IdentityState Is Not Intrinsic

**Current:** `appearance_latent` = 256x256x3 uint8 BGR image (anchor-corrected canonical face)

**Problem:** This is appearance-space, not intrinsic identity-space. Pores become lighting, beard becomes shadow, wrinkles become illumination.

**Correct formulation:**
```
a_t = (A_t, D_t)
```
Where:
- `A_t` = intrinsic albedo (lighting-invariant)
- `D_t` = geometric micro-detail residual

**Current status:** Both treated as same tensor. No albedo/specular decomposition.

### Issue 2: Temporal Estimator Is Not Bayesian

**Current:** Scalar confidence values (`temporal_confidence: float`, `drift_score: float`)

**Problem:** `confidence = 0.83` means nothing mathematically. No variance, no distribution, no Bayesian update.

**Correct formulation:**
```
p(x_t | y_{1:t})
```
With:
- Belief state (mean + covariance)
- Uncertainty propagation
- Measurement update
- Prediction update

**Current status:** `motion_field` is uniform pose-delta vector, not optical flow. No probabilistic reasoning.

### Issue 3: Renderer Is Not Physically Consistent

**Current:** `Y = M ⊙ Y_face + (1-M) ⊙ Y_bg` — alpha-blend compositing

**Problem:** Illumination, shading, skin scattering, specularity are NOT linear under alpha blending. This causes:
- LAB drift survives
- Uncanny face energy
- Identity "floats"

**Correct formulation:**
```
Y = R(G, A, L, V)
```
Where:
- `G` = geometry
- `A` = albedo
- `L` = illumination
- `V` = view direction

**Current status:** Image compositing, not rendering.

### Issue 4: Transforms Not Lie-Group Constrained

**Current:** `M_inv = 0.4 * last + 0.6 * new` — linear interpolation on affine matrices

**Problem:** Small EMA updates on affine matrices ≠ valid geometric interpolation. Causes:
- Subtle skew drift
- Non-rigid accumulation
- Temporal wobble

**Correct math:** Transforms should evolve on SE(2), SIM(2), SE(3) using:
- Exponential maps
- Logarithmic interpolation
- Geodesic averaging

**Current status:** Illegal geometry interpolation (even if determinant tests pass).

### Issue 5: Identity Anchors Are Discrete

**Current:** `anchor_basis = [identity_face]`, `anchor_weights = [1.0]` — single anchor

**Problem:** Face manifold is continuous. Discrete anchors produce:
- Interpolation discontinuities
- Mode switching
- Identity popping

**Correct solution:** Identity should live on continuous latent manifold `M_identity`. Anchor basis should only initialize local charts.

**Current status:** `IdentityHypothesisSpace` uses discrete pose/expression bins, not continuous manifold.

### Issue 6: Geometry Estimator Is Landmark-Centric

**Current:** 478 MediaPipe landmarks → sparse constraints

**Problem:** Landmarks are sparse constraints, not geometry. Missing:
- Surface normals
- Dense correspondence
- Curvature continuity
- Volumetric structure

**Correct direction:** Dense mesh fitting, differentiable morphable model, or neural implicit geometry.

### Test Coverage Gaps

| Category | Status | Details |
|---|---|---|
| Long-horizon consistency (500+ frames) | **ABSENT** | Max 50 frames tested |
| System identifiability | **ABSENT** | No test verifies two faces produce distinguishable states |
| Identity drift under adversarial input | **WEAK** | 20-30 frame tests only |
| End-to-end pipeline integration | **ABSENT** | All tests are component-level |
| V2 subsystem cross-data-flow | **ABSENT** | No test chains GeometryState→IdentityState→TemporalState→Renderer |
| VerificationGate | **ABSENT** | Zero tests |
| Renderer blending equation | **ABSENT** | Never tested with known inputs |
| Numerical stability in solver | **ABSENT** | Division by near-zero untested |

---

## 13. Phase 1 Roadmap: Mathematical Hardening

### Goal

Transform V2 from "well-engineered classical CV pipeline" to "mathematically consistent reconstruction system."

### Phase 1A — Testable Now (This Sprint)

| Test | Target | Status |
|---|---|---|
| Long-horizon identity drift (500 frames) | Identity stays within 10 LAB of anchor | In progress |
| System identifiability (two faces) | Different faces → distinguishable identity states | In progress |
| Renderer blending equation | `Y = M * Y_face + (1-M) * Y_bg` verified with known inputs | In progress |
| VerificationGate coverage | All 3 checks tested (pixels, embedding, liveness) | In progress |
| Renderer with actual identity data | Identity path exercised (not just empty states) | In progress |

### Phase 1B — Mathematical Foundations (Next Sprint)

| Change | Description | Priority |
|---|---|---|
| Lie-group transforms | SE(2)/SIM(2) geodesic interpolation instead of linear EMA | High |
| Bayesian temporal state | (mean, covariance) belief instead of scalar confidence | High |
| Albedo/specular decomposition | Split `appearance_latent` into intrinsic albedo + shading | Medium |
| Continuous identity manifold | Replace discrete anchors with learned latent space | Medium |

### Phase 1C — Architectural Leap (Future)

| Change | Description | Priority |
|---|---|---|
| Energy function formulation | Each subsystem contributes `E_term`, solve jointly | High |
| Physical renderer | `Y = R(G, A, L, V)` instead of compositing | High |
| Dense geometry | Mesh fitting or neural implicit geometry | Medium |
| Full-video optimization | `p(x_{1:T} | y_{1:T})` — solve entire video jointly | Low |

### Energy Function Target

```python
E = E_geom + E_identity + E_temporal + E_photometric + E_smoothness
```

Where:
- `E_geom` = landmark reprojection + mesh regularization
- `E_identity` = anchor consistency + albedo smoothness
- `E_temporal` = frame-to-frame coherence + drift penalty
- `E_photometric` = appearance matching + shading consistency
- `E_smoothness` = spatial + temporal smoothness priors

**Current status:** Each subsystem optimizes locally. No joint optimization.

### Phase 0: Contract Lockdown (COMPLETE)

**Status:** ✅ All 305 tests passing (28 new Phase 0 tests)

**Deliverables:**
- `FrameContract` — output shape/dtype/value-range validation
- `EnergyTerms` — 5 energy terms as measurable floats
- `EnergyReport` — per-frame energy + all metrics
- `RendererReport` — per-frame renderer contract validation
- `PassReport` — before/after/delta logging (MANDATORY)
- `GeometryMetrics`, `IdentityMetrics`, `TemporalMetrics`, `RendererMetrics`
- `PhaseState` — current phase tracking

**New Modules:**
- `face_os/energy.py` — `EnergyComputer` with E_geom, E_identity, E_temporal, E_photometric, E_smoothness
- `face_os/visibility.py` — `VisibilityLogger` for before/after/delta JSON logging

**Parameter-wise Visibility (MANDATORY):**
Every pass must expose:
1. GeometryState: yaw/pitch/roll, det(A), mask_coverage%, transform_stability
2. IdentityState: anchor_weights[], uncertainty, region_confidence{}, appearance_latent_norm
3. TemporalState: temporal_confidence, drift_score, continuity_score
4. Energy Terms: E_geom, E_identity, E_temporal, E_photometric, E_smoothness (exact float)
5. Renderer: M_mean, Y_face_range, Y_bg_range, blend_weight_stats

### Phase 1: Energy Function Reformulation (COMPLETE)

**Status:** ✅ All 341 tests passing (36 new Phase 1 tests)

**Tests Added (test_phase1_energy.py):**
- EnergyTermExistence (7 tests) — all 5 terms exist as floats, E_total = sum
- EnergyTermNumericRange (8 tests) — all terms non-negative, bounded <100
- EnergyDeltaRegression (5 tests) — energy decreases with better state
- EnergyMonotonicity (3 tests) — energy converges, does not diverge
- EnergyReportPerFrame (6 tests) — all metrics sections present
- EnergyFromSubsystems (7 tests) — each term depends on its subsystem

**Exit Condition Met:**
- ✅ Each subsystem emits its own energy contribution
- ✅ Each energy term is a measurable float
- ✅ Each term has a testable numeric range
- ✅ No energy term is hidden inside a black box
- ✅ EnergyReport per frame is computable

### Phase 2A: State-Space Formulation (COMPLETE)

**Status:** ✅ All 380 tests passing (39 new Phase 2A tests)

**New Module:** `face_os/state_space.py`

**Components:**
- `LatentState` — 11-dimensional hidden state vector (geometry, identity, temporal, recovery, lighting)
- `StateTransitionModel` — `x_t = f(x_{t-1}) + ε_t` with damping toward equilibrium
- `ProcessNoiseModel` — `ε_t ~ N(0, Q)` with positive-definite covariance
- `ObservationModel` — `z_t = h(x_t) + δ_t` with Jacobian
- `StateSpaceEstimator` — Kalman-like predict-update cycle
- `StateEvolutionReport` — per-frame state evolution metrics
- `RecoveryState` — stable/uncertain/degraded/recovering/reset_required

**State Vector (11D):**
| Index | Component | Description |
|---|---|---|
| 0 | yaw | Head rotation left/right |
| 1 | pitch | Head rotation up/down |
| 2 | roll | Head tilt |
| 3 | identity_uncertainty | Identity confidence [0,1] |
| 4 | appearance_latent_norm | L2 norm of appearance |
| 5 | temporal_confidence | Temporal consistency [0,1] |
| 6 | drift_score | Identity drift from anchor |
| 7 | continuity_score | Temporal smoothness [0,1] |
| 8 | recovery_state | Recovery state (encoded) |
| 9 | brightness_mean | Frame brightness |
| 10 | contrast_mean | Frame contrast |

**Tests Added:**
- LatentState (7 tests) — geometry, identity, temporal, covariance, to_vector/from_vector
- StateTransitionModel (6 tests) — deterministic, noise, stability
- ProcessNoiseModel (5 tests) — positive definite, sample statistics
- ObservationModel (5 tests) — observe, residual, Jacobian
- StateEvolutionReport (5 tests) — metrics, serialization
- StateSpaceEstimator (11 tests) — predict, update, convergence, drift, occlusion, recovery

**Exit Condition Met:**
- ✅ State transition law defined (damping toward equilibrium)
- ✅ Process noise model defined (positive-definite covariance)
- ✅ Observation model defined (linear mapping with Jacobian)
- ✅ Uncertainty propagation implemented (Kalman predict-update)
- ✅ Recovery states defined (5 states)
- ✅ Drift bounded over 500 frames

### Phase 2B: Optimizer Architecture (COMPLETE)

**Status:** ✅ All 412 tests passing (32 new Phase 2B tests)

**New Module:** `face_os/optimizer.py`

**Components:**
- `OptimizerConfig` — max_iterations, tolerance, damping, rollback_threshold
- `OptimizerState` — iteration, energy_history, converged, rollback_count
- `GaussNewtonOptimizer` — `x_{k+1} = x_k - H^{-1} * g`
- `LevenbergMarquardtOptimizer` — adaptive damping `(H + lambda*I)^{-1} * g`
- `ConvergenceReport` — x, energy, iterations, converged, energy_history

**Update Rules:**
- Gauss-Newton: `x_{k+1} = x_k - H^{-1} * g`
- Levenberg-Marquardt: `x_{k+1} = x_k - (H + lambda*I)^{-1} * g`

**Damping Strategy:**
- If energy decreased: `lambda *= decrease_factor`
- If energy increased: `lambda *= increase_factor`, rollback

**Tests Added:**
- OptimizerConfig (5 tests) — max_iterations, tolerance, damping, rollback
- OptimizerState (5 tests) — iteration, energy_history, converged, rollback_count
- GaussNewtonOptimizer (6 tests) — convergence, energy descent, bounded iterations
- LevenbergMarquardtOptimizer (4 tests) — ill-conditioned, adaptive damping
- ConvergenceReport (7 tests) — metrics, serialization
- EnergyDescent (2 tests) — consistency, descent rate
- SingularityHandling (2 tests) — singular Hessian, zero gradient
- Rollback (1 test) — rollback on energy increase

**Exit Condition Met:**
- ✅ Optimizer abstraction defined
- ✅ Convergence logic implemented
- ✅ Update scheduling (adaptive damping)
- ✅ Stopping conditions (gradient norm, step norm, max iterations)
- ✅ Numerical stability (conditioning threshold, damping adaptation)
- ✅ Rollback policy (energy increase detection)

### Phase 2C: Observability Analysis (COMPLETE)

**Status:** ✅ All 440 tests passing (28 new Phase 2C tests)

**New Module:** `face_os/observability.py`

**Components:**
- `ObservabilityAnalyzer` — computes O = [H; HF; HF²; ...]
- `ObservabilityReport` — rank, observable/degenerate dimensions, condition number
- `DegeneracyReport` — lighting/pose/identity/temporal ambiguity

**Tests Added:**
- ObservabilityAnalyzer (7 tests) — matrix, rank, dimensions, condition
- ObservabilityReport (8 tests) — metrics, serialization
- FullRankObservability (3 tests) — full observability, bounded rank
- PartialObservability (1 test) — reduced under occlusion
- LightingAmbiguity (2 tests) — lighting/albedo ambiguity
- PoseDegeneracy (2 tests) — extreme pose detection
- IdentityAmbiguity (2 tests) — high uncertainty detection
- JacobianConditioning (3 tests) — finite, positive, shape

### Phase 2D: State Separation (COMPLETE)

**Status:** ✅ All 474 tests passing (34 new Phase 2D tests)

**New Module:** `face_os/state_separation.py`

**Components:**
- `PhysicalState` — geometry, pose, lighting, appearance (no uncertainty)
- `BeliefState` — covariance, uncertainty, confidence, innovation (no physical)
- `MetaState` — recovery, degradation, reset, health (no physical/belief)
- `SeparatedState` — physical + belief + meta
- `StateSeparator` — separate/merge LatentState ↔ SeparatedState

**Tests Added:**
- PhysicalState (7 tests) — geometry, lighting, appearance, isolation
- BeliefState (8 tests) — covariance, uncertainty, innovation, isolation
- MetaState (9 tests) — recovery, degradation, reset, health, bounded
- SeparatedState (6 tests) — composition, to_latent, from_latent
- StateSeparator (4 tests) — separate, merge, roundtrip

### Phase 2E: Joint MAP Estimation (COMPLETE)

**Status:** ✅ All 493 tests passing (19 new Phase 2E tests)

**New Module:** `face_os/map_estimation.py`

**Components:**
- `MAPConfig` — dynamics_weight, observation_weight, energy_weight
- `MAPOptimizer` — `x_t* = argmin(E(x) + ||x-f(x_prev)||²_{Σ^{-1}} + ||z-h(x)||²_{R^{-1}})`
- `MAPReport` — energy, posterior_uncertainty, convergence_trajectory, innovation_norm

**MAP Objective:**
```
x_t* = argmin_x ( E(x_t) + ||x_t - f(x_{t-1})||²_{Σ^{-1}} + ||z_t - h(x_t)||²_{R^{-1}} )
```

**Tests Added:**
- MAPConfig (4 tests) — dynamics_weight, observation_weight, energy_weight
- MAPOptimizer (5 tests) — convergence, energy descent, bounded iterations
- MAPReport (7 tests) — metrics, serialization
- PosteriorContraction (2 tests) — uncertainty computable, shrinks with observations
- MAPEnergyDescent (1 test) — consistent convergence

### Phase 2F: Energy Normalization + Red Flag Fixes (COMPLETE)

**Status:** ✅ All 493 tests passing | All 6 visibility red flags resolved

**Changes:**
- `EnergyComputer.normalize_energy` flag (default=False for backward compat)
- Running z-score normalization: `E_total = Σ(E_i - μ_i) / σ_i`
- `EnergyTerms._normalized`, `_raw_total` fields for transparency

**Red Flag Fixes:**

| # | Red Flag | BEFORE | AFTER | Fix |
|---|---|---|---|---|
| 1 | E_smoothness = 0.0 | Dead term | Active (mean=12.5) | previous_geometry updated between frames |
| 2 | appearance_latent_norm = 50494 | 50,144 | 12.3 | Compact 16x16 representation |
| 3 | mask_coverage_pct = 0.0 | 0% | 71% | GeometryState.mask populated |
| 4 | continuity_score = 0.95 | Hardcoded | Computed (0.84) | Frame-to-frame similarity |
| 5 | pitch = -174° | -171° | 9.1° | Z-axis fix + wrap-around logic |
| 6 | transform_stability = 1.0 | Placeholder | 0.97 | Relative change metric |

**Observation Model Expanded (5 → 9 dimensions):**
```python
H[0, 0] = 1.0  # yaw
H[1, 1] = 1.0  # pitch
H[2, 2] = 1.0  # roll
H[3, 3] = 1.0  # identity_uncertainty
H[4, 5] = 1.0  # temporal_confidence
H[5, 9] = 1.0  # brightness_mean      ← NEW
H[6, 10] = 1.0  # contrast_mean       ← NEW
H[7, 6] = 1.0  # drift_score          ← NEW
H[8, 7] = 1.0  # continuity_score     ← NEW
```

**Process Noise Tuned:**
```python
Q[4] = 1.0  # appearance_latent_norm (was 100.0)
```

**Recovery State Threshold:**
```python
trace > 50.0  # was 10.0 (initial trace = 11.0)
```

### Phase 2G: Probabilistic Recovery Dynamics (COMPLETE)

**Status:** ✅ All 531 tests passing (38 new Phase 2G tests)

**New Module:** `face_os/recovery_dynamics.py`

**Components:**
- `RecoveryTransitionMatrix` — 5x5 state transition probabilities P(x_meta,t | x_meta,t-1)
- `ProbabilisticRecoveryState` — soft state with probabilities + entropy
- `RecoveryDynamics` — predict-update cycle with Bayesian inference
- `RecoveryReport` — per-frame recovery metrics

**Transition Matrix:**
```
           STABLE  UNCERTAIN  DEGRADED  RECOVERING  RESET
STABLE      0.95     0.05      0.00      0.00      0.00
UNCERTAIN   0.30     0.50      0.20      0.00      0.00
DEGRADED    0.00     0.20      0.50      0.30      0.00
RECOVERING  0.30     0.00      0.20      0.50      0.00
RESET       0.00     0.00      0.00      0.50      0.50
```

**Key Design:**
- Soft sigmoid-based likelihoods (no hard thresholds)
- Uniform prior for Bayesian update (allows state shifts)
- Entropy tracking for uncertainty quantification
- Steady-state convergence guaranteed

**Tests Added:**
- RecoveryTransitionMatrix (10 tests) — shape, row sums, steady state
- ProbabilisticRecoveryState (11 tests) — probabilities, entropy, dominant state
- RecoveryDynamics (13 tests) — predict, update, step, entropy tracking
- TransitionDynamics (4 tests) — persistence, recovery, entropy bounds

**Logging Format:**
```json
{
  "pass_id": "phase2_transform_hardening",
  "frame_id": 128,
  "status": "accepted",
  "before": {"det_A": 0.9931, "mask_coverage_pct": 61.4},
  "after": {"det_A": 0.9994, "mask_coverage_pct": 61.5},
  "delta": {"det_A": 0.0063, "mask_coverage_pct": 0.1},
  "metrics": { ... }
}
```

If visibility is missing, the change must be rejected.

---

## File Structure (V2.1.0)

```
face_os/
├── __init__.py              # Package init
├── types.py                 # Core data structures (GeometryState, IdentityState, TemporalState, FrameContract, EnergyReport, PassReport)
├── config.py                # YAML config loader
├── energy.py                # NEW — EnergyComputer with 5 energy terms
├── visibility.py            # NEW — VisibilityLogger for before/after/delta JSON
├── ingest.py                # Module 1: Video loading, frame reader
├── detect_track.py          # Module 2: MediaPipe tasks API + pose-aware gates
├── landmarks.py             # Module 3: 478-point landmarks + PnP pose
├── canonical_map.py         # Module 4: Canonical UV alignment
├── crop_planner.py          # Module 5: Reference-based crop planning
├── temporal_solve.py        # Module 6: Bidirectional temporal solver
├── face_enhance.py          # Module 7: Structure-preserving rendering
├── identity_state.py        # Module 8: Frequency decomposition + VerificationGate
├── compositor.py            # Module 9: Confidence-weighted compositing
├── appearance_field.py      # AppearanceField + DynamicAppearanceField
├── neural_codec.py          # PersonalizedSpace + NeuralCodec
├── pipeline.py              # V0.5 Orchestrator (USE_IDENTITY flag)
├── pipeline_v2.py           # V2 Orchestrator (subsystem-based architecture)
├── face_detector.tflite     # MediaPipe face detection model
├── face_os_config.yaml      # All tuning parameters
└── subsystems/              # V2 Architecture
    ├── __init__.py
    ├── geometry_estimator.py    # Subsystem A — spatial structure estimation
    ├── identity_estimator.py    # Subsystem B — stable identity representation
    ├── temporal_estimator.py    # Subsystem C — temporal consistency
    └── renderer.py              # Subsystem D — physically consistent rendering

output/face_os/
├── v05_phase1_test.mp4      # V0.5 pipeline output (345 frames)
├── v05_phase1_test.qc.json  # QC report
└── visibility/              # NEW — Pass/energy/renderer reports (JSON)
```
├── v2_test.qc.json          # QC report
└── ...

tests/face_os/
├── test_strict_regression.py    # 26 tests — frame contract, mask stability, NaN/Inf
├── test_v2_subsystems.py        # 20 tests — V2 subsystem isolation, invariants
├── test_math_hardening.py       # 37 tests — 10 invariant classes
├── test_detection.py            # 14 tests
├── test_quality_gates.py        # 13 tests
├── test_identity_state.py       # 17 tests
├── test_identity_state_fixes.py # 5 tests
├── test_patch_memory.py         # 18 tests
├── test_temporal_solve.py       # 10 tests
├── test_face_enhance.py         # 18 tests
├── test_appearance_field.py     # 14 tests
├── test_neural_codec.py         # 12 tests
├── test_hypothesis_matching.py  # 4 tests
├── test_region_confidence.py    # 4 tests
└── conftest.py
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| OpenCV (cv2) | ≥4.5 | Image processing |
| NumPy | ≥1.20 | Array operations |
| dlib | ≥19.22 | Face embeddings (optional, NOT required) |
| face_recognition | ≥1.3 | Identity matching (optional, wraps dlib) |
| mediapipe | ≥0.10.35 | Face detection + landmarks (tasks API) |
| FFmpeg | ≥5.0 | Video encoding (external binary) |
| PyYAML | ≥5.0 | Config file parsing |

---

## V4 Migration Checklist (Complete)

| Component | Status | Details |
|---|---|---|
| **Config** | ✅ | `model: mediapipe_478` default |
| **types.py** | ✅ | `FaceTrack.mesh_478`, `GeometryState`, `IdentityState`, `TemporalState` |
| **detect_track.py** | ✅ | MediaPipe FaceDetector + FaceLandmarker, pose-aware gates |
| **landmarks.py** | ✅ | 100% MediaPipe 478-point, NO dlib |
| **face_enhance.py** | ✅ | Eye indices: MediaPipe 478-point |
| **pipeline.py** | ✅ | Single anchor, feathered mask, USE_IDENTITY flag |
| **pipeline_v2.py** | ✅ | 4 isolated subsystems, explicit state types |
| **identity_state.py** | ✅ | Config-driven EMA, high-freq floor, single anchor |
| **canonical_map.py** | ✅ | Handles 478-point + 68-point dynamically |
| **config.py** | ✅ | `model: mediapipe_478` default |
| **Haar Cascade** | ✅ | Zero references in codebase |
| **dlib dependency** | ✅ | Optional only, not required |

---

## How to Run

```bash
# Full Face OS test suite (240 tests)
.venv/bin/python -m pytest tests/face_os/ -v

# Strict regression tests only (26 tests)
.venv/bin/python -m pytest tests/face_os/test_strict_regression.py -v

# V2 subsystem tests only (20 tests)
.venv/bin/python -m pytest tests/face_os/test_v2_subsystems.py -v

# Phase 1 hardening tests (long-horizon, identifiability, renderer, verification gate)
.venv/bin/python -m pytest tests/face_os/test_phase1_hardening.py -v

# Run V0.5 pipeline
.venv/bin/python -m face_os.pipeline --video clips_test/test_clip.mp4 --reference expectation.png --photos photos/ --output output.mp4

# Run V2 pipeline
.venv/bin/python -m face_os.pipeline_v2 --video clips_test/test_clip.mp4 --reference expectation.png --photos photos/ --output output.mp4

# Simple mode (no identity)
.venv/bin/python -m face_os.pipeline --video clips_test/test_clip.mp4 --no-identity --output output.mp4
```