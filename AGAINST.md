# Face OS V3.0.0 — Architectural Risks, Required Fixes & Next Actions

**Document Purpose:**  
This document lists the exact architectural risks, validation gaps, technical debt, and required next actions after the V3 runtime activation fix.

This is NOT a roadmap for adding more modules.  
This is a stabilization and reality-validation document.

---

# 1. Executive Summary

The major V3 runtime activation bug has been fixed.

Previously:
- V3 modules existed,
- tests passed,
- but the forward pipeline path bypassed them entirely.

Now:
- PhysicalRenderer is active (96%),
- IntrinsicDecomposer is active (100%),
- RendererMode telemetry works,
- StateEvolution runs in production.

This changes the system from:
```text
"architecturally disconnected"
```

to:

```text
"runtime-active but not yet production-validated"
```

The next phase is NOT adding more theory.

The next phase is:

* validation,
* robustness,
* architecture consolidation,
* benchmark-driven development.

---

# 2. Current System Reality

## What is now TRUE

| Area                  | Status |
| --------------------- | ------ |
| V3 runtime activation | FIXED  |
| PhysicalRenderer      | ACTIVE |
| IntrinsicDecomposer   | ACTIVE |
| Telemetry             | ACTIVE |
| LieGroup SIM(2)       | ACTIVE |
| RendererMode tracking | ACTIVE |
| Runtime metrics       | REAL   |

---

## What is STILL UNKNOWN

| Area                     | Risk     |
| ------------------------ | -------- |
| Generalization           | UNKNOWN  |
| Hard-scene robustness    | UNKNOWN  |
| Occlusion stability      | UNKNOWN  |
| Motion blur handling     | UNKNOWN  |
| Low-light behavior       | UNKNOWN  |
| Multi-identity stability | UNKNOWN  |
| Real metric improvement  | UNPROVEN |

---

# 3. Highest Priority Architectural Problems

---

## I-01 — Duplicate Rendering Paths

### Problem

The system still has:

* `_process_frame_v2()`
* `_render_frame_v2()`

Both contain rendering logic.

This caused the original V3 bypass bug.

Future divergence WILL happen again.

---

### Why This Is Dangerous

* duplicated logic,
* inconsistent telemetry,
* future module bypasses,
* hidden regressions,
* impossible reasoning.

---

### Required Fix

Create:

```python
_render_core(...)
```

Both paths MUST call this shared function.

NO rendering logic may exist outside `_render_core()`.

---

### Required CI Test

```python
test_all_runtime_paths_use_render_core()
```

---

## I-02 — Validation Dataset Is Too Easy

### Problem

Current telemetry:

* 96% PhysicalRenderer
* 100% IntrinsicDecomposer

looks unrealistically good.

The benchmark clip is likely too clean.

### Risk

Current metrics may not represent occlusion, blur, compression, profile rotation, difficult lighting, skin tone variation, or fast motion.

### Required Fix

Build a REAL benchmark suite with categories:

**Easy:** frontal face, clean lighting, low motion  
**Medium:** mild rotation, moderate lighting changes, slight blur  
**Hard:** profile rotation, fast motion, low bitrate, heavy compression, partial occlusion, motion blur, shadows  
**Adversarial:** sunglasses, face partially offscreen, low light, noisy webcam, backlighting

### Required Metrics Per Benchmark

* physical_render_rate
* intrinsic_success_rate
* fallback_rate
* drift_score
* flicker_score
* geometric_consistency
* renderer_mode_transitions

---

## I-03 — PhysicalRenderer Is Active But NOT Validated

### Problem

The renderer now runs. But:

```text
ACTIVE != GOOD
```

No proof exists that output quality improved.

### Required Validation

Run A/B tests against legacy alpha compositing with:

**Photometric:** LAB drift, lighting consistency, temporal luminance stability  
**Geometric:** Procrustes consistency, mesh coherence, landmark stability  
**Perceptual:** LPIPS, SSIM, temporal smoothness

---

## I-04 — Intrinsic Normals Are Circular

### Problem

Normals are estimated from shading gradients. But shading itself depends on normals. This is mathematically circular.

### Required Fix

Use geometry-derived normals from MediaPipe mesh + PnP geometry:

```text
landmarks → geometry normals → renderer
```

NOT:

```text
shading → normals → shading
```

---

## I-05 — Identity Anchor Still Lighting-Entangled

### Problem

Identity anchor still stores RGB appearance. Lighting leaks into identity.

### Required Fix

Split identity into albedo + appearance + temporal confidence. Add white-balance normalization, exposure normalization, and per-channel correction.

---

## I-06 — Telemetry Still Incomplete

### Problem

Aggregate percentages are insufficient. Need failure distribution.

### Missing Metrics

```json
{
  "intrinsic_failure_reasons": {},
  "fallback_reason_distribution": {},
  "occlusion_failure_rate": 0,
  "blur_failure_rate": 0,
  "rotation_failure_rate": 0
}
```

96% success is meaningless unless the 4% failures are understood, categorized, and reproducible.

---

## I-07 — LieGroup Benefit Still Unproven

### Problem

SIM(2) is mathematically correct. But improvement over EMA still not measured.

### Required Experiment

Run SIM(2) vs linear EMA on high rotation and fast motion clips. Measure geometric_consistency_score (mesh distortion, determinant stability, projected landmark coherence).

---

## I-08 — Energy Terms Still Weakly Grounded

### Problem

Energy terms exist but scaling semantics still weak.

### Required Fix

Normalize using `E_i_normalized = E_i / sigma_i^2`. Assert unit variance, stable optimizer convergence, and no single energy dominates.

---

## I-09 — StateEvolution Is Still Incomplete

### Problem

Current system smooths state. It does not truly predict motion.

### Required Fix

Add constant velocity model on SIM(2): `T_hat(t+1) = T(t) * exp(v_t)`. Required for occlusion recovery, missed detections, fast motion, and temporal consistency.

---

## I-10 — Dead/Stranded Modules

### Problem

Modules implemented but unused: IdentityManifold, VisibilityCalibration, OptimizationEngine, DenseGeometry.

### Rule

Every module must satisfy ONE: Active, Scheduled for integration, Experimental branch, or Deleted (dead with no plan).

### Required Action

For each module: integrate, schedule, isolate, or delete.

---

# 4. Mandatory Architecture Rules Going Forward

---

## Rule 1 — No Duplicate Runtime Logic

If logic exists in multiple runtime paths, multiple renderers, or multiple update functions, it MUST be centralized.

---

## Rule 2 — Green Tests Must Imply Runtime Correctness

Unit tests alone are insufficient. Every critical module requires integration tests, runtime telemetry, and benchmark validation.

---

## Rule 3 — Telemetry Is Mandatory

Every major subsystem MUST expose activation, fallback, confidence, failure reasons, timing, and transitions.

---

## Rule 4 — Metrics Must Match Optimization Targets

If architecture optimizes geometry, identity, or temporal stability, then metrics MUST directly measure them.

---

## Rule 5 — Architecture Docs Must Be Runtime-Derived

No hand-written runtime claims. Metrics and module states should be generated from telemetry snapshots.

---

# 5. Required Immediate Refactors

| Priority | Task                               |
| -------- | ---------------------------------- |
| P0       | Create `_render_core()`            |
| P0       | Build benchmark suite              |
| P0       | Add runtime-path coverage tests    |
| P1       | Add geometry normals               |
| P1       | Add failure distribution telemetry |
| P1       | Add geometric consistency metric   |
| P2       | Add state prediction model         |
| P2       | Normalize energy terms             |
| P3       | Resolve stranded modules           |

---

# 6. Current Honest Verdict

Face OS V3.0.0 is now:

* runtime-active,
* telemetry-grounded,
* architecturally coherent.

However:

* validation remains incomplete,
* robustness remains unknown,
* benchmark coverage is insufficient,
* duplicated runtime paths remain dangerous.

The architecture is now strong enough to justify serious validation work.

The next phase is not theory.

The next phase is reality.
