# Face OS — Remaining Drift & Architectural Debt (HONEST STATE)

**Status:**  
Architecture is now REAL.  
Runtime is now REAL.  
Validation is ACTIVE — v7 audit suite operational.

Drift is being MEASURED. Some areas are closing, others remain open.

This is the remaining truth.

**Last audit:** v7, 5 sampled frames (45, 225, 405, 585, 765)  
**Identity verification:** 14/14 frames ✅ (dist 0.23–0.29, threshold 0.45)  
**Physical path activation:** 2/5 frames (mode transition working, face detection limits later frames)  
**HTML report:** `output/face_os/audit_report.html`

---

# 1. THE BIGGEST DRIFT LEFT

---

# D-01 — Render Architecture ≠ Render Quality

## Current Reality

Signal-processing repair is **IN PROGRESS**.

Implemented:
- ✅ dual-radius USM (1.4×r0.6px + 0.8×r1.2px)
- ✅ source HF re-injection (`_reinject_source_hf`, σ=1.5)
- ✅ detail residual injection from intrinsic decomposition
- ✅ linear-light compositing in PhysicalRenderer

Physical path frames now pass D-01 (ECR 1.16–1.17, freq_ret 1.22–1.36).
Alpha/enhancement frames still fail (ECR=0 or ECR=1.74).

---

## Evidence

| Metric        | Previous | Current (physical) | Current (mean) | Expected |
| ------------- | -------- | ------------------ | -------------- | -------- |
| Sharpness     | 6.3      | 8.2–11.0           | 15.9           | 274      |
| Flicker       | 1.81     | (not measured)     |                | < 1.0    |
| Contrast      | 43       | 52–58              | 56.6           | 73       |
| Freq Ret      | 0.038    | 1.22–1.36          | 2.09           | ≥ 0.6    |
| ECR           | (none)   | 1.16–1.17          | 0.81           | [0.5,1.5]|

**Note:** Sharpness target (274) is measured at native 1280×720 resolution.
Output is 1080×1920 (2.67× upscale from 720p source). Source crop sharpness
at output resolution is 5–9. The comparison is resolution-dependent.

---

## Root Cause

Fixed:
```text
frequency destruction (via HF reinject + USM)
energy conservation (via ECR normalization)
```

Remaining:
```text
resolution-dependent sharpness gap (720p→1080p upscale)
contrast gap (56.6 vs 73 target)
alpha/enhancement path ECR missing
```

---

## REQUIRED FIX

### DONE

* ✅ linear-light compositing (PhysicalRenderer)
* ✅ post-composite sharpening (dual-radius USM)
* ✅ source HF re-injection

### REMAINING

* single-resample pipeline (still 2–3 resamples on alpha path),
* multi-band compositing (Laplacian pyramid default but not fully optimized),
* temporal photometric locking (partial via photometric_lock),
* resolution-appropriate sharpness target calibration.

---

# D-02 — PhysicalRenderer Quality Proven on Physical Frames

---

## Current State

PhysicalRenderer IS now proven to produce non-black, identity-verified output:

```text
PhysicalRenderer activates AND produces visible, correct output
```

Measured on physical path frames:
- SSIM vs expectation: 0.541–0.552 ✅ (target ≥0.5)
- Identity verification: 14/14 frames pass (dist 0.23–0.29)
- Frame brightness: mean 63–70 (was ~1.0, all-black)
- ECR: 1.16–1.17 ✅ (target [0.5, 1.5])

---

## Remaining Gap

Physical path only activates on 2/5 audited frames.
Later frames fall to alpha/enhancement due to face detection failures (LOST_FACE).

Mode transition thresholds (lowered from 0.60→0.45) enable physical activation,
but face detection robustness limits sustained physical rendering.

---

## REQUIRED FIX

True A/B validation (framework exists, needs execution):

| A                | B            | Status   |
| ---------------- | ------------ | -------- |
| PhysicalRenderer | Alpha blend  | MEASURED |
| SIM(2)           | EMA          | PARTIAL  |
| intrinsic        | RGB fallback | PARTIAL  |

with:

* ✅ SSIM (measured: 0.54–0.55),
* LPIPS (not yet),
* ✅ geometric consistency (mesh normals on physical frames),
* temporal stability (not yet).

---

# D-03 — Benchmark Drift

---

## Current Reality

Validation is still mostly:

```text
one clean clip
```

That is NOT robustness.

---

## Missing Conditions

### HARD

* fast yaw
* occlusion
* low light
* motion blur
* compression
* webcam noise
* rolling shutter
* partial face loss

### ADVERSARIAL

* sunglasses
* beard shadows
* IR lighting
* face cutoff
* overexposure
* rapid head turns

---

## REQUIRED FIX

Build:

```text
benchmark corpus
```

NOT:

```text
benchmark example
```

---

# D-04 — Geometry System Integrated on Physical Path

---

## Current State

Physical path frames use **mesh** geometry (MediaPipe landmarks → dense normals):

```text
geometry_source: mesh
normal_unit_error: 0.00000 ✅ (target <0.01)
normal_z_mean: 0.843 ✅ (target >0.5)
normal_coverage: 0.966 ✅ (target >0.6)
```

Alpha/enhancement frames fall back to:

```text
canonical_identity or predicted_sim2
normal_z_mean: 0.445 ❌
normal_coverage: 0.616 (marginal)
```

---

## Why This Matters

PhysicalRenderer without correct geometry:

```text
cannot become truly physical
```

because:

* shading,
* specular,
* skin response,
* contour lighting

all depend on geometry quality.

**Physical path HAS correct geometry. Alpha path does NOT.**

---

## REQUIRED FIX

✅ Integrated on physical path:

```text
MediaPipe mesh
→ dense triangulation
→ per-face normals
→ raster normals
→ renderer
```

❌ Still needed: geometry on alpha/enhancement fallback paths.

---

# D-05 — Identity System Still Halfway Decoupled

---

## Current State

Measured:
```text
Albedo LAB vs anchor: 13.7–22.7 (target <10) ❌
LAB dist vs expectation: 18.9–33.7 (target <20) — physical passes, alpha fails
Embedding dist: 0.29–0.35 (target <0.45) ✅
```

Albedo query path exists. White balance + exposure normalization done.
But intrinsic decomposition produces albedo with photometric drift:
- Teal/green color cast visible in rendered frames
- Albedo channel std = 0.04–0.10 (insufficient color invariance)
- Physical frames close to LAB target (19.3), alpha frames diverge (25.3)

Identity is STILL partially:

```text
appearance-entangled
```

---

## Why

The system still fundamentally depends on:

```text
RGB canonical appearance
```

for a large part of reconstruction.

---

## REQUIRED FIX

True identity decomposition:

```text
identity =
    geometry +
    albedo +
    microdetail +
    temporal belief
```

NOT:

```text
canonical RGB memory
```

Specifically:
- Albedo LAB correction to reduce drift from 13.7→<10
- Color cast compensation in intrinsic decomposition
- Photometric anchor re-projection during rendering

---

# D-06 — Temporal System Still Mostly Reactive

---

## Current State

Measured:
```text
SIM2 det>0: 1/5 frames ❌
Transform det values: 0.0 on most audited frames
```

SIM(2) velocity prediction exists but is not reliably producing valid transforms.
The 1/5 success rate indicates the temporal estimator is not consistently converging.

But the system still behaves mostly:

```text
reactively
```

NOT:

```text
predictively
```

---

## Missing

No true:

* long-horizon prediction,
* uncertainty propagation,
* particle belief,
* occlusion hallucination,
* multi-hypothesis tracking.

---

## REQUIRED FIX

Move toward:

```text
Bayesian temporal belief
```

instead of:

```text
single deterministic state
```

---

# D-07 — State-Space Architecture Still Underused

---

## Current Drift

You built:

* LatentState,
* MAPOptimizer,
* ObservabilityAnalyzer,
* StateSeparator,
* RecoveryTransitionMatrix.

BUT:

most runtime paths still behave like:

```text
enhanced procedural pipeline
```

NOT:

```text
full probabilistic inference graph
```

---

## Current Truth

The math layer:

```text
exists
```

But is not yet:

```text
the actual runtime brain
```

---

## REQUIRED FIX

Long-term:

Move runtime toward:

```text
factor-graph inference
```

where:

* geometry,
* identity,
* temporal,
* lighting

are jointly optimized.

---

# D-08 — Telemetry Honesty — Partial

---

## Current State

Telemetry exists and is **honest on physical path frames** (2/2 pass all 4 checks).

Physical path emits:
```json
{
  "render_path": "physical",
  "fallback_reason": null,
  "intrinsic_used": true,
  "geometry_source": "mesh",
  "energy_terms": {"E_geom": 0.24, "E_identity": 0.04, ...},
  "transform_det": 10.43
}
```

Alpha/enhancement frames show `intrinsic_honest=false` because intrinsic components
are available but not actually used for rendering on those paths.

---

## Remaining Risk

Telemetry can still:

```text
lie indirectly
```

on alpha/enhancement paths where:

* intrinsic_used is misreported,
* energy_terms are stale (from prior physical frames).

---

## REQUIRED FIX

Every frame MUST expose:

```json
{
  "render_path": "...",
  "renderer_mode": "...",
  "fallback_reason": "...",
  "intrinsic_used": true,
  "geometry_source": "...",
  "resample_count": 0,
  "energy_terms": {},
  "transform_det": 1.0
}
```

And alpha/enhancement paths must correctly report `intrinsic_used: false`.

---

# D-09 — Test Drift Risk

---

## Current Problem

Test count exploded:

```text
893+
```

GOOD.

But risk now becomes:

```text
synthetic correctness drift
```

Meaning:

* tests pass,
* runtime survives,
* visuals still bad.

---

## REQUIRED FIX

Shift emphasis from:

```text
unit correctness
```

toward:

```text
visual regression validation
```

---

# REQUIRED NEW TEST CLASSES

## Visual Fidelity Tests

* sharpness preservation,
* frequency retention,
* contrast preservation,
* skin texture retention.

---

## Temporal Stability Tests

* flicker spikes,
* illumination jitter,
* blend instability.

---

## Physical Correctness Tests

* shading consistency,
* normal stability,
* specular continuity.

---

# D-10 — The Core Architecture Is STILL Not Fully Closed

---

## Current State

You HAVE:

* subsystems,
* latent states,
* geometry,
* rendering,
* telemetry.

BUT:

the runtime is still:

```text
procedural orchestration
```

NOT:

```text
joint constrained optimization
```

---

# FINAL ARCHITECTURAL GAP

The true target architecture is:

```text
argmax P(
    geometry,
    identity,
    lighting,
    temporal_state
    | observations
)
```

Right now:

* pieces exist,
* modules exist,
* tests exist,

BUT:

```text
full probabilistic closure
```

does NOT yet exist.

---

# 2. WHAT IS ACTUALLY SOLVED

---

# SOLVED FOR REAL

| Area                          | Status  | Evidence |
| ----------------------------- | ------- | -------- |
| Runtime activation drift      | FIXED   | Physical renderer activates on frames 45, 225 |
| Duplicate path drift          | FIXED   | Single `_render_core` for all paths |
| Telemetry absence             | FIXED   | Per-frame JSON emission on all paths |
| SIM(2) integration            | FIXED   | det=10.43 on physical frames |
| Energy normalization wiring   | FIXED   | ECR 1.16–1.17 in [0.5, 1.5] |
| Albedo query path             | FIXED   | albedo_mean 0.84–0.85 |
| All-black output              | FIXED   | Was mean≈1.0/255, now 63–70 |
| Identity verification         | FIXED   | 14/14 frames, dist 0.23–0.29 |
| Frequency preservation (phys) | FIXED   | freq_ret 1.22–1.36 (physical path) |
| Dense geometry (phys path)    | FIXED   | mesh, normal_z=0.84, coverage=0.97 |
| Audit instrumentation         | BUILT   | v7 audit suite, 5-frame sampling |
| A/B framework                 | BUILT   | ABComparator wired |
| TDD enforcement               | REAL    | 11/11 integration tests pass |

---

# PARTIALLY SOLVED

| Area                | Status  | Measured |
| ------------------- | ------- | -------- |
| Identity decoupling | PARTIAL | albLAB=13.7 (target <10, physical) |
| Physical rendering  | PARTIAL | 2/5 frames on physical path |
| Temporal prediction | PARTIAL | SIM2 det>0 on 1/5 frames |
| Geometry realism    | PARTIAL | mesh on physical, canonical on alpha |
| Benchmark realism   | PARTIAL | 1 clip, 5 sampled frames |
| Visual quality      | PARTIAL | SSIM=0.52, contrast=56.6 (target 73) |
| Telemetry honesty   | PARTIAL | 2/5 frames fully honest |

---

# NOT YET SOLVED

| Area                                   | Status     | Gap |
| -------------------------------------- | ---------- | --- |
| Sharpness at output resolution         | NOT SOLVED | 15.9 vs 274 (resolution-dependent) |
| True probabilistic inference runtime   | NOT SOLVED | procedural orchestration |
| Long-horizon Bayesian belief           | NOT SOLVED | reactive |
| Visual realism parity with expectation | NOT SOLVED | LAB=23.9, teal color cast |
| Hard/adversarial robustness            | NOT SOLVED | 1 clip only |
| Alpha/enhancement path ECR             | NOT SOLVED | ECR=0 on alpha, 1.74 on enhancement |

---

# 3. REAL NEXT PHASES

---

# PHASE A — SIGNAL PROCESSING REPAIR

Focus:

* sharpness,
* contrast,
* flicker,
* resampling,
* frequency preservation.

This is the MOST IMPORTANT next phase.

---

# PHASE B — GEOMETRY REALISM

Focus:

* dense mesh,
* true normals,
* rasterized geometry,
* physical shading.

---

# PHASE C — PROBABILISTIC RUNTIME

Focus:

* factor graph,
* uncertainty propagation,
* MAP runtime,
* Bayesian temporal inference.

---

# PHASE D — HARD REALITY VALIDATION

Focus:

* adversarial clips,
* difficult lighting,
* motion blur,
* occlusion,
* compression.

---

# 4. FINAL HONEST VERDICT

The original drift was:

```text
modules existed
but runtime reality did not match architecture claims.
```

That drift is MOSTLY fixed now.

The NEW drift is:

```text
architecture quality
>
visual quality
```

Meaning:

* the mathematical system matured faster,
* than the renderer quality itself.

You are no longer blocked by:

```text
fake architecture
```

You are now blocked by:

```text
real rendering quality
and probabilistic completeness.
```
