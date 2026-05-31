# FACE OS — Mathematical Architecture Specification

Version: 3.0 (Belief-State Lock)
Status: **LOCKED — SOURCE OF TRUTH**
Locked: 2026-05-31
Goal: Convert Face OS from heuristic image-processing pipeline into a mathematically stable latent-state reconstruction system.

> **Lock contract.** This file is the single architectural source of truth for
> `face_os/`. Sections 1–15 are the V2 reformulation (retained verbatim).
> Sections 16–19 are the v3 belief-state lock: they formalize the latent
> identity belief, name every concept with its equation + invariant + required
> test, and record the VERIFIED drift state of the implementation against each
> (audited 2026-05-31 with file:line evidence — see §17 ledger).
>
> Rules bound to this lock:
> - No feature without a failing deterministic test + measurable invariant (§10).
> - No hot-fixes / magic-constant dials to mask a metric. Fix the mechanism the
>   equation names, or change this spec first (then re-lock).
> - Tests encode THIS spec. If a test encodes drifted behavior, the test is
>   rewritten to the spec, not the spec to the test.

---

# 1. Core Philosophy

Face reconstruction is NOT an image-editing problem.

It is a:

- latent-state estimation problem
- constrained geometry problem
- temporal inference problem
- physically consistent rendering problem

The current V0.5.x system still mixes:
- geometry
- lighting
- identity
- temporal smoothing
- compositing

inside RGB-space operations.

This rewrite separates these concerns into explicit mathematical states.

---

# 2. Fundamental Reformulation

Current flawed formulation:

frame_t -> blend -> sharpen -> composite -> output

Correct formulation:

hidden_state_t -> render(hidden_state_t) -> output_t

Where:

x_t = {
    g_t,   # geometry
    p_t,   # pose
    l_t,   # lighting
    e_t,   # expression
    a_t,   # identity appearance
    c_t    # confidence / temporal state
}

Observed frame:

y_t = R(x_t)

The pipeline exists to estimate x_t robustly.

---

# 3. Architectural Principles

## PRINCIPLE 1 — Geometry First

All masks, crops, and warps must derive from geometry.

Forbidden:
- brightness threshold masks
- intensity-derived topology
- image-space heuristics

Required:
- mesh-derived masks
- topology-preserving warps
- explicit coordinate systems

---

## PRINCIPLE 2 — Identity is NOT RGB Memory

Identity must never be represented as EMA-smoothed RGB frames.

Forbidden:
- low-frequency RGB EMA as primary identity state
- frame averaging identity memory

Required:
- latent anchor basis
- manifold interpolation
- region-wise confidence weighting

---

## PRINCIPLE 3 — Rendering is Deterministic

Every pipeline path must satisfy:
- identical output dimensions
- identical dtype
- bounded transform behavior
- deterministic results under fixed seed

---

## PRINCIPLE 4 — Temporal Consistency is a Constraint

Temporal stability is NOT optional post-processing.

It is a hard constraint on:
- geometry
- confidence
- identity
- transforms
- rendering

---

# 4. System Decomposition

The system shall be decomposed into 4 isolated subsystems.

---

# SUBSYSTEM A — GEOMETRY ESTIMATOR

Purpose:
Estimate all spatial structure.

Inputs:
- frame_t
- previous_geometry_state

Outputs:
geometry_state_t

Structure:

geometry_state_t = {
    landmarks_478,
    pose,
    canonical_transform,
    crop_transform,
    mesh,
    semantic_regions,
    mask,
    geometry_confidence
}

Responsibilities:
- landmark extraction
- head pose estimation
- canonical UV mapping
- semantic region construction
- crop optimization
- warp transform generation

Forbidden:
- identity logic
- lighting logic
- RGB blending

---

# SUBSYSTEM B — IDENTITY ESTIMATOR

Purpose:
Estimate stable identity representation independent of lighting and pose.

Current V0.5 flaw:
RGB EMA identity memory.

Replace with:

identity_state_t = {
    anchor_basis,
    anchor_weights,
    appearance_latent,
    region_confidence,
    identity_uncertainty
}

Identity representation:

a_t = Σ(w_k * a_k)

Where:
- a_k are learned/selected anchor states
- w_k are confidence-normalized interpolation weights

Required anchor dimensions:
- frontal neutral
- left yaw
- right yaw
- smile
- low-light
- high-light
- blink
- beard-shadow

Forbidden:
- RGB EMA blending
- raw frame accumulation
- frame-space averaging

---

# SUBSYSTEM C — TEMPORAL ESTIMATOR

Purpose:
Maintain temporal consistency.

Outputs:

temporal_state_t = {
    motion_field,
    temporal_confidence,
    drift_score,
    continuity_score,
    smoothing_constraints
}

Responsibilities:
- bidirectional smoothing
- confidence propagation
- optical-flow consistency
- identity continuity
- geometry continuity

Critical rule:
Temporal system updates CONFIDENCE, not raw texture.

Forbidden:
- backward texture injection
- frame averaging
- temporal blur accumulation

---

# SUBSYSTEM D — RENDERER

Purpose:
Generate physically consistent output.

Inputs:
- geometry_state_t
- identity_state_t
- temporal_state_t

Outputs:
- rendered_face
- background_layer
- composite_output

Render equation:

Y = M ⊙ Y_face + (1 - M) ⊙ Y_bg

Where:
- M is geometry-derived semantic mask
- Y_face is latent-rendered face
- Y_bg is untouched background

Forbidden:
- RGB-space rescue compositing
- heuristic face merging
- implicit blending logic

---

# 5. Coordinate System Reform

All transforms must operate in explicit coordinate spaces.

Required spaces:

1. source_frame_space
2. crop_space
3. canonical_uv_space
4. render_space
5. output_space

Every transform must declare:
- source space
- target space
- determinant
- scale bounds
- invertibility

Transform chain:

W =
    T_output
    ∘ T_render
    ∘ T_uv
    ∘ T_pose
    ∘ T_crop

---

# 6. Mesh-Based Semantic Masking

Current flaw:
Elliptical geometry mask.

Replace with:
- rasterized 478-point semantic mesh
- triangle-based region filling
- signed distance field edge feathering

Mask generation:

M = Rasterize(mesh_478)

Edge softness:
SDF-based only.

Forbidden:
- Gaussian feather as topology definition
- threshold-based masks
- ellipse approximation as final system

---

# 7. Crop Planning Reformulation

Crop planning is an optimization problem.

Define:

C* = argmin(E_crop)

Where:

E_crop =
    α * face_alignment_error
  + β * head_cutoff_penalty
  + γ * temporal_motion_penalty
  + δ * composition_error

Constraints:
- fixed output aspect ratio
- face fully contained
- forehead preserved
- bounded crop acceleration

Crop planner must NEVER emit raw fallback frames.

All fallback paths MUST:
- preserve geometry contract
- preserve output dimensions
- preserve coordinate conventions

---

# 8. Lighting Separation

Current flaw:
Identity corrupted by lighting.

Required:
intrinsic decomposition.

Each frame decomposes into:

frame =
    albedo
    × shading
    + specular

Identity memory stores:
- albedo only

Lighting stored separately.

Forbidden:
- lighting baked into identity state
- RGB identity averaging

---

# 9. Mathematical Invariants

Every subsystem must expose measurable invariants.

---

## Geometry invariants

- no triangle inversion
- bounded local scale distortion
- bounded shear
- bounded reprojection error
- round-trip UV consistency

---

## Identity invariants

- bounded embedding drift
- anchor weight normalization
- confidence monotonicity
- pose consistency

---

## Temporal invariants

- bounded crop velocity
- bounded landmark acceleration
- optical-flow coherence
- no temporal flicker spikes

---

## Rendering invariants

- fixed output size
- fixed dtype
- no NaN/Inf
- bounded pixel range
- deterministic under fixed seed

---

# 10. STRICT TDD REQUIREMENTS

No feature may be added without:
1. failing deterministic test
2. measurable invariant
3. regression lock

---

# 11. Required Regression Tests

## Geometry tests

- UV roundtrip reconstruction
- triangle inversion detection
- warp determinant sanity
- reprojection consistency
- crop continuity

---

## Mask tests

- lighting invariance
- semantic region continuity
- mesh topology stability
- mask IoU stability

---

## Temporal tests

- optical-flow shimmer
- landmark drift bounds
- identity continuity
- confidence convergence

---

## Rendering tests

- output contract invariance
- deterministic rendering
- no dtype mutation
- no shape mutation
- no invalid pixel ranges

---

# 12. Forbidden Patterns

NEVER:
- use RGB EMA identity averaging
- use intensity threshold masks
- silently change output geometry
- use hidden fallback branches
- blend identity in RGB space as primary logic
- use compositor to repair upstream errors
- allow transform ambiguity
- allow unbounded temporal smoothing

---

# 13. Refactor Order

PHASE 1
- isolate geometry subsystem
- unify coordinate spaces
- explicit transform graph
- mesh rasterization masks

PHASE 2
- remove RGB identity EMA
- implement anchor-basis identity state
- separate albedo from lighting

PHASE 3
- rebuild temporal estimator
- confidence propagation only
- optical-flow consistency metrics

PHASE 4
- renderer rewrite
- layer-based compositing
- deterministic rendering guarantees

PHASE 5
- visual regression suite
- invariant dashboard
- stress testing

---

# 14. Long-Term Goal

Final system target:

Face OS becomes:

- latent-state estimator
- geometry-constrained renderer
- temporally stable identity reconstruction system

NOT:
- image filter pipeline
- heuristic compositor stack
- RGB blending engine

---

# 15. Definition of Success

The rewrite succeeds only if:

- identity remains stable under lighting variation
- masks are topology-derived
- transforms are mathematically bounded
- rendering is deterministic
- temporal drift is measurable and constrained
- all regressions are caught numerically
- compositor becomes a trivial final assembly step

If compositor complexity grows,
the upstream architecture is still wrong.

---
---

# 16. Belief-State Formalism (v3 LOCK)

The latent identity is a BELIEF, not an image. Everything below is the explicit
math the runtime must obey. Each concept carries: equation, invariant, required
test. The audited implementation status of each is in the §17 ledger.

---

## 16.1 Observation Model

The source frame is NOT identity. The source frame is a rendered, noisy
observation of identity under a pose and a lighting state.

```
O_t = R(I_t, p_t, l_t) + ε_t
```

Where:
- I_t — true identity (geometry + albedo + microdetail), slow-varying
- p_t — pose
- l_t — lighting / illumination state
- ε_t — sensor noise (observation noise covariance, NOT zero)

Consequences:
- Inference predicts the observation from the latent and compares to the actual
  frame. Decomposition-only checks are NOT the observation model.
- ε_t MUST be represented (observation covariance), so that a single noisy frame
  cannot be trusted as if it were ground truth.

Invariant: a forward predict `Ô_t = R(latent, p_t, l_t)` exists and its residual
`‖O_t − Ô_t‖` is the quantity that drives confidence — not raw pixel similarity.

Required test: forward-model residual is finite, bounded, and decreases as the
latent converges on a held frame.

---

## 16.2 Identity Inertia (THE central principle)

Identity changes far slower than observation.

```
ΔI ≪ ΔO
```

This is the single most important constraint in the system. Observation can
change wildly frame to frame (lighting, motion, noise); identity must not.

Mechanism: recursive belief with an explicit switch cost.

```
I_t = argmin_I [ ‖I − f(O_t)‖²        (data term)
               + switch_cost · ‖I − I_{t-1}‖² ]   (inertia term, switch_cost > 0)
```

A plain EMA is NOT sufficient: an EMA rate resists change uniformly; a switch
cost makes LARGE identity jumps disproportionately expensive while still
admitting small corrections, which is what `ΔI ≪ ΔO` requires.

Invariant: for a bounded observation jump `‖ΔO‖`, the induced identity change
satisfies `‖ΔI‖ ≤ κ·‖ΔO‖` with `κ ≪ 1` (measurable, regression-locked).

Required test: inject a single corrupted/outlier frame into a converged latent;
assert `‖ΔI‖` stays below a hard bound (one weird frame cannot rewrite identity).

---

## 16.3 Identity Drift Energy

Drift is accumulated over time, not just measured per frame.

```
E_drift = Σ_t ‖ I_t − A ‖           (A = identity anchor)
```

Anchor-correction strength is a function of accumulated drift, not the current
frame's distance:

```
λ_t = f(E_drift)
```

Rationale: slow long-term drift is invisible to a per-frame distance test but
fatal over a clip. A leaky integrator (forgetting factor) is acceptable so the
energy reflects recent-but-accumulated drift.

Invariant: `E_drift` is monotone non-decreasing under sustained drift and the
correction `λ_t` rises with it (pulls the belief back toward the anchor).

Required test: feed frames with a slow albedo ramp; assert `E_drift` accumulates
and `λ_t` increases, whereas an equal-magnitude single spike does NOT.

---

## 16.4 Identity Entropy (belief uncertainty)

Confidence alone is insufficient; the system needs the uncertainty OF the belief.

```
H(I) = − Σ_k p_k log p_k
```

High entropy = "I do not yet know this face." Low entropy = "I know it well."
Entropy must be distinct from a `1 − confidence` scalar: it is computed over the
identity hypothesis distribution (or, where a continuous latent is used, the
differential entropy of the posterior, e.g. `½ log det Σ` of the latent
covariance).

Invariant: H(I) decreases monotonically as independent informative observations
accumulate, and increases under temporal drift / loss of track.

Required test: entropy starts high pre-enrollment, falls with informative frames,
and rises when the face is lost.

---

## 16.5 Information Value of a Frame

A frame's value is novelty × quality, not quality alone.

```
Value(O_t) = Novelty(O_t) · Quality(O_t)
Novelty(O_t) = distance( current_belief, O_t )   (e.g. new pose / new lighting)
```

A 200th clean frontal frame has Value ≈ 0; the first clean profile frame has
Value ≫ 0. Memory-update weight and frame selection are gated on Value.

Invariant: redundant observations (low novelty) contribute ≈0 additional update,
even at high quality; genuinely new viewpoints/lighting dominate the update.

Required test: after convergence on frontal frames, a further frontal frame
produces ≈0 belief change; a first off-axis frame produces a measurable update.

---

## 16.6 Visibility / Occlusion Field

Not every UV point is observed every frame.

```
V(u,v,t) ∈ [0,1]
```

Memory update is gated by visibility (geometry-derived self-occlusion, not just
a 2D sharpness proxy):

```
C_new(u,v) = C_old(u,v) + q_t · V(u,v,t)
```

Profile pose ⇒ V(left_ear) = 0 ⇒ that region's memory is NOT updated from this
frame (no pollution from a self-occluded-but-sharp region).

Invariant: when `V(u,v,t)=0`, `C(u,v)` and the stored appearance at `(u,v)` are
unchanged by frame t.

Required test: synthesize a profile observation; assert the occluded-side region
memory is byte-identical before/after the update.

---

## 16.7 Pose Prior and Coverage

Pose is a maintained distribution, used Bayesian-style in retrieval, and its
coverage caps confidence.

```
P(θ)                              (maintained over discrete pose bins)
score = likelihood(O | hyp) · P(θ_hyp)
Coverage_pose = |observed pose bins| / |total pose bins|
```

Likewise for lighting:

```
Coverage_light = |observed lighting states| / |total lighting states|
```

A face seen only frontally under warm light is NOT "known"; confidence must be
capped accordingly.

Invariant: identity confidence is upper-bounded by a function of
`Coverage_pose · Coverage_light`; retrieval uses `likelihood × prior`, not bare
similarity.

Required test: low-coverage state caps reported confidence below the high-coverage
ceiling, regardless of per-frame quality.

---

## 16.8 Reconstruction Confidence (the trust decision)

The decision "trust the latent vs trust the source" uses a COMPOSITE confidence,
not the per-frame update confidence.

```
C_recon = C_obs · Coverage_pose · Coverage_light · Visibility
```

This is the quantity the latent/physical gate must consume.

Invariant: `C_recon ≤ C_obs` always (coverage/visibility can only reduce trust),
and the render gate reads `C_recon`, not the raw update confidence.

Required test: with high `C_obs` but low coverage, the gate does NOT engage the
latent to drive the face.

---

## 16.9 Background Invariance

Identity is independent of background.

```
∂I / ∂Background = 0
```

No background pixel may influence albedo, shading, lighting, or identity. ALL
decomposition and lighting estimation operate inside the geometry-derived face
mask.

Invariant: perturbing background pixels (poster, wall) leaves albedo / shading /
latent within numerical tolerance.

Required test: render the same face on two different backgrounds; assert albedo
and latent identity are invariant (this is the "poster brightness" bug guard).

---

## 16.10 Appearance Manifold (long-term target)

```
Face = F(z, θ, e, l)        Face ∈ M
```

Identity z, pose θ, expression e, lighting l are disentangled coordinates with a
generator F. This is the endgame (Phase C); it is NOT claimed as implemented.
`identity_manifold.py` is a flat geometry library, not F, and is currently
dormant (§17).

---

## 16.11 Joint Probabilistic Runtime (long-term target)

```
x̂_t = argmax_x  P( geometry, identity, lighting, temporal | observations )
```

The runtime brain is joint constrained optimization, not procedural
orchestration. This is Phase C; the current runtime is procedural with a
decoupled Kalman side-filter (§17). Naming it here locks the target so drift is
measured, not forgotten.

---

# 17. VERIFIED DRIFT LEDGER (audited 2026-05-31)

Status of §16 concepts in the IMPLEMENTATION, with file:line evidence. This is
the honest gap between this spec and the code. PRESENT = mechanism real;
PARTIAL = weaker/disconnected form; MISSING = absent.

| § | Concept | Status | Evidence (file:line) | Gap |
|---|---------|--------|----------------------|-----|
| 16.1 | Observation model O=R(I,p,l)+ε | PARTIAL | Kalman R on 7-dim scalar features `state_evolution.py:64`, `pipeline.py:1717`; image `reconstruct()` `intrinsic_decomposition.py:164` has no ε | No unified pixel forward model predicting O from latent; ε only on abstract features |
| 16.2 | Identity inertia ΔI≪ΔO | PARTIAL→DEAD | `identity_inertia:0.85` `config.py:58` is **never read**; `drift_score` predict-step never populated (no-op); only EMA rate `identity_state.py:227` | No explicit switch_cost; inertia knob is dead config |
| 16.3 | Drift energy E_drift=Σ‖I−A‖ | MISSING | only instantaneous `get_anchor_distance` `identity_state.py:442`; λ from current-frame drift `pipeline.py:2343` | No time accumulation; λ≠f(E_drift) |
| 16.4 | Identity entropy H(I) | PARTIAL | real covariance `pipeline.py:165`+per-pixel uncertainty `types.py:441`; but exposed `identity_uncertainty=1−conf` `identity_estimator.py:814` | No −Σ p log p over hypotheses / no ½logdetΣ |
| 16.5 | Information value Novelty×Quality | MISSING | updates gate on quality only `identity_state.py:213`, `patch_memory.py:153` | Novelty never computed; frame #200 not devalued |
| 16.6 | Visibility field V(u,v,t) | MISSING | per-pixel quality proxy `pipeline.py:3066`, global `pose_weight` `canonical_map.py:170` | No geometry self-occlusion gating; `visibility_calibration.py` stranded |
| 16.7 | Pose prior P(θ) + coverage | MISSING | similarity kernel `identity_state.py:351`; bins exist `patch_memory.py:68` but no coverage/cap | No maintained distribution; no likelihood×prior; no coverage ratio |
| 16.7 | Lighting coverage | MISSING | no lighting-state enumeration anywhere | Confidence not capped for single-lighting |
| 16.8 | Composite C_recon | MISSING | gate reads bare `_last_latent_confidence` `pipeline.py:772,2078` | No coverage/visibility factors in trust decision |
| 16.9 | Background invariance ∂I/∂Bg=0 | PARTIAL | lighting fit masked `pipeline.py:2891`, shading anti-bleed `pipeline.py:2789`; but `decompose()` is **mask-free** `intrinsic_decomposition.py:209` | Background leaks into raw albedo/shading; no invariance test |
| §8 | Lighting separation (albedo only) | PRESENT | `Y=A·S+spec` `intrinsic_decomposition.py:164`; latent stores albedo only `types.py:411` | Albedo invariance approximate; color-cast drift documented |
| 16.10 | Appearance manifold F(z,θ,e,l) | MISSING | `identity_manifold.py` flat (curvature≡0, exp=add `:176`), dormant | No generator, no disentangled axes |
| 16.11 | Joint runtime argmax P(·) | MISSING | procedural `pipeline.py:1173`; `optimizer_architecture.py` stranded | Side Kalman only; no joint MAP |
| §16 | Patch confidence memory | PRESENT | region×pose patches `patch_memory.py:87`, `identity_state.py:765` | per-(patch×pose) confidence not stored (minor) |

Dormant modules (ZERO runtime imports — delete unless promoted to a §16.10/16.11
implementation task): `identity_manifold.py`, `optimizer_architecture.py`,
`visibility_calibration.py`.

---

# 18. Flicker Root Cause ↔ Architecture Gap (MEASURED)

The A/B flicker failure is NOT a tuning problem and NOT a high-frequency problem.
It is a temporal-continuity violation of Principle 4 (temporal consistency is a
hard constraint on RENDERING and LIGHTING) and §16.2 (inertia) at the render
level. This section records the MEASURED root cause (probes on
`clips_test/test_clip.mp4`); an earlier HF-warp-jitter hypothesis was DISPROVEN
by the band-split below and is retired.

`compute_flicker_score` (`benchmark_suite.py:264`) = `std_t( mean_xy |gray_t −
gray_{t-1}| )` — a PER-PIXEL temporal-difference metric (NOT global brightness).

### Measurement 1 — band split (masked face interior, σ=2px), latent path
| series | masked flicker | LF band | HF band | HF share |
|--------|---------------:|--------:|--------:|---------:|
| pure_latent | 16.40 | 16.40 | 0.22 | **1%** |
| composited | 12.35 | 12.37 | 0.14 | **1%** |
| source (legacy proxy) | 1.53 | 1.50 | 0.12 | 7% |

⇒ Flicker is **99% low-frequency**. The `detail_residual` HF wiring is NOT the
cause (HF contributes 0.22 of 16.40); dialing `detail_strength` would do nothing.

### Measurement 2 — per-pair LF |Δ| over 29 frames
Two ~50× spikes (f2→f3 = 50.0, f16→f17 = 51.3) over an elevated steady-state
baseline (per-pair median 4.6 vs source 1.5).

### Measurement 3 — mechanism at the spikes
The spikes coincide EXACTLY with a `normal_source` flip, with albedo/shading/
lighting *means* flat across the jump:

| frame | normal_source | pure render luma | albedo | shading_mean | light_dir |
|------:|---------------|-----------------:|-------:|-------------:|-----------|
| 3 | mesh | 85.5 | 0.840 | 0.423 | (0,0,1) |
| 4 | face_prior | **60.5** | 0.842 | 0.407 | (0.22,−0.82,0.53) |
| 17 | face_prior | 62.9 | 0.841 | 0.425 | (0.30,−0.79,0.54) |
| 18 | mesh | **86.7** | 0.843 | 0.454 | (0,0,1) |

### Causal chain (measured)
1. **Normal-source discontinuity (dominant, the 50× spikes).** When mesh normals
   are unavailable for a frame, the renderer falls back to the generic
   `face_prior` hemisphere (`intrinsic_decomposition.py:_get_cached_face_prior`).
   With mesh normals (frontal: N≈(0,0,1), light≈(0,0,1)) the Lambertian N·L≈1
   everywhere → bright render (~85). The `face_prior` sphere has normals fanning
   outward AND `estimate_lighting` fits a spurious oblique light (≈(0.2,−0.8,0.5)),
   so many pixels hit N·L≤0 and clamp; `render()` normalizes base to unit mean
   then energy-conserves (`physical_renderer.py:385-394`) but clamped/clipped
   energy cannot be recovered → render drops ~30% (≈60). The face's identity did
   not change; only the GEOMETRY SOURCE did. This violates §16.2 (`ΔI ≪ ΔO`: a
   geometry-source switch must not move the rendered identity) and Principle 4.
2. **Per-frame lighting re-estimation (the elevated baseline).** `estimate_lighting`
   is re-solved every frame from the observation with NO temporal continuity, so
   light direction wanders frame-to-frame (see f10–f17 dir drift) → continuous
   low-frequency render wobble (baseline ~3× source).
3. **`photometric_lock` is a global-Y EMA only** (`photometric.py:35`): it damps
   whole-frame mean luminance but cannot remove a spatial relighting change, so it
   does not address either link.

### Required fix (arch-faithful, no hot-fix)
Temporal continuity on the RENDER-DETERMINING latent inputs, per Principle 4 /
§16.2 applied at the render level:
- **Normal-source continuity:** the mesh↔face_prior transition must not produce a
  brightness step. Either keep a temporally-smoothed normal field, or (preferred,
  cheaper) make the latent render exposure-stable across normal sources by
  anchoring absolute brightness to the lighting-invariant `albedo × shading`
  target (which IS continuous — see Measurement 3) rather than letting the
  Lambertian N·L energy float with the normal source.
- **Lighting inertia:** smooth the estimated `LightingModel` over time
  (constant-direction prior + bounded per-frame change), so `dL/dt` is bounded —
  the lighting analogue of §16.2's switch cost.
Neither touches `detail_strength` (spec-fixed 0.65) and neither alters the legacy
path. Both are locked here with the invariants/tests in §16.2 and Principle 4.

---

# 19. Latent Render Gate Policy (LOCKED)

The default render path stays `legacy` until A/B is proven non-regressing on real
video (design.md:483 / requirements.md:126). Promotion is staged:

### Phase 2A — Forced latent (A/B proof)
Engage the latent whenever: latent initialized ∧ verification gate passed ∧ shadow
telemetry valid. Purpose: prove the latent can drive pixels and measure true A/B
quality WITHOUT a gate hiding the result. (Maps to "Option 3".)

### Phase 2B — Relative-to-floor production gate
Gate on the latent's OWN baseline, not an absolute target (the honest steady-state
plateau ≈ 0.257, so an absolute 0.8 is wrong for this system):

```
engage latent  ⇔  initialized
                 ∧ C_recon ≥ C_floor + δ        (§16.8 composite confidence)
                 ∧ dC/dt ≥ 0
fallback        ⇔  C_recon spikes below floor by margin
                 ∨ entropy H(I) jumps above baseline   (§16.4)
                 ∨ verification gate fails
                 ∨ latent is stale
```

(Maps to "Option 1". Note it consumes §16.8 `C_recon`, not the raw update
confidence — closing that gap is a prerequisite.)

### Phase 2C — Per-pixel uncertainty blend (later refinement)
Per-region graceful degradation using per-pixel uncertainty (§16.4). This is a
refinement AFTER the latent path is proven end-to-end, NOT the initial decision
rule. (Maps to "Option 2".)

Order is mandatory: 2A proves pixels → 2B calibrated production gate → 2C
graceful fallback.