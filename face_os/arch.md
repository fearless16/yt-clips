# FACE OS V2 — Mathematical Architecture Specification

Version: 2.0 (Architectural Rewrite)
Status: Design Spec
Goal: Convert Face OS from heuristic image-processing pipeline into a mathematically stable latent-state reconstruction system.

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