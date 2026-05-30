# Requirements Document

## Introduction

This document specifies the requirements for **Latent Identity Rendering** (drift item D-05, Identity decoupling) in the Face OS project. The requirements are **derived from** the approved design (`design.md`) and grounded in the measured state and required fix recorded in `LOCKED_ARCHITECTURE.md` (§D-05, with telemetry-honesty ties to §D-08).

Face OS is a state-estimation engine founded on the belief that **Identity ≠ Pixels**: the true face is a latent state `X = {Geometry, Identity, Appearance, Lighting, Temporal, Uncertainty}`, video frames are noisy observations, and the system estimates `P(X | Y)`. Today the core identity memory is still an RGB pixel buffer updated by RGB EMA, and the physical render path re-decomposes the current source crop and relights it (paste-then-relight), with the stored identity only nudging the result through a `0.4` albedo blend and drift-bucket anchor heuristics.

This feature promotes a **lighting-invariant identity latent** to be the renderer's **primary input**. The renderer must synthesize the stored identity warped into the current geometry and shaded under estimated lighting, instead of re-decomposing and relighting the current source crop. The feature retires the RGB-EMA identity buffer, the `0.4` albedo blend, drift-bucket anchor heuristics, source high-frequency reinjection inside the face mask, and silent channel sanitizers — replacing the last with enforced type assertions. The migration is phased so the existing 28 integration tests stay green while telemetry proves the latent — not the source crop — drove each rendered face pixel.

Each requirement maps back to the design's correctness properties (P1–P8) where applicable, so traceability between requirements, design properties, and property-based tests is explicit.

## Glossary

- **Face_OS**: The overall state-estimation engine that estimates and renders a face latent state from video observations.
- **Identity_Estimator**: Subsystem B (`identity_estimator.py`). The sole owner of the identity latent; exposes `update_latent`, `synthesize_identity`, and `query_uncertainty`.
- **Geometry_Estimator**: Subsystem A (`geometry_estimator.py`). Produces a `GeometryState` (canonical transform, mesh, normals, mask, pose).
- **Temporal_Estimator**: Subsystem C (`temporal_estimator.py`). Predicts latent uncertainty and SIM(2) motion; updates confidence/uncertainty only, never texture.
- **Face_Renderer**: Subsystem D (`renderer.py`). Exposes `render_from_latent` as the primary synthesis entry point.
- **Pipeline**: The thin orchestrator that sequences Geometry → Identity update → Temporal → Identity synthesis → Renderer → Compositor.
- **Compositor**: The trivial final-assembly step (`multiband_blend`) that combines the synthesized face with the background outside the face mask.
- **Telemetry_System**: The per-frame emitter of `LatentRenderTelemetry` and frame telemetry records.
- **Identity_Latent**: A lighting-invariant identity representation in canonical UV space holding reflectance and structure only (`albedo`, `appearance_code`, `microdetail`, `uncertainty` fields, `wb_reference`). It never stores illumination or raw RGB frames. It is the renderer's primary input source.
- **Albedo**: Diffuse reflectance in canonical UV space, white-balance normalized. The only color identity store in the latent.
- **Microdetail**: Identity high-frequency residual (pores, beard edges) in canonical UV space, maintained best-observation-only and never averaged.
- **Appearance_Code**: A geometry-conditioned low-dimensional appearance vector (16-D `IdentityManifold` point) representing pose/expression-aware reflectance modulation; lighting-invariant.
- **Canonical_UV**: The `(256, 256)` atlas space used by `canonical_map` in which the latent is defined; pose-decoupled because pose is re-applied at synthesis time.
- **WB_Reference**: The white-balance reference used to normalize incoming albedo so the latent canonicalizes color temperature.
- **Lighting_Invariance**: The property that the identity latent stores reflectance and structure only, so the same identity observed under different lightings produces (nearly) identical latent albedo.
- **No_Leak**: The property that, when latent confidence is high, no face-interior output pixel is traceable to the source crop; the face interior is synthesized from the latent.
- **Latent_Primary**: A per-frame telemetry boolean that is true if and only if the face interior was synthesized from the latent.
- **Source_Pixel_Fraction**: A per-frame telemetry measure of the fraction of face-mask pixels traceable to the source crop; target near zero on the latent path.
- **IntrinsicComponents_Contract**: The enforced type contract at the Identity→Renderer boundary (`assert_intrinsic_contract`) that validates `IntrinsicComponents` shapes, channel counts, dtype, range, and finiteness, raising `ContractViolation` instead of silently clamping.
- **IntrinsicComponents**: The reused carrier (`intrinsic_decomposition.py`) passed from the Identity subsystem to the renderer, holding `albedo`, `normal_map`, `detail_residual`, `shading`, and `confidence`.
- **Render_Source**: The feature flag selecting the render path, with values `legacy` and `latent`, defaulting to `legacy` until the latent path is proven on real video.
- **Frame_Contract**: The output validity contract (fixed size, float32, bounded range `[0,1]`, no NaN/Inf) per `arch.md` §9.
- **Physical_Frame**: A frame rendered through the physical render path (as opposed to alpha/enhancement fallback paths).
- **ContractViolation**: The error raised when the IntrinsicComponents contract is not satisfied.

## Requirements

### Requirement 1: Lighting-Invariant Identity Latent Representation

**User Story:** As a Face OS architect, I want a lighting-invariant identity latent owned behind the Identity subsystem, so that identity is stored as reflectance, structure, microdetail, and uncertainty rather than as RGB pixel memory.

#### Acceptance Criteria

1. WHEN the same identity albedo is observed under two different lightings, THE Identity_Estimator SHALL update the latent such that the LAB distance between the two resulting latent albedos is below the defined lighting-invariance threshold. *(Validates: Property 1 — lighting invariance)*
2. WHILE observation quality decreases during occlusion and no region is improved, THE Identity_Estimator SHALL maintain per-region albedo uncertainty values that are monotonically non-decreasing. *(Validates: Property 4 — uncertainty monotonicity)*
3. THE Identity_Latent SHALL store the fields albedo, appearance_code, microdetail, per-field uncertainty maps, and wb_reference in canonical UV space.
4. THE Identity_Latent SHALL exclude illumination and raw RGB frames from all stored fields.
5. WHEN fusing a new observation, THE Identity_Estimator SHALL update microdetail only in regions where the incoming observation quality exceeds the stored best observation quality.
6. THE Identity_Estimator SHALL be the sole owner of the Identity_Latent, and THE Pipeline SHALL access the latent only through the public methods update_latent, synthesize_identity, and query_uncertainty.
7. THE Identity_Estimator SHALL update the latent using uncertainty-weighted fusion rather than a fixed-rate RGB exponential moving average.
8. THE Face_OS SHALL enforce Identity_Latent ownership through architectural discipline and code review rather than runtime access enforcement.

### Requirement 2: Latent-Primary Synthesis and Rendering

**User Story:** As a Face OS architect, I want the renderer to synthesize from the latent as the primary path, so that the face interior is generated from the stored identity warped into the current geometry and shaded under estimated lighting instead of being pasted from the source crop and relit.

#### Acceptance Criteria

1. WHEN the latent is synthesized into two different poses and canonicalized back, THE Identity_Estimator SHALL produce albedos whose LAB distance is below the defined pose-preservation threshold. *(Validates: Property 2 — identity preservation across pose)*
2. WHILE latent confidence is high, THE Face_Renderer SHALL produce a face interior whose source_pixel_fraction is below the defined no-leak threshold. *(Validates: Property 3 — no source-pixel leak)*
3. WHEN render_from_latent is called twice with identical inputs and a fixed seed, THE Face_Renderer SHALL produce byte-identical output. *(Validates: Property 6 — synthesis determinism)*
4. WHEN rendering on the latent path, THE Face_Renderer SHALL synthesize the face interior as a function of the IntrinsicComponents, the estimated lighting, and the geometry only, and SHALL NOT accept the source crop as a render input.
5. WHEN synthesizing identity, THE Identity_Estimator SHALL fill albedo and detail_residual from the stored latent warped into the current geometry and SHALL leave shading as a neutral unit field so the Face_Renderer applies lighting.
6. WHEN rendering a frame, THE Face_Renderer SHALL apply lighting estimated from the current observation and SHALL NOT read lighting from the latent.

### Requirement 3: Enforced IntrinsicComponents-to-Renderer Type Contract

**User Story:** As a Face OS maintainer, I want the IntrinsicComponents-to-renderer contract enforced by assertions, so that malformed tensors fail loudly at the boundary instead of being silently sanitized.

#### Acceptance Criteria

1. IF the shading tensor of an IntrinsicComponents value has more than one channel, THEN THE IntrinsicComponents_Contract SHALL raise a ContractViolation rather than clamping the tensor. *(Validates: Property 5 — type-contract enforcement)*
2. WHEN render_from_latent produces output, THE Face_Renderer SHALL return a tensor that is float32, bounded within `[0,1]`, free of NaN and Inf values, and shaped to the expected geometry render size. *(Validates: Property 8 — frame contract on synthesized output)*
3. WHEN rendering on the latent path, THE Face_Renderer SHALL evaluate the IntrinsicComponents_Contract as a fatal check before synthesis.
4. WHERE the legacy render path is active during early migration phases, THE Face_OS SHALL evaluate the IntrinsicComponents_Contract in warn-only mode and log each contract violation.
5. WHERE the contract mode is explicitly configured to fatal during legacy migration, THE Face_OS SHALL honor the fatal configuration and raise a ContractViolation rather than forcing warn-only mode.

### Requirement 4: Subsystem Boundary Integrity

**User Story:** As a Face OS architect, I want clean subsystem boundaries with identity ownership fully behind the Identity subsystem and the GeometryEstimator instantiated, so that the pipeline is a thin orchestrator that does not reach into identity internals.

#### Acceptance Criteria

1. WHEN rendering on the latent path, THE Pipeline SHALL NOT access the identity private members `_anchor_albedo`, `_intrinsic_decomposer`, or `_gate`.
2. THE Pipeline SHALL instantiate the Geometry_Estimator and route latent updates and synthesis through a GeometryState value produced by the Geometry_Estimator.
3. THE Geometry_Estimator SHALL own all warps and masks, and SHALL NOT perform identity, lighting, or RGB blending.
4. THE Temporal_Estimator SHALL provide predicted uncertainty and motion as a read input to identity fusion and to render gating.
5. IF Geometry_Estimator instantiation fails, THEN THE Pipeline SHALL continue using fallback geometry handling rather than crashing.

### Requirement 5: Color Stability Without Drift-Bucket Heuristics

**User Story:** As a Face OS quality engineer, I want albedo color stability achieved through white-balance normalization and uncertainty-weighted fusion, so that identity color does not diverge under stable lighting and drift-bucket anchor heuristics are no longer needed.

#### Acceptance Criteria

1. WHEN the latent receives repeated observations under stable lighting, THE Identity_Estimator SHALL keep the final albedo drift from wb_reference no greater than the initial drift plus the defined white-balance tolerance. *(Validates: Property 7 — white-balance convergence)*
2. WHEN fusing an observation, THE Identity_Estimator SHALL normalize incoming albedo against wb_reference before fusion.
3. THE Identity_Estimator SHALL replace the drift-bucket mean-correction heuristic with uncertainty-weighted fusion on the default render path.

### Requirement 6: Measurable Identity-Quality Targets

**User Story:** As a Face OS quality engineer, I want measurable identity-quality targets from the locked architecture, so that progress on identity decoupling is objectively verifiable.

#### Acceptance Criteria

1. WHEN a physical frame is audited, THE Face_OS SHALL produce an albedo LAB drift from anchor of less than 10. *(Reduces measured 13.7–22.7 toward target; LOCKED_ARCHITECTURE D-05)*
2. WHEN a physical frame is audited, THE Face_OS SHALL produce a LAB distance versus expectation of less than 20.
3. WHEN identity verification runs on an audited frame, THE Face_OS SHALL produce an embedding distance of less than 0.45.
4. WHEN intrinsic decomposition produces albedo, THE Identity_Estimator SHALL apply color-cast compensation so that the rendered teal/green color cast is removed and albedo color invariance is improved beyond the measured channel standard deviation range of 0.04–0.10.
5. IF a color-cast compensation cannot simultaneously remove the teal/green cast AND improve albedo color invariance, THEN THE Identity_Estimator SHALL reject that compensation.
6. THE Identity_Latent SHALL represent identity as the combination of geometry, albedo, microdetail, and temporal belief, and SHALL NOT represent identity as canonical RGB memory.

### Requirement 7: Phased Migration with Green Tests and Runtime-Truth Telemetry

**User Story:** As a Face OS maintainer, I want a phased migration that keeps the existing 28 integration tests green while emitting runtime-truth telemetry, so that I can prove the latent drives the render without a green suite masking a paste-then-relight runtime.

#### Acceptance Criteria

1. WHILE the migration is in progress, THE Face_OS SHALL keep all 28 existing integration tests passing through additive changes.
2. WHEN a frame is processed, THE Telemetry_System SHALL emit a LatentRenderTelemetry record containing render_path, latent_primary, source_pixel_fraction, latent_confidence, albedo_drift_from_anchor, uncertainty_mean, and contract_assertions_passed.
3. WHEN the latent render path runs across the physical frames of the real video, THE Telemetry_System SHALL report latent_primary as true and source_pixel_fraction below 0.02 for at least 90 percent of physical frames.
4. THE Pipeline SHALL select the render path using the Render_Source flag with a hard fallback, and SHALL default to the legacy path until the latent path is proven non-regressing on real video.
5. WHEN the Render_Source flag is explicitly set to latent, THE Pipeline SHALL use the latent path immediately regardless of proof status, overriding the legacy default.
6. THE Face_OS SHALL provide an architectural test that asserts no identity private members are accessed on the latent path and a runtime-truth test that validates latent_primary and source_pixel_fraction on the real video.

### Requirement 8: Honest Per-Frame Telemetry

**User Story:** As a Face OS operator, I want honest per-frame telemetry, so that each frame correctly reports whether the latent drove the render and never reports stale data.

#### Acceptance Criteria

1. WHEN any frame is processed, THE Telemetry_System SHALL expose render_path, renderer_mode, fallback_reason, intrinsic_used, geometry_source, resample_count, energy_terms, and transform_det.
2. WHILE an alpha or enhancement path renders a frame, THE Telemetry_System SHALL report intrinsic_used as false.
3. WHEN emitting per-frame telemetry, THE Telemetry_System SHALL report energy_terms computed for the current frame rather than values carried over from prior frames.
4. WHEN the IntrinsicComponents_Contract is evaluated for a frame, THE Telemetry_System SHALL report contract_assertions_passed reflecting the outcome for that frame.

### Requirement 9: Observable Anti-Pattern Retirement

**User Story:** As a Face OS architect, I want anti-pattern retirement to be observable, so that the removal of forbidden mechanisms is verifiable rather than assumed.

#### Acceptance Criteria

1. WHEN the default render path is the latent path, THE Face_OS SHALL demote the RGB-EMA identity buffer to a diagnostic-only role and SHALL NOT use it as a render input.
2. WHEN rendering on the default latent path, THE Face_OS SHALL omit the fixed `0.4` albedo blend.
3. WHEN rendering on the default latent path, THE Face_OS SHALL omit the drift-bucket anchor mean-correction.
4. WHEN rendering on the latent path, THE Face_OS SHALL omit source high-frequency reinjection inside the face mask, and SHALL source face-interior microdetail from the latent.
5. THE Face_OS SHALL replace the silent channel sanitizers with IntrinsicComponents_Contract assertions.

### Requirement 10: Graceful Degradation Preserving the Frame Contract

**User Story:** As a Face OS operator, I want graceful degradation when the latent is unavailable or low-confidence, so that every emitted frame still satisfies the frame contract.

#### Acceptance Criteria

1. IF the latent is not initialized, THEN THE Identity_Estimator SHALL return neutral components, THE Face_Renderer SHALL decline latent rendering, and THE Telemetry_System SHALL report latent_primary as false.
2. IF geometry is unavailable because landmarks are missing, THEN THE Identity_Estimator SHALL skip the latent update so the latent is not corrupted by a misaligned observation.
3. IF the lighting estimate is degenerate with near-zero intensities, THEN THE Face_Renderer SHALL clamp lighting to a documented minimum ambient value.
4. WHILE latent confidence is low across the face, THE Pipeline SHALL gate to a hybrid or alpha path and blend the latent with the observation by uncertainty.
5. WHEN any fallback path is taken, THE Face_OS SHALL produce output that satisfies the Frame_Contract.
