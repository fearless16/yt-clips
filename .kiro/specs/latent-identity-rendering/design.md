# Design Document: Latent Identity Rendering (D-05 Identity Decoupling)

## Overview

Face OS is a state-estimation engine, not a face-swap or filter. Its founding belief is **Identity ≠ Pixels**: the true face is a latent state `X = {Geometry, Identity, Appearance, Lighting, Temporal, Uncertainty}`, video frames are noisy observations `Y`, and the system should estimate `P(X | Y)`. The renderer's job is to answer *"what would this stored identity look like under the current geometry and lighting?"* — synthesis from latent — not *"what pixels should I paste and relight?"*.

Today the code does the opposite. The core identity memory is still an RGB pixel buffer (`BeliefPixel`, `identity_state.py:186-265`) updated by RGB EMA (`identity_state.py:233`), which `arch.md` §8/§12 explicitly forbids. The physical render path **re-decomposes the current source crop** and relights *that* (`_render_with_physical_renderer`, `pipeline.py:1937-2143`, decomposition at `pipeline.py:1969`), with the stored identity only nudging it through a mean-correction (`pipeline.py:2010-2017`) and a fixed `0.4` albedo blend (`pipeline.py:1262-1264`, `pipeline.py:1384-1386`). The net effect is **paste-then-relight**, not **synthesize-from-latent**.

This feature promotes a **lighting-invariant identity latent** to be the renderer's **primary input**. The latent stores *reflectance and structure* (albedo + geometry-conditioned appearance + microdetail) plus *uncertainty*, owned entirely behind the Identity subsystem. A new synthesis entry point, `render_from_latent(...)`, warps the stored identity albedo into the current geometry and shades it under the *estimated* lighting, instead of decomposing the source. This is the architectural root for D-05 and is a precondition that unblocks D-07 (state-space runtime: the latent becomes the state the runtime reasons over) and D-10 (factor-graph closure: the latent + uncertainty are the variable nodes). The design is phased so the existing 28 integration tests (`tests/face_os/test_integration.py`) keep passing while telemetry *proves* the latent — not the source crop — drives each rendered pixel.

This document covers both the **High-Level Design** (latent representation, data flow, component/subsystem separation, data models, diagrams) and the **Low-Level Design** (concrete signatures, pseudocode, and the enforced `IntrinsicComponents ↔ renderer` type contract). It also defines the migration/coexistence strategy, correctness properties for property-based testing, and an explicit anti-pattern retirement ledger.

---

## Architectural Assessment (verified against code)

This grounds the design in current runtime truth. CODE is treated as the source of truth; doc claims that contradict code are flagged.

| # | Finding | Evidence | Implication for this design |
|---|---|---|---|
| A-1 | Core identity belief is an **RGB pixel buffer**, updated by RGB EMA | `BeliefPixel` `identity_state.py:186-265`; EMA `self.best_low = self.best_low*(1-low_rate_3d)+low*low_rate_3d` `identity_state.py:233` | Forbidden by `arch.md` §2/§8/§12. Must be **demoted** behind a reflectance latent. |
| A-2 | Albedo path exists but is **secondary**; blended into RGB identity at weight ≈ `mean(conf)*0.4` | `query_albedo` `identity_state.py:608-636`; blend `albedo_weight = float(np.mean(albedo_conf)) * 0.4` `pipeline.py:1262-1264`, `pipeline.py:1384-1386` | The latent must become **primary**, not a 0.4 nudge. Retire the magic 0.4. |
| A-3 | Renderer is **observation-driven**: decomposes the CURRENT source crop and renders that | `source_intrinsic = source_decomposer.decompose(source_rgb)` `pipeline.py:1969`; method `_render_with_physical_renderer` `pipeline.py:1937-2143` | Replace the input to the renderer with the stored latent warped into current geometry. |
| A-4 | Stored identity only **nudges** the source via drift-bucketed mean correction | `if drift>0.05: lambda_corr=min(0.3, drift*2.0)` `pipeline.py:2010-2017` | Retire the drift-bucket anchor heuristic; replace with uncertainty-weighted latent fusion. |
| A-5 | Output is **paste-then-relight**: multiband blend of source `cropped` with render + source-HF reinjection | `multiband_blend(cropped, rendered_face, ...)` `compositor.py:92` via `pipeline.py:2096`; `_reinject_source_hf(..., strength=0.80)` `pipeline.py:1810`, `pipeline.py:388` | When latent confidence is high, the face interior must contain **no source-crop pixels** (no-leak property). |
| A-6 | Subsystem boundaries **leak**: pipeline reaches into identity privates | `self.identity_state._anchor_albedo` `pipeline.py:1992-1994`; `self.identity_state._intrinsic_decomposer` `pipeline.py:1967`; `._gate` accessed via update path | Move identity ownership fully behind `IdentityEstimator`; forbid private reach-through. |
| A-7 | `GeometryEstimator` is imported but **never instantiated**; identity/temporal wrappers are bypassed | `subsystems/geometry_estimator.py` exists; pipeline uses `self._dense_geometry.estimate(...)` `pipeline.py:2063` and `canonical_map` directly | Synthesis must consume a `GeometryState`-shaped input so the renderer is identity/geometry-driven, not pipeline-internal. |
| A-8 | A real Kalman predict/update runs but its state is **unused by rendering** | `predict_update_full` `state_evolution.py:128`, called `pipeline.py:1522-1572`; `E_temporal` computed `pipeline.py:1709` but gate reads only `E_geom>0.8`/`E_photometric<0.1` `pipeline.py:1790-1796` | Latent confidence/uncertainty must become a *read* input to synthesis, not just telemetry. |
| A-9 | Render gating uses **magic constants** only | `pipeline.py:1790-1796`; `renderer_mode.py` thresholds `0.45`/`0.20` | New gating reads latent uncertainty; constants become named, justified parameters (deferred full removal). |
| A-10 | **Silent channel sanitizers** clamp >3-channel "shading" tensors | `pipeline.py:1656` ("BHENCHOD SANITIZER"), `_render_with_physical_renderer` `pipeline.py:1980-1984`, `physical_renderer.py:80-95` (`_ensure_shading`) | Convert to **enforced assertions** at the type-contract boundary, not silent clamps. |
| A-11 | `IdentityManifold` (16-D Riemannian) is **stranded** | `identity_manifold.py:48,133`; `STRANDED_MODULES.md` | Candidate substrate for the geometry-conditioned appearance code; reuse rather than re-invent. |
| A-12 | Doc inconsistency: `STRANDED_MODULES.md` says D-07 "NOT NEEDED"; `STATE.md` tracks D-05/D-07/D-10 PARTIAL | `STRANDED_MODULES.md` "Architecture Honesty"; `STATE.md` Drift Status | This design treats the latent as the enabling substrate for D-07/D-10 and does not depend on the "NOT NEEDED" claim. |

---

## Goals and Non-Goals

**Goals**
- Define a lighting-invariant identity **latent** (`IdentityLatent`) as the renderer's primary input.
- Define `render_from_latent(...)`: warp stored albedo into current geometry, shade under estimated lighting.
- Move identity ownership fully behind the Identity subsystem; stop the pipeline from reaching into identity privates.
- Enforce the `IntrinsicComponents ↔ renderer` contract with assertions, not silent clamps.
- Provide a phased migration that keeps the 28 integration tests truthful and proves latent-driven rendering via telemetry.
- Specify executable correctness properties for property-based testing.

**Non-Goals (deferred)**
- Full factor-graph runtime (D-10) and full state-space runtime brain (D-07) — this design *unblocks* them but does not implement them.
- Removing all magic constants in `renderer_mode.py` — these are *named and justified* here, full removal deferred.
- Neural identity codecs / learned anchors — the latent is analytic (reflectance + structure), consistent with current modules.
- Replacing `multiband_blend`/compositor — it stays as the trivial final assembly step (`arch.md` §15).

---

## High-Level Design

### The Identity Latent Representation

The latent is **lighting-invariant by construction**: it stores reflectance and structure, never illumination. It is defined in **canonical UV space** (the `(256,256)` atlas used by `canonical_map`), so it is also pose-decoupled — pose is re-applied at synthesis time by warping into the current geometry. It has four reflectance/structure fields and one explicit uncertainty field, mirroring `X = {Geometry, Identity, Appearance, Lighting, Temporal, Uncertainty}` (lighting is *excluded* on purpose; geometry/temporal live in their own subsystems).

| Latent field | Meaning | Lighting-invariant? | Source module to build on |
|---|---|---|---|
| `albedo` | Diffuse reflectance in canonical UV, white-balance normalized | Yes (reflectance) | `IntrinsicComponents.albedo` (`intrinsic_decomposition.py`), `query_albedo` |
| `appearance_code` | Geometry-conditioned low-D appearance vector (pose/expression-aware reflectance modulation) | Yes (parameterized by geometry, not light) | `IdentityManifold` 16-D point (`identity_manifold.py:48,68`) — currently stranded |
| `microdetail` | Identity high-frequency residual (pores, beard edges) in canonical UV — *never* averaged | Yes (reflectance HF) | `IntrinsicComponents.detail_residual`; HF "best observation only" rule (`identity_state.py` docstring) |
| `uncertainty` | Per-field, per-region epistemic uncertainty maps | n/a (it *is* the uncertainty) | `IntrinsicComponents.albedo_uncertainty`, `BeliefPixel.variance`/`get_confidence` |
| `wb_reference` | White-balance reference used to normalize incoming albedo | Yes (canonicalizes color temp) | `_normalize_white_balance`, `_wb_scale_ema` (`identity_state.py`) |

Key properties:
- **Albedo is the only color identity store.** The RGB `BeliefPixel` is demoted to a *diagnostic/legacy* role behind a flag (Phase 1), not a render input.
- **Microdetail uses best-observation-only**, consistent with the existing high-frequency rule — never an EMA of pixels.
- **Uncertainty is first-class**: it is *read* by synthesis (fusion weights, gating) and by telemetry, closing the gap in A-8 where Kalman state is computed but unused.

### Why this unblocks D-07 and D-10

- **D-07 (state-space runtime):** Once the latent (not the source crop) is the render input, the runtime has a concrete *state* to predict/update. `StateEvolution.predict_update_full` (`state_evolution.py:128`) can target the latent's `appearance_code` + `uncertainty` instead of an 11-vector whose result is discarded (A-8).
- **D-10 (factor-graph closure):** The latent fields become variable nodes and `uncertainty` becomes the information matrix; `IdentityManifold` provides the geodesic prior factor. No factor graph is built here, but the data model is shaped to accept one.

## Architecture

### Target component diagram

```mermaid
graph TD
    subgraph Geometry["Subsystem A — GeometryEstimator (geometry_estimator.py)"]
        GE["estimate() -> GeometryState<br/>landmarks_478, pose, canonical M, mesh, mask"]
    end
    subgraph Identity["Subsystem B — IdentityEstimator (identity_estimator.py) — OWNS the latent"]
        IL["IdentityLatent<br/>albedo + appearance_code + microdetail + uncertainty"]
        LU["update_latent(obs, geometry, quality)"]
        SY["synthesize_identity(geometry) -> IntrinsicComponents"]
    end
    subgraph Temporal["Subsystem C — TemporalEstimator (temporal_estimator.py)"]
        TE["predict()/update() -> TemporalState<br/>SIM(2) velocity, latent covariance"]
    end
    subgraph Renderer["Subsystem D — FaceRenderer (renderer.py)"]
        RF["render_from_latent(IntrinsicComponents, GeometryState, LightingModel)"]
    end

    SRC["Source frame Y_t (OBSERVATION ONLY)"] --> GE
    SRC -. "observation for latent update only" .-> LU
    GE -->|GeometryState| LU
    GE -->|GeometryState| SY
    GE -->|GeometryState| RF
    LU --> IL
    IL --> SY
    TE -->|uncertainty / motion| LU
    TE -->|uncertainty| RF
    SY -->|IntrinsicComponents in current geometry| RF
    RF -->|Y_face (latent-synthesized)| COMP["Compositor.multiband_blend<br/>(trivial final assembly)"]
    SRC -->|background only, OUTSIDE face mask| COMP
    COMP --> OUT["Output frame"]

    classDef forbidden fill:#fdd,stroke:#c00;
    class SRC forbidden;
```

The single solid arrow from `SRC` into the face render path is **removed** by this design: the source frame feeds geometry and the *latent update*, but the **face interior of the output is synthesized from the latent**, not pasted from `cropped`. The source only contributes the background outside the face mask.

### Data flow: observation → latent update → synthesis

```mermaid
sequenceDiagram
    participant P as Pipeline (orchestration)
    participant G as GeometryEstimator (A)
    participant I as IdentityEstimator (B) [owns IdentityLatent]
    participant T as TemporalEstimator (C)
    participant R as FaceRenderer (D)
    participant C as Compositor

    P->>G: estimate(frame, detection)
    G-->>P: GeometryState g_t (M, mesh, mask, pose)

    Note over P,I: UPDATE PHASE — source is an observation of the latent
    P->>I: update_latent(canonical_face=warp(frame,g_t), g_t, quality_map)
    I->>I: decompose -> albedo/detail; WB-normalize; fuse by uncertainty
    I->>T: report appearance_code + residual
    T-->>I: predicted uncertainty / motion (read by next update)

    Note over P,R: SYNTHESIS PHASE — latent drives the render
    P->>I: synthesize_identity(g_t)
    I-->>P: IntrinsicComponents (albedo warped into g_t, identity normals/detail)
    P->>R: render_from_latent(components, g_t, lighting=estimate_lighting(frame,g_t))
    R-->>P: Y_face (linear-light, [0,1])
    P->>C: multiband_blend(background=frame_bg, face=Y_face, mask=g_t.mask)
    C-->>P: output frame
```

Contrast with today's `_render_with_physical_renderer` (`pipeline.py:1937-2143`), where the *synthesis phase* input is `source_decomposer.decompose(source_rgb)` (`pipeline.py:1969`) — the current crop — and the latent only mean-corrects it (`pipeline.py:2010-2017`).

## Components and Interfaces

### Subsystem separation (the 4 clean subsystems)

`arch.md` §4 mandates four isolated subsystems. The current leaks (A-6, A-7) are closed as follows:

- **A — Geometry (`geometry_estimator.py`)**: produces `GeometryState` (canonical transform `M`, mesh, normals, mask, pose). Must be **instantiated** in the pipeline (today it is not — A-7). Owns all warps/masks. Forbidden: identity/lighting/RGB blending (enforced by boundary contract already in the file's docstring).
- **B — Identity (`identity_estimator.py`)**: **sole owner** of `IdentityLatent`. Exposes `update_latent`, `synthesize_identity`, `query_uncertainty`. The pipeline may **only** call these public methods. Reaching into `identity_state._anchor_albedo`, `._intrinsic_decomposer`, `._gate` (A-6) becomes a contract violation caught by an architectural test.
- **C — Temporal (`temporal_estimator.py`)**: predicts latent uncertainty + SIM(2) motion; updates *confidence/uncertainty only*, never texture (`arch.md` §C). Its output becomes a **read input** to identity fusion and render gating (closing A-8).
- **D — Renderer (`renderer.py`)**: `render_from_latent` is the new primary entry. Consumes `IntrinsicComponents` (from B) + `GeometryState` (from A) + `LightingModel`. Forbidden: estimating geometry/identity, RGB-space rescue compositing.

The pipeline becomes a thin orchestrator: A → (B.update) → (C) → (B.synthesize) → D → Compositor. Synthesis/lighting/masking logic that currently lives in `pipeline.py` (`_render_with_physical_renderer`) moves **into the Renderer subsystem** so `FaceRenderer` is a real delegate rather than a thin wrapper around `physical_renderer.render` (current `renderer.py`).

## Data Models

```python
# face_os/types.py — NEW dataclasses (additive; existing types unchanged in Phase 1)

@dataclass
class IdentityLatent:
    """Lighting-invariant identity latent in CANONICAL UV space.

    This is the renderer's PRIMARY input source. It stores reflectance and
    structure only — NEVER illumination, NEVER raw RGB frames.

    Invariants (enforced by IdentityEstimator.update_latent):
      - albedo, microdetail are in canonical UV (atlas_size), float32 [0,1] / residual
      - albedo is white-balance normalized against wb_reference
      - uncertainty fields are float32 in [0,1], same HxW as their data field
      - microdetail is best-observation-only (no EMA of pixels)
    """
    atlas_size: Tuple[int, int]                  # (H, W) canonical UV, e.g. (256, 256)

    albedo: np.ndarray                           # (H, W, 3) float32 [0,1] reflectance
    appearance_code: np.ndarray                  # (D,) float32, D=ManifoldConfig.dimension (16)
    microdetail: np.ndarray                      # (H, W, 3) float32 zero-mean HF residual
    wb_reference: np.ndarray                     # (3,) float32 white-balance reference

    albedo_uncertainty: np.ndarray               # (H, W) float32 [0,1]
    appearance_uncertainty: float                # scalar [0,1] (epistemic, from manifold)
    microdetail_uncertainty: np.ndarray          # (H, W) float32 [0,1]

    observation_count: np.ndarray                # (H, W) float32 — for confidence
    initialized: bool = False

    def mean_confidence(self) -> float: ...       # 1 - mean(albedo_uncertainty), face region only


@dataclass
class LatentRenderTelemetry:
    """Proves the latent (not the source crop) drove the render. Emitted per frame."""
    frame_idx: int
    render_path: str                             # 'latent' | 'physical_legacy' | 'alpha' | 'enhancement'
    latent_primary: bool                         # True iff face interior synthesized from latent
    source_pixel_fraction: float                 # fraction of face-mask pixels traceable to source (target 0)
    latent_confidence: float
    albedo_drift_from_anchor: float
    uncertainty_mean: float
    contract_assertions_passed: bool
```

`IntrinsicComponents` (`intrinsic_decomposition.py`) is **reused unchanged** as the carrier from B→D, so `PhysicalRenderer.render_with_intrinsic` keeps working. The difference is *where its fields come from*: `synthesize_identity` fills `albedo`/`normal_map`/`detail_residual` from the **latent warped into current geometry**, with `shading` deliberately left as a neutral unit field (lighting is the renderer's responsibility, from `LightingModel`).

---

## Low-Level Design

### Core Interfaces / Signatures

```python
# face_os/subsystems/identity_estimator.py — IdentityEstimator becomes the latent owner

class IdentityEstimator:
    def __init__(self, identity_state, manifold: "IdentityManifold", atlas_size=(256, 256)): ...

    def set_anchor(self, reference_face_bgr: np.ndarray) -> None:
        """Initialize the latent from an enrollment reference (WB-normalized albedo + manifold point)."""

    def update_latent(
        self,
        canonical_face: np.ndarray,         # (H,W,3) uint8 BGR, source warped into canonical UV
        geometry: "GeometryState",          # provides pose, mesh_478, warp M for normals
        quality_map: np.ndarray,            # (H,W) float32 [0,1]
        temporal: Optional["TemporalState"] = None,  # read-only: predicted uncertainty/motion
    ) -> IdentityLatent:
        """Fuse one observation into the latent. Uncertainty-weighted, NOT fixed-rate EMA."""

    def synthesize_identity(self, geometry: "GeometryState") -> "IntrinsicComponents":
        """PRIMARY render input: warp stored albedo+microdetail into current geometry,
        attach geometry normals, leave shading neutral (renderer applies lighting)."""

    def query_uncertainty(self, geometry: "GeometryState") -> np.ndarray:
        """(H,W) float32 [0,1] uncertainty in CURRENT geometry — read by render gating."""

    # Public, lighting-invariant accessors kept; private reach-through (A-6) is forbidden.


# face_os/subsystems/renderer.py — FaceRenderer gains the primary entry point

class FaceRenderer:
    def render_from_latent(
        self,
        components: "IntrinsicComponents",  # from IdentityEstimator.synthesize_identity
        geometry: "GeometryState",          # normals/mesh/mask come from here
        lighting: "LightingModel",          # estimated from current frame (NOT baked in latent)
        view_direction: Optional[np.ndarray] = None,
    ) -> np.ndarray:                        # (H,W,3) float32 linear-light [0,1], face interior only
        """Synthesize the stored identity under current geometry + lighting.
        Enforces the IntrinsicComponents<->renderer contract via assert_intrinsic_contract()."""


# face_os/intrinsic_decomposition.py (or a small contracts.py) — enforced contract

def assert_intrinsic_contract(c: "IntrinsicComponents", expect_hw: Tuple[int, int]) -> None:
    """Replace silent channel sanitizers (A-10) with HARD assertions at the B->D boundary."""


# face_os/pipeline.py — lighting estimator (kept explicit, replaces inline LightingModel(...) literals)

def estimate_lighting(frame_bgr: np.ndarray, geometry: "GeometryState") -> "LightingModel":
    """Estimate scene illumination from the OBSERVATION. Lighting is never stored in the latent."""
```

### Key Functions with Formal Specifications

#### `IdentityEstimator.update_latent(...)`

**Preconditions**
- `canonical_face.shape == (*atlas_size, 3)`, dtype `uint8`.
- `quality_map.shape == atlas_size`, values in `[0, 1]`.
- `geometry.pose` is defined; `geometry.mesh_478`/`warp_M` may be `None` (then face-prior normals).
- The latent is owned solely here; no external caller has mutated latent fields.

**Postconditions**
- Returns the updated `IdentityLatent` with `initialized == True`.
- `albedo` is white-balance normalized: `||mean(albedo) - wb_reference||` is non-increasing in expectation under stable lighting.
- Fusion is **uncertainty-weighted**, not fixed-rate: regions with lower incoming `albedo_uncertainty` move more; high-uncertainty regions are conservative.
- `microdetail` only changes where the incoming observation quality exceeds the stored best (best-observation-only).
- `uncertainty` is **monotonically non-decreasing** when `quality_map` drops (e.g., occlusion) and the observation does not improve any region.
- No RGB-EMA of a primary identity buffer occurs (A-1/A-2 retired).

**Loop invariants** (per-region fusion loop)
- After processing region `k`, all regions `0..k` satisfy `0 ≤ uncertainty ≤ 1` and `albedo ∈ [0,1]`.
- The white-balance reference is applied identically across regions (no per-region color drift).

#### `IdentityEstimator.synthesize_identity(geometry)`

**Preconditions**
- Latent `initialized == True`.
- `geometry` provides a valid canonical transform `M` (and `mesh_478` when mesh normals are desired).

**Postconditions**
- Returns `IntrinsicComponents` whose `albedo`/`detail_residual` are the **stored latent warped into current geometry** — *not* derived from any current source crop.
- `normal_map` comes from `geometry` (mesh) or face-prior fallback; `shading` is a neutral unit field (`ones`), because lighting is applied by the renderer.
- `confidence` field equals `1 - query_uncertainty(geometry)`.
- Output satisfies `assert_intrinsic_contract(result, expect_hw=geometry render size)`.

**Loop invariants**: none (no iteration); pure warp + assemble.

#### `FaceRenderer.render_from_latent(...)`

**Preconditions**
- `assert_intrinsic_contract(components, expect_hw)` passes (raises otherwise — no silent clamp).
- `lighting` is a valid `LightingModel`; `geometry.mask` defines the face region.

**Postconditions**
- Output is the identity albedo lit by `lighting` under `geometry` normals; deterministic under fixed seed (`arch.md` §3).
- **No-source-leak**: output face-interior pixels are a function of `components` + `lighting` + `geometry` **only**; the source crop is not an argument and cannot leak in.
- Output dtype/shape/range satisfy the frame contract (`arch.md` §9: fixed size, no NaN/Inf, bounded range).

### Algorithmic Pseudocode

#### Latent update (uncertainty-weighted fusion — replaces RGB EMA)

```pascal
ALGORITHM update_latent(canonical_face, geometry, quality_map, temporal)
INPUT:  canonical_face (H,W,3 uint8), geometry, quality_map (H,W in [0,1]), temporal (optional)
OUTPUT: latent : IdentityLatent
BEGIN
    ASSERT canonical_face.shape = (atlas_H, atlas_W, 3)
    ASSERT quality_map.shape = (atlas_H, atlas_W)

    // 1. Decompose the OBSERVATION (source is telemetry, not memory)
    obs_lin        <- srgb_to_linear(canonical_face)
    intrinsic      <- intrinsic_decomposer.decompose(obs_lin, geometry.mesh_478, geometry.warp_M)
    obs_albedo     <- normalize_white_balance(intrinsic.albedo, latent.wb_reference)
    obs_detail     <- intrinsic.detail_residual
    obs_albedo_unc <- intrinsic.albedo_uncertainty           // per-pixel epistemic

    IF NOT latent.initialized THEN
        latent.albedo               <- obs_albedo
        latent.microdetail          <- obs_detail
        latent.albedo_uncertainty   <- obs_albedo_unc
        latent.observation_count    <- quality_map
        latent.appearance_code      <- manifold.log_map(origin, encode(obs_albedo))
        latent.initialized          <- TRUE
        RETURN latent
    END IF

    // 2. Optional temporal inflation of uncertainty BEFORE fusion (predict step)
    IF temporal is not NULL THEN
        latent.albedo_uncertainty <- inflate(latent.albedo_uncertainty, temporal.drift_score)
    END IF

    // 3. Per-region uncertainty-weighted fusion (Kalman-like gain, NOT fixed EMA rate)
    FOR each region r IN REGION_DEFS DO
        ASSERT all_uncertainty_in_[0,1](latent, regions <= r)        // loop invariant
        // gain high where stored is uncertain AND observation is confident
        gain_r <- (latent.unc[r]) / (latent.unc[r] + obs_albedo_unc[r] + eps)
        gain_r <- gain_r * quality_map[r]
        latent.albedo[r] <- (1 - gain_r) * latent.albedo[r] + gain_r * obs_albedo[r]
        latent.albedo_uncertainty[r] <- (1 - gain_r) * latent.albedo_uncertainty[r]
    END FOR

    // 4. Microdetail: BEST-OBSERVATION-ONLY (never average pores)
    better <- quality_map > latent.observation_count_quality
    latent.microdetail <- WHERE(better, obs_detail, latent.microdetail)

    // 5. Appearance code on the manifold (geometry-conditioned, bounded)
    target_code <- encode(latent.albedo, geometry.pose)
    latent.appearance_code <- manifold.interpolate(latent.appearance_code, target_code, t=gain_mean)

    latent.observation_count <- latent.observation_count + quality_map
    ASSERT no_nan_inf(latent.albedo) AND in_[0,1](latent.albedo)
    RETURN latent
END
```

#### Synthesis (`render_from_latent` core — warp identity, shade under current light)

```pascal
ALGORITHM render_from_latent(components, geometry, lighting, view_dir)
INPUT:  components (IntrinsicComponents from synthesize_identity), geometry, lighting, view_dir
OUTPUT: Y_face (H,W,3 float32 linear [0,1])   // face interior ONLY, no source pixels
BEGIN
    // CONTRACT: hard assertions, NOT silent clamps (retires A-10 sanitizers)
    assert_intrinsic_contract(components, expect_hw = geometry.render_hw)

    albedo  <- components.albedo            // stored identity, already warped into geometry
    normals <- components.normal_map        // from geometry (mesh) or face-prior
    detail  <- components.detail_residual   // identity HF (NOT source HF)

    // Lighting comes from the OBSERVATION's estimate, never from the latent
    ambient  <- albedo * lighting.ambient
    N_dot_L  <- max(0, dot(normals, lighting.diffuse_direction))
    diffuse  <- albedo * lighting.diffuse_intensity * N_dot_L
    half     <- normalize(lighting.diffuse_direction + view_dir)
    N_dot_H  <- max(0, dot(normals, half))
    specular <- lighting.specular_intensity * pow(N_dot_H, lighting.specular_power)

    base   <- w_a*ambient + w_d*diffuse + w_s*specular
    Y_face <- base + detail_strength * detail        // identity microdetail, edge-masked
    Y_face <- clip(Y_face, 0, 1)

    ASSERT shape(Y_face) = geometry.render_hw + (3,)
    ASSERT no_nan_inf(Y_face)
    RETURN Y_face
END
```

This reuses `PhysicalRenderer.render` math (`physical_renderer.py`) verbatim — the change is the **input provenance** (latent-warped albedo vs. `source_decomposer.decompose(source_rgb)` at `pipeline.py:1969`) and the removal of source-HF reinjection inside the face mask.

#### Enforced `IntrinsicComponents ↔ renderer` type contract (assertions, not clamps)

```pascal
ALGORITHM assert_intrinsic_contract(c, expect_hw)
INPUT:  c : IntrinsicComponents, expect_hw : (H, W)
OUTPUT: none (raises ContractViolation on failure)
BEGIN
    ASSERT c.albedo.ndim = 3 AND c.albedo.shape[2] = 3
        ELSE RAISE ContractViolation("albedo must be (H,W,3), got " + shape)
    ASSERT c.albedo.shape[:2] = expect_hw
    ASSERT c.shading.ndim = 3 AND c.shading.shape[2] = 1
        ELSE RAISE ContractViolation("shading must be (H,W,1); >3 channels is a TYPE BUG, not a clamp target")
    ASSERT c.normal_map.shape = expect_hw + (3,)
    ASSERT dtype(c.albedo) = float32 AND in_[0,1](c.albedo)
    ASSERT no_nan_inf(c.albedo) AND no_nan_inf(c.shading) AND no_nan_inf(c.normal_map)
    // The >3-channel "shading" leak (A-10) now FAILS LOUDLY instead of np.mean(...)-clamping
END
```

The current code silently repairs this at three sites: `pipeline.py:1656` (`_render_core`), `pipeline.py:1980-1984` (`_render_with_physical_renderer`), and `physical_renderer.py:80-95` (`_ensure_shading`). Those become assertion call sites; the upstream producer (`IntrinsicDecomposer` / `synthesize_identity`) is fixed so the contract holds by construction.

### Example Usage

```python
# Pipeline orchestration after migration (Phase 3) — thin, subsystem-delegated
geom = self._geometry_estimator.estimate(frame, detection)            # Subsystem A
canonical = canonical_map.warp_to_canonical(frame, geom.landmarks)[0]

# UPDATE: source frame is an OBSERVATION of the latent
self._identity_estimator.update_latent(canonical, geom, quality_map,
                                        temporal=self._temporal_state)  # Subsystem B

# SYNTHESIS: latent drives the render
components = self._identity_estimator.synthesize_identity(geom)         # Subsystem B -> IntrinsicComponents
lighting   = estimate_lighting(frame, geom)                            # from observation, not latent
y_face     = self._face_renderer.render_from_latent(components, geom, lighting)  # Subsystem D

# ASSEMBLY: compositor stays trivial; source only OUTSIDE the face mask
output = multiband_blend(bg=frame_in_output_space, fg=to_uint8_bgr(y_face),
                         mask=geom.mask, levels=4)                      # compositor.py:92
```

---

## Migration / Coexistence Strategy

The goal is to introduce the latent as the primary render input **without breaking the 28 integration tests** (`tests/face_os/test_integration.py`) and without a "green tests hiding broken runtime" situation (an explicit anti-pattern). Truthfulness is enforced by telemetry that *proves* the latent drove the render, not by counters that infer it.

The render path is selected by a feature flag `render_source ∈ {legacy, latent}` with a hard fallback, so legacy behavior is the default until the latent path is proven on real video.

### Phase 0 — Contracts and telemetry (no behavior change)
- Add `assert_intrinsic_contract(...)` and call it *alongside* the existing sanitizers (assert in a "warn-only" mode first: log a contract violation but still clamp). This surfaces every place a >3-channel tensor leaks (A-10) before we make it fatal.
- Add `LatentRenderTelemetry` to the per-frame telemetry log (`_emit_frame_telemetry`, `pipeline.py:1519`). For legacy frames, `latent_primary=False`, `source_pixel_fraction≈1.0` — this *documents the current truth* (A-3/A-5).
- **Tests stay green** (additive only). New tests assert the telemetry schema exists.

### Phase 1 — Build the latent behind the Identity subsystem (dormant)
- Add `IdentityLatent` to `types.py` and the `update_latent`/`synthesize_identity`/`query_uncertainty` methods to `IdentityEstimator`. Wire `IdentityManifold` (`identity_manifold.py`, currently stranded — A-11) as the `appearance_code` substrate.
- `update_latent` runs every frame in shadow mode (populates the latent) but **does not drive rendering yet**. The RGB `BeliefPixel` and the `0.4` albedo blend remain in place.
- Instantiate `GeometryEstimator` (closes A-7) and route `update_latent` through `GeometryState` instead of pipeline-internal warps.
- Telemetry now logs `latent_confidence`, `albedo_drift_from_anchor`, `uncertainty_mean` even though the render is still legacy. This lets us validate latent quality on the real `input/video.mp4` slow tests before flipping the switch.

### Phase 2 — Latent-primary render path (flagged, A/B)
- Implement `FaceRenderer.render_from_latent(...)` and a `render_source='latent'` branch in `_render_core` (`pipeline.py:~1727`) that:
  1. calls `synthesize_identity(geom)` for the render input (replaces `source_decomposer.decompose(source_rgb)`, `pipeline.py:1969`),
  2. calls `render_from_latent(...)`,
  3. composites with `multiband_blend` using `geom.mask` and **does not** call `_reinject_source_hf` inside the face mask (`pipeline.py:1810`).
- Make `assert_intrinsic_contract` **fatal** on the latent path (sanitizers removed there); keep warn-only on the legacy path until Phase 4.
- Use the existing A/B harness (`ab_validation.py`, `compare_render_methods`, `process_frame(..., render_mode_override=...)`) to compare latent vs legacy on SSIM, LAB drift, sharpness, flicker. Gate promotion on: no regression in detection rate, identity drift, flicker (metrics tracked in `FULL_REFERENCE.md` §7).
- **No-leak telemetry**: `source_pixel_fraction` on latent frames must be ≈0 inside the face mask (only background is source). This is the runtime proof that the latent — not the crop — drives the face.

### Phase 3 — Flip default to latent, demote RGB belief
- Default `render_source='latent'` once A/B is non-regressing on real video.
- Demote `BeliefPixel` to diagnostic-only behind `USE_LEGACY_RGB_BELIEF` (default off). It is no longer a render input; it may remain for the `get_anchor_distance` LAB telemetry.
- Retire the `0.4` albedo blend (`pipeline.py:1262-1264`, `pipeline.py:1384-1386`) and the drift-bucket mean correction (`pipeline.py:2010-2017`) on the default path.

### Phase 4 — Cleanup
- Remove the silent sanitizers (`pipeline.py:1656`, `pipeline.py:1980-1984`, `physical_renderer.py:80-95`) once no contract violations have been observed for N clips; `assert_intrinsic_contract` is the only guard.
- Make render gating read `query_uncertainty(...)` instead of magic `E_geom>0.8`/`E_photometric<0.1` (`pipeline.py:1790-1796`); name and justify any remaining thresholds.

### Test-truthfulness guardrails
- An **architectural test** asserts the pipeline does not access identity privates (`_anchor_albedo`, `_intrinsic_decomposer`, `_gate`) on the latent path (closes A-6). Implemented by attribute-access tracing or a lint check on `pipeline.py`.
- A **runtime-truth test** runs the real video and asserts `LatentRenderTelemetry.latent_primary == True` and `source_pixel_fraction < 0.02` for ≥90% of physical frames — preventing a green suite from masking a still-paste-then-relight runtime.

---

## Correctness Properties

(For property-based testing.) These are executable properties for `hypothesis` (Python). They define the validation contract for the latent and the synthesis path, per the project's "tests before implementation" and "property-based testing is the validation method" philosophy.

**Property Test Library:** `hypothesis` (Python), with synthetic faces/albedos/lighting as strategies; deterministic seeds per `arch.md` §3.

### Property 1: Lighting invariance of the identity latent

**Validates: Requirements 1.1** (lighting-invariant identity latent)

For the same identity albedo `A` observed under two different lightings `L1`, `L2`, the updated `latent.albedo` must be (nearly) identical — identity must not absorb illumination. Guards against RGB EMA identity memory (A-1).

```python
@given(albedo=albedos(), L1=lightings(), L2=lightings(), geom=geometries())
def prop_latent_lighting_invariance(albedo, L1, L2, geom):
    obs1 = shade(albedo, geom, L1)              # render observation under L1
    obs2 = shade(albedo, geom, L2)              # render observation under L2
    lat1 = fresh_identity().update_latent(obs1, geom, full_quality).albedo
    lat2 = fresh_identity().update_latent(obs2, geom, full_quality).albedo
    assert mean_lab_distance(lat1, lat2) < EPS_LIGHTING   # invariant to lighting
```

### Property 2: Identity preservation across pose

**Validates: Requirements 2.1** (synthesis warps stored identity into current geometry)

Synthesizing the latent into two poses then canonicalizing back yields a stable albedo. Guards against paste-then-relight (A-3).

```python
@given(latent=initialized_latents(), pose_a=poses(), pose_b=poses())
def prop_identity_preserved_across_pose(latent, pose_a, pose_b):
    ca = canonicalize(latent.synthesize_into(pose_a))
    cb = canonicalize(latent.synthesize_into(pose_b))
    assert mean_lab_distance(ca.albedo, cb.albedo) < EPS_POSE
```

### Property 3: No source-pixel leak when latent confidence is high

**Validates: Requirements 2.2** (latent is the primary render input; no paste-then-relight)

When latent confidence is high, no face-interior output pixel equals the source crop. Guards against source-HF reinjection / the 0.4 blend (A-5).

```python
@given(latent=high_confidence_latents(), frame=frames(), geom=geometries())
def prop_no_source_leak_high_confidence(latent, frame, geom):
    components = latent.synthesize_identity(geom)
    y_face = renderer.render_from_latent(components, geom, estimate_lighting(frame, geom))
    out = composite(frame, y_face, geom.mask)
    leak = source_pixel_fraction(out, frame, geom.mask)   # fraction matching source within mask
    assert leak < EPS_LEAK                                 # face is synthesized, not pasted
```

### Property 4: Honest uncertainty (Kalman shrink + occlusion floor + predict-step inflation)

**Validates: Requirements 1.2** (uncertainty is first-class and honest)

Uncertainty must behave like a real Bayesian posterior, consistent with the fusion law in the algorithm block above (`unc <- (1-gain)*unc`, `gain = unc/(unc+obs_unc+eps)*quality`):

- **P4b — shrink under information:** any observation with `quality > 0` is evidence, so per-region uncertainty is **non-increasing** under fusion (more evidence ⇒ tighter posterior). This is what stops `latent_confidence` from collapsing to ~0 and keeps the latent render path live. It explicitly forbids a "running-max ratchet" that inflates whenever a frame fails to beat the best-seen quality.
- **P4a — occlusion floor:** when `quality → 0` the observation carries no information (`gain → 0`), so uncertainty **holds** (`unc <- (1-0)*unc`) — it must not *decrease*. No evidence cannot make us more certain.
- **P4c — predict-step inflation:** the **only** source of uncertainty increase is the temporal predict step (`inflate(unc, temporal.drift_score)`), reflecting genuine drift between observations.

```python
@given(latent=initialized_latents(), seq=positive_quality_sequences())
def prop_uncertainty_shrinks_under_information(latent, seq):
    prev = float(np.mean(latent.albedo_uncertainty))
    for q in seq.qualities():                  # q > 0, NO temporal drift
        latent = latent.update_latent(seq.frame(q), seq.geom, q)
        cur = float(np.mean(latent.albedo_uncertainty))
        assert cur <= prev + EPS_MONO          # information only TIGHTENS
        prev = cur

def prop_zero_quality_holds_uncertainty(latent):
    before = mean(latent.albedo_uncertainty)
    latent = latent.update_latent(frame, geom, quality=0.0)   # no information
    assert mean(latent.albedo_uncertainty) >= before - EPS    # never reduced

def prop_temporal_drift_inflates(latent):
    before = mean(latent.albedo_uncertainty)
    latent = latent.update_latent(frame, geom, q=0.02, temporal=Drift(0.6))
    assert mean(latent.albedo_uncertainty) > before           # predict step only
```

> **Resolved inconsistency (algorithm block is source of truth):** an earlier draft of this property asserted uncertainty was *non-decreasing under any decreasing quality*. That contradicts the fusion law above (positive-quality evidence must shrink the posterior) and was only satisfiable by a running-max ratchet that collapsed `latent_confidence` to ~0 on real video, leaving the latent render path permanently dormant. The honest model (shrink on information; hold at the occlusion floor; inflate only on temporal drift) is the contract.

### Property 5: Type-contract enforcement (assertions, not clamps)

**Validates: Requirements 3.1** (enforced IntrinsicComponents↔renderer contract)

A >3-channel "shading" tensor must RAISE, never be silently clamped. Guards against silent channel sanitizers (A-10).

```python
@given(bad_channels=integers(min_value=4, max_value=256))
def prop_contract_rejects_bad_shading(bad_channels):
    c = make_components(shading_channels=bad_channels)
    with pytest.raises(ContractViolation):
        assert_intrinsic_contract(c, expect_hw=(256, 256))
```

### Property 6: Synthesis determinism

**Validates: Requirements 2.3** (deterministic latent-driven synthesis)

`render_from_latent` is deterministic under fixed inputs/seed (`arch.md` §3). Guards against hidden heuristic branches.

```python
@given(latent=initialized_latents(), geom=geometries(), light=lightings())
def prop_render_deterministic(latent, geom, light):
    c = latent.synthesize_identity(geom)
    y1 = renderer.render_from_latent(c, geom, light)
    y2 = renderer.render_from_latent(c, geom, light)
    assert np.array_equal(y1, y2)
```

### Property 7: White-balance convergence (latent stays color-stable)

**Validates: Requirements 5.1** (retire drift-bucket anchor heuristic)

Repeated updates under stable lighting do not drift albedo away from `wb_reference`. Guards against the drift-bucket anchor heuristic (A-4).

```python
@given(albedo=albedos(), lighting=lightings(), n=integers(2, 50))
def prop_wb_non_divergence(albedo, lighting, n):
    latent = fresh_identity()
    drifts = []
    for _ in range(n):
        latent = latent.update_latent(shade(albedo, canonical_geom, lighting), canonical_geom, full_quality)
        drifts.append(np.linalg.norm(latent.albedo.mean((0,1)) - latent.wb_reference))
    assert drifts[-1] <= drifts[0] + EPS_WB           # non-divergent
```

### Property 8: Frame contract on synthesized output

**Validates: Requirements 3.2** (output frame contract on synthesized faces)

Synthesized output satisfies the frame contract (`arch.md` §9). Guards against NaN/Inf/shape drift.

```python
@given(latent=initialized_latents(), geom=geometries(), light=lightings())
def prop_output_frame_contract(latent, geom, light):
    y = renderer.render_from_latent(latent.synthesize_identity(geom), geom, light)
    assert y.dtype == np.float32 and y.min() >= 0.0 and y.max() <= 1.0
    assert not np.any(np.isnan(y)) and not np.any(np.isinf(y))
```

### Coverage matrix

| Property | Validates | Anti-pattern it guards against |
|---|---|---|
| Property 1 lighting invariance | Latent stores reflectance, not light | RGB EMA identity memory (A-1) |
| Property 2 pose preservation | Identity stable across viewpoint | paste-then-relight (A-3) |
| Property 3 no source leak | Face synthesized from latent | source-HF reinjection / blend (A-5) |
| Property 4 honest uncertainty (shrink/floor/predict) | Uncertainty is a real posterior, not a ratchet | "green tests hiding broken runtime"; confidence-collapse ratchet |
| Property 5 contract enforcement | Types enforced, not clamped | silent channel sanitizers (A-10) |
| Property 6 determinism | Deterministic render | hidden heuristic branches |
| Property 7 WB convergence | Color stability w/o anchor buckets | drift-bucket anchor heuristic (A-4) |
| Property 8 frame contract | Output validity | NaN/Inf/shape drift |

---

## Error Handling

| Scenario | Condition | Response | Recovery |
|---|---|---|---|
| Contract violation (bad shading channels) | `assert_intrinsic_contract` fails on latent path | Raise `ContractViolation` (fatal on latent path) | Phase 0–3: warn-only on legacy path; producer fixed so it cannot recur |
| Latent not initialized | First frames before enrollment/observation | `synthesize_identity` returns neutral components; renderer declines | Fall back to legacy/alpha path; telemetry `latent_primary=False` |
| Geometry unavailable (no landmarks) | `GeometryState` empty | Skip latent update; do not corrupt latent with mis-aligned obs | Use last-known geometry via Temporal subsystem; uncertainty inflates |
| Lighting estimate degenerate | `estimate_lighting` returns near-zero intensities | Clamp to a documented minimum ambient (named constant, justified) | Energy conservation in `PhysicalRenderer` calibrates output |
| Latent confidence low | `query_uncertainty` high across face | Gate to HYBRID/alpha; blend latent with observation by uncertainty | As confidence recovers, latent reasserts primacy |

All fallbacks **preserve the frame contract** (shape/dtype/range) per `arch.md` §7, and every fallback emits explicit telemetry (no inferred branch truth, per D-08).

---

## Testing Strategy

### Unit testing
- `IdentityLatent` invariants (shape/dtype/range of each field), `assert_intrinsic_contract` accept/reject cases.
- `synthesize_identity` provenance: assert output albedo derives from the latent (mock the decomposer; verify the source crop is *not* read).
- `estimate_lighting` bounds and determinism.

### Property-based testing
- Properties P1–P8 above, with `hypothesis` strategies for albedos, lightings, poses, geometries, occlusion sequences. Deterministic seeds.

### Integration testing (must keep all 28 green)
- Reuse `tests/face_os/test_integration.py` classes: `TestPipelineOutputValidity`, `TestPhysicalRendererBrightness`, `TestFaceDetectionOnOutput`, `TestEnergyConservation`, `TestProcessFrameContract`.
- Add a `TestLatentDrivesRender` class: on the latent path, assert `LatentRenderTelemetry.latent_primary` and `source_pixel_fraction < 0.02` for ≥90% of physical frames on real video (`input/video.mp4`, slow marker).
- Add `TestSubsystemBoundaries`: assert the pipeline does not touch identity privates on the latent path (A-6).

### A/B validation
- `ab_validation.compare_render_methods` (`ab_validation.py:270`) for latent vs legacy: SSIM, LAB drift, Procrustes, flicker. Promotion gate = no regression vs `FULL_REFERENCE.md` §7 baselines.

---

## Performance Considerations

- `synthesize_identity` adds one canonical→output warp of stored albedo/microdetail per frame; this replaces the per-frame `source_decomposer.decompose(source_rgb)` (`pipeline.py:1969`), so net decomposition cost in synthesis drops (decomposition runs only in `update_latent`). Expected neutral-to-positive on the ~3.8 fps baseline (`FULL_REFERENCE.md` §7).
- `IdentityManifold` ops are 16-D (`identity_manifold.py:48`), negligible cost.
- Mesh-normal raster is already bounded (`_normal_raster_shape`, `pipeline.py:2146`); reused unchanged.

## Security Considerations

Not applicable to this feature (offline video processing, no network/auth/PII surface introduced). The `VerificationGate` (`identity_state.py`) that gates identity updates on liveness/identity-match remains owned by the Identity subsystem and is unchanged.

---

## Anti-Pattern Retirement Ledger

Explicit per the request: which anti-patterns this design **removes** vs **defers**.

| Anti-pattern | Location | Disposition | Phase | Notes |
|---|---|---|---|---|
| RGB EMA identity memory | `BeliefPixel` `identity_state.py:186-265`, EMA `:233` | **Removed as render input** (demoted to diagnostic behind flag) | 3 | Forbidden by `arch.md` §2/§8/§12. Kept readable for LAB telemetry only. |
| `0.4` albedo blend into RGB identity | `pipeline.py:1262-1264`, `:1384-1386` | **Removed** | 3 | Replaced by latent-primary synthesis. |
| Drift-bucket anchor mean-correction (`if drift>0.05: lambda_corr=min(0.3, drift*2.0)`) | `pipeline.py:2010-2017` | **Removed** on default path | 3 | Replaced by uncertainty-weighted fusion (P7 guards regression). |
| Drift-bucket anchor `lambda` ladders (`>30/>15/>5`) | `identity_state.py` `query`/`query_identity`/`query_albedo` | **Removed** from render-driving path | 3 | These query methods stop driving the render; kept only if diagnostic. |
| Silent channel sanitizers (256→clamp) | `pipeline.py:1656`, `:1980-1984`, `physical_renderer.py:80-95` | **Replaced by assertions** (fatal on latent path) | 2 (assert) / 4 (delete) | `assert_intrinsic_contract`. |
| Source-HF reinjection inside face mask | `pipeline.py:1810`, `:1860`, `:1909`, `_reinject_source_hf` `:388` | **Removed inside face mask** on latent path | 2 | Microdetail comes from the latent, not the source. Background HF unaffected. |
| Magic render-gate constants (`E_geom>0.8`, `E_photometric<0.1`) | `pipeline.py:1790-1796` | **Deferred** (named/justified now, replaced by uncertainty later) | 4 | Becomes `query_uncertainty`-driven. |
| `renderer_mode.py` thresholds `0.45`/`0.20` | `renderer_mode.py` | **Deferred** | post-4 | Documented as named parameters; full removal out of scope. |
| `GeometryEstimator` never instantiated | `subsystems/geometry_estimator.py` (A-7) | **Fixed** (instantiated, feeds `GeometryState`) | 1 | Required for subsystem separation. |
| Pipeline reaching into identity privates | `_anchor_albedo`/`_intrinsic_decomposer`/`_gate` (A-6) | **Fixed** (public methods only; architectural test) | 1–3 | Identity ownership moves fully behind Subsystem B. |

---

## Dependencies

- **Reused unchanged:** `intrinsic_decomposition.py` (`IntrinsicComponents`, `IntrinsicDecomposer`), `physical_renderer.py` (Lambertian + Blinn-Phong math, `render_with_intrinsic`/`render_with_mesh`), `compositor.py` (`multiband_blend`), `canonical_map.py` (alignment), `dense_geometry.py` (mesh normals), `state_evolution.py` (Kalman, for Temporal), `lie_group.py` (SIM(2)), `photometric.py` (LAB lock), `types.py` (existing contracts).
- **Reactivated:** `identity_manifold.py` (`IdentityManifold`, `ManifoldConfig`, `IdentityPoint`) — currently stranded (`STRANDED_MODULES.md`), used for `appearance_code`.
- **New:** `IdentityLatent` + `LatentRenderTelemetry` dataclasses (`types.py`); `assert_intrinsic_contract`; `IdentityEstimator.update_latent`/`synthesize_identity`/`query_uncertainty`; `FaceRenderer.render_from_latent`; `estimate_lighting`; `render_source` flag and latent branch in `_render_core`.
- **Test deps:** `hypothesis` (property-based), existing `pytest` + `dlib` integration harness (`tests/face_os/test_integration.py`).

> Doc-vs-code note (A-12): `STRANDED_MODULES.md` marks D-07 "NOT NEEDED" while `STATE.md` tracks D-05/D-07/D-10 as PARTIAL. This design follows the CODE as ground truth and positions the latent as the substrate that *enables* D-07/D-10 without committing to building either here.
