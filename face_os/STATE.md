# Face OS v3.8 — Compact State Reference

**Last updated:** 2026-05-30 | **Tests:** 278 collected in tests/face_os/ (265 fast + 13 slow runtime-truth; 9 skip without input/video.mp4) | **Source:** ~15,200 lines

This file is the current source of truth for Face OS. Older status files may
describe historical drift; use this file for current runtime alignment.

---

## Architecture Map

```
pipeline.py (single orchestration runtime)
    ├── detect_track.py / landmarks.py / crop_planner.py
    ├── canonical_map.py
    ├── identity_state.py / patch_memory.py
    ├── intrinsic_decomposition.py
    ├── dense_geometry.py
    ├── physical_renderer.py
    ├── compositor.py
    ├── photometric.py
    ├── state_evolution.py / temporal_solve.py / lie_group.py
    ├── energy_scaling.py (now gates rendering decisions)
    ├── ab_validation.py (A/B comparison harness)
    └── subsystems/
        ├── identity_estimator.py
        ├── temporal_estimator.py
        └── renderer.py
```

## Drift Status

| ID | Requirement | Status | Current Evidence |
|---|---|---|---|
| D-01 | Signal-preserving render path | **ALIGNED** | Linear-light compositing via sRGB↔linear conversion. HF detail injection in linear space (fixed C-01). Consistent detail→sharpen→photometric order across all paths. |
| D-02 | PhysicalRenderer improves quality vs alpha | **ALIGNED MECHANISM** | Real Lambertian+Blinn-Phong shading, real mesh normals. A/B framework with SSIM, LAB drift, Procrustes metrics. |
| D-03 | Benchmark corpus | **PARTIAL** | Real video integration tests exist using input/video.mp4. Synthetic hard-condition tests also present. |
| D-04 | Dense geometry integration | **ALIGNED** | Physical path calls DenseGeometryEstimator.estimate() with anatomical anchor-based landmark mapping (fixed H-07). |
| D-05 | Identity decoupling | **PARTIAL — Phase 2B render quality + gate + per-pixel hybrid PROVEN (default still legacy)** | Lighting-invariant `IdentityLatent` is owned by `IdentityEstimator`. Fusion is an **honest per-pixel Kalman filter** (design.md:354-361): pure shrink `unc <- (1-gain)*unc` on every positive-quality observation; the ONLY inflation source is the temporal predict step (`drift_score`). A running-max "ratchet" (removed) had collapsed confidence to ~0 on real video (0.234→0.006); the corrected model rises with evidence to a fixed-point plateau (0.234→0.257), governed honestly by the decomposer's `albedo_uncertainty`. Pinned by P4a/P4b/P4c. **Phase 2 (latent DRIVES pixels):** with `render_source='latent'` the latent renders the face via `synthesize_identity` → `estimate_lighting` (closed-form Lambertian inverse, from the OBSERVATION) → `FaceRenderer.render_from_latent` (delegates to `render_with_intrinsic`, `observed=None`, fatal contract), composited as a PEER branch in `_render_core` that skips the source-HF tail — retiring A-2/A-3/A-5 on that path. PROVEN on real video: `TestLatentRenderModeOnRealVideo` 4/4 (latent_primary=True, render_path='latent', source_pixel_fraction≈0.80). **Default stays `legacy`** (shadow `TestLatentShadowModeOnRealVideo` 4/4 unchanged; fast 228, 0 regressions). **Phase 2B OPEN (root cause PROVEN by crop-space measurement 2026-05-30, all 6 frames).** Earlier notes blamed face-prior normals; that is REFUTED — the pipeline already feeds real `mesh_478` normals on 4/6 frames (`geometry_source=[mesh×4, face_prior×2]`) and the mesh frames are the FLAT ones (mask-interior latent std 1.3 vs face_prior 65.9). The honest mask-interior A/B (inside the real `crop_mask`, NOT the diluted landmark bbox; the composited-bbox SSIM 0.99 / 0.93× were ~80% shared-background dilution) is: latent face mean **194 (≈0.76) vs source 93 (≈0.36) = 2.1×**, render-vs-source ≈102/255 (the latent genuinely drives the face, leak≈0 — Phase 2A real). The 2.1×-too-bright observation was therefore CORRECT at mask scale (I wrongly retracted it off the diluted bbox number). ROOT CAUSE (captured `LightingModel`, every frame): `ambient=0.030` (=`_MIN_AMBIENT` floor), `diffuse_intensity=0.000`, `components.shading=1.000±0.000`. The lighting fit COLLAPSES to its degenerate floor because `estimate_lighting` fits RAW observed luminance against normals — but luminance = albedo×shading, and the face's albedo variation (eyes/brows/lips) dominates and is not explained by normals, so least-squares recovers `b≈0` (collapse fires even with strongly-varying face-prior normals → proof it is albedo conflation, not normals). Then the renderer ENERGY-NORMALIZES the output to the latent albedo's own brightness (~0.84), discarding scene exposure → 2.1× bright + flat. **FIX LANDED & PROVEN (2026-05-30):** the renderer's REAL contract is that the SHADING field carries absolute scene exposure (`render()` normalizes the LightingModel amplitude away then energy-conserves to `mean(albedo*shading)`, physical_renderer.py:374-386) — so `synthesize_identity`'s NEUTRAL unit shading pinned the output to albedo brightness. The spec's "neutral shading" prose (design.md:211/239/303) loses to the as-built renderer contract per the doc-inconsistency rule. The latent path now replaces shading with `_observation_shading = lowpass(observed_luminance / latent_albedo)` (pipeline.py): `albedo*shading` reconstructs scene luminance, the latent still stores NO illumination (it only supplies albedo), and the low-pass passes only smooth illumination so no source-HF/identity leaks. Re-measured gate (mask interior): latent mean **194→92.4** (= scene 92.9), per-frame std **1.3→9–89** (flat collapse gone, structure restored), render-vs-source 102→41 (still real synthesis, not paste), leak **0.004–0.010 < 0.02** (no re-leak from the new shading). 9 RED→GREEN unit tests (`_observation_shading` + `_source_pixel_fraction`); fast suite **237** (0 regressions); slow real-video **8/8** + two NEW runtime-truth guards (`test_latent_render_matches_scene_exposure`, `test_latent_render_is_not_flat` — per-frame min std, a guard the OLD collapse would have failed). NOT hot-fixed by rescaling. **PRODUCTION GATE LANDED & PROVEN (2026-05-30):** the latent now only DRIVES a frame when the pure static `_evaluate_latent_gate` ENGAGES — RELATIVE-TO-FLOOR because measured real-video confidence lives in a tiny band (seed 0.2335 → plateau 0.2567), so an ABSOLUTE threshold would never fire. Precedence: `uninitialized` → `confidence_spike` (`C_prev−C_t ≥ 0.05`, instability, checked before floor) → `below_floor` (`C_t < C_floor+0.01`, no evidence earned past enrollment) → else `engaged`. NOTE the earlier `dC/dt≥0` prose is WRONG for the as-built and is retired: the PLATEAU (dC/dt=0, above floor) MUST engage (it is the measured steady state); only a *sharp* drop refuses, normal jitter (|Δ|≤~0.006) stays engaged. Measured real-video sequence (margin 0.01 → engage at 0.2435): frame 0 conf 0.2401 → `below_floor` (legacy/alpha), frames 1–5 conf 0.2458→0.2567 → `engaged` (latent drives) — so the gate is demonstrably NOT a no-op in either direction. New `gate_state` telemetry field (9th in `LatentRenderTelemetry`). 8 RED→GREEN gate unit tests + 2 slow runtime-truth guards: `test_gate_state_couples_to_render` (biconditional `engaged ⟺ latent_primary` — the anti-decorative-telemetry guard proving the decision CONTROLS the render) and `test_gate_engages_on_real_video` (not a total regression). Re-ran standing gate WITH the gate active: 5 latent_primary frames, mask-interior lat_mean **93.7 ≈ src 92.8**, lat_std **41.2 ≈ src 40.6**, render-vs-src **43** — engaged-frame quality identical to the pre-gate brightness-fix run (gate routed frame 0 to legacy without degrading the rest). Fast suite **245**, full suite **257** (0 regressions). **PER-PIXEL HYBRID LANDED & PROVEN (2026-05-30):** within an engaged frame the latent no longer drives the interior all-or-nothing — `_hybrid_face` blends the rendered face TOWARD the observation per pixel BY UNCERTAINTY (design.md:665, requirements 10.4). `alpha = 1 − query_uncertainty·blend_max` (blend_max=0.5 → latent keeps ≥50% authority on every pixel); blends toward `LOWPASS(observation)` only (smooth illumination/chroma crosses, source HF never returns per-pixel — same anti-leak low-pass as `_observation_shading`). ROOT CAUSE found by measurement when the naive blend tripped the leak guard (0.022 > 0.02): the offline estimate (0.009) skipped the `multiband_blend` composite; the wired diagnostic PROVED 100% of hybrid-induced leak lived in the FEATHER TRANSITION BAND (where the composite already mixes source), not in the blend strength — lowering blend_max barely moved it (0.022→0.022). FIX is architectural, not a dial: RESTRICT the hybrid to the SOLID interior (`feathered_mask>0.99`); there `|latent−source| ≫ tol` so leak == pure-latent (<0.01) even at full blend_max=0.5 (proven: `erode099` leak identically equalled pure-latent on all frames). Real-video: `hybrid_alpha_mean` ~0.62–0.77 (broad uncertainty engages the blend, cap holds). New `hybrid_alpha_mean` telemetry (10th `LatentRenderTelemetry` field). 10 RED→GREEN helper unit tests (`_hybrid_blend_alpha`, `_hybrid_face` incl. the source-HF-rejection anti-leak guard) + slow `test_hybrid_blend_engages_and_respects_cap`; the existing `test_latent_render_reduces_source_fraction` auto-covers leak<0.02 on the hybrid composite. Pure-latent debug capture preserved, so the exposure/flatness guards still measure synthesis quality un-diluted (mask metrics unchanged: lat_mean 93.7, std 41.2). NOT hot-fixed by rescaling or by relaxing the guard. Fast **255**, full **268** (0 regressions). **TASK 5.2 LANDED & PROVEN (2026-05-30):** the magic H-03 render-gate constants are now a pure, tested decision — extracted to static `_evaluate_physical_gate(energy_terms, latent_uncertainty_mean=None, geom_extreme_z=0.8, photometric_low_z=0.1, latent_uncertainty_max=0.95)`. Two stale-spec corrections found by recon: the gate is at pipeline.py:2043-2052 (not ~1790, that's telemetry) and it gates the LEGACY physical path; and the constants compare against Z-SCORE-normalized terms (`EnergyScaler` default zscore, energy_scaling.py:18), so `0.8`/`0.1` are now NAMED + z-score-justified params (E_geom = pose-mag z; E_photometric = decomposition-QUALITY z, gated low). Closes A-8: the latent's epistemic uncertainty (`1 - mean_confidence()`, scalar of the same albedo_uncertainty query_uncertainty exposes) is now a first-class READ input — INITIALIZED-GUARDED (caller passes None pre-enrollment where query_uncertainty is all-ones, so legacy-only runs are byte-identical). New `latent_uncertainty_max=0.95` veto (reason `latent_uncertainty_high`) sits ABOVE the measured operating point (real U seed~0.77/plateau~0.74/spike~0.8) so it's inert in normal op and fires only on near-total collapse (U→1); energy vetoes keep precedence so existing reason vocabulary is unchanged. 10 RED→GREEN `TestPhysicalGate` incl. ANTI-DECORATIVE (synthetic U=0.99 flips the decision) + NON-REGRESSION (measured U=0.77 stays inert). **TASK 5.3/5.4 VERIFIED COMPLETE:** per-clause runtime-truth audit confirmed 10.1 (uninitialized→neutral/decline/latent_primary=False), 10.2 (missing-landmarks→canonical_face=None→estimator returns latent unchanged, identity_estimator.py:253-259), 10.3 (degenerate→documented `_MIN_AMBIENT=0.03` on every fit return path), 10.4 (hybrid), 10.5 (all 4 fallbacks funnel through `_postprocess_rendered_crop`→uint8) are all IMPLEMENTED+COVERED. 5.4 closed the one real gap: strengthened `test_fit_lighting_degenerate_inputs_return_safe_floor` from `ambient>=0.0` to `ambient>=_MIN_AMBIENT` (lock proven tight — degenerate fit returns exactly 0.0300). Full suite **278** collected, 0 regressions (9 of them skip only when `input/video.mp4` is absent — env, not code; slow latent class uses `clips_test/test_clip.mp4` fallback, 13/13 green). **Still OPEN:** flipping the default from `legacy` to `latent` (Phase 3, after formal ABComparator non-regression); full removal of the energy constants + `renderer_mode.py` 0.45/0.20 thresholds (design.md:48/716 mark Deferred — named/justified, not deleted); Req 4.4 temporal-predicted-uncertainty read input (only identity-latent uncertainty wired so far). Standing gate: real-video HTML A/B report after every change.
| D-06 | Predictive temporal belief | **ALIGNED MECHANISM** | SIM(2) velocity prediction computed and used for 1-2 frame occlusion recovery (fixed H-02). True SE(2)/SIM(2) Lie algebra exp/log (fixed H-04). |
| D-07 | State-space runtime brain | **PARTIAL** | Energy scaling now gates rendering decisions (fixed H-03). Runtime remains procedural. optimizer_architecture.py is stranded. |
| D-08 | Per-frame truthful telemetry | **ALIGNED** | `_emit_frame_telemetry()` accepts explicit branch truth. All paths emit explicit telemetry including energy gating reasons. |
| D-09 | Visual regression validation | **ALIGNED** | Integration tests validate sharpness, contrast ratio, flicker, NaN/Inf, telemetry schema on real video. |
| D-10 | Probabilistic architectural closure | **PARTIAL / PHASE C** | Subsystem wrappers are real delegates. Factor graph not implemented. |

**Current honest summary:** 5 aligned mechanisms/areas, 5 partial architectural areas.

## Key Entry Points

| Function | File:Line | Purpose |
|---|---:|---|
| `process_frame` | pipeline.py:698 | Public single-frame API for tests and A/B validation |
| `_process_frame_v2` | pipeline.py:~1000 | Forward-only frame processing |
| `_render_core` | pipeline.py:~1680 | Single render branch selector with energy gating |
| `_render_with_physical_renderer` | pipeline.py:~1770 | Physical renderer + dense-geometry path |
| `_emit_frame_telemetry` | pipeline.py:~1519 | Per-frame explicit telemetry schema |
| `_postprocess_rendered_crop` | pipeline.py:~1600 | Shared sharpen + photometric lock |
| `multiband_blend` | compositor.py:92 | Laplacian pyramid blend |
| `photometric_lock` | photometric.py:29 | LAB temporal luminance lock |
| `render_with_mesh` | physical_renderer.py:329 | Mesh-derived normal rendering |
| `compare_render_methods` | ab_validation.py:270 | Physical vs alpha A/B harness |
| `SE2Transform.exp/log` | lie_group.py:43-100 | True SE(2) Lie algebra maps |
| `SIM2Transform.exp/log` | lie_group.py:100-140 | True SIM(2) Lie algebra maps |

## Test Suite

| File | Current Role |
|---|---|
| test_integration.py | End-to-end pipeline, telemetry, render quality, compositor, Lie group, geometry, identity, A/B |

## Run Commands

```bash
# All tests
.venv/bin/python -m pytest tests/face_os/ -v

# Fast tests only (no video)
.venv/bin/python -m pytest tests/face_os/ -v -m "not slow"

# Slow tests only (requires input/video.mp4)
.venv/bin/python -m pytest tests/face_os/ -v -m "slow"
```
