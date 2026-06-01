# Architecture

> Face OS status note: use `face_os/STATE.md` as the current source of truth.
> Runtime percentages and test counts below are historical snapshots unless
> repeated in `face_os/STATE.md`.

Two parallel systems co-exist in this codebase:

1. **Face OS** (primary) — Identity-reconstruction pipeline for portrait-mode studio video
2. **Legacy cricket pipeline** — 16:9 live stream → 9:16 shorts (MediaPipe/YOLOv8-face + GFPGAN)

---

## Face OS Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  FACE OS — Identity Belief State Engine                             │
│  (pipeline.py, face_os/*)                                           │
│                                                                     │
│  Core equation:  OUTPUT = source * (1 - conf) + identity * conf    │
│  Frequency-aware: low-freq trust identity, high-freq trust source   │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 1: ENROLL                                                    │
│  expectation.png + photos/* → identity embeddings + canonical atlas │
│  MediaPipe FaceLandmarker (478-point mesh)                          │
│  PnP head pose from 6 key landmarks                                 │
│  Verification gate: embedding distance + face pixels + liveness     │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 2: PER-FRAME PROCESSING (forward path)                       │
│  ┌─ Detect & track ───────────────────────────────────────────────┐ │
│  │  MediaPipe FaceDetector + FaceLandmarker                        │ │
│  │  Identity matching (face_recognition embeddings)                │ │
│  │  Occupancy gate (face_area/bbox_area < 0.25 → reject)          │ │
│  │  No fallback to non-target tracks                               │ │
│  ├─ Geometry ──────────────────────────────────────────────────────┤ │
│  │  478-point landmarks + PnP head pose → SE(2)/SIM(2) transform  │ │
│  │  Canonical warp via LieGroup interpolation                      │ │
│  │  Geometry-based elliptical mask (brightness-invariant)          │ │
│  ├─ Identity ──────────────────────────────────────────────────────┤ │
│  │  Query identity belief state (frequency decomposition)          │ │
│  │  Query intrinsic (albedo/shading/specular) from IntrinsicDecomp │ │
│  │  Query patch memory (pose-conditioned retrieval)                │ │
│  ├─ Render ────────────────────────────────────────────────────────┤ │
│  │  _render_core() — SINGLE source of truth for ALL rendering      │ │
│  │    1. PhysicalRenderer (96%): albedo + shading + specular       │ │
│  │    2. Identity composite fallback: warp anchor face + blend     │ │
│  │    3. Enhancement last resort: sharpen + denoise                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 3: BIDIRECTIONAL SOLVE (offline, optional)                   │
│  Forward pass: collect all frames + quality metrics                 │
│  Temporal solve: future frames repair past frames                   │
│  Render pass: query solved identity for each frame                  │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 4: EXPORT + QC                                               │
│  VideoExporter (1080x1920, H.264, audio muxing)                     │
│  Fade in/out transitions (configurable duration)                    │
│  QC checks: identity drift, sharpness, flicker, face detection rate │
└─────────────────────────────────────────────────────────────────────┘
```

### V3 Module Integration Status

```
Module                Integrated   Active    Validated    Default
──────────────────────────────────────────────────────────────────
IntrinsicDecomposer   ✅ Yes       ✅ 100%   ❌ No        ✅ Yes
PhysicalRenderer      ✅ Yes       ✅ 96%    ❌ No        ✅ Yes
LieGroup SIM(2)       ✅ Yes       ✅ Yes    ⚠️ Partial   ✅ Yes
RendererMode          ✅ Yes       ✅ Yes    ❌ No        ✅ Yes
StateEvolution        ✅ Yes       ✅ Yes    ❌ No        ✅ Yes
EnergyScaler          ✅ Yes       ⚠️ Opt-in ❌ No        ❌ No
OptimizationEngine    ❌ No        ❌ No     ❌ No        ❌ No
DenseGeometry         ❌ No        ❌ No     ❌ No        ❌ No
IdentityManifold      ❌ No        ❌ No     ❌ No        ❌ No
VisibilityCalibration ❌ No        ❌ No     ❌ No        ❌ No
```

**Key:** ACTIVE ≠ VALIDATED. PhysicalRenderer runs 96% of frames but no proof yet that output quality improved over alpha compositing.

### V2 Subsystem Architecture

Face OS decomposes into 4 isolated subsystems (face_os/subsystems/):

1. **Geometry Estimator** — all spatial structure, no identity/lighting logic
2. **Identity Estimator** — stable identity, no RGB blending
3. **Temporal Estimator** — temporal consistency, no texture injection
4. **Renderer** — physically consistent output, no heuristic compositing

---

## Legacy Cricket Pipeline (automation/)

The automation pipeline is managed by `automation/orchestrator.py`, which runs
**9 independently-skippable phases**. Every phase is wrapped in `run_phase()`
(see *Observability* below) so a structured `stage` + `duration_ms` + `status`
record is emitted on success **and** failure, all correlated by a per-run
`run_id`.

| Phase | Stage key | Module | Notes |
|------|-----------|--------|-------|
| 0 | `transcript_fetch` | `automation/transcript.py` | YouTube transcript API → yt-dlp VTT; skips Whisper if segments found |
| 1 | `download` (`drive_pull`) | `download.py` / `sync.py` | yt-dlp + aria2c; or pull inputs from Google Drive |
| 2 | `transcribe` | `transcribe.py` | faster-whisper (Hindi/English), batched on GPU |
| 3 | `highlight` | `highlight.py` | audio RMS + transcript scoring + LLM refinement |
| 4 | `export` | `export.py` + crop planner | 9:16 crop + encode |
| 4.5 | `enhancement` | `ref_grade.py` / `face_mapper.py` | optional color grade / face-region grade |
| 5 | `seo` | `automation/seo/seo.py` | LLM cricket-aware SEO (escalate-not-degrade) |
| 5.5 | `thumbnails` | `thumbnail.py` | thumbnail generation |
| 6 | `sync` | `sync.py` | Google Drive backup |
| 7 | `upload` | `upload.py` | YouTube Data API v3 + jittered scheduling |
| 8 | `analytics` | `automation/seo/analytics.py` | performance metrics → SEO learning feedback |

Two analysis paths exist for crop planning during export:
- **Cheap** (`frame_analyzer.py`): **MediaPipe BlazeFace** via `utils/face_detect.py`
  (`face_detector.tflite`) + dlib `face_recognition` identity match + EMA crop (CPU).
- **Premium** (`premium_analyzer.py` + `premium_render.py`): **YOLOv8-face**
  (GPU, explicit device + batched inference) + ByteTrack + Kalman + bezier crop,
  with RIFE + GFPGAN (GPU).

> **Note:** there is **no Haar Cascade** anywhere in the pipeline. Earlier docs
> and comments referenced "Haar"; the actual cheap detector has been MediaPipe
> BlazeFace. A regression test (`tests/test_face_detection_gpu.py`) guards
> against reintroducing Haar.

### Module Ownership (legacy pipeline)

| Concern | Owner module(s) | Key entry points |
|---|---|---|
| Orchestration / phases | `automation/orchestrator.py` | `run()` |
| Structured logging | `utils/logger.py` | `get_logger()`, `run_phase()`, `new_run_id()` |
| LLM orchestration | `utils/ai_client.py` | `AIClient.generate_text()`, `generate_fastest_first()` |
| Resilience primitives | `utils/resilience.py` | `CircuitBreaker`, `retry_with_backoff` |
| SEO generation | `automation/seo/seo.py` | `generate_clip_seo()`, `process_all_seo()` |
| Cricket facts | `automation/seo/cricket_context.py` | `correct_cricket_spelling()`, `find_canonical_entities()` |
| Trends/scorecard | `automation/seo/trends.py` | `get_trending_context()` |
| Self-learning | `automation/seo/seo_learner.py`, `automation/seo/analytics.py` | `SEOLearner`, `fetch_advanced_metrics()` |
| Transcription | `transcribe.py`, `automation/transcript.py`, `utils/transcript_postproc.py` | `transcribe()`, `fetch()`, `correct_segments()` |
| Upload | `upload.py` | `upload_video()` |
| Crop planning | `frame_analyzer.py` (cheap) / `premium_analyzer.py` (premium) | `FaceDetector`, `analyze_clip()` |

### Reliability & quality behaviors (2026 overhaul)

- **Observability:** `run_phase()` emits start/ok/failed records with
  `stage, status, phase, phase_index, phase_total, run_id, duration_ms,
  error_type, metadata`; failures are logged at the site with `exc_info` (full
  traceback) instead of a bare warning at exit.
- **LLM orchestration:** `generate_fastest_first()` now routes every candidate
  through the shared token-bucket + circuit-breaker, honors HTTP 429
  `Retry-After` (per-provider cooldown), and keeps slow-but-valid responses
  (tier deadline instead of a 5 s `result()` timeout). On total failure it
  returns empty so callers **escalate**, never silently degrade.
- **SEO (escalate-not-degrade):** no generic/template fallback. On unparseable
  output it retries with a stricter JSON-only prompt via the failover chain;
  if still failing it raises `SEOGenerationError` and the clip is **queued**
  (`{clip}_seo_failed.json`, no `_metadata.json` → upload skips it). Shorts keep
  the LLM's short description; player/team/venue names are corrected on the
  transcript **and** title/scorecard before generation; the IPL season is
  derived from the clock (no hardcoded year).
- **Transcription:** cricket spelling correction is centralized
  (`utils/transcript_postproc.py`) and applied to **all** sources (api/vtt/
  whisper), with a guarded lexicon (no `sky→SKY` / `stark→Starc` corruption) and
  validated LLM corrections. `batch_size` now drives a real
  `BatchedInferencePipeline` on GPU.
- **Self-learning:** ingests **real** YouTube Analytics signals (retention/CTR/
  impressions) via `fetch_advanced_metrics()`, fixing the previously-dead
  retention/CTR scoring branches; pattern keys exclude raw numeric features
  (so patterns accumulate); a single `_recompute_best_model()` prefers real
  performance over the synthetic benchmark.
- **Upload:** finite resumable chunk size (real progress + resume), byte-safe
  5000-**byte** description truncation, `categoryId` validation via
  `videoCategories.list`, bounded retries + wall-clock deadline,
  `containsSyntheticMedia`/`madeForKids` from metadata/config.

> The full design rationale, evidence, and per-module findings live in
> `docs/IMPROVEMENT_PLAN.md`.

---

## Key Design Decisions (Face OS)

### Why Geometry-Based Mask (Not Intensity Threshold)
Old: `mask[gray < 5] = 0.0` → beard, shadows, dark skin erased → flicker
New: fixed elliptical geometry mask → brightness-invariant, deterministic

### Why Direct Blend
Both frames use `src * (1-mask) + identity * mask` (not compositor.composite()).
Identity face is already anchor-corrected in canonical space and warped to crop space.
Re-introducing compositor would de-correct the anchor.

### Why EMA at 0.4/0.6
Old 0.7/0.3 caused 10-frame lag (~300ms at 30fps, visible ghosting).
New converges in 5 frames (~150ms), smooths jitter without visible lag.

### Why Last Good Crop Plan
When face is lost mid-clip, `_last_good_crop_plan` preserves the last valid crop position.
Prevents jarring 16:9 full-frame output when face temporarily disappears.

### Why _render_core()
Both `_process_frame_v2()` (forward) and `_render_frame_v2()` (bidirectional) had duplicated rendering logic. This caused the V3 modules to be bypassed in the forward path. `_render_core()` is now the single source of truth for all rendering: PhysicalRenderer → identity composite → enhancement fallback.

---

## Test Suite

Refer to `face_os/STATE.md` and `automation/AGENTS.md` for current test suite status.
To run the full suite:
```bash
.venv/bin/python -m pytest tests/
```

---

## Runtime Validation

Run with `.venv/bin/python validate_metrics.py`

### Latest Dashboard (100 frames, test_clip.mp4)

| Claim | Value | Status |
|---|---|---|
| PhysicalRenderer active | 96.0% | ✅ |
| IntrinsicDecomposer active | 100.0% | ✅ |
| Frame contract (1920x1080x3, uint8) | 50/50 frames | ✅ |
| RendererMode stable | 1 transition | ✅ |
| Avg intrinsic confidence | 0.758 | ✅ |
| Avg decomposition error | 0.053 | ✅ |
| Fallback reason telemetry | renderer_mode_alpha=4 | ✅ |
| No NaN/Inf in output | 50/50 clean | ✅ |
| Telemetry key coverage | 14/14 keys | ✅ |
| PhysicalRenderer dominant | 4% alpha fallback | ✅ |

---

## Configuration

Two config files:

| File | Purpose |
|---|---|
| `face_os_config.yaml` | Face OS tuning (identity, renderer, crop, export, enhancement) |
| `config.yaml` | Legacy pipeline config (download, transcription, premium face-detection, SEO, LLM resilience, upload) |

---

## Stale/Unresolved (Face OS)

| Issue | Status |
|---|---|
| I-01 Duplicate render paths | ✅ FIXED (_render_core()) |
| I-02 Benchmark suite | ❌ PENDING |
| I-03 Normals circular (shading→normals→shading) | ❌ PENDING |
| I-05 Identity anchor RGB-entangled | ❌ PENDING |
| I-07 SIM(2) benefit unmeasured | ❌ PENDING |
| I-09 State prediction (constant velocity) | ❌ PENDING |
| I-10 Stranded modules | ❌ PENDING |
| ARCHITECTURE.md stale | ✅ UPDATED |

See `face_os/STATE.md` for current drift status.
