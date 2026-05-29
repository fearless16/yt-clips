# Architecture

> Face OS status note: use `face_os/STATE.md` as the current source of truth.
> Runtime percentages and test counts below are historical snapshots unless
> repeated in `face_os/STATE.md`.

Two parallel systems co-exist in this codebase:

1. **Face OS** (primary) — Identity-reconstruction pipeline for portrait-mode studio video
2. **Legacy cricket pipeline** — 16:9 live stream → 9:16 shorts (Haar/YOLO + GFPGAN)

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

**Key:** ACTIVE ≠ VALIDATED. PhysicalRenderer runs 96% of frames but no proof yet that output quality improved over alpha compositing. See `AGAINST.md`.

### V2 Subsystem Architecture

Face OS decomposes into 4 isolated subsystems (face_os/subsystems/):

1. **Geometry Estimator** — all spatial structure, no identity/lighting logic
2. **Identity Estimator** — stable identity, no RGB blending
3. **Temporal Estimator** — temporal consistency, no texture injection
4. **Renderer** — physically consistent output, no heuristic compositing

---

## Legacy Cricket Pipeline (automation/)

The automation pipeline is managed by `automation/orchestrator.py` which executes in 8 skippable phases:

1. **Download (Phase 1):** `download.py` (yt-dlp + aria2c)
2. **Transcribe (Phase 2):** `transcribe.py` (faster-whisper, Hindi/English)
   - *Phase 2.5 (Video Analysis):* `video_analyzer.py` (face/lighting map)
3. **Highlight Detection (Phase 3):** `highlight.py` (Audio RMS + transcript scoring + LLM refinement)
4. **Export Shorts (Phase 4):** `export.py` + `frame_analyzer.py` (crop + encode + Haar/YOLO)
   - *Phase 4.25 (Enhancement):* `ref_grade.py` or `face_mapper.py`
   - *Phase 4.5 (SEO & Thumbnails):* `seo.py` + `thumbnail.py`
5. **SEO Generation (Phase 5):** `automation/seo/seo.py` (LLM-based cricket-aware SEO)
6. **Sync (Phase 6):** `sync.py` (Google Drive backup)
7. **Upload (Phase 7):** `upload.py` (YouTube Data API v3 + jittered scheduling)
8. **Analytics (Phase 8):** `automation/seo/analytics.py` (Performance metrics + learning feedback)

Two analysis paths exist for crop planning during export:
- **Cheap** (`frame_analyzer.py`): Haar Cascade / OpenCV DNN + heuristics (CPU)
- **Premium** (`premium_analyzer.py` + `premium_render.py`): YOLOv8-face + ByteTrack + Kalman + RIFE + GFPGAN (GPU)

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
| `config.yaml` | Legacy pipeline config (download, transcription, premium toggle) |

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

See `AGAINST.md` for full analysis.
