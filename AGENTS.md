# AGENTS.md — Current State, Gaps & Fix Plan

Last updated: 2026-05-19

## Current State Summary

The codebase has a **working pipeline** (download → transcribe → highlight → export → SEO → upload) that produces 9:16 Shorts from 16:9 YouTube VODs. A **3-pass selective enhancement pipeline** was added as standalone modules but is **NOT integrated** into the main pipeline and has critical bugs.

### What Works
- Full 6-phase pipeline via `pipeline.py`
- 16:9 → 9:16 smart cropping in `export.py` (face tracking, center crop, chat exclusion)
- Cheap analysis: `frame_analyzer.py` (Haar Cascade + heuristics)
- Premium analysis: `premium_analyzer.py` (YOLOv8-face + ByteTrack + Kalman)
- Premium render: `premium_render.py` (RIFE interpolation + GFPGAN + two-pass VBR)
- Super-resolution: `utils/super_res.py` (Real-ESRGAN 4x + GFPGAN + reference guidance)
- SEO generation with 3-tier fallback + self-improving loop
- Colab/Kaggle bridge architecture (tunnel + watcher + job queue)
- 219+ tests

### What's Broken / Not Integrated

#### CRITICAL: 3-Pass Enhancement Pipeline

Three new modules exist but are **standalone** — not called from `pipeline.py`:

| Module | Purpose | Status |
|---|---|---|
| `state_analyzer.py` | Pass 1: Per-frame state classification (heavy/light/skip) | Standalone only |
| `selective_enhancer.py` | Pass 2: Conditional GFPGAN/sharpening/propagation | Standalone only, **wrong input** |
| `temporal_consistency.py` | Pass 3: Flicker removal + drift correction | Standalone only |
| `test_full_pipeline.py` | End-to-end test of Pass 1+2+3 | Works but **produces wrong output** |

---

## Identified Gaps (Priority Order)

### GAP 1: Wrong Pipeline Order (BROKEN OUTPUT)

**Problem**: `test_full_pipeline.py` feeds raw 16:9 video into `selective_enhancer.py`, which does:
```python
frame = cv2.resize(frame, (target_w, target_h))  # 640x360 → 1080x1920
```
This **stretches** the 16:9 frame to 9:16 — the "sandwiched/flat" look the user reported.

**Correct order**:
```
16:9 source → export.py (smart crop) → 9:16 clip → selective_enhancer → temporal_consistency → final
```

**Fix**: The 3-pass enhancement must run on `export.py`'s output (already-cropped 9:16 video), NOT on raw source.

### GAP 2: Not Integrated into Pipeline

`pipeline.py` has no awareness of the 3-pass enhancement. The export phase ends at Phase 4, and the enhancement modules are never called.

**Fix**: Add Phase 4.25 (Selective Enhancement) between export and SEO, controlled by a config toggle.

### GAP 3: Double Enhancement

`export.py` already applies per-frame FFmpeg filters:
```python
# export.py line 600
filter_base += ",unsharp=5:5:1.0:5:5:0.0,eq=saturation=1.15:contrast=1.15:brightness=0.04"
```

If `selective_enhancer.py` also runs, frames get sharpened + contrast-boosted **twice**.

**Fix**: When selective enhancement is enabled, disable the FFmpeg filters in `export.py` (or make them conditional).

### GAP 4: GFPGAN Never Actually Works

`selective_enhancer.py` hardcodes:
```python
model_path="experiments/pretrained_models/GFPGANv1.4.pth"
```
This file doesn't exist. The module logs a warning and falls back to mild OpenCV sharpening — no actual face restoration happens.

**Fix**: Use `utils/super_res.py`'s model loading logic (auto-download weights) or share the `SuperResEnhancer` instance.

### GAP 5: Duplicate Face Detection

Three separate Haar Cascade instances:
- `state_analyzer.py` → `_detect_face()`
- `selective_enhancer.py` → `_detect_face()`
- `temporal_consistency.py` → `_detect_face()`

Plus `frame_analyzer.py` and `premium_analyzer.py` have their own detection.

**Fix**: Extract shared face detection into a common utility (or reuse `utils/face_matcher.py`).

### GAP 6: No Config Toggle

`config.yaml` has no setting to enable/disable selective enhancement or its sub-features.

**Fix**: Add to config.yaml:
```yaml
enhancement:
  selective: false          # Enable 3-pass selective enhancement
  gfpgan_strength: 0.7     # Face restoration strength
  temporal_alpha: 0.7      # Temporal smoothing (0=smooth, 1=raw)
  drift_threshold: 65      # Identity drift detection threshold
```

### GAP 7: No Pipeline Integration Tests

No tests verify that the 3-pass enhancement integrates correctly with `export.py`.

### GAP 8: Temp Files Not Cleaned on Failure

If `selective_enhancer.py` or `temporal_consistency.py` crash mid-process, `/tmp/yt_clips_enhance/` and `/tmp/yt_clips_consistency/` are left behind.

### GAP 9: Doc Drift

`ARCHITECTURE.md` and `README.md` don't mention the 3-pass enhancement pipeline at all.

---

## Correct Architecture (Target State)

```
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1: DOWNLOAD (yt-dlp + aria2c)                             │
├──────────────────────────────────────────────────────────────────┤
│  PHASE 2: TRANSCRIBE (faster-whisper)                            │
├──────────────────────────────────────────────────────────────────┤
│  PHASE 2.5: VIDEO ANALYSIS (face/lighting map)                   │
├──────────────────────────────────────────────────────────────────┤
│  PHASE 3: HIGHLIGHT DETECTION (audio RMS + transcript + AI)      │
├──────────────────────────────────────────────────────────────────┤
│  PHASE 4: EXPORT (16:9 → 9:16 crop + encode)                     │
│  ┌─ Standard: Haar/EMA → FFmpeg crop → single-pass encode ──┐   │
│  ┌─ Premium:  YOLO/ByteTrack → Kalman/bezier → RIFE+GFPGAN ┐│   │
│  └─────────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────┤
│  PHASE 4.25: SELECTIVE ENHANCEMENT (NEW — config toggle)         │
│  ┌─ Pass 1: state_analyzer.py — per-frame classification ────┐  │
│  │   heavy/light/skip based on mouth, eyes, pose, lighting   │  │
│  ├─ Pass 2: selective_enhancer.py — conditional enhancement ─┤  │
│  │   heavy: GFPGAN face restore                              │  │
│  │   light: conservative sharpen + color                     │  │
│  │   skip:  temporal propagation from nearest enhanced       │  │
│  ├─ Pass 3: temporal_consistency.py — flicker removal ───────┤  │
│  │   IIR face smoothing, drift correction, boundary blend    │  │
│  └─────────────────────────────────────────────────────────────┘│
│  Input:  export.py output (already 9:16 cropped)                 │
│  Output: enhanced 9:16 video (replaces export output)            │
├──────────────────────────────────────────────────────────────────┤
│  PHASE 4.5: SEO & THUMBNAILS                                     │
├──────────────────────────────────────────────────────────────────┤
│  PHASE 5: SYNC (optional)                                        │
├──────────────────────────────────────────────────────────────────┤
│  PHASE 6: UPLOAD (optional)                                      │
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Rules

1. **Selective enhancement operates on 9:16 cropped video** — never on raw 16:9 source
2. **When selective enhancement is ON, disable FFmpeg filters** in `export.py` (no double processing)
3. **GFPGAN must use auto-download weights** — share logic with `utils/super_res.py`
4. **Face detection should be shared** — not duplicated across 3 modules
5. **Config toggle** controls the entire 3-pass pipeline
6. **Temp files cleaned up** in `finally` blocks

### Integration Point in `pipeline.py`

```python
# After Phase 4 (export), before Phase 4.5 (SEO):
if cfg.get("enhancement", {}).get("selective", False):
    from selective_enhancer import enhance_clip
    from temporal_consistency import apply_temporal_consistency
    from state_analyzer import analyze_clip

    for clip_path in exported:
        # Pass 1: Analyze
        analysis = analyze_clip(str(clip_path))
        # Pass 2: Enhance (in-place or temp file)
        enhanced = enhance_clip(str(clip_path), analysis_path=...)
        # Pass 3: Temporal consistency
        final = apply_temporal_consistency(enhanced, analysis_path=...)
        # Replace original
        Path(final).rename(clip_path)
```

### Integration Point in `export.py`

When `enhancement.selective` is true, skip the FFmpeg enhancement filters:
```python
# In _build_enhance_stack():
if not cfg.get("enhancement", {}).get("selective", False):
    filter_base += ",unsharp=5:5:1.0:5:5:0.0,eq=saturation=1.15:contrast=1.15:brightness=0.04"
```

---

## Colab/Kaggle Status

### Tunnel Architecture
```
Local Mac                    Google Drive              Colab T4
─────────                    ────────────              ────────
automate.sh → push_code.py → code files ────────────→ colab_setup.py
automate.sh → bridge.py ───→ job file ───────────────→ watcher.py
                             shorts/ ←───────────────── pipeline.py
```

### Known Colab Issues
- Flask app on port 5000 blocks watcher.py — must `!fuser -k 5000/tcp` first
- Ngrok tunnel goes offline frequently (ERR_NGROK_3200)
- Drive-mounted paths cause cv2 VideoWriter corruption — use `/tmp/` for intermediate files
- GFPGAN weights must be downloaded on Colab (not included in repo)

### Colab GPU Memory Budget (T4, 16GB VRAM)
| Operation | VRAM | Notes |
|---|---|---|
| YOLOv8-face | ~0.5 GB | Always loaded in premium mode |
| GFPGAN | ~2.5 GB | Loaded on demand |
| Real-ESRGAN 4x | ~3 GB | Loaded on demand |
| FILM/RIFE | ~3 GB | Frame interpolation |
| Whisper (CUDA) | ~1 GB | Transcription only |
| Peak total | ~10 GB | Fits in T4's 16GB |

---

## File Reference

### Core Pipeline
| File | Lines | Purpose |
|---|---|---|
| `pipeline.py` | 371 | Main orchestrator — 6 phases |
| `download.py` | — | yt-dlp + aria2c |
| `transcribe.py` | — | faster-whisper |
| `highlight.py` | — | Audio RMS + transcript scoring |
| `export.py` | 1208 | 16:9→9:16 crop + FFmpeg encode |
| `seo.py` | — | SEO generation (3-tier fallback) |
| `upload.py` | — | YouTube API upload |
| `sync.py` | — | Google Drive sync |

### Analysis
| File | Lines | Purpose |
|---|---|---|
| `frame_analyzer.py` | 702 | Cheap: Haar Cascade + heuristics |
| `premium_analyzer.py` | 956 | Premium: YOLOv8 + ByteTrack + Kalman |
| `video_analyzer.py` | 642 | Pre-analysis: face/lighting map for full VOD |
| `state_analyzer.py` | 774 | **NEW**: Per-frame enhancement classification |

### Enhancement
| File | Lines | Purpose |
|---|---|---|
| `premium_render.py` | 397 | Premium: RIFE + GFPGAN + two-pass VBR |
| `utils/super_res.py` | 439 | Real-ESRGAN 4x + GFPGAN + reference |
| `selective_enhancer.py` | 627 | **NEW**: Conditional enhancement (Pass 2) |
| `temporal_consistency.py` | 489 | **NEW**: Flicker removal (Pass 3) |

### Infrastructure
| File | Lines | Purpose |
|---|---|---|
| `watcher.py` | 230 | Colab/Kaggle job listener |
| `bridge.py` | — | Local→cloud job pusher |
| `push_code.py` | — | Code sync to Drive |
| `colab_setup.py` | — | Colab dependency installer |

### Config & Docs
| File | Purpose |
|---|---|
| `config.yaml` | All configuration (193 lines) |
| `ARCHITECTURE.md` | Pipeline design, GPU/CPU split |
| `README.md` | Quick start, features, project structure |
| `Colab.md` | Colab setup instructions |
| `AGENTS.md` | **This file** — current state & gaps |

---

## Next Steps (Ordered)

1. **Fix selective_enhancer.py input** — accept 9:16 video, not raw 16:9
2. **Add config toggle** — `enhancement.selective: false` in config.yaml
3. **Integrate into pipeline.py** — Phase 4.25 after export
4. **Fix double enhancement** — conditional FFmpeg filters in export.py
5. **Fix GFPGAN loading** — use auto-download or share with SuperResEnhancer
6. **Extract shared face detection** — common utility module
7. **Add temp cleanup** — finally blocks in all 3 modules
8. **Add integration tests** — test selective enhancement with export output
9. **Update docs** — ARCHITECTURE.md, README.md, Colab.md
