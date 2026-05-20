# AGENTS.md — Current State, Gaps & Fix Plan

Last updated: 2026-05-19

## Current State Summary

The codebase has a **working pipeline** (download → transcribe → highlight → export → SEO → upload) that produces 9:16 Shorts from 16:9 YouTube VODs. **Reference-derived color grading** (`ref_grade.py`) is integrated as Phase 4.25 with a `--mode` CLI flag.

### What Works
- Full 6-phase pipeline via `pipeline.py` with `--mode {face_mapper,ref_grade}` flag
- 16:9 → 9:16 smart cropping in `export.py` (face tracking, center crop, chat exclusion)
- **ref_grade.py**: Target-based color grading — enrollment-once, apply-always. Blends source TOWARD reference (not beyond). LUT-based a,b transform, cached vignette, split-tone LUT. 128 tests pass.
- Cheap analysis: `frame_analyzer.py` (Haar Cascade + heuristics)
- Premium analysis: `premium_analyzer.py` (YOLOv8-face + ByteTrack + Kalman)
- Premium render: `premium_render.py` (RIFE interpolation + GFPGAN + two-pass VBR)
- Super-resolution: `utils/super_res.py` (Real-ESRGAN 4x + GFPGAN + reference guidance)
- SEO generation with 3-tier fallback + self-improving loop
- Colab/Kaggle bridge architecture (tunnel + watcher + job queue)
- `video_analyzer.py`: Auto-detect CUDA/VideoToolbox hwaccel
- `push_code.py`: Syncs `tests/*.py`, prevents Drive "(1)" duplicates
- `monitor.py`: Poll Colab pipeline status via tunnel
- **128 tests pass, 1 skipped, 0 failures**

### T4 GPU Performance
| Operation | T4 CPU | Mac M1 |
|---|---|---|
| 1080p apply_grade | 8fps (130ms) | 29fps (35ms) |
| 720p grade_video pipe | 8fps | 14fps |
| Enrollment | 0.5s | 0.3s |

### Parameter Tuning (Current)
ref_grade uses TARGET-BASED BLENDING (not multipliers):
- **L brightness**: Blends 25% toward reference mean per frame
- **a,b LUTs**: Blend toward reference target by 25%
- **Contrast**: Moderate stretch (ratio ~1.22) centered on per-frame mean
- **Split tone**: Gentle (shadow=0.06, highlight=0.04)
- **Vignette**: Cached by resolution

### Known Issues
1. **Face detection fails on portrait content**: Haar cascade can't find faces in cropped 9:16 clips → export falls back to "degraded center-crop mode"
2. **Colab inactivity timeout**: ngrok tunnel disconnects after ~10 min of inactivity. `monitor.py --watch` pings health endpoint to prevent this.
3. **Drive sync latency**: Google Drive mount on Colab has caching delays (30-60s). Files may appear stale.
4. **Drive "(1)" duplicates**: Fixed in push_code.py with `_find_file_by_name` fallback.

### Abandoned / Not Integrated
| Module | Status |
|---|---|
| `state_analyzer.py` | Abandoned (3-pass pipeline) |
| `selective_enhancer.py` | Abandoned (3-pass pipeline) |
| `temporal_consistency.py` | Abandoned (3-pass pipeline) |

---

## File Reference

### Core Pipeline
| File | Lines | Purpose |
|---|---|---|
| `pipeline.py` | 452 | Main orchestrator — 6 phases + `--mode` flag |
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
| `video_analyzer.py` | 655 | Pre-analysis: face/lighting map + auto hwaccel |

### Enhancement
| File | Lines | Purpose |
|---|---|---|
| `ref_grade.py` | 327 | Target-based color grade (LUT + vignette cache) |
| `face_mapper.py` | 642 | Per-frame 6-step pipeline + region-aware grading |
| `premium_render.py` | 397 | Premium: RIFE + GFPGAN + two-pass VBR |
| `utils/super_res.py` | 439 | Real-ESRGAN 4x + GFPGAN + reference |

### Infrastructure
| File | Lines | Purpose |
|---|---|---|
| `watcher.py` | 230 | Colab/Kaggle job listener |
| `bridge.py` | 150 | Local→cloud job pusher |
| `push_code.py` | 269 | Code sync to Drive (tests/*.py, no duplicates) |
| `colab_setup.py` | — | Colab dependency installer |
| `monitor.py` | 202 | Poll Colab pipeline status via tunnel |

### Tests
| File | Tests | Purpose |
|---|---|---|
| `tests/test_ref_grade.py` | 37 | ref_grade enrollment, grading, flicker, video |
| `tests/test_face_mapper.py` | 35 | face_mapper enhancement |
| `tests/test_video_analyzer.py` | 37 | video_analyzer analysis |
| `tests/test_t4_compat.py` | 17 | T4 GPU compatibility (also runs under pytest) |
| **Total** | **128** | **0 failures** |

### Config & Docs
| File | Purpose |
|---|---|
| `config.yaml` | All configuration (213 lines) |
| `AGENTS.md` | **This file** — current state & gaps |

---

## Next Steps

1. **Reconnect Colab** and re-run with new target-based params
2. **Fix face detection** for portrait content (use YOLO or larger Haar cascade)
3. **Tune parameters** based on re-run results (L blend strength, contrast ratio)
4. **Add `--mode face_mapper`** test on real content
5. **Update docs** — ARCHITECTURE.md, README.md
