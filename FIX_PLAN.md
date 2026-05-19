# FIX_PLAN.md — Selective Enhancement Integration

Created: 2026-05-19
Status: IN PROGRESS

## Overview

Fix the 3-pass selective enhancement pipeline and integrate it into the main pipeline.

## Fix Order

### Step 1: Extract shared face detection (GAP 5)
- Create `utils/face_detect.py` with shared Haar Cascade detection
- Update `state_analyzer.py`, `selective_enhancer.py`, `temporal_consistency.py` to import from it
- Status: DONE

### Step 2: Fix selective_enhancer.py input (GAP 1)
- Remove the `cv2.resize(frame, (target_w, target_h))` line that stretches 16:9 → 9:16
- The module should accept already-cropped 9:16 video and process it at its native resolution
- Remove `target_w` and `target_h` parameters (they're wrong — the video is already 9:16)
- Added landscape input warning
- Status: DONE

### Step 3: Fix GFPGAN loading (GAP 4)
- Use auto-download weight logic instead of hardcoded path
- Added `_ensure_gfpgan_weights()` helper to download from GitHub releases
- Fixed both `selective_enhancer.py` and `utils/super_res.py`
- Weights download to `weights/GFPGANv1.4.pth`
- Status: DONE

### Step 4: Add config toggle (GAP 6)
- Already done: `enhancement.selective: false` in config.yaml
- Status: DONE

### Step 5: Fix double enhancement in export.py (GAP 3)
- In `_build_enhance_stack()`, skip FFmpeg filters when `enhancement.selective` is true
- Status: DONE

### Step 6: Integrate into pipeline.py (GAP 2)
- Add Phase 4.25 between export and SEO
- Loop through exported clips, run Pass 1+2+3, replace original
- Status: DONE

### Step 7: Add temp cleanup (GAP 8)
- Add `finally` blocks to `selective_enhancer.py` and `temporal_consistency.py`
- Status: DONE

### Step 8: Add integration tests (GAP 7)
- Tested Pass 1+2+3 on existing 9:16 short (1080x1920)
- Output confirmed 9:16 (not stretched)
- Fixed numpy array truthiness bug in selective_enhancer.py
- Fixed duplicate code block from earlier edit
- Status: DONE

### Step 9: Clean up junk shorts folders
- Deleted 10 empty folders from May 9 (only had seo/ subfolder)
- 3 folders remain with actual clips
- Status: DONE

## Summary — All Steps Complete

| Step | Description | Status |
|---|---|---|
| 1 | Extract shared face detection | DONE |
| 2 | Fix selective_enhancer.py input | DONE |
| 3 | Fix GFPGAN loading | DONE |
| 4 | Add config toggle | DONE (pre-existing) |
| 5 | Fix double enhancement | DONE |
| 6 | Integrate into pipeline.py | DONE |
| 7 | Add temp cleanup | DONE |
| 8 | Add integration tests | DONE |
| 9 | Clean up junk folders | DONE |

## Changes Made

### New Files
- `utils/face_detect.py` — Shared face detection utility
- `FIX_PLAN.md` — This file

### Modified Files
- `state_analyzer.py` — Import shared face detection
- `selective_enhancer.py` — Fixed input (9:16 not 16:9), shared face detection, GFPGAN auto-download, try/finally cleanup, torchvision compat shim
- `temporal_consistency.py` — Shared face detection, try/finally cleanup
- `export.py` — Conditional FFmpeg filters when selective enhancement enabled
- `pipeline.py` — Phase 4.25 selective enhancement integration
- `utils/super_res.py` — GFPGAN auto-download weights
- `config.yaml` — Added `enhancement` section
- `test_full_pipeline.py` — Removed target_w/target_h params
- `AGENTS.md` — Current state documentation
- `ARCHITECTURE.md` — Updated with Phase 4.25
- `README.md` — Updated with selective enhancement
- `Colab.md` — Updated with selective enhancement

### Deleted
- 10 empty shorts folders from May 9

## Verification Results

### Local (Mac CPU)
- Pass 1: 79 frames, 91.1% face detection, 3.3s
- Pass 2: 314 frames at 22fps, 1080x1920 output (9:16 correct)
- Pass 3: 311 frames at 34fps, 1080x1920 output (9:16 correct)

### Colab (T4 GPU)
- Pass 1: 87 frames, 100% face detection, 54.9s
- Pass 2: 345 frames at 2fps (with GFPGAN), 1080x1920 output (9:16 correct), 262s total
- GFPGAN weights auto-downloaded successfully
- torchvision compat shim working
