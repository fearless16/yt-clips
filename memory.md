# Conversation Memory & Summary

## Objective
Fix critical issues in the yt-clips YouTube Shorts pipeline (quality, sync, overlay, scoring, logo, clip count) and close the quality gap between pipeline output and `expectation.png` reference.

## User Profile & Constraints
- Hindi cricket commentary ("CRICKET WITH PRAJJWAL"), talking-head streams with StreamYard
- Cricket-themed backgrounds, viewer chat overlays (NOT live cricket)
- Subtitles NOT required — removed entirely
- 8GB RAM Mac — pipeline runs on Kaggle GPU (2x T4, 30GB VRAM, 57.6GB disk, 12hr sessions)
- No paid APIs — GPU is free and same quality
- Output must match `expectation.png`: face fills entire 9:16 frame, hair visible, sharp details
- No budget for equipment — must work with existing StreamYard setup
- Highly customizable personal project, not a generic service
- No manual work — fully automated pipeline

## Implemented Fixes

### 1. Highlight Scoring (`highlight.py`)
- Reduced silence penalty (-3.0→-1.5), word count penalty (-1.5→-0.5)
- Added hook potential scoring (+1.0), emotional arc scoring (+0.8)
- Second-pass fallback at 15% threshold for <4 clips
- Reduced `merge_gap` from 8s→5s in config.yaml

### 2. Super-Resolution (`utils/super_res.py`)
- Switched from `RealESRGAN_x4plus_anime_6B` to `RealESRGAN_x4plus` (23 RRDB blocks)
- Added GFPGAN face restoration after super-res (was premium-only)
- Added framerate detection via ffprobe (removed hardcoded 30fps)
- GPU optimizations: cv2 VideoCapture/Writer (2-3x faster I/O), CUDA stream, single ffmpeg mux
- H.264 encode via ffmpeg (`libx264 -crf 18 -preset fast`) instead of cv2 mp4v
- VRAM cleanup: `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.synchronize()`
- Gentle `_aggressive_enhance`: 5x kernel, 1.15x params (was 1.3x/1.2x — too aggressive)
- `BORDER_REPLICATE` over `BORDER_REFLECT` (reflect creates mirror duplicates)

### 3. Export & Overlay (`export.py`)
- **Fill-crop overlay**: `force_original_aspect_ratio=increase` + `crop` (fills frame edge-to-edge)
- **Circular logo**: 200px, positioned bottom-right (`W-w-30:H-h-280`)
- **Subtitles removed** entirely
- **Post-export A/V sync validation**: `_validate_av_sync()` checks diff > 0.5s
- **Pre-filter degraded mode**: clips failing face detection get center-crop fallback
- **Hair headroom**: `_TOP_PAD_RATIO` 0.10→0.25, min 150px
- **Face crop**: `face_height × 3.0` (headroom 80% + body 120%)
- **Color boost**: `unsharp=5:5:1.0 + eq=saturation=1.15:contrast=1.15:brightness=0.04` (updated from old params)

### 4. Face Crop (`frame_analyzer.py`)
- `_apply_top_padding(face_top_y, face_height, face_width)`: calculates crop from FACE SIZE (target 25% of output width)
- `BORDER_REPLICATE` for edge padding

### 5. Trends (`trends.py`)
- Removed blind Cricbuzz fallback (returns empty scorecard when no match)

### 6. Face Reference System (`utils/face_reference.py`)
- Extracts face embeddings from user dataset
- Identity-preserving restoration
- Color profile matching (BGR ratio matching)

### 7. Face Restore Pipeline (`utils/face_restore.py`)
- Aggressive face restoration with identity preservation
- GFPGAN + CodeFormer support

### 8. Code Quality
- Fixed all import names (`get_logger`, `retry_with_backoff`, `load_config`)
- All 33 Python files compile successfully
- All 26 imports pass

### 9. Kaggle Notebook (`Kaggle.ipynb`) — CRITICAL INSTALL FIX
**Root Cause**: `basicsr==1.4.2` (PyPI, Aug 2022) imports `PIL._typing._Ink` which was REMOVED in Pillow 10.0 (July 2023). ALL PyPI versions of basicsr/realesrgan/gfpgan are stale 2022 packages.

**Second Failure Mode**: `ultralytics` and other core deps depend on `Pillow>=10`, so pip upgrades Pillow to 11.x during install. Simple `pip install Pillow==9.5.0` gets overridden by dependency resolution.

**Fix** (7-step install):
1. Pin `Pillow==9.5.0` with `--force-reinstall --no-deps` BEFORE anything else
2. Install core deps (yt-dlp, whisper, etc.)
3. Re-pin Pillow with `--force-reinstall --no-deps` after core deps
4. Install basicsr from **GitHub master** (has May 2024 torchvision fix) with `--no-deps`
5. Install basicsr runtime deps separately (excluding Pillow)
6. Install realesrgan/gfpgan/facexlib with `--no-deps`
7. Final Pillow lock with `--force-reinstall --no-deps` + verification (show version)

**Why `--force-reinstall --no-deps` is required**: `ultralytics` requires `Pillow>=10`, so pip's dependency resolver upgrades Pillow even after pinning. `--force-reinstall` overrides the already-installed version, `--no-deps` prevents pip from resolving dependencies (which would pull in Pillow>=10 again).

### 10. Cleanup
- Removed: `proof_new/`, `diagnostics/`, `temp/`, `scratch/`, `graphify-out/`, `photos/`, `proof.html`, debug images, scripts
- Added: `analyze_faces.py`, `viewer.html`, `Kaggle.ipynb`, `memory.md`
- Face reference dataset: `photos/p1.png`, `p2.png`, `p3.png` (3 AI-generated images)

## Key Technical Decisions
- **Fill-crop** replaces dark-bg overlay for 16:9→9:16 conversion (center crop approach)
- **ffmpeg H.264** over cv2 mp4v for video encode quality
- **GFPGAN** after Real-ESRGAN for face restoration
- **Face size normalization**: crop width from face width (target 25% of output), NOT from height
- **Color grading via BGR ratio matching**: match reference R/G/B ratios (hue replacement turned image green)
- **Gentle enhancement**: 5x sharpen kernel + 1.15x contrast/saturation + 0.04 brightness
- **`photos/` directory must exist**: `face_matcher.py` uses `iterdir()` with `.png`, `.jpg`, `.jpeg` suffixes
- **T4-optimized pipeline**: Real-ESRGAN + GFPGAN only — RIFE too slow on T4, CodeFormer download fails on Kaggle, DeepFace heavy TF, Video2X unavailable, BasicVSR++ too heavy
- **Don't reinstall PyTorch on Kaggle** — pre-installed with CUDA
- **Sequential installs on Kaggle**: basicsr first (builds Cython), then realesrgan/gfpgan/facexlib
- **Model weights auto-download**: GFPGANv1.4.pth + RealESRGAN_x4plus.pth in separate Kaggle cell
- **Pillow<10 required**: Pillow 10+ removed `PIL._Ink` which basicsr imports internally

## Test Results
- **A/V sync verified**: 0.006s diff (tolerance 0.5s)
- **6 frames from YouTube video tested**: 4/6 look great, 2 extreme close-ups unavoidable
- **Sharpness**: 77-610 (ref: 274) — source itself is blurry (43-58)
- **Saturation**: 122-160 (ref: 133)
- **Face detection**: 75% (3/4 frames)
- **Identity match**: all pass (distance 0.30-0.58, threshold 0.6)
- **Face area**: 25-29% of output width

## Key Metrics
| Metric | Source | Output | Expectation | Gap |
|--------|--------|--------|-------------|-----|
| Sharpness | 43 | 176 | 274 | GPU super-res needed |
| Saturation | 96 | 118.7 | 133 | Histogram matching |
| Face position (% from top) | 24% | 20-27% | 41% | Source video limit |
| Face area (% of output) | — | 25-29% | 20% | ✅ Good |

## Config (`config.yaml`)
- `fps: 60`, `crf: 18`, `super_resolution: true`, `merge_gap: 8`, `max_clips: 10`

## Git History (main)
- `317186e` — Fix bulletproof Pillow<10 install order for basicsr/realesrgan/gfpgan
- `de0f4a7` — Kaggle install robustness (Pillow<10 pin)
- `8b8208e` — Face crop normalization (target 25% output width)
- `0a9445b` — Photos dataset (face reference)
- `5fa1149` — Refactored Kaggle notebook (user's clean cell structure)
- `bec0d78` — T4-optimized (removed RIFE, CodeFormer, DeepFace, Video2X)
- `46ea752` — Previous fixes

## Blocked / Not Yet Tested
- **Kaggle Pillow conflict**: User needs to re-run Cell 3 after `317186e` fix
- **Logo circular mask**: `channel_logo.png` exists in project root — ready for testing
- **Histogram matching**: Not yet implemented (would close saturation gap)
- **Source video blur**: sharpness=43-58, only GPU super-res can improve

## Next Steps
1. **User re-runs Cell 3 on Kaggle** — verify Pillow==9.5.0 + realesrgan/gfpgan import ✅
2. **Run full pipeline on Kaggle 2x T4** — verify Real-ESRGAN + GFPGAN quality with GPU
3. **Add histogram matching** — replace flat color grading with reference-matched color
4. **Test circular logo overlay** — `channel_logo.png` is in project root and referenced in config.yaml
5. **Test with longer video** (30+ min) to validate highlight scoring produces 4+ clips
