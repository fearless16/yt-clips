# Conversation Memory & Summary

## Objective
Fix critical issues in the yt-clips YouTube Shorts pipeline (quality, sync, overlay, scoring, logo, clip count) and close the quality gap between pipeline output and `expectation.png` reference.

**Key user constraints:**
- User does NOT stream live cricket — talking-head streams with StreamYard, cricket-themed backgrounds
- Subtitles are NOT required — removed entirely
- Content is Hindi cricket commentary ("CRICKET WITH PRAJJWAL")
- User wants best-in-class solution, not band-aids
- 8GB RAM Mac — pipeline runs on Colab/Kaggle GPU
- No paid APIs (Replicate etc.) — GPU is free and same quality
- Output must match `expectation.png`: face fills entire 9:16 frame, hair visible, sharp details
- User has **Kaggle 2x T4 GPU** (30GB total VRAM, 57.6GB disk, 12hr sessions)

## Implemented Fixes

### 1. Highlight Scoring (`highlight.py`)
- Reduced silence penalty (-3.0→-1.5), word count penalty (-1.5→-0.5)
- Added hook potential scoring (+1.0), emotional arc scoring (+0.8)
- Second-pass fallback at 15% threshold for <4 clips
- Reduced `merge_gap` from 8s→5s in config.yaml

### 2. Super-Resolution (`utils/super_res.py`)
- Switched from `RealESRGAN_x4plus_anime_6B` to `RealESRGAN_x4plus` (23 RRDB blocks)
- Added GFPGAN face restoration after super-res
- Added framerate detection via ffprobe (removed hardcoded 30fps)
- GPU optimizations: cv2 VideoCapture/Writer, CUDA stream, single ffmpeg mux
- H.264 encode via ffmpeg (`libx264 -crf 18 -preset fast`) instead of cv2 mp4v

### 3. Export & Overlay (`export.py`)
- **Fill-crop overlay**: `force_original_aspect_ratio=increase` + `crop` (fills frame edge-to-edge, removes dark background)
- **Circular logo**: 200px, positioned bottom-right (`W-w-30:H-h-280`)
- **Subtitles removed** entirely
- **Post-export A/V sync validation**: `_validate_av_sync()` checks diff > 0.5s
- **Pre-filter degraded mode**: clips failing face detection get center-crop fallback
- **Hair headroom**: `_TOP_PAD_RATIO` 0.10→0.25, min 150px
- **Face crop**: `face_height × 3.0` (headroom 80% + body 120%)
- **Color boost**: `eq=saturation=1.35:contrast=1.08:brightness=0.03` after lighting fix

### 4. Face Crop (`frame_analyzer.py`)
- `_apply_top_padding()`: face_height-based crop with generous headroom
- Face area increased from 10% to ~15%
- Position: 20-27% from top (source video limit)

### 5. Trends (`trends.py`)
- Removed blind Cricbuzz fallback (returns empty scorecard when no match)

### 6. Code Quality
- Fixed all import names (`get_logger`, `retry_with_backoff`, `load_config`)
- All 33 Python files compile successfully
- All 26 imports pass (including pipeline, export, frame_analyzer, etc.)

### 7. Kaggle Notebook (`Kaggle.ipynb`)
- 7 cells: GPU check, system deps, Python deps (RealESRGAN + GFPGAN + CodeFormer + RIFE + DeepFace + Video2X), model verification, torchvision compat, worker + tunnel, monitor
- Optimized for Kaggle 2x T4 GPU

### 8. Cleanup
- Removed: `proof_new/` (46M), `diagnostics/` (9.5M), `diagnostics_clip2/` (3M), `temp/` (8.5M), `scratch/` (3.9M), `graphify-out/` (7.6M), `photos/` (2M), `proof.html` (3.3M), root debug images, `benchmark_llms.py`, `remote_job_result.json`, `opencode.sh`, `.push_cache.json`, 73 empty shorts dirs
- Added: `analyze_faces.py`, `viewer.html`, `Kaggle.ipynb`, `memory.md`

## Key Technical Decisions
- **Fill-crop** replaces dark-bg overlay for 16:9→9:16 conversion
- **ffmpeg H.264** over cv2 mp4v for video encode quality
- **GFPGAN** after Real-ESRGAN for face restoration
- **2026 research**: FlashVSR (A100 80GB), STCDiT (24GB+), Vivid-VR (25-43GB) — all too heavy for Kaggle T4; Real-ESRGAN + CodeFormer + RIFE is best practical pipeline
- **CodeFormer > GFPGAN** for face restoration (native video, adjustable fidelity)

## Git History (main, ahead of origin by 0)
- `1e1cecd` — Add Kaggle notebook, viewer HTML, face analysis script, memory notes; remove opencode.sh
- `5c0b766` — Face crop repositioning (headroom 25%, face area 15%)
- `9e84c45` — Replace cv2 mp4v with ffmpeg H.264 encode
- `b8af6ff` — GPU optimizations (cv2 I/O, CUDA stream, ffmpeg mux)
- `c8562f9` — CodeFormer research, Kaggle notebook
- `0cb87ae` — Remove unused files, fix imports, add memory.md
- `d67132b` — Hair headroom fix (_TOP_PAD_RATIO 0.25)
- `3839271` — Fill-crop overlay, circular logo, color boost
- `7b26d9c` — Previous session commits

## Key Metrics
- **A/V sync**: 0.006s diff (tolerance 0.5s)
- **Sharpness**: source=43, output=176, expectation=274 (GPU super-res needed)
- **Saturation**: output=96, after boost=118.7, expectation=133 (histogram matching needed)
- **Face position**: output=20-27% from top, expectation=41% (source video limit)
- **Detection rate**: 75% (3/4 frames)
- **Config**: fps=60, crf=18, super_resolution=true, merge_gap=5

## Next Steps
1. Run full pipeline on Kaggle 2x T4 — verify Real-ESRGAN + GFPGAN quality
2. Integrate CodeFormer as alternative to GFPGAN
3. Add histogram matching for color grading
4. Add `channel_logo.png` to test circular logo overlay
5. Test with longer video (30+ min) to validate highlight scoring
6. Update `memory.md` with results after Kaggle run

## Remaining Gaps
- GPU super-res + GFPGAN cannot be tested locally (no GPU on Mac)
- Logo circular mask cannot be visually verified (no `channel_logo.png`)
- Source video itself is blurry (sharpness=43-58) — only GPU super-res can improve
- Face position limited by source video framing
