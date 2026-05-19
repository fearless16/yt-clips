# yt-clips — AI-Powered YouTube Shorts Automation

Convert 16:9 live streams → studio-grade 9:16 shorts automatically. Two modes:

## Quick Start

```bash
# Local (Mac/PC — CPU only, no GPU)
./automate.sh "https://youtu.be/VIDEO_ID"   # select option 1

# Remote GPU (Kaggle 2x T4 — recommended)
# 1. Open Kaggle notebook, run all cells
# 2. From Mac:
./automate.sh "https://youtu.be/VIDEO_ID"   # select option 2
python kaggle_monitor.py --monitor            # watch progress
```

## Features

### Cheap Mode (default — works anywhere)
| Component | Tech |
|---|---|
| Face detection | OpenCV Haar Cascade |
| Crop smoothing | EMA filter |
| Layout detection | Heuristic (edge + brightness) |
| Frame interpolation | FFmpeg `framerate` filter |
| Encoding | Single-pass CRF (h264_videotoolbox/h264_nvenc/libx264) |

### Premium Mode (Colab T4 — enable `premium.enabled: true`)
| Component | Tech | Quality Gain |
|---|---|---|
| Face detection | YOLOv8-face | ~95% acc vs ~60% (Haar) |
| Face tracking | ByteTrack (Kalman + IoU) | Persistent IDs across VOD |
| Crop trajectory | Kalman filter + cubic bezier | Buttery smooth, no jitter |
| Layout classification | Smart heuristic | solo/dual/screen-share/black |
| Chat detection | Edge + variance analysis | Excludes chat from crop |
| Frame interpolation | RIFE / FFmpeg | 30→60fps, real motion |
| Face enhancement | GFPGAN | Eye/skin detail on keyframes |
| Encoding | Two-pass VBR | Optimal bit allocation |
| Speed variation | Gaussian-smoothed 1.0-1.25x | Dynamic pacing |

### Selective Enhancement (Phase 4.25 — enable `enhancement.selective: true`)
3-pass enhancement on 9:16 cropped output. Fixes flicker, ghosting, uncanny results.

| Pass | Module | What It Does |
|---|---|---|
| Pass 1 | `state_analyzer.py` | Per-frame classification: heavy/light/skip based on mouth, eyes, pose, lighting |
| Pass 2 | `selective_enhancer.py` | Conditional: GFPGAN (heavy), sharpen (light), propagate (skip) |
| Pass 3 | `temporal_consistency.py` | IIR face smoothing, drift correction, boundary blending |

**Key design:** Operates on 9:16 cropped video from Phase 4, NOT raw 16:9 source. When enabled, FFmpeg filters in export.py are disabled to prevent double processing.

### Pre-Generation Test Guard
Controlled by `testing.enabled` in config.yaml (default: `false` for speed).
Set `testing.enabled: true` to auto-run `pytest tests/ -x --timeout=120` before any operation.
Use `--skip-tests` to bypass.

## Project Structure

```
yt-clips/
├── pipeline.py          # Main orchestrator — 7 phases
├── download.py          # yt-dlp + aria2c download
├── transcribe.py        # faster-whisper (Hindi/English)
├── highlight.py         # Audio RMS + transcript scoring
├── frame_analyzer.py    # Cheap analysis (Haar + heuristics)
├── premium_analyzer.py  # Premium analysis (YOLO + ByteTrack + Kalman)
├── video_analyzer.py    # Pre-analysis: face/lighting map for full VOD
├── premium_render.py    # Premium render (RIFE + GFPGAN + VBR)
├── export.py            # Clip export + FFmpeg encoding
├── state_analyzer.py    # Pass 1: Per-frame enhancement classification
├── selective_enhancer.py # Pass 2: Conditional enhancement (GFPGAN/sharpen/propagate)
├── temporal_consistency.py # Pass 3: Flicker removal + drift correction
├── seo.py               # SEO generation (Gemini AI)
├── seo_learner.py       # Self-improving SEO from past performance
├── analytics.py         # YouTube analytics dashboard + SEO feedback loop
├── thumbnail.py         # Thumbnail generation
├── upload.py            # YouTube upload
├── sync.py              # Google Drive sync
├── bridge.py            # Colab/Kaggle bridge job pusher
├── push_code.py         # Code sync to Drive
├── channel_watcher.py   # Auto-pilot: watch channel for new VODs
├── scheduler.py         # Upload scheduling
├── config.yaml          # All configuration
├── Kaggle.ipynb         # Kaggle GPU worker notebook (2x T4)
├── kaggle_monitor.py    # Mac-side progress monitor
├── watcher.py           # Job listener (tunnel + file poll)
├── tests/               # 219+ tests
└── utils/
    ├── logger.py         # Rich + JSON structured logging
    ├── config.py         # YAML config loader
    ├── ai_client.py      # Gemini/OpenAI client
    ├── drive_auth.py     # Google Drive auth
    ├── face_matcher.py   # Face matching utility
    ├── face_reference.py # Reference face system
    ├── face_restore.py   # Face restoration utility
    ├── super_res.py      # Real-ESRGAN 4x + GFPGAN
    ├── subtitles.py      # ASS subtitle generation
    ├── torchvision_compat.py # Torchvision compatibility
    ├── resilience.py     # Retry/error handling
    └── reports.py        # Run report generation
```

## Configuration

Key config toggles in `config.yaml`:

```yaml
premium:
  enabled: false              # Set true for YOLO+ByteTrack
  face_enhancement: true      # GFPGAN (auto on Kaggle)
  frame_interpolation: true   # FILM 30→60fps

enhancement:
  selective: false            # 3-pass selective enhancement (Phase 4.25)
  gfpgan_strength: 0.7        # Face restoration strength (0-1)
  temporal_alpha: 0.7         # Face temporal smoothing (0=smooth, 1=raw)
  drift_threshold: 65         # Identity drift detection threshold

download:
  format: "bv*+ba/b"         # Best available (no cap)
  use_aria2c: true           # 2-3x faster (local only; disabled on Kaggle)

transcription:
  language: "hi"             # Hinglish/Hindi detection
  model: "small"             # tiny | base | small | medium | large-v3
  device: "cuda"             # GPU on Kaggle, CPU on local
  compute_type: "float16"    # GPU optimized

export:
  fps: 60                    # Target FPS
  super_resolution: true     # RealESRGAN 4x (auto on Kaggle)
  encoder: "libx264"         # auto-detects h264_nvenc on Kaggle
```

## Pipeline Flow

```
URL → Download (yt-dlp + aria2c)
    → Transcribe (faster-whisper GPU)
    → Video Analysis (face/lighting map)
    → Highlight Detection (audio RMS + transcript scoring + Gemini AI)
    → Frame Analysis (cheap=Haar / premium=YOLO+ByteTrack)
    → Export (crop + enhance + interpolate + encode)
    → Selective Enhancement (3-pass: state→enhance→temporal) [optional]
    → SEO + Thumbnails (Gemini)
    → Upload to YouTube (optional)
```

## Kaggle Workflow

1. Open `Kaggle.ipynb` on [kaggle.com](https://kaggle.com)
2. Set **Settings → Accelerator → GPU × 2 (T4)**
3. Run all cells → Worker prints tunnel URL
4. From Mac, send job:
   ```bash
   ./automate.sh "https://youtu.be/VIDEO_ID"
   ```
5. Monitor progress:
   ```bash
   python kaggle_monitor.py                # quick status
   python kaggle_monitor.py --monitor      # live tail
   python kaggle_monitor.py --files        # list Kaggle files
   python kaggle_monitor.py --log          # tail watcher log
   python kaggle_monitor.py --exec "ls"    # run command on Kaggle
   ```

### Kaggle Remote Control
The watcher exposes HTTP endpoints via tunnel:
- `GET /health` — pipeline status
- `GET /files` — list files in working dir
- `GET /download/<path>` — download a file
- `POST /exec {"cmd": "..."}` — run shell command
- `POST /write {"path": "...", "content": "..."}` — write file
- `POST /job {"url": "...", "flags": [...]}` — submit pipeline job

### Pipeline Flags
| Flag | What |
|---|---|
| `--skip-download` | Use existing `input/video.mp4` |
| `--skip-transcribe` | Use existing transcript |
| `--sync-from-drive` | Pull video+transcript from Google Drive |
| `--sync` | Auto-sync shorts to Google Drive |
| `--upload` | Auto-upload shorts to YouTube |
| `--schedule` | Auto-schedule uploads (2-hour intervals) |

## Tests

```bash
pytest tests/ -v                  # 219+ tests, ~60s
pytest tests/ -m "not slow"       # Skip slow integration tests
pytest tests/ --timeout=120       # With 2-min timeout per test
```

## Docs

| File | What |
|---|---|
| `ARCHITECTURE.md` | Pipeline design, GPU/CPU split, config reference |
| `Kaggle.ipynb` | Kaggle GPU worker notebook (2x T4) |
| `kaggle_monitor.py` | Mac-side progress monitor for Kaggle |
| `watcher.py` | Job listener — accepts jobs via HTTP tunnel |
| `bridge.py` | Pushes jobs from Mac to Kaggle via tunnel |
