# yt-clips — AI-Powered YouTube Shorts Automation

Convert 16:9 live streams → studio-grade 9:16 shorts automatically. Two modes:

## Quick Start

```bash
# Cheap (local, CPU-based)
./automate.sh "https://youtu.be/VIDEO_ID"   # select option 1

# Premium (Colab T4, GPU) — one-time Colab setup required
./automate.sh "https://youtu.be/VIDEO_ID"   # select option 2
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

### Pre-Generation Test Guard
Before any expensive operation, `pytest tests/ -x --timeout=120` runs automatically.
Aborts on failure. Use `--skip-tests` to bypass.

## Project Structure

```
yt-clips/
├── pipeline.py          # Main orchestrator — 6 phases
├── download.py          # yt-dlp + aria2c download
├── transcribe.py        # faster-whisper (Hindi/English)
├── highlight.py         # Audio RMS + transcript scoring
├── frame_analyzer.py    # Cheap analysis (Haar + heuristics)
├── premium_analyzer.py  # Premium analysis (YOLO + ByteTrack + Kalman)
├── premium_render.py    # Premium render (RIFE + GFPGAN + VBR)
├── export.py            # Clip export + FFmpeg encoding
├── seo.py               # SEO generation (Gemini AI)
├── seo_learner.py       # Self-improving SEO from past performance
├── analytics.py         # YouTube analytics dashboard + SEO feedback loop
├── thumbnail.py         # Thumbnail generation
├── upload.py            # YouTube upload
├── sync.py              # Google Drive sync
├── bridge.py            # Colab bridge job pusher
├── push_code.py         # Code sync to Drive
├── channel_watcher.py   # Auto-pilot: watch channel for new VODs
├── scheduler.py         # Upload scheduling
├── config.yaml          # All configuration
├── colab_setup.py       # Colab GPU worker setup (one-shot)
├── watcher.py           # Colab job listener (tunnel + file poll)
├── tests/               # 185 tests
└── utils/
    ├── logger.py         # Rich + JSON structured logging
    ├── config.py         # YAML config loader
    ├── ai_client.py      # Gemini/OpenAI client
    ├── drive_auth.py     # Google Drive auth
    └── subtitles.py      # ASS subtitle generation
```

## Configuration

Key config toggles in `config.yaml`:

```yaml
premium:
  enabled: false              # Set true on Colab T4
  face_enhancement: true      # GFPGAN
  frame_interpolation: true   # RIFE 30→60fps

download:
  format: "bv*+ba/b"         # Best available (no cap)
  use_aria2c: true           # 2-3x faster downloads (install aria2c)
  concurrent_fragments: 8    # Max connections

transcription:
  language: "hi"             # Hinglish/Hindi detection
  model: "small"             # small/medium/large (larger = more accurate)
  device: "cuda"             # GPU on Colab, CPU on local

export:
  fps: 60                    # Target FPS
  enable_variable_speed: true
  video_bitrate: "25M"
  encoder: "h264_nvenc"      # h264_videotoolbox on Mac
```

## Pipeline Flow

```
URL → Download (yt-dlp + aria2c)
    → Transcribe (faster-whisper GPU)
    → Highlight Detection (audio RMS + transcript scoring + Gemini AI)
    → Frame Analysis (cheap=Haar / premium=YOLO+ByteTrack)
    → Export (crop + enhance + interpolate + encode)
    → SEO + Thumbnails (Gemini)
    → Upload to YouTube (optional)
```

## Tests

```bash
pytest tests/ -v                  # 185 tests, ~60s
pytest tests/ -m "not slow"       # Skip slow integration tests
pytest tests/ --timeout=120       # With 2-min timeout per test
```

## Docs

| File | What |
|---|---|
| `ARCHITECTURE.md` | Pipeline design, GPU/CPU split, config reference |
| `Colab.md` | Colab setup guide |
| `colab_setup.py` | Colab GPU worker setup (mounts Drive, installs deps, starts tunnel) |
| `watcher.py` | Colab job listener — accepts jobs via HTTP tunnel or file poll |
```
