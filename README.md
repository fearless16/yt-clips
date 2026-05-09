# yt-clips — YouTube Shorts Automation Pipeline

> Local-first Python pipeline: Download → Transcribe → Detect Highlights → Export Vertical Shorts → Sync to Drive → Upload to YouTube

---

## Quick Start

The entire process is automated. Just run:

```bash
./automate.sh https://youtu.be/YOUR_VIDEO_ID
```

This script will:
1.  Check/Install **FFmpeg** and **Python 3.10+** via Homebrew.
2.  Create a virtual environment and install dependencies.
3.  Run the full pipeline end-to-end (sync + upload + schedule by default).
4.  Optionally prompt to sync Shorts to Google Drive after completion.

### Execution Modes

| Mode | Description |
|---|---|
| `1` Local Run | Full pipeline on your machine's CPU |
| `2` Remote Run | Offload heavy processing to Google Colab GPU |
| `3` Sync Only | Upload existing shorts/ to Google Drive |
| `4` Auto-Pilot | Watch your channel for new VODs, auto-process them |

### Fast Iteration (Skip Phases)

```bash
# Skip download and transcription, only redo highlights and export
./automate.sh https://youtu.be/XXXX --skip-download --skip-transcribe

# Only re-run the export phase (e.g., after changing layout in config.yaml)
./automate.sh https://youtu.be/XXXX --skip-download --skip-transcribe --skip-highlight
```

### One-Click Sync (Standalone)

```bash
# Sync all shorts to Google Drive
python sync.py

# Sync specific folder
python sync.py --folder shorts/2026-05-03_093000
```

### Output Structure

```
shorts/
  2026-05-03_093000/
    clip1.mp4
    clip1_metadata.json
    clip2.mp4
    clip2_metadata.json
    ...
```

---

## Pipeline Phases

| Phase | Module | Description |
|---|---|---|
| 1 — Download | `download.py` | Download YouTube VOD at up to 4K quality |
| 2 — Transcribe | `transcribe.py` | Speech-to-text via faster-whisper |
| 3 — Highlights | `highlight.py` | Detect highlight moments (audio + heuristics) |
| 4 — Export | `export.py` | 9:16 vertical reframing + pro enhancements |
| 4.5 — SEO/Thumbnails | `seo.py` / `thumbnail.py` | AI-driven titles and aesthetic thumbnails |
| 5 — Sync | `sync.py` | Upload Shorts to Google Drive (optional) |
| 6 — Upload | `upload.py` | Upload to YouTube with auto-scheduling (optional) |

### Individual Phase Commands

| Phase | Command |
|---|---|
| Download only | `python download.py <url>` |
| Transcribe only | `python transcribe.py` |
| Detect highlights | `python highlight.py` |
| Full pipeline | `python pipeline.py <url>` |
| Full + sync | `python pipeline.py <url> --sync` |
| Full + upload | `python pipeline.py <url> --upload` |
| Full + schedule | `python pipeline.py <url> --upload --schedule` |

### Skip Flags

```bash
# Re-use existing download, redo everything else
python pipeline.py <url> --skip-download

# Re-use download + transcript, redo highlights + export
python pipeline.py <url> --skip-download --skip-transcribe

# Only re-export (change layout config, re-run export)
python pipeline.py <url> --skip-download --skip-transcribe --skip-highlight

# Full pipeline + auto sync + auto upload with 2-hour scheduling
python pipeline.py <url> --sync --upload --schedule
```

---

## Folder Structure

```
yt-clips/
├── pipeline.py          # Main orchestrator (6 phases)
├── download.py          # Phase 1: yt-dlp downloader (up to 4K)
├── transcribe.py        # Phase 2: faster-whisper transcription
├── highlight.py         # Phase 3: highlight detection heuristics
├── export.py            # Phase 4: FFmpeg vertical reframing + export
├── sync.py              # Phase 5: Google Drive sync
├── upload.py            # Phase 6: YouTube uploader
├── seo.py               # SEO metadata generator (titles, tags, descriptions)
├── scheduler.py         # Smart 2-hour upload scheduling
├── trends.py            # Trending hashtag/hook engine
├── bridge.py            # Local→Cloud job bridge
├── watcher.py           # Colab-side job watcher
├── channel_watcher.py   # Autonomous channel monitor (Auto-Pilot)
├── config.yaml          # All settings (no hardcoded values)
├── automate.sh          # One-command entry point
├── setup.sh             # One-shot environment setup
├── setup_youtube.py     # Guided YouTube API setup wizard
├── requirements.txt
├── utils/
│   ├── config.py        # YAML config loader
│   └── logger.py        # Premium colored logging
├── input/               # Downloaded source video
├── temp/                # Intermediate files (auto-cleaned)
├── transcripts/         # JSON transcripts
├── highlights/          # YAML highlight timestamps + text
├── shorts/              # ← Final vertical Shorts (date-stamped)
│   └── 2026-05-03_093000/
│       ├── clip1.mp4
│       ├── clip1_metadata.json
│       └── clip2.mp4
└── logs/
    └── pipeline.log
```

---

## Configuration (`config.yaml`)

All values live in `config.yaml`. Important ones to customise:

### Download Quality

```yaml
download:
  # Shorts-friendly cap: much faster than 4K, still enough for 1080x1920 output
  format: "bv*[height<=1440]+ba/b[height<=1440]/bv*+ba/b"
  concurrent_fragments: 8
  sleep_requests: 0
  progress_interval_seconds: 10
  progress_percent_step: 5
  po_token: "YOUR_TOKEN"   # For bypassing bot detection
  proxy: ""               # Optional proxy
```

### Hardware Encoder

| Machine | Encoder setting |
|---|---|
| macOS (Apple Silicon) | `h264_videotoolbox` ✅ default |
| NVIDIA GPU | `h264_nvenc` |
| CPU fallback | `libx264` |

```yaml
export:
  encoder: "h264_videotoolbox"
  video_bitrate: "25M"        # Premium quality
  audio_bitrate: "320k"       # Studio-grade audio
  fps: 60
```

### Audio Enhancement (Built-in)

The export pipeline automatically applies:
- **High-pass filter** (80Hz) — removes rumble/noise
- **Compressor** — evens out dynamic range
- **Loudness normalization** (EBU R128, -14 LUFS) — YouTube-optimal levels
- **48kHz resampling** — broadcast quality

### Video Enhancement (Built-in)

The export pipeline automatically applies:
- **hqdn3d denoising** — removes grain/noise
- **Unsharp mask** — crisp detail enhancement
- **Deband filter** — smooth gradients, removes YouTube compression artifacts
- **Contrast/saturation boost** — punchy, vibrant output
- **Lanczos scaling** — highest quality upscale algorithm
- **Circular branding logo** — auto-applied if `channel_logo.png` exists

### Facecam Layout

Set the pixel coordinates of your facecam overlay in the **source** video:

```yaml
layout:
  has_facecam: true
  facecam:
    x: 0        # left edge of facecam in source frame
    y: 540      # top edge
    width: 320
    height: 180
  facecam_output_height: 400   # how tall the facecam strip is in the Short
```

> **Smart validation:** If facecam coordinates exceed the actual source video dimensions, the system automatically falls back to full-frame center crop mode. No crashes.

Set `has_facecam: false` if your stream has no facecam overlay.

### Highlight Sensitivity

```yaml
highlight:
  audio_energy_threshold: 0.75   # raise to only pick the loudest moments
  min_duration: 15               # seconds
  max_duration: 29               # YouTube Shorts limit (<30s)
  max_clips: 5
  fast_speech_wpm: 160
```

### YouTube Upload

```yaml
youtube:
  privacy_status: "private"       # private | unlisted | public
  category_id: "17"               # 17 = Sports
  upload_enabled: false
  schedule_interval_hours: 2      # Automate posting every 2 hours
  niche: "Cricket"                # Used for SEO trend analysis
```

---

## Output Format

- Resolution: **1080 × 1920** (9:16)
- Codec: **H.264** + **AAC**
- Video Bitrate: **25 Mbps** (premium quality)
- Audio: **320kbps AAC** @ 48kHz
- FPS: **source FPS** (up to 60)
- Audio processing: loudnorm + highpass + acompressor

---

## Highlight Detection Heuristics

No LLMs. No emotion AI. Pure signal:

| Signal | Weight |
|---|---|
| Audio RMS energy spike | ×3.0 (dominant) |
| Peak energy (top 10%) | +0.5 per bucket |
| Fast speech (WPM > 160) | +1.5 |
| Moderate speech (WPM > 128) | +0.5 |
| Reaction keywords (wow, insane, wicket, six, …) | +0.4 per hit |
| Exclamation/question marks | +0.2 per mark |
| Long silence penalty | −0.5 |

---

## Google Drive Sync

### Setup (one-time)

```bash
# Install gcloud CLI, then:
gcloud auth application-default login
```

### Usage

```bash
# Sync all shorts
python sync.py

# Sync specific folder
python sync.py --folder shorts/2026-05-03_093000

# Auto-sync during pipeline
python pipeline.py <url> --sync
```

Drive folder structure: `My Drive/yt-clips/shorts/<date-folder>/`

---

## YouTube Upload Setup

Run the guided setup wizard:

```bash
python setup_youtube.py
```

This will walk you through:
1. Creating a Google Cloud project
2. Enabling the YouTube Data API v3
3. Setting up OAuth credentials
4. Authenticating with your YouTube channel

---

## Remote Execution (Google Colab)

See [Colab.md](Colab.md) for instructions on running the pipeline on Google Colab with GPU acceleration.

The bridge system supports three modes:
1. **Direct Google Drive API** — instant job delivery
2. **Tunnel bridge** — real-time HTTP connection to Colab
3. **Local sync folder** — reliable file-based fallback

---

## 💎 Pro-Grade Features

The pipeline includes advanced features for high-performance, studio-quality automated production:

### 🧠 Intelligent Frame Analysis (`frame_analyzer.py`)
- **Multi-Timestamp Sampling**: Upgraded from single-frame analysis to 5-timestamp `np.linspace` sampling per segment, selecting the most prominent face candidate.
- **Poster Rejection Heuristics**: Implemented "Poster Nuke" logic to reject static background posters in gaming/cricket setups using area-to-height ratios.
- **Temporal Crop Smoothing**: Uses weighted alpha smoothing for `crop_x` transitions to prevent jarring, jittery camera movements between cuts.
- **Auto-Layout Detection**: Detects split-screen or multi-panel layouts and intelligently decides between `SOLO` and `VERTICAL STACK` modes.
- **Lighting Correction**: Detects backlit or underexposed scenes and automatically applies analysis-driven gamma and exposure fixes.
- **Leading Black Trim**: Automatically detects and trims leading black frames (up to 1s) for professional, "instant action" starts.

### 🎬 Immersive Cinema-Grade Export (`export.py`)
- **Global 1.25x Speedup**: Integrated global speed factor for high-retention pacing across all clips.
- **A/V Sync Lock**: Robust timestamp resampling (`aresample=async=1`) ensures audio and video stay perfectly synced at high speeds.
- **Poor Lighting Enhancement**: Advanced filter chain with **Gamma Boosting (1.2x)**, **Saturation Lift (1.2x)**, and **Deband** filters for studio-quality output from 720p stream sources.
- **Butter-Smooth Motion**: Auto-detects source FPS and applies motion-blending interpolation for high-quality 60fps output.

### 📈 Hardened SEO Engine (`seo.py`)
- **Rate-Limited Batching**: Implemented a 3-clip batch queue with 8s mandatory pauses and exponential backoff to eliminate `429 Too Many Requests` errors.
- **Phonetic Hallucination Correction**: System instructions forced to identify and correct phonetic transcription errors (e.g., "Chakris Gale" → "Chris Gayle").
- **Strict Limit Enforcement**: Guarantees compliance with YouTube's character limits (Title: 100, Desc: 5000) before processing, preventing metadata rejection.
- **Premium Run Reporting**: Generates a detailed `run_report.md` with performance snapshots, failure summaries, and formatted SEO previews for every run.

### 🖼️ Aesthetic Thumbnail Generation (`thumbnail.py`)
- **Free-Tier Optimization**: Defaulting to **Nano Banana 2 (Gemini 2.5 Flash Image)** for high-volume, free-tier AI thumbnail generation.
- **Contrast Optimization**: Adds semi-transparent gradient overlays and blurred drop shadows for premium, high-impact visuals.
- **A/B Variants**: Generates multiple variants for manual or automated A/B testing.

### 🚀 Production Stability
- **Smart Sync**: Supports `--dry-run` to verify uploads before committing to Drive.
- **Robust Config**: Type-safe nested configuration with dot-notation access.
- **Safe Pre-flight**: Validates all source files and metadata before starting long-running API operations.

---

## 🧪 Testing & Quality Assurance

The pipeline includes a robust test suite to ensure export quality and stability.

### Running Tests

```bash
# Run all tests using pytest
./.venv/bin/python -m pytest tests/

# Run a specific test file
./.venv/bin/python -m pytest tests/test_export.py
```

### What's Tested?
- **Export Logic**: Encoder detection, hardware smoke tests, fallback to `libx264`, and 9:16 reframing.
- **Frame Analysis**: Black frame detection, lighting analysis, and layout voting.
- **Tempo Intelligence**: Analysis-driven speed adjustments based on transcript WPM.
- **Edge Cases**: Handling of videos with no audio, invalid timestamps, and missing assets.

---

## Troubleshooting

**`h264_videotoolbox` fails** → The pipeline automatically falls back to `libx264`.  
**No highlights detected** → Lower `audio_energy_threshold` in `config.yaml` (try `0.5`).  
**Transcript is empty** → Try `transcription.model: small` or `medium`.  
**Download fails** → Update yt-dlp: `pip install -U yt-dlp`  
**Frame clipping broken** → Check `layout.facecam` coords match your source video dimensions.  
**Low quality download** → Check that yt-dlp isn't being rate-limited; the format pulls up to 4K.  
**Sync fails** → Run `gcloud auth application-default login` to authenticate.  
**Upload fails** → Run `python setup_youtube.py` and ensure `yt_token.json` exists.
