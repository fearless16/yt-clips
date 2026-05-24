# yt-clips — Face OS + YouTube Shorts Automation

> Face OS status note: use `face_os/STATE.md` as the current source of truth.
> Older metric snapshots in this README are historical until refreshed.

Two pipelines in one repo:

1. **Face OS** — Identity-reconstruction pipeline for portrait-mode studio video
2. **Legacy cricket pipeline** — 16:9 live stream → 9:16 shorts automation

---

## Face OS Pipeline

**Philosophy:** Every frame is a noisy photon observation. Maintain an identity belief state. Query memory, don't enhance pixels.

```
Frame → Detect (MediaPipe) → Landmarks (478-point) → Canonical warp
  → Query identity state + intrinsic (albedo/shading/specular)
  → Plan 9:16 crop → _render_core():
      1. PhysicalRenderer (96% of frames)
      2. Identity composite fallback
      3. Enhancement last resort
  → Export 1080x1920 H.264
```

### Quick Start

```bash
.venv/bin/python -m face_os.pipeline --video clips_test/test_clip.mp4 \
    --reference expectation.png --photos photos/
```

Or validate all metrics claims:

```bash
.venv/bin/python validate_metrics.py
```

### Test Suite (773 tests)

```bash
.venv/bin/python -m pytest tests/face_os/ -v
.venv/bin/python -m pytest tests/face_os/test_strict_regression.py -v
```

### V3 Runtime Status (100 frames, test_clip.mp4)

| Metric | Value |
|---|---|
| PhysicalRenderer activation | 96% |
| IntrinsicDecomposer success | 100% |
| Frame contract (1920x1080x3 uint8) | 50/50 pass |
| Avg intrinsic confidence | 0.758 |
| Avg decomposition error | 0.053 |
| RendererMode transitions | 1 |

---

## Legacy Cricket Pipeline

Convert 16:9 live streams → 9:16 shorts automatically.

```bash
./automate.sh "https://youtu.be/VIDEO_ID"
```

### Modes

**Cheap** (default): Haar Cascade + heuristics, works anywhere  
**Premium** (`premium.enabled: true`): YOLOv8-face + ByteTrack + Kalman + GFPGAN, GPU required  
**Selective Enhancement** (`enhancement.selective: true`): 3-pass state→enhance→temporal

### Pipeline Flow

```
URL → Download (yt-dlp + aria2c)
    → Transcribe (faster-whisper, Hindi/English)
    → Highlight Detection (audio RMS + transcript scoring + Gemini AI)
    → Frame Analysis (cheap=Haar / premium=YOLO+ByteTrack)
    → Export (crop + enhance + interpolate + encode)
    → Selective Enhancement (3-pass) [optional]
    → SEO + Thumbnails (Gemini + OpenRouter + Groq + NVIDIA)
    → Upload to YouTube [optional]
```

### Config

Edit `config.yaml` for legacy pipeline.  
Edit `face_os_config.yaml` for Face OS tuning.

### Kaggle GPU Worker

```bash
./automate.sh "https://youtu.be/VIDEO_ID"   # select option 2
python kaggle_monitor.py --monitor            # watch progress
```

---

## Reports & Analytics

All generated reports live in `reports/` (gitignored). Each run overwrites.

### Face Detection Report

One-shot pipeline: sample → detect → crop → compare → HTML report.

```bash
# Full pipeline
python tools/run_report.py

# Reuse existing sampled frames
python tools/run_report.py --skip-sampling

# Open in browser when done
python tools/run_report.py --open

# Validate existing outputs
python tools/run_report.py --validate-only
```

Input: `expectation.png` (reference face photo) + `input/video.mp4`  
Output: `reports/face_detection/face_detection_report.html`

The report includes:
- Face ROI comparison (expectation vs best video frame)
- Per-frame detection stats (confidence, area, position, sharpness)
- After-cropping quality analysis (sharpness, contrast, saturation, brightness drops)
- Face position consistency across 30 sampled frames

### SEO Analytics Dashboard

Standalone report from collected performance data + YouTube analytics.

```bash
# Generate analytics dashboard
python automation/seo/analytics_report.py

# Custom output path
python automation/seo/analytics_report.py -o my_report.html

# Open in browser
python automation/seo/analytics_report.py --open
```

Input: `data/seo_performance.json` + `logs/analytics_*.json`  
Output: `reports/analytics/analytics_report.html`

---

## Docs

| File | What |
|---|---|
| `ARCHITECTURE.md` | Full architecture, Face OS + legacy |
| `AGENTS.md` | Source of truth, known bugs, next steps |
| `AGAINST.md` | Architectural risks, required fixes |
| `face_os/FULL_REFERENCE.md` | Detailed Face OS audit |
| `validate_metrics.py` | Runtime metrics validation |
