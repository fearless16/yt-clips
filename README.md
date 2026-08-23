# yt-clips — Face OS + YouTube Shorts Automation

> Face OS status note: use `face_os/STATE.md` as the current source of truth.
> Older metric snapshots in this README are historical until refreshed.

Two pipelines in one repo:

1. **Face OS** — Identity-reconstruction pipeline for portrait-mode studio video
2. **Cricket Shorts pipeline** — 16:9 live stream → 9:16 shorts automation
   (selection, packaging, SEO, upload, outcome learning)

---

## 🚀 One-Click Run (Windows)

**Desktop shortcut: `YT Clips Pipeline.lnk`**
→ points at `run_pipeline.bat` (icon: `pipeline_icon.ico`, cwd: repo root).

Double-click se ye hota hai:

```
[1/3] OAuth token check → expired ho toh browser re-auth khud kholta hai
      (pehle PERSONAL Google account, phir CHANNEL account select karo)
[2/3] Mode menu:
        1) FULL     — URL poochega → download, transcribe, select, export,
                      SEO, YouTube upload (scheduled slots pe)
        2) QUICK    — existing input\video.mp4 + transcript reuse,
                      local export only (no upload/sync/schedule)
        3) DRY RUN  — saare external APIs stubbed, sirf plumbing check
[3/3] "PIPELINE FINISHED" + pause
```

| Mode | Command jo chalta hai | Upload? |
|---|---|---|
| FULL | `.venv\Scripts\python.exe pipeline.py "<URL>"` | ✅ scheduled |
| QUICK | `pipeline.py "https://youtu.be/local" --skip-download --skip-transcribe --no-upload --no-sync --no-schedule` | ❌ |
| DRY RUN | `dry_run.py "https://youtu.be/test" --skip-download --skip-transcribe` | ❌ |

### Local-only one-click (kabhi upload nahi karta)

`make_shorts.bat` → `.venv\Scripts\python.exe -m automation.cli "<URL>"`
— orchestrator flow (ingest → transcript → cricket-only selection → export →
SEO → telemetry + canonical DB persistence). Output `shorts\<date-folder>\`
mein jaata hai. Upload/sync ke liye explicit flags chahiye:

```bash
.venv/Scripts/python.exe -m automation.cli "<URL>" --upload    # + scheduled upload
.venv/Scripts/python.exe -m automation.cli "<URL>" --sync      # + Drive sync
.venv/Scripts/python.exe -m automation.cli --learn             # shelf + Analytics sync + model refit
```

Useful flags: `--skip-download --skip-transcribe --skip-highlight
--skip-export --skip-seo --skip-enhancement --sample-minutes N`.

**Outputs:** clips + `<clip>_metadata.json` → `shorts/<date-folder>/`;
selection YAML → `highlights/`; logs → `logs/pipeline.log`.
Secrets (`*_token.json`, `client_secrets.json`, `cookies.txt`) gitignored
hain — Drive pe synced hote hain, kabhi commit mat karna.

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
.venv/bin/python -m face_os.pipeline --video <input.mp4> \
    --reference expectation.png --photos photos/
```

### Test Suite

```bash
.venv/Scripts/python.exe -m pytest face_os/tests/ -v
```

See `face_os/STATE.md` for architecture map, drift status and entry points.

---

## Cricket Shorts Pipeline (current behavior)

Convert 16:9 live streams → 9:16 shorts automatically.

```bash
.venv/Scripts/python.exe -m automation.cli "https://youtu.be/VIDEO_ID"
```

### Pipeline Flow

```
URL → Download (yt-dlp + aria2c)
    → Transcribe (faster-whisper, Hindi/English)
    → Complete-thought segmentation + cricket-only gate
    → 7-agent clip selection (+ LLM arbiter with full clip scripts,
      match facts, live YouTube search demand; max 3 clips/stream,
      empty selection allowed on weak batches)
    → Frame Analysis / Face crop
    → Export 1080x1920 (natural pace: target 25s thoughts, speedup rare)
    → SEO packaging v3 (promise_v3_english):
        - Full-English public copy, Devanagari banned everywhere
        - Title <=60 chars, one premise, 1-2 mandatory emojis
        - Description 350-900 chars, moment-first, AI-slop gate
        - Grounded search terms/tags (API-only), 3 hashtags
    → Upload to YouTube [opt-in via flags]
    → shorts_intelligence outcome learning (engaged views + avg %viewed)
```

### Selection & Packaging Notes

- Har candidate ko `content_type` (moment/comedy/debate/news) milta hai;
  arbiter moments-with-stakes ko debate/chatter pe prefer karta hai.
- LLM arbiter ke paas candidate ka POORA script hota hai (150-char fragment
  nahi) — self-containedness isi se judge hoti hai.
- Quality gate real hai: weak batch = zero clips, force-backfill nahi.
- Learner (`shorts_intelligence.db`) sirf calibrated evidence use karta hai;
  selection adjustments bounded hain (clip boundaries kabhi nahi badalte).

```bash
.venv/Scripts/python.exe -m shorts_intelligence status   # model state
.venv/Scripts/python.exe -m shorts_intelligence sync     # shelf + outcomes
```

### Modes

**Cheap** (default): MediaPipe BlazeFace + heuristics, works anywhere  
**Premium** (`premium.enabled: true`): YOLOv8-face + ByteTrack + Kalman + GFPGAN, GPU required  

### Config

Edit `config.yaml` for the cricket pipeline (`highlight`, `clip_selection`,
`seo`, `upload_schedule`, `shorts_intelligence`).  
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

### Shorts Intelligence Dashboard

Standalone report from the canonical cricket Shorts database.

```bash
# Generate analytics dashboard
python automation/seo/analytics_report.py

# Custom output path
python automation/seo/analytics_report.py -o my_report.html

# Open in browser
python automation/seo/analytics_report.py --open

# Explicitly sync the shelf + Analytics and refit (no source URL needed)
python -m automation.cli --learn
```

Input: `shorts_intelligence.db` (exact channel Shorts shelf + YouTube Analytics)

Output: `reports/analytics/analytics_report.html`

---

## Docs

| File | What |
|---|---|
| `ARCHITECTURE.md` | Full pipeline and Face OS architecture |
| `AGENTS.md` | Source of truth, testing gates, module pointers |
| `face_os/STATE.md` | Face OS state reference & drift status |
| `automation/AGENTS.md` | Orchestrator + learning ownership rules |

---

## Validation Gates (commit se pehle, hamesha)

```bash
.venv/Scripts/python.exe -m pytest tests/ -v --ignore=tests/test_face_detect.py
.venv/Scripts/python.exe dry_run.py https://youtu.be/test --skip-download --skip-transcribe
```

Dono pass hone chahiye — fail ho toh commit nahi.
