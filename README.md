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
[1/3] Mode menu:
        1) NEW RUN  — URL poochega → download, transcribe, select,
                      export, SEO. Upload? alag se y/N poochhta hai.
        2) QUICK    — existing input\video.mp4 + transcript reuse,
                      local export only
        3) DRY RUN  — saare external APIs stubbed, sirf plumbing check
[2/3] OAuth check only if upload requested
[3/3] Truthful outcome + artifact paths + pause
```

| Mode | Command jo chalta hai | Upload? |
|---|---|---|
| NEW RUN | `automation.cli "<URL>" --source-scoped --skip-instagram` | ❌ local drafts |
| NEW RUN + upload | `automation.cli "<URL>" --source-scoped --sync --upload --schedule --skip-instagram` | scheduled slots, only after rights/release checks |
| QUICK | `automation.cli "https://youtu.be/local" --skip-download --skip-transcribe` | ❌ |
| DRY RUN | `dry_run.py "https://youtu.be/test" --skip-download --skip-transcribe` | ❌ |

**Ek hi dwaar:** `.venv/Scripts/python.exe -m automation.cli` — Windows entry
sirf `run_pipeline.bat` hai (desktop shortcut isi ko chalata hai). Direct CLI:

```bash
.venv/Scripts/python.exe -m automation.cli "<URL>" --source-scoped --skip-instagram # local only
.venv/Scripts/python.exe -m automation.cli "<URL>" --source-scoped --upload --schedule --skip-instagram # gated delivery
.venv/Scripts/python.exe -m automation.cli --learn               # shelf + Analytics sync + model refit
```

Useful flags: `--skip-download --skip-transcribe --skip-highlight
--skip-export --skip-seo --skip-enhancement --sample-minutes N`.

**Default:** two distinct source moments with `cricket-gameplay-stadium-v1`,
existing EA07 stock, seeded continuous gameplay excerpts, muted gameplay and
original source voice at natural speed. Weak/incomplete cuts are never forced
to meet the target. Branding: **CRICKET WITH PRAJJWAL**, no handle/2.0 in artwork.
On a fresh Windows environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m playwright install chromium
```

FFmpeg/ffprobe must be on PATH, and the reviewed licensed Arial Black font must
be installed. Preflight checks the artwork/font, launches a local headless
Chromium page, and requires approved continuous gameplay covering the configured
clip ceiling (currently 38 seconds). No browser or media is downloaded at runtime.
The configured `input/gameplay/library-clean-v2.json` is enabled stock (`STOCK` /
`AVAILABLE`), with seeded excerpts embedded automatically. Drive sync is allowed;
scheduled YouTube upload still requires the configured exact-file owner release.
Provenance hashes, attribution and unresolved rights records are preserved, not
silently relabelled cleared. Legacy private-review manifests stay private.
See [stock runtime and recovery](docs/STOCK_GAMEPLAY_RUNTIME.md) and
[stadium setup and release requirements](docs/STADIUM_PIPELINE_INTEGRATION.md).

Resume a specific existing batch with `--skip-export --export-dir shorts/BATCH`.
Multiple batches require that explicit choice; nested review renders are not
auto-discovered. Existing covers are preserved, and Drive/DB backup requires
`--sync`. See the [integration audit](docs/editorial/INTEGRATION_AUDIT_20261102.md)
for verified checks and remaining production limitations.

**NEW RUN outputs:** clips + `<clip>_metadata.json` → `shorts/<source-id>/<batch>/`;
input, transcript and selection YAML also live under their source ID. The earlier
canonical `input/video.mp4` is not overwritten. Legacy QUICK/direct CLI without
`--source-scoped` retains its configured paths. Logs → `logs/pipeline.log`.

Stock ranges are decoded frame-by-frame for dark transitions before selection;
the delivered gameplay panel is checked again during encoded QC. Only a clean
continuous excerpt long enough for the whole Short is usable—no looping or speed
changes. Search queries use the actual clip's grounded players/topics rather than
the full stream roster, and final internal terms are entity-checked after ranking.
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
.venv/Scripts/python.exe -m automation.cli "<YouTube video URL>" --source-scoped --skip-instagram
```

### Pipeline Flow

```
URL → Source-scoped download (yt-dlp + aria2c)
    → Original YouTube transcript / local Hindi ASR
    → Whole-transcript discovery → word-ID boundaries → final text review
      (target two distinct moments; bounded speech recovery; no forced quota)
    → Frame Analysis / Face crop
    → Stadium export 1080x1920, 1x voice + muted seeded EA07 stock
    → Encoded audio/A–V/full-decode QC
    → Source-grounded SEO + source-derived covers:
        - Roman Hinglish / English title <=60 chars, one supported premise
        - Cover 1-3 words; no mandatory emojis or keyword padding
        - Bounded title/cover format correction; failures remain resumable
    → Exact-file owner/rights review → YouTube [opt-in, existing schedule]
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
| `docs/REPOSITORY_CLEANUP_20261104.md` | Main integration, removed unused modules and preserved cleanup candidates |

---

## Validation Gates (commit se pehle, hamesha)

```bash
.venv/Scripts/python.exe -m pytest tests/ -v --ignore=tests/test_face_detect.py
.venv/Scripts/python.exe dry_run.py https://youtu.be/test --skip-download --skip-transcribe
```

Dono pass hone chahiye — fail ho toh commit nahi.
