# Running yt-clips on Google Colab

Since Google Colab provides free high-end GPUs (T4) and ultra-fast internet, it is the perfect place to run this pipeline without lagging your local PC.

## Steps

1. **Sync code to Drive:** On your Mac, run `./automate.sh` → option 3 (Sync Only)
2. **Open notebook:** Upload `Colab.ipynb` to [colab.research.google.com](https://colab.research.google.com)
3. **Set runtime:** Runtime → Change runtime type → **T4 GPU**
4. **Run all cells:** The worker will start and show a tunnel URL
5. **Send job:** On your Mac, run `./automate.sh "URL"` → option 2 (Remote Run)

## What the notebook does

- Installs all deps: ffmpeg, aria2, Deno, Python packages (**Face Recognition**, PyTorch CUDA, YOLOv8, GFPGAN)
- Automatically matches the host's face against reference photos in the `photos/` folder.
- Writes GPU-optimized `config.yaml` (`premium.enabled: true`, `h264_nvenc`)
- Starts `watcher.py` (job listener) + localtunnel
- Shows tunnel URL for bridge communication

## Selective Enhancement (Phase 4.25)

When `enhancement.selective: true` in config.yaml, a 3-pass enhancement runs on each exported 9:16 clip:

| Pass | Module | What It Does | Device |
|---|---|---|---|
| Pass 1 | `state_analyzer.py` | Per-frame classification: heavy/light/skip | CPU |
| Pass 2 | `selective_enhancer.py` | GFPGAN (heavy), sharpen (light), propagate (skip) | GPU |
| Pass 3 | `temporal_consistency.py` | IIR smoothing, drift correction, boundary blend | CPU |

**Key design:** Operates on 9:16 cropped video from export.py, NOT raw 16:9 source. When enabled, FFmpeg filters in export.py are disabled to prevent double processing.

Enable in config.yaml:
```yaml
enhancement:
  selective: true           # Enable 3-pass selective enhancement
  gfpgan_strength: 0.7      # Face restoration strength
  temporal_alpha: 0.7       # Face temporal smoothing
  drift_threshold: 65       # Identity drift detection threshold
```

## Premium Mode (Colab T4 Only)

Set `premium.enabled: true` in `config.yaml` on Colab for studio-grade quality:

| Feature | Cheap (CPU) | Premium (GPU) |
|---|---|---|
| Face Detection | Haar Cascade (2005) | YOLOv8-face (95%+ acc) |
| Face Tracking | None | ByteTrack (persistent IDs) |
| Crop Smoothing | EMA (lags) | Kalman + Bezier (buttery) |
| Layout Detection | Heuristic | Smart classifier |
| Frame Interpolation | FFmpeg framerate | FILM (true AI 30→60fps) |
| Face Enhancement | None | GFPGAN (eye/texture detail) |
| Speed Variation | Discrete steps | Gaussian-smoothed map |

## Pre-Generation Test Guard

Tests are DISABLED by default on Colab (`testing.enabled: false`).
Set `testing.enabled: true` in the Drive-synced config to enable the pre-run guard:
```bash
pytest tests/ -x --timeout=120
```
Use `--skip-tests` to bypass even when enabled.

## Key Differences: Colab vs Local

| Setting | Local (Mac) | Colab (GPU) |
|---|---|---|
| `transcription.device` | `cpu` | `cuda` |
| `transcription.compute_type` | `int8` | `float16` |
| `export.encoder` | `h264_videotoolbox` | `h264_nvenc` or `libx264` |
| `premium.enabled` | `false` | `true` |
| `enhancement.selective` | `false` (CPU too slow) | `true` (GPU for GFPGAN) |

## Why Colab is Better for Heavy Workloads

- **No PC Lag**: All processing happens on Google's servers.
- **Fast Transcription**: Whisper runs on a GPU (CUDA), taking seconds instead of minutes.
- **Fast Download**: Colab has gigabit internet — use aria2c for 2-3x faster downloads
- **Premium Pipeline**: YOLOv8-face + ByteTrack + FILM + GFPGAN = studio-grade shorts
- **Selective Enhancement**: 3-pass enhancement with GPU-accelerated GFPGAN face restoration
- **Auto-Sync**: Once finished, your Shorts will be in the `shorts/` folder or synced to Google Drive with `--sync`.

## Colab Code Snippets

### 1. Setup & Installation
Run this in a Colab cell to install the new dynamic facial recognition engine:
```python
!pip install face_recognition
!python colab_setup.py
```

### 2. Manual Run
If you want to run a specific video manually on Colab:
```python
!python pipeline.py "YOUR_YOUTUBE_URL" --sync
```
