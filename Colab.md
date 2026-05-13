# Running yt-clips on Google Colab

Since Google Colab provides free high-end GPUs (T4) and ultra-fast internet, it is the perfect place to run this pipeline without lagging your local PC.

## Steps

1. **Sync code to Drive:** On your Mac, run `./automate.sh` → option 3 (Sync Only)
2. **Open notebook:** Upload `Colab.ipynb` to [colab.research.google.com](https://colab.research.google.com)
3. **Set runtime:** Runtime → Change runtime type → **T4 GPU**
4. **Run all cells:** The worker will start and show a tunnel URL
5. **Send job:** On your Mac, run `./automate.sh "URL"` → option 2 (Remote Run)

## What the notebook does

- Mounts Google Drive (gets code from `yt-clips/` folder)
- Installs all deps: ffmpeg, aria2, Deno, Python packages, **PyTorch CUDA + YOLOv8 + GFPGAN**
- Writes GPU-optimized `config.yaml` (`premium.enabled: true`, `h264_nvenc`)
- Starts `watcher.py` (job listener) + localtunnel
- Shows tunnel URL for bridge communication

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

Before ANY expensive operation, the pipeline auto-runs:
```bash
pytest tests/ -x --timeout=120
```
Aborts immediately if tests fail. Use `--skip-tests` to bypass.

## Key Differences: Colab vs Local

| Setting | Local (Mac) | Colab (GPU) |
|---|---|---|
| `transcription.device` | `cpu` | `cuda` |
| `transcription.compute_type` | `int8` | `float16` |
| `export.encoder` | `h264_videotoolbox` | `h264_nvenc` or `libx264` |
| `premium.enabled` | `false` | `true` |

## Why Colab is Better for Heavy Workloads

- **No PC Lag**: All processing happens on Google's servers.
- **Fast Transcription**: Whisper runs on a GPU (CUDA), taking seconds instead of minutes.
- **Fast Download**: Colab has gigabit internet — use aria2c for 2-3x faster downloads
- **Premium Pipeline**: YOLOv8-face + ByteTrack + FILM + GFPGAN = studio-grade shorts
- **Auto-Sync**: Once finished, your Shorts will be in the `shorts/` folder or synced to Google Drive with `--sync`.
