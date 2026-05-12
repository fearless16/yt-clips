# Running yt-clips on Google Colab 🚀

Since Google Colab provides free high-end GPUs (T4) and ultra-fast internet, it is the perfect place to run this pipeline without lagging your local PC.

## Option A: Colab Bridge (Recommended)

Use the pre-built `Colab_Bridge.ipynb` notebook for a seamless experience:

1. Upload `Colab_Bridge.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. Set Runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
3. Run all cells — the worker will start and listen for jobs.
4. On your local machine, run: `./automate.sh` and select **Remote Run** (option 2).

The bridge automatically syncs your code and beams jobs to Colab.

## Option B: Manual Setup (Colab_Setup.ipynb)

Upload `Colab_Setup.ipynb` to Colab — it has 6 cells:
1. Install deps (ffmpeg, aria2c, python packages, premium GPU deps)
2. Upload code files via 📁 sidebar
3. Write optimized config.yaml with `premium.enabled: true`
4. Run pre-flight tests (185 tests)
5. Paste YouTube URL → run pipeline
6. Download shorts.zip

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
