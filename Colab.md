# Running yt-clips on Google Colab

Since Google Colab provides free high-end GPUs (T4) and ultra-fast internet, it is the perfect place to run this pipeline without lagging your local PC.

## Setup: `colab_setup.py` (Recommended)

Upload `colab_setup.py` + `watcher.py` + all `.py` files + `utils/` to Colab and run:

```
!python colab_setup.py
```

Or sync from Google Drive (run `./automate.sh` → option 3 on your Mac first).

What it does:
1. Mounts Google Drive
2. Installs all deps (ffmpeg, aria2, Deno, Python packages, PyTorch + CUDA, YOLOv8, GFPGAN)
3. Writes GPU-optimized `config.yaml` with `premium.enabled: true`
4. Starts `watcher.py` (HTTP server on port 5000) + localtunnel
5. Shows tunnel URL — use with `./automate.sh` → **Remote Run**

## Remote Job Worker

`watcher.py` runs on Colab and accepts pipeline jobs two ways:
- **Tunnel (instant):** bridge.py POSTs to the tunnel URL → watcher processes it
- **File poll (fallback):** watches for `remote_job.json` in Drive sync

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
