# Running yt-clips on Google Colab 🚀

Since Google Colab provides free high-end GPUs (T4) and ultra-fast internet, it is the perfect place to run this pipeline without lagging your local PC.

## Option A: Colab Bridge (Recommended)

Use the pre-built `Colab_Bridge.ipynb` notebook for a seamless experience:

1. Upload `Colab_Bridge.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. Set Runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
3. Run all cells — the worker will start and listen for jobs.
4. On your local machine, run: `./automate.sh` and select **Remote Run** (option 2).

The bridge automatically syncs your code and beams jobs to Colab.

## Option B: Manual Setup

### 1. Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com) and create a **New Notebook**.

### 2. Set to GPU Mode (Crucial for Speed)
- In the top menu, go to **Runtime** > **Change runtime type**.
- Select **T4 GPU** (or any available GPU).
- Click **Save**.

### 3. Copy and Paste this into the first cell

```python
# 1. Install system dependencies
!apt-get install -y ffmpeg

# 2. Install Python packages
!pip install yt-dlp faster-whisper PyYAML google-api-python-client google-auth-httplib2 google-auth-oauthlib requests

# 3. Create the directory structure
import os
for folder in ['input', 'temp', 'transcripts', 'highlights', 'shorts', 'logs', 'utils']:
    os.makedirs(folder, exist_ok=True)

# 4. Write the Colab-optimized config.yaml
# IMPORTANT: device=cuda and compute_type=float16 for GPU acceleration
config_content = """
paths:
  input:       input/
  temp:        temp/
  transcripts: transcripts/
  highlights:  highlights/
  shorts:      shorts/
  logs:        logs/

download:
  format: "bestvideo[height<=2160]+bestaudio/bestvideo+bestaudio/best"
  output_filename: "video.mp4"

transcription:
  model: "base"
  language: "en"
  device: "cuda"
  compute_type: "float16"

highlight:
  audio_energy_threshold: 0.75
  min_duration: 15
  max_duration: 29
  merge_gap: 8
  max_clips: 5
  fast_speech_wpm: 160
  silence_penalty_seconds: 1.5

layout:
  has_facecam: true
  facecam: {x: 0, y: 540, width: 320, height: 180}
  facecam_output_height: 400
  gameplay_output_height: 1520

export:
  width: 1080
  height: 1920
  fps: 60
  video_bitrate: "25M"
  audio_bitrate: "320k"
  crf: 18
  encoder: "libx264"
  encoder_preset: "medium"
  crop_smooth_factor: 0.2
  transitions:
    fade_in_duration: 0.5
    fade_out_duration: 0.5
    audio_fade_in: 0.3
    audio_fade_out: 0.4

youtube:
  privacy_status: "private"
  category_id: "17"
  self_declared_made_for_kids: false
  upload_enabled: false
  schedule_interval_hours: 2
  niche: "Cricket"

quality:
  black_threshold: 20
  backlit_brightness_threshold: 80
  overexposed_brightness_threshold: 210
  silence_threshold_db: -35
  min_silence_duration: 1.0
  slow_wpm_threshold: 100
  frame_sample_count: 5
  solo_preference_weight: 1.5

logging:
  level: "INFO"
  log_file: "logs/pipeline.log"
"""
with open('config.yaml', 'w') as f:
    f.write(config_content)
```

### 4. Upload your files
1.  Click the **Folder icon** on the left sidebar of Colab.
2.  Drag and drop all your `.py` files and the `utils` folder into that space.
3.  Run a cell with:
    ```python
    !python pipeline.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
    ```

## Key Differences: Colab vs Local

| Setting | Local (Mac) | Colab (GPU) |
|---|---|---|
| `transcription.device` | `cpu` | `cuda` |
| `transcription.compute_type` | `int8` | `float16` |
| `export.encoder` | `h264_videotoolbox` | `libx264` |

## Why Colab is Better for Heavy Workloads

- **No PC Lag**: All processing happens on Google's servers.
- **Fast Transcription**: Whisper runs on a GPU (CUDA), taking seconds instead of minutes.
- **Fast Download**: Colab has gigabit internet, bypassing many 403 errors.
- **Auto-Sync**: Once finished, your Shorts will be in the `shorts/` folder or synced to Google Drive with `--sync`.
