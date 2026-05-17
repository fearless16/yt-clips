# Conversation Memory & Summary

## Objective
The objective was to fix clipping-related issues in the `yt-clips` automated YouTube Shorts pipeline and ensure that there is no runtime payoff (penalty) for clips that are improperly cut off, along with testing small video slices due to low processing power on a MacBook Air.
The user also wanted to ensure the subject (the speaker) remains centered in the 9:16 frame, just like AI Studio outputs (e.g. Opus Clip).

## Implemented Fixes

1. **Lightweight Random Testing (`--sample-minutes`)**
   - Added `--sample-minutes` parameter to `pipeline.py`.
   - Updated `download.py` to use `yt-dlp`'s `--download-sections` argument. This downloads a randomly selected continuous snippet of N minutes instead of downloading the entire stream, drastically reducing transcription and processing overhead.

2. **No Runtime Penalty for Bad Clips (Strict Validation)**
   - Updated `_validate_output` in `export.py` to enforce a hard absolute minimum duration of 5 seconds for any output clip.
   - Introduced an `expected_duration` validation check. If FFmpeg produces a clip that deviates by more than 2.0 seconds from the requested duration (meaning it ended abruptly or was improperly cut), it is instantly rejected.
   - Any rejected clip will NOT be queued for SEO generation or YouTube uploads, preventing wasted API calls and runtime.

3. **9:16 AI-Studio Centering**
   - Verified that `frame_analyzer.py` utilizes Haar Cascades (`detect_face_crop`) to pinpoint the dominant face in the frame.
   - It computes a 9:16 bounding box (`target_w = target_h * 9/16`) specifically centered around `crop_x = x + w // 2 - target_w // 2`.
   - `export.py` maps this face-aware bounding box into FFmpeg's `crop={cw}:{ch}:{cx}:{cy}` filter. This guarantees that the user remains the dead center of the 9:16 portrait video frame, completely matching the expectation of standard AI studio crops.

4. **Testing Phase**
   - Autonomous tests were executed across the workspace's test suite to validate the integrity of the updated scripts.
