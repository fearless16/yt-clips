"""
channel_watcher.py — Version 4: Autonomous Channel Monitor.
Checks a target YouTube channel for new videos and triggers the pipeline.

Features:
  - Automatic detection of new uploads
  - Full pipeline trigger with sync + upload + schedule
  - Error recovery with max retry limits
  - Proper logging via utils/logger.py
"""
import time
import subprocess
import argparse
from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("channel_watcher", cfg["logging"]["log_file"], cfg["logging"]["level"])

# Maximum consecutive failures before backing off
MAX_CONSECUTIVE_ERRORS = 5
BACKOFF_MULTIPLIER = 2  # Double the interval after max errors


def monitor(channel_url: str, interval_minutes: int = 60):
    """
    Infinite loop that checks for new videos and processes them.

    Args:
        channel_url: YouTube channel URL to monitor
        interval_minutes: How often to check (in minutes)
    """
    processed_file = Path("processed_videos.txt")
    if not processed_file.exists():
        processed_file.touch()

    consecutive_errors = 0
    current_interval = interval_minutes

    log.info(f"🔭 Channel Monitor started for: {channel_url}")
    log.info(f"⏰ Check interval: {interval_minutes} minutes")

    while True:
        log.info(f"[{time.strftime('%H:%M:%S')}] Checking for new videos...")

        # Use yt-dlp to get the ID of the latest video
        cmd = [
            "yt-dlp",
            "--get-id",
            "--playlist-items", "1",
            "--no-warnings",
            channel_url,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

            video_id = result.stdout.strip()
            if not video_id:
                raise ValueError("Empty video ID returned")

            video_url = f"https://www.youtube.com/watch?v={video_id}"

            processed = processed_file.read_text().splitlines()

            if video_id not in processed:
                log.info(f"✨ NEW VIDEO DETECTED: {video_url}")

                # Trigger the pipeline with full automation
                pipeline_cmd = [
                    "python", "pipeline.py",
                    video_url,
                    "--sync",
                    "--upload",
                    "--schedule",
                ]
                pipeline_result = subprocess.run(pipeline_cmd)

                if pipeline_result.returncode == 0:
                    # Mark as processed only on success
                    with open(processed_file, "a") as f:
                        f.write(f"{video_id}\n")
                    log.info(f"✅ Pipeline complete for {video_id}")
                    consecutive_errors = 0
                    current_interval = interval_minutes
                else:
                    log.error(f"❌ Pipeline failed for {video_id} (exit code {pipeline_result.returncode})")
                    consecutive_errors += 1
            else:
                log.info("   No new videos found.")
                consecutive_errors = 0
                current_interval = interval_minutes

        except subprocess.TimeoutExpired:
            log.warning("⏱ yt-dlp timed out checking channel. Will retry.")
            consecutive_errors += 1
        except Exception as e:
            log.error(f"❌ Error checking channel: {e}")
            consecutive_errors += 1

        # Back off if too many consecutive errors
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            current_interval = interval_minutes * BACKOFF_MULTIPLIER
            log.warning(
                f"⚠️ {consecutive_errors} consecutive errors. "
                f"Backing off to {current_interval} minute intervals."
            )

        log.info(f"😴 Sleeping for {current_interval} minutes...")
        time.sleep(current_interval * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor a YouTube channel for new uploads.")
    parser.add_argument("url", help="YouTube Channel URL to monitor")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in minutes")
    args = parser.parse_args()

    monitor(args.url, args.interval)
