"""
download.py — Phase 1: Download a YouTube VOD/stream with yt-dlp.

Downloads at the HIGHEST available quality (up to 4K/2160p).
Prefers VP9/AV1 for quality, remuxes to MP4 container.

Usage:
    python download.py <youtube_url> [--output input/video.mp4]
"""

import argparse
import subprocess
import sys
from typing import Optional
from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("download", cfg["logging"]["log_file"], cfg["logging"]["level"])


def download(url: str, output_path: Optional[str] = None) -> Path:
    """
    Download a YouTube video using yt-dlp at the highest available quality.

    Args:
        url:         YouTube URL (video, live-stream, or VOD).
        output_path: Destination file path. Falls back to config value.

    Returns:
        Path to the downloaded file.
    """
    dl_cfg = cfg["download"]
    paths_cfg = cfg["paths"]

    if output_path is None:
        output_path = str(Path(paths_cfg["input"]) / dl_cfg["output_filename"])

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # yt-dlp writes to a temp name then renames; use a template so we know
    # the final filename regardless of container.
    stem = dest.stem
    template = str(dest.parent / f"{stem}.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", dl_cfg["format"],
        "--merge-output-format", "mp4",
        # Prefer higher resolution, then higher bitrate
        "--format-sort", "res:2160,vbr,abr",
        "--output", template,
        "--no-playlist",
        "--progress",
        "--no-warnings",
        # Retry and resilience
        "--retries", "5",
        "--fragment-retries", "5",
        # Keep yt-dlp defaults so it can auto-pick the best working YouTube client.
        "--no-check-certificate",
        "--verbose",
        url,
    ]

    log.info("Starting download: %s", url)
    log.info("Target path: %s", dest)
    log.info("Format: %s", dl_cfg["format"])
    log.debug("yt-dlp command: %s", " ".join(cmd))

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        log.error("yt-dlp exited with code %d", result.returncode)
        sys.exit(result.returncode)

    # yt-dlp may produce video.mp4 or video.webm → normalise to dest
    produced = dest.parent / f"{stem}.mp4"
    if not produced.exists():
        # Fallback: find any file with matching stem
        candidates = list(dest.parent.glob(f"{stem}.*"))
        # Exclude partial download files
        candidates = [c for c in candidates if not c.suffix.endswith(".part")]
        if not candidates:
            log.error("Download completed but output file not found in %s", dest.parent)
            sys.exit(1)
        produced = candidates[0]

    # Rename to exactly the configured filename if needed
    if produced != dest:
        produced.rename(dest)
        produced = dest

    # Log quality info
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,codec_name",
        "-of", "default=noprint_wrappers=1",
        str(produced),
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    log.info("Download quality info:\n%s", probe_result.stdout.strip())

    size_mb = produced.stat().st_size / 1_048_576
    log.info("Download complete → %s (%.1f MB)", produced, size_mb)
    return produced


# ─── CLI entry-point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download a YouTube VOD or stream.")
    parser.add_argument("url", help="YouTube URL to download")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: input/video.mp4 from config.yaml)",
    )
    args = parser.parse_args()

    download(args.url, args.output)


if __name__ == "__main__":
    main()
