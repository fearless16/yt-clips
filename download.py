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

    # ─── Capture Video Metadata ──────────────────────────────────────────────
    try:
        title_cmd = ["yt-dlp", "--get-title", "--no-warnings", url]
        title_res = subprocess.run(title_cmd, capture_output=True, text=True, timeout=10)
        video_title = title_res.stdout.strip() or "Cricket Highlights"
        log.info(f"📹 Video Title: {video_title}")
        
        # Save title for later phases
        meta_file = Path(paths_cfg["input"]) / "video_metadata.json"
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"title": video_title, "url": url}, f, indent=4)
    except Exception as e:
        log.warning(f"Failed to capture video title: {e}")
        video_title = "Cricket Highlights"

    # yt-dlp writes to a temp name then renames; use a template so we know
    # the final filename regardless of container.
    stem = dest.stem
    template = str(dest.parent / f"{stem}.%(ext)s")

    # ─── Autonomous Client Rotation Loop ──────────────────────────────────────
    # We try multiple player clients to find one that isn't bot-blocked.
    # ios and android are generally less restricted than web.
    # Feb 2025: web_creator and android_embedded are strong fallbacks.
    player_clients = [
        ("ios", "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"),
        ("android", "com.google.android.youtube/19.25.39 (Linux; U; Android 14; en_US) gzip"),
        ("web_creator", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"),
        ("mweb", "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36"),
        ("tv", "Mozilla/5.0 (Web0S; Linux/SmartTV) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.105 Safari/537.36 SmartTV/10.0.0"),
    ]
    
    proxy = dl_cfg.get("proxy")
    po_token = dl_cfg.get("po_token")
    cookies_path = Path("cookies.txt")

    success = False
    import time
    for client, ua in player_clients:
        log.info(f"🚀 Attempting download with client: {client}...")
        
        current_cmd = [
            "yt-dlp",
            "--format", dl_cfg["format"],
            "--merge-output-format", "mp4",
            "--format-sort", "res:2160,vbr,abr",
            "--output", template,
            "--no-playlist",
            "--progress",
            "--no-warnings",
            "--retries", "2",
            "--user-agent", ua,
            "--extractor-args", f"youtube:player-client={client}",
            "--js-runtimes", "deno",  # Modern JS challenge bypass
            "--no-check-certificate",
        ]

        if proxy:
            current_cmd.extend(["--proxy", proxy])
        
        if cookies_path.exists():
            current_cmd.extend(["--cookies", str(cookies_path)])
            
        if po_token:
            # PO Token works best with 'web' context but can be applied to others
            current_cmd[current_cmd.index("--extractor-args") + 1] += f";po_token=web+{po_token}"

        current_cmd.append(url)
        
        result = subprocess.run(current_cmd, check=False)
        
        if result.returncode == 0:
            success = True
            log.info(f"✅ Success with client: {client}")
            break
        else:
            log.warning(f"⚠️ Client {client} failed. Rotating...")
            time.sleep(2) # Grace period

    if not success:
        log.error("❌ ALL clients failed. Bot detection is blocking this IP.")
        log.error("💡 ZERO INTERVENTION TIP: Use a Proxy in config.yaml OR download the video LOCALLY first.")
        sys.exit(1)

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
