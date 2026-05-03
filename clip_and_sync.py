"""
clip_and_sync.py — Lightened script to re-run clipping and sync to Drive.
Useful for applying new export settings without re-downloading or re-analyzing.
"""
import sys
import time
from pathlib import Path

from export import export_all
from sync import sync_to_drive
from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("clip_sync", cfg["logging"]["log_file"], cfg["logging"]["level"])


def run_clip_sync():
    """Run Phase 4 (Export) and Phase 5 (Sync) only."""
    log.info("🚀 Starting Clip & Sync Only Mode...")

    # ─── Locate Inputs ──────────────────────────────────────────────────────
    # Try to find the most likely candidates if not specified
    video_path = Path(cfg["paths"]["input"]) / cfg["download"]["output_filename"]
    highlights_path = Path(cfg["paths"]["highlights"]) / "video.yaml"

    if not video_path.exists():
        log.error(f"❌ Video not found: {video_path}")
        log.info("👉 Run the full pipeline first or check your config.yaml paths.")
        return

    if not highlights_path.exists():
        log.error(f"❌ Highlights file not found: {highlights_path}")
        return

    # ─── Phase 4: Export ─────────────────────────────────────────────────────
    log.info(f"🎞️  Clipping video using: {highlights_path}")
    t0 = time.perf_counter()
    try:
        exported_clips = export_all(str(highlights_path), str(video_path))
    except Exception as e:
        log.error(f"❌ Export failed: {e}")
        return

    if not exported_clips:
        log.warning("⚠️ No clips were exported.")
        return

    log.info(f"✅ Exported {len(exported_clips)} clips in {time.perf_counter() - t0:.1f}s")

    # ─── Phase 5: Sync ───────────────────────────────────────────────────────
    log.info("📤 Syncing to Google Drive...")
    t0 = time.perf_counter()
    try:
        # Use the parent folder of the first clip (the date-stamped folder)
        export_folder = str(exported_clips[0].parent)
        sync_to_drive(folder_path=export_folder)
        log.info(f"✅ Sync complete in {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        log.error(f"❌ Sync failed: {e}")


if __name__ == "__main__":
    run_clip_sync()
