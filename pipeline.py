"""
pipeline.py — Main orchestrator. Runs all phases end-to-end.

Usage:
    python pipeline.py <youtube_url>
    python pipeline.py <youtube_url> --skip-download    # use existing input/video.mp4
    python pipeline.py <youtube_url> --skip-transcribe  # use existing transcript
    python pipeline.py <youtube_url> --sync             # auto-sync to Google Drive after export
    python pipeline.py <youtube_url> --upload           # auto-upload Shorts to YouTube
    python pipeline.py <youtube_url> --schedule         # auto-schedule uploads (2-hour intervals)
"""

import argparse
import sys
import time
from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("pipeline", cfg["logging"]["log_file"], cfg["logging"]["level"])


def _banner(phase: str) -> None:
    bar = "─" * 60
    log.info(bar)
    log.info("  %s", phase)
    log.info(bar)


def run(
    url: str,
    skip_download: bool = False,
    skip_transcribe: bool = False,
    skip_highlight: bool = False,
    skip_export: bool = False,
    skip_sync: bool = False,
    auto_sync: bool = False,
    auto_upload: bool = False,
    auto_schedule: bool = False,
) -> None:
    start_total = time.perf_counter()
    failures = []

    paths   = cfg["paths"]
    dl_cfg  = cfg["download"]
    video_path = str(Path(paths["input"]) / dl_cfg["output_filename"])

    # ── Phase 1: Download ──────────────────────────────────────────────────────
    _banner("PHASE 1 — DOWNLOAD")
    if skip_download:
        log.info("Skipping download (--skip-download). Using: %s", video_path)
        if not Path(video_path).exists():
            log.error("Video not found at %s — cannot skip download.", video_path)
            sys.exit(1)
    else:
        # CLEAN SLATE: Delete old video if it exists
        if Path(video_path).exists():
            log.warning("🧹 Cleaning old video file for a fresh start...")
            Path(video_path).unlink()
        
        t0 = time.perf_counter()
        from download import download
        video_path = str(download(url, video_path))
        log.info("Phase 1 complete in %.1f s", time.perf_counter() - t0)

    stem = Path(video_path).stem
    transcript_path = str(Path(paths["transcripts"]) / f"{stem}.json")
    highlights_path = str(Path(paths["highlights"])  / f"{stem}.yaml")

    # ── Phase 2: Transcribe ────────────────────────────────────────────────────
    _banner("PHASE 2 — TRANSCRIPTION")
    if skip_transcribe:
        log.info("Skipping transcription (--skip-transcribe). Using: %s", transcript_path)
        if not Path(transcript_path).exists():
            log.error("Transcript not found at %s — cannot skip transcription.", transcript_path)
            sys.exit(1)
    else:
        t0 = time.perf_counter()
        from transcribe import transcribe
        transcribe(video_path, transcript_path)
        log.info("Phase 2 complete in %.1f s", time.perf_counter() - t0)

    # ── Phase 3: Highlight Detection ───────────────────────────────────────────
    _banner("PHASE 3 — HIGHLIGHT DETECTION")
    if skip_highlight:
        log.info("Skipping highlight detection (--skip-highlight). Using: %s", highlights_path)
        if not Path(highlights_path).exists():
            log.error("Highlights not found at %s — cannot skip detection.", highlights_path)
            sys.exit(1)
    else:
        t0 = time.perf_counter()
        from highlight import detect_highlights
        highlights = detect_highlights(transcript_path, video_path, highlights_path)
        log.info("Phase 3 complete in %.1f s — %d highlights", time.perf_counter() - t0, len(highlights))

    # Verify highlights file exists before proceeding
    if not Path(highlights_path).exists():
        log.error("Highlights file missing: %s — cannot export.", highlights_path)
        sys.exit(1)

    # ── Phase 4: Export Shorts ─────────────────────────────────────────────────
    _banner("PHASE 4 — EXPORT SHORTS")
    exported = []
    if skip_export:
        log.info("Skipping export (--skip-export). Finding existing clips...")
        shorts_base = Path(cfg["paths"]["shorts"])
        if shorts_base.exists():
            # Get all subdirectories and sort by name (timestamp format YYYY-MM-DD_HHMMSS)
            folders = sorted([d for d in shorts_base.iterdir() if d.is_dir()], reverse=True)
            if folders:
                exported = sorted(list(folders[0].glob("*.mp4")))
                if exported:
                    log.info("🚀 Found %d clips in most recent export: %s", len(exported), folders[0])
            
            # Fallback: Search all .mp4 files in shorts_base recursively
            if not exported:
                exported = sorted(list(shorts_base.rglob("*.mp4")))
                if exported:
                    log.info("Found %d clips recursively in %s", len(exported), shorts_base)
        else:
            log.warning("Shorts base directory %s not found.", shorts_base)
    else:
        t0 = time.perf_counter()
        from export import export_all
        exported = export_all(highlights_path, video_path, transcript_path=transcript_path)
        log.info("Phase 4 complete in %.1f s — %d clips exported", time.perf_counter() - t0, len(exported))

    # ── Phase 4.5: SEO & Thumbnails ──────────────────────────────────────────
    if exported:
        _banner("PHASE 4.5 — SEO & THUMBNAILS")
        t0 = time.perf_counter()
        from seo import process_all_seo
        from thumbnail import process_all_thumbnails
        
        export_dir = str(exported[0].parent)
        
        # 1. Generate Metadata (Batch AI SEO)
        # This writes {clip_id}_metadata.json for each highlight in the directory
        process_all_seo(highlights_path, export_dir)
        
        # 2. Generate Thumbnails (Frame extraction or AI)
        # This searches for mp4s and matching metadata in the folder
        process_all_thumbnails(export_dir)
        
        log.info("Phase 4.5 complete in %.1f s", time.perf_counter() - t0)

    # ── Phase 5: Sync to Google Drive (optional) ──────────────────────────────
    if auto_sync and exported and not skip_sync:
        _banner("PHASE 5 — SYNC TO GOOGLE DRIVE")
        t0 = time.perf_counter()
        try:
            from sync import sync_to_drive
            export_folder = str(exported[0].parent)
            sync_to_drive(folder_path=export_folder)
            log.info("Phase 5 complete in %.1f s", time.perf_counter() - t0)
        except Exception as e:
            log.error("Sync failed: %s", e)
            failures.append({"phase": "Drive Sync", "error": str(e)})
    elif skip_sync:
        log.info("Skipping Drive sync (--skip-sync)")

    # ── Phase 6: YouTube Upload (optional) ────────────────────────────────────
    uploaded_count = 0
    if auto_upload and exported:
        _banner("PHASE 6 — YOUTUBE UPLOAD")
        t0 = time.perf_counter()
        from upload import upload_video
        from scheduler import get_next_slot, format_for_youtube

        interval = cfg["youtube"].get("schedule_interval_hours", 2)

        for clip_path in exported:
            meta_path = clip_path.with_name(f"{clip_path.stem}_metadata.json")
            if meta_path.exists():
                publish_at = None
                if auto_schedule:
                    slot = get_next_slot(interval)
                    publish_at = format_for_youtube(slot)

                try:
                    upload_video(
                        str(clip_path),
                        str(meta_path),
                        privacy=cfg["youtube"]["privacy_status"],
                        publish_at=publish_at,
                    )
                    uploaded_count += 1
                except Exception as e:
                    log.error("Upload failed for %s: %s", clip_path.name, e)
                    failures.append({"phase": "YouTube Upload", "clip_id": clip_path.name, "error": str(e)})
            else:
                log.warning("No metadata found for %s — skipping upload.", clip_path.name)

        log.info("Phase 6 complete in %.1f s — %d/%d clips uploaded",
                 time.perf_counter() - t0, uploaded_count, len(exported))

    # ── Summary ────────────────────────────────────────────────────────────────
    total = time.perf_counter() - start_total
    
    try:
        from utils.reports import generate_run_report
        export_dir_path = str(exported[0].parent) if exported else str(Path(cfg["paths"]["shorts"]))
        if not Path(export_dir_path).exists():
            Path(export_dir_path).mkdir(parents=True, exist_ok=True)
            
        report_file = generate_run_report(
            export_dir=export_dir_path,
            pipeline_duration_sec=total,
            exported_clips=exported,
            uploaded_clips=uploaded_count,
            failures=failures
        )
        log.info("📄 Pipeline Report generated: %s", report_file)
    except Exception as e:
        log.error("Failed to generate run report: %s", e)

    log.info("─" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info("  Total time : %.1f s (%.1f min)", total, total / 60)
    log.info("  Shorts ready:")
    for p in exported:
        size_mb = p.stat().st_size / 1_048_576
        log.info("    → %s  (%.1f MB)", p, size_mb)
    if not auto_sync:
        log.info("  To sync to Drive: python sync.py")
    log.info("─" * 60)


# ─── CLI entry-point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="yt-clips — Convert a YouTube VOD into vertical Shorts automatically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (download → transcribe → highlight → export)
  python pipeline.py https://youtu.be/XXXX

  # Full run + auto-sync to Google Drive
  python pipeline.py https://youtu.be/XXXX --sync

  # Re-use existing download + transcript, redo highlights + export
  python pipeline.py https://youtu.be/XXXX --skip-download --skip-transcribe

  # Re-use everything, only re-export clips
  python pipeline.py https://youtu.be/XXXX --skip-download --skip-transcribe --skip-highlight
""",
    )
    parser.add_argument("url", help="YouTube URL (VOD or stream)")
    parser.add_argument("-skip-download", "--skip-download",   action="store_true", help="Skip Phase 1 (use existing input/video.mp4)")
    parser.add_argument("-skip-transcribe", "--skip-transcribe", action="store_true", help="Skip Phase 2 (use existing transcript)")
    parser.add_argument("-skip-highlight", "--skip-highlight",  action="store_true", help="Skip Phase 3 (use existing highlights)")
    parser.add_argument("-skip-export", "--skip-export",     action="store_true", help="Skip Phase 4 (use existing clips in shorts/)")
    parser.add_argument("-skip-sync", "--skip-sync",       action="store_true", help="Skip Phase 5 (Drive Sync)")
    parser.add_argument("-sync", "--sync",            action="store_true", help="Auto-sync exported Shorts to Google Drive")
    parser.add_argument("-upload", "--upload",          action="store_true", help="Auto-upload exported Shorts to YouTube")
    parser.add_argument("-schedule", "--schedule",        action="store_true", help="Auto-schedule uploads (2-hour intervals)")
    args = parser.parse_args()

    run(
        url=args.url,
        skip_download=args.skip_download,
        skip_transcribe=args.skip_transcribe,
        skip_highlight=args.skip_highlight,
        skip_export=args.skip_export,
        skip_sync=args.skip_sync,
        auto_sync=args.sync,
        auto_upload=args.upload,
        auto_schedule=args.schedule or args.upload, # Always schedule if uploading
    )


if __name__ == "__main__":
    main()
