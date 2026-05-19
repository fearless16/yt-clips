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
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from utils.config import load_config
from utils.logger import get_logger, phase_tracker

cfg = load_config()
log = get_logger("pipeline", cfg["logging"]["log_file"], cfg["logging"]["level"])


def _run_tests(skip: bool = False) -> None:
    """Run pytest guard before any expensive generation. Aborts on failure."""
    if skip:
        log.warning("Skipping pre-generation test guard (--skip-tests)")
        return
    phase_tracker.begin("GUARD — Pre-Generation Tests")
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-x", "--timeout=120", "-q"],
        capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        log.error("Pre-generation tests FAILED (%d failures, %.1fs)\n%s",
                  result.returncode, elapsed, result.stdout.strip() or result.stderr[:1000])
        phase_tracker.end("failed")
        sys.exit(1)
    log.info("Pre-generation tests PASSED (%.1fs)", elapsed)
    phase_tracker.end("done")


def _banner(phase: str) -> None:
    phase_tracker.begin(phase)


def run(
    url: str,
    skip_download: bool = False,
    skip_transcribe: bool = False,
    skip_highlight: bool = False,
    skip_export: bool = False,
    skip_sync: bool = False,
    skip_seo: bool = False,
    auto_sync: bool = False,
    auto_upload: bool = False,
    auto_schedule: bool = False,
    skip_tests: bool = False,
    sample_minutes: Optional[int] = None,
    sync_from_drive: bool = False,
) -> None:
    cfg = load_config()
    if not skip_tests and not cfg.get("testing", {}).get("enabled", False):
        skip_tests = True
    _run_tests(skip_tests)
    start_total = time.perf_counter()
    failures = []

    paths   = cfg["paths"]
    dl_cfg  = cfg["download"]
    video_path = str(Path(paths["input"]) / dl_cfg["output_filename"])

    # ── Phase 1: Download ──────────────────────────────────────────────────────
    _banner("PHASE 1 — DOWNLOAD")
    if sync_from_drive:
        log.info("Pulling video + transcript from Google Drive...")
        from sync import download_from_drive
        # Download video to input/
        download_from_drive(
            filenames=["video.mp4"],
            dest_dir=paths["input"],
        )
        # Download transcript to transcripts/
        download_from_drive(
            filenames=["video.json"],
            dest_dir=paths["transcripts"],
        )
        skip_download = True
        skip_transcribe = True
        log.info("Drive sync complete — files ready.")
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
        video_path = str(download(url, video_path, sample_minutes=sample_minutes))
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

    # ── Phase 2.5: Video Analysis (face/lighting map) ─────────────────────────
    _banner("PHASE 2.5 — VIDEO ANALYSIS")
    analysis_path = str(Path(cfg["paths"]["temp"]) / f"{stem}_analysis.json")
    try:
        from video_analyzer import analyze_video
        analysis = analyze_video(
            video_path=video_path,
            photos_dir="photos/",
            reference_image="expectation.png",
            sample_interval=2.0,
            output_path=analysis_path,
        )
        log.info("Phase 2.5 complete — face rate: %.0f%%, avg quality: %.3f",
                 analysis["summary"]["face_detection_rate"],
                 analysis["summary"]["avg_quality"])
    except Exception as e:
        log.warning("Video analysis failed (non-fatal): %s", e)
        analysis = None

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

    # ── Phase 4.25: Selective Enhancement (optional) ─────────────────────────
    if exported and cfg.get("enhancement", {}).get("selective", False):
        _banner("PHASE 4.25 — SELECTIVE ENHANCEMENT")
        t0 = time.perf_counter()
        try:
            from state_analyzer import analyze_clip as analyze_states
            from selective_enhancer import enhance_clip as selective_enhance
            from temporal_consistency import apply_temporal_consistency

            enhanced_exported = []
            for clip_path in exported:
                clip_str = str(clip_path)
                clip_id = clip_path.stem

                # Pass 1: State analysis
                log.info("[%s] Pass 1: State analysis...", clip_id)
                analysis_path = str(Path(cfg["paths"]["temp"]) / f"{clip_id}_state_analysis.json")
                analysis = analyze_states(
                    video_path=clip_str,
                    sample_rate=2,
                    segment_size_sec=2.0,
                    output_path=analysis_path,
                )
                if "error" in analysis:
                    log.warning("[%s] State analysis failed — skipping enhancement", clip_id)
                    enhanced_exported.append(clip_path)
                    continue

                # Pass 2: Selective enhancement
                log.info("[%s] Pass 2: Selective enhancement...", clip_id)
                enhanced_path = str(Path(cfg["paths"]["temp"]) / f"{clip_id}_enhanced.mp4")
                enhanced = selective_enhance(
                    video_path=clip_str,
                    analysis_path=analysis_path,
                    output_path=enhanced_path,
                )

                # Pass 3: Temporal consistency
                log.info("[%s] Pass 3: Temporal consistency...", clip_id)
                final_path = str(Path(cfg["paths"]["temp"]) / f"{clip_id}_final.mp4")
                final = apply_temporal_consistency(
                    video_path=enhanced,
                    analysis_path=analysis_path,
                    output_path=final_path,
                )

                # Replace original with enhanced version
                import shutil
                shutil.move(final, clip_str)
                enhanced_exported.append(clip_path)
                log.info("[%s] Enhancement complete", clip_id)

            exported = enhanced_exported
            log.info("Phase 4.25 complete in %.1f s — %d clips enhanced",
                     time.perf_counter() - t0, len(exported))
        except Exception as e:
            log.warning("Selective enhancement failed (non-fatal): %s", e)

    # ── Phase 4.5: SEO & Thumbnails ──────────────────────────────────────────
    if exported:
        _banner("PHASE 4.5 — SEO & THUMBNAILS")
        t0 = time.perf_counter()
        from thumbnail import process_all_thumbnails

        export_dir = str(exported[0].parent)

        if skip_seo:
            log.info("Skipping SEO (--skip-seo)")
        else:
            from seo import process_all_seo
            seo_results = [f for f in Path(export_dir).glob("*_metadata.json")]
            if not seo_results:
                process_all_seo(highlights_path, export_dir)
            else:
                log.info("SEO metadata already exists for %d clips — skipping", len(seo_results))

        # Generate Thumbnails (Frame extraction or AI)
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

        interval = cfg.get("upload_schedule", {}).get("interval_hours",
                   cfg["youtube"].get("schedule_interval_hours", 1))

        # Pre-generate schedule with jitter + prime-time assignment
        if auto_schedule:
            from scheduler import assign_clips_to_slots, format_for_youtube
            # Optional: rank clips by SEO quality score
            clip_scores = {}
            for clip_path in exported:
                meta_path = clip_path.with_name(f"{clip_path.stem}_metadata.json")
                if meta_path.exists():
                    try:
                        import json
                        with open(meta_path) as f:
                            meta = json.load(f)
                        score = meta.get("quality_score") or meta.get("seo_score") or 0.0
                        clip_scores[clip_path.stem] = float(score)
                    except (json.JSONDecodeError, OSError, ValueError):
                        pass
            assignments = assign_clips_to_slots(
                clips=[p.stem for p in exported],
                interval_hours=interval,
                clip_scores=clip_scores if clip_scores else None,
            )
            slot_map = {stem: dt for stem, dt in assignments}
            log.info("Schedule generated for %d clips (jittered hourly)", len(assignments))

        for clip_path in exported:
            meta_path = clip_path.with_name(f"{clip_path.stem}_metadata.json")
            if meta_path.exists():
                publish_at = None
                if auto_schedule:
                    slot = slot_map.get(clip_path.stem)
                    if slot:
                        publish_at = format_for_youtube(slot)
                        log.info("⏰ Slot: %s → %s", clip_path.stem, slot.strftime("%Y-%m-%d %H:%M IST"))

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

    # ── Phase 7: Analytics & SEO Learning ──────────────────────────────────────
    if auto_upload:
        _banner("PHASE 7 — ANALYTICS & SEO LEARNING")
        try:
            from analytics import generate_daily_insights
            generate_daily_insights()
        except FileNotFoundError as e:
            if "yt_analytics_token.json" in str(e) or "client_secrets" in str(e):
                log.warning("Analytics skipped — need yt_analytics_token.json with youtube.readonly scope")
            else:
                log.warning("Analytics failed: %s", e)
        except Exception as e:
            log.warning("Analytics failed: %s", e)

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

    from rich import print as rprint
    rprint(phase_tracker.summary_table())

    log.info("─" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info("  Total time : %.1f s (%.1f min)", total, total / 60)
    if exported:
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
    parser.add_argument("-skip-seo", "--skip-seo",         action="store_true", help="Skip SEO generation in Phase 4.5")
    parser.add_argument("-sync", "--sync",            action="store_true", help="Auto-sync exported Shorts to Google Drive")
    parser.add_argument("-upload", "--upload",          action="store_true", help="Auto-upload exported Shorts to YouTube")
    parser.add_argument("-schedule", "--schedule",        action="store_true", help="Auto-schedule uploads (2-hour intervals)")
    parser.add_argument("-skip-tests", "--skip-tests",    action="store_true", help="Skip pre-generation pytest guard")
    parser.add_argument("--sample-minutes", type=int, default=None, help="Download only a random N-minute sample of the video")
    parser.add_argument("--sync-from-drive", action="store_true", help="Pull video + transcript from Google Drive instead of downloading")
    args = parser.parse_args()

    run(
        url=args.url,
        skip_download=args.skip_download,
        skip_transcribe=args.skip_transcribe,
        skip_highlight=args.skip_highlight,
        skip_export=args.skip_export,
        skip_sync=args.skip_sync,
        skip_seo=args.skip_seo,
        auto_sync=args.sync,
        auto_upload=args.upload,
        auto_schedule=args.schedule,
        skip_tests=args.skip_tests,
        sample_minutes=args.sample_minutes,
        sync_from_drive=args.sync_from_drive,
    )


if __name__ == "__main__":
    main()
