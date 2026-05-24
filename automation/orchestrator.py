"""orchestrator.py — 8-phase pipeline runner for yt-clips automation.

Orchestrates the full pipeline: transcript → download → transcribe →
highlight → export → sync → SEO → upload. Each phase is independently
skippable via flags.

Usage::

    from .orchestrator import run

    result = run(url="https://youtu.be/...",
                 auto_sync=True, auto_upload=True)
"""

import time
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataclasses import dataclass, field
from typing import Optional

from . import config

log = logging.getLogger("orchestrator")


@dataclass
class PipelineResult:
    """Result container for the pipeline run.

    Attributes:
        exported: List of exported file paths.
        uploaded_count: Number of videos uploaded.
        failures: List of phase error messages.
        total_seconds: Total wall-clock runtime.
        transcript_source: "api", "vtt", "none", or "unavailable".
    """
    exported: list = field(default_factory=list)
    uploaded_count: int = 0
    failures: list = field(default_factory=list)
    total_seconds: float = 0.0
    transcript_source: str = "none"


def _t(start: float) -> str:
    return "%.1fs" % (time.monotonic() - start)


def run(url: str, skip_download=False, skip_transcribe=False,
        skip_highlight=False, skip_export=False, skip_sync=False,
        skip_seo=False, auto_sync=False, auto_upload=False,
        auto_schedule=False, skip_tests=False, sample_minutes=None,
        sync_from_drive=False, mode=None) -> PipelineResult:
    """Execute the yt-clips pipeline.

    Each phase is a lazy import (zero cost if skipped).

    Args:
        url: YouTube video URL.
        skip_*: Skip the corresponding phase.
        auto_sync: Sync outputs to Google Drive after export.
        auto_upload: Upload to YouTube after SEO generation.
        auto_schedule: Schedule the upload.
        skip_tests: Skip pre-generation pytest guard.
        sample_minutes: Limit download to first N minutes.
        sync_from_drive: Sync input from Google Drive first.
        mode: Enhancement mode ("face_mapper" or "ref_grade").

    Returns:
        PipelineResult with exported list, counts, and failures.
    """
    start = time.monotonic()
    result = PipelineResult()

    cfg = config.load()
    paths = cfg.get("paths", {})
    dl_cfg = cfg.get("download", {})

    input_dir = paths.get("input", "input")
    transcripts_dir = paths.get("transcripts", "transcripts")
    highlights_dir = paths.get("highlights", "highlights")
    shorts_dir = paths.get("shorts", "shorts")
    output_filename = dl_cfg.get("output_filename", "video.mp4")

    video_path = str(Path(input_dir) / output_filename)
    stem = Path(video_path).stem
    transcript_path = str(Path(transcripts_dir) / ("%s.json" % stem))
    highlights_path = str(Path(highlights_dir) / ("%s.yaml" % stem))

    # ── Phase 0: YouTube transcript (API, skip Whisper if available) ──────────
    if not skip_transcribe:
        t0 = time.monotonic()
        log.info("[phase 0] YouTube transcript")
        try:
            from .transcript import fetch as fetch_transcript
            transcript = fetch_transcript(url, output_path=transcript_path)
            if transcript.get("segments"):
                skip_transcribe = True
                result.transcript_source = transcript.get("source", "api")
                log.info("[phase 0] done %s source=%s segments=%d", _t(t0), result.transcript_source, len(transcript["segments"]))
        except Exception as e:
            result.failures.append("phase0: %s" % e)

    # ── Phase 1: Download ─────────────────────────────────────────────────────
    if sync_from_drive:
        log.info("[phase 1] Pulling video + transcript + shorts from Google Drive")
        try:
            from sync import download_from_drive
            download_from_drive(filenames=["video.mp4"], dest_dir=input_dir)
            download_from_drive(filenames=["%s.json" % stem], dest_dir=transcripts_dir)
            # Also pull any exported shorts so post-export phases can run
            download_from_drive(filenames=None, dest_dir=shorts_dir)
            skip_download = True
            skip_transcribe = True
            log.info("[phase 1] Drive sync complete")
        except Exception as e:
            result.failures.append("phase1 sync: %s" % e)

    if not skip_download:
        t0 = time.monotonic()
        log.info("[phase 1] Download")
        try:
            from download import download
            video_path = str(download(url, video_path, sample_minutes=sample_minutes))
            stem = Path(video_path).stem
            transcript_path = str(Path(transcripts_dir) / ("%s.json" % stem))
            highlights_path = str(Path(highlights_dir) / ("%s.yaml" % stem))
            log.info("[phase 1] done %s", _t(t0))
        except Exception as e:
            result.failures.append("phase1: %s" % e)

    # ── Phase 2: Transcribe ───────────────────────────────────────────────────
    if not skip_transcribe:
        t0 = time.monotonic()
        log.info("[phase 2] Transcribe")
        try:
            from transcribe import transcribe
            transcribe(video_path, transcript_path)
            log.info("[phase 2] done %s", _t(t0))
        except Exception as e:
            result.failures.append("phase2: %s" % e)

    # ── Phase 3: Highlight Detection ──────────────────────────────────────────
    if not skip_highlight:
        t0 = time.monotonic()
        log.info("[phase 3] Highlight")
        try:
            from highlight import detect_highlights
            detect_highlights(transcript_path, video_path, highlights_path)
            log.info("[phase 3] done %s", _t(t0))
        except Exception as e:
            result.failures.append("phase3: %s" % e)

    # ── Phase 4: Export Shorts ────────────────────────────────────────────────
    if not skip_export:
        t0 = time.monotonic()
        log.info("[phase 4] Export")
        try:
            from export import export_all
            result.exported = export_all(
                highlights_path, video_path,
                transcript_path=transcript_path,
                generate_seo=False,
            )
            log.info("[phase 4] done %s exported=%d", _t(t0), len(result.exported))
        except Exception as e:
            result.failures.append("phase4: %s" % e)
    else:
        # Populate from existing files on disk so post-export phases can run
        # Clips live in dated subdirs (e.g. shorts/2026-05-23_155847/clip1.mp4)
        existing = sorted(Path(shorts_dir).rglob("*.mp4")) if Path(shorts_dir).exists() else []
        # Exclude test files and prefer the most recent dated subdir
        existing = [p for p in existing if "test_output" not in p.name]
        if existing:
            # Use only the latest batch (most recent dated subdir)
            latest_dir = max(set(p.parent for p in existing), key=lambda d: d.name)
            existing = sorted(latest_dir.glob("*.mp4"))
        if existing:
            result.exported = existing
            log.info("[phase 4] skipped — using %d existing clips from %s", len(existing), shorts_dir)
        else:
            log.warning("[phase 4] skipped — no .mp4 files found in %s", shorts_dir)

    # ── Phase 4.5: Enhancement (optional, mode-driven) ───────────────────────
    if result.exported and mode:
        t0 = time.monotonic()
        log.info("[phase 4.5] Enhancement (%s)", mode)
        try:
            import shutil
            ref_path = cfg.get("enhancement", {}).get("reference", "expectation.png")
            enhanced = []

            if mode == "ref_grade":
                from ref_grade import grade_video
                for clip_path in result.exported:
                    graded = str(Path(paths.get("temp", "temp")) / ("%s_graded.mp4" % clip_path.stem))
                    r = grade_video(str(clip_path), ref_path, graded)
                    if r == graded and Path(graded).exists():
                        shutil.move(graded, str(clip_path))
                        enhanced.append(clip_path)
                        log.info("[%s] Color grade applied", clip_path.stem)
                    else:
                        enhanced.append(clip_path)
                        log.warning("[%s] Color grade failed — keeping original", clip_path.stem)

            elif mode == "face_mapper":
                from face_mapper import enhance_video
                for clip_path in result.exported:
                    enhanced_path = str(Path(paths.get("temp", "temp")) / ("%s_enhanced.mp4" % clip_path.stem))
                    r = enhance_video(str(clip_path), ref_path, enhanced_path,
                                      use_region_grading=True)
                    if r == enhanced_path and Path(enhanced_path).exists():
                        shutil.move(enhanced_path, str(clip_path))
                        enhanced.append(clip_path)
                        log.info("[%s] Face-mapper applied", clip_path.stem)
                    else:
                        enhanced.append(clip_path)
                        log.warning("[%s] Face-mapper failed — keeping original", clip_path.stem)

            result.exported = enhanced
            log.info("[phase 4.5] done %s enhanced=%d", _t(t0), len(result.exported))
        except Exception as e:
            result.failures.append("phase4.5: %s" % e)

    # ── Phase 5: SEO Generation ──────────────────────────────────────────────
    if not skip_seo and result.exported:
        t0 = time.monotonic()
        log.info("[phase 5] SEO")
        try:
            from .seo.seo import process_all_seo
            export_dir = str(result.exported[0].parent)
            process_all_seo(highlights_path, export_dir)
            log.info("[phase 5] done %s", _t(t0))
        except Exception as e:
            result.failures.append("phase5: %s" % e)

    # ── Phase 5.5: Thumbnails ────────────────────────────────────────────────
    if result.exported:
        t0 = time.monotonic()
        log.info("[phase 5.5] Thumbnails")
        try:
            from thumbnail import process_all_thumbnails
            export_dir = str(result.exported[0].parent)
            process_all_thumbnails(export_dir)
            log.info("[phase 5.5] done %s", _t(t0))
        except Exception as e:
            result.failures.append("phase5.5: %s" % e)

    # ── Phase 6: Sync to Google Drive ────────────────────────────────────────
    do_sync = auto_sync and not skip_sync
    if do_sync and result.exported:
        t0 = time.monotonic()
        log.info("[phase 6] Sync")
        try:
            from sync import sync_to_drive
            export_folder = str(result.exported[0].parent)
            sync_to_drive(folder_path=export_folder)
            log.info("[phase 6] done %s", _t(t0))
        except Exception as e:
            result.failures.append("phase6: %s" % e)
    elif skip_sync:
        log.info("[phase 6] Skipping sync (--skip-sync)")

    # ── Phase 7: Upload to YouTube ───────────────────────────────────────────
    if (auto_upload or auto_schedule) and result.exported:
        t0 = time.monotonic()
        log.info("[phase 7] Upload")
        try:
            from upload import upload_video
            privacy = cfg.get("youtube", {}).get("privacy_status", "private")
            interval = cfg.get("upload_schedule", {}).get(
                "interval_hours",
                cfg.get("youtube", {}).get("schedule_interval_hours", 1)
            )

            slot_map = {}
            if auto_schedule:
                from scheduler import assign_clips_to_slots, format_for_youtube
                clip_scores = {}
                for clip_path in result.exported:
                    meta_path = clip_path.with_name("%s_metadata.json" % clip_path.stem)
                    if meta_path.exists():
                        try:
                            import json
                            with open(meta_path) as f:
                                meta = json.load(f)
                            score = meta.get("quality_score") or meta.get("seo_score") or 0.0
                            clip_scores[clip_path.stem] = float(score)
                        except (ValueError, OSError):
                            pass
                assignments = assign_clips_to_slots(
                    clips=[p.stem for p in result.exported],
                    interval_hours=interval,
                    clip_scores=clip_scores if clip_scores else None,
                )
                slot_map = {stem: dt for stem, dt in assignments}
                log.info("Schedule generated for %d clips", len(assignments))

            for clip_path in result.exported:
                meta_path = clip_path.with_name("%s_metadata.json" % clip_path.stem)
                if not meta_path.exists():
                    log.warning("No metadata for %s — skipping upload", clip_path.name)
                    continue
                publish_at = None
                if auto_schedule:
                    slot = slot_map.get(clip_path.stem)
                    if slot:
                        publish_at = format_for_youtube(slot)
                try:
                    upload_video(
                        str(clip_path), str(meta_path),
                        privacy=privacy, publish_at=publish_at,
                    )
                    result.uploaded_count += 1
                except Exception as e:
                    result.failures.append("phase7 %s: %s" % (clip_path.name, e))

            log.info("[phase 7] done %s uploaded=%d", _t(t0), result.uploaded_count)
        except Exception as e:
            result.failures.append("phase7: %s" % e)

    # ── Phase 8: Analytics & SEO Learning ──────────────────────────────────────
    if auto_upload:
        t0 = time.monotonic()
        log.info("[phase 8] Analytics")
        try:
            from .seo.analytics import generate_daily_insights
            generate_daily_insights()
            log.info("[phase 8] done %s", _t(t0))
        except Exception as e:
            result.failures.append("phase8: %s" % e)

    result.total_seconds = time.monotonic() - start
    log.info("pipeline done %.1fs failures=%d", result.total_seconds, len(result.failures))
    return result
