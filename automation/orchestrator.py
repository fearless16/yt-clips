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

from utils.logger import get_logger, run_phase, new_run_id
log = get_logger("orchestrator")


@dataclass
class PipelineResult:
    """Result container for the pipeline run.

    Attributes:
        exported: List of exported file paths.
        uploaded_count: Number of videos uploaded.
        failures: List of phase error messages.
        total_seconds: Total wall-clock runtime.
        transcript_source: "api", "vtt", "none", or "unavailable".
        run_id: Short correlation id shared by every log record of this run.
    """
    exported: list = field(default_factory=list)
    uploaded_count: int = 0
    failures: list = field(default_factory=list)
    total_seconds: float = 0.0
    transcript_source: str = "none"
    run_id: str = ""


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
    result.run_id = new_run_id()
    rid = result.run_id
    log.info("[START] pipeline run_id=%s url=%s", rid, url,
             extra={"stage": "pipeline_start", "status": "start",
                    "run_id": rid, "duration_ms": 0})

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
        try:
            with run_phase(log, "phase 0 Transcript", "transcript_fetch", run_id=rid) as ph:
                from .transcript import fetch as fetch_transcript
                transcript = fetch_transcript(url, output_path=transcript_path)
                if transcript.get("segments"):
                    skip_transcribe = True
                    result.transcript_source = transcript.get("source", "api")
                    ph.set(source=result.transcript_source,
                           segments=len(transcript["segments"]))
                else:
                    ph.set(source="none", segments=0)
        except Exception as e:
            result.failures.append("phase0: %s" % e)

    # ── Phase 1: Download ─────────────────────────────────────────────────────
    if sync_from_drive:
        try:
            with run_phase(log, "phase 1 Drive pull", "drive_pull", run_id=rid):
                from sync import download_from_drive
                download_from_drive(filenames=["video.mp4"], dest_dir=input_dir)
                download_from_drive(filenames=["%s.json" % stem], dest_dir=transcripts_dir)
                # Also pull any exported shorts so post-export phases can run
                download_from_drive(filenames=None, dest_dir=shorts_dir)
                skip_download = True
                skip_transcribe = True
        except Exception as e:
            result.failures.append("phase1 sync: %s" % e)

    if not skip_download:
        try:
            with run_phase(log, "phase 1 Download", "download", run_id=rid) as ph:
                from download import download
                video_path = str(download(url, video_path, sample_minutes=sample_minutes))
                stem = Path(video_path).stem
                transcript_path = str(Path(transcripts_dir) / ("%s.json" % stem))
                highlights_path = str(Path(highlights_dir) / ("%s.yaml" % stem))
                ph.set(video_path=video_path)
        except Exception as e:
            result.failures.append("phase1: %s" % e)

    # ── Phase 2: Transcribe ───────────────────────────────────────────────────
    if not skip_transcribe:
        try:
            with run_phase(log, "phase 2 Transcribe", "transcribe", run_id=rid):
                from transcribe import transcribe
                transcribe(video_path, transcript_path)
        except Exception as e:
            result.failures.append("phase2: %s" % e)

    # ── Phase 3: Highlight Detection ──────────────────────────────────────────
    if not skip_highlight:
        try:
            with run_phase(log, "phase 3 Highlight", "highlight", run_id=rid):
                from highlight import detect_highlights
                detect_highlights(transcript_path, video_path, highlights_path)
        except Exception as e:
            result.failures.append("phase3: %s" % e)

    # ── Phase 4: Export Shorts ────────────────────────────────────────────────
    if not skip_export:
        try:
            with run_phase(log, "phase 4 Export", "export", run_id=rid) as ph:
                from export import export_all
                result.exported = export_all(
                    highlights_path, video_path,
                    transcript_path=transcript_path,
                    generate_seo=True,
                )
                ph.set(exported=len(result.exported))
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
        try:
            with run_phase(log, "phase 4.5 Enhancement (%s)" % mode, "enhancement", run_id=rid) as ph:
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
                ph.set(enhanced=len(result.exported), mode=mode)
        except Exception as e:
            result.failures.append("phase4.5: %s" % e)

    # ── Phase 5: SEO Generation ──────────────────────────────────────────────
    if not skip_seo and result.exported:
        try:
            with run_phase(log, "phase 5 SEO", "seo", run_id=rid) as ph:
                from .seo.seo import process_all_seo, retry_failed_seo
                export_dir = str(result.exported[0].parent)
                process_all_seo(highlights_path, export_dir)
                # Close the escalate-not-degrade loop: retry any clips that were
                # queued on failure (no generic fallback was written for them).
                retry = retry_failed_seo(export_dir)
                ph.set(clips=len(result.exported),
                       seo_recovered=retry.get("recovered", 0),
                       seo_still_queued=retry.get("still_failed", 0))
        except Exception as e:
            result.failures.append("phase5: %s" % e)

    # ── Phase 5.5: Thumbnails ────────────────────────────────────────────────
    if result.exported:
        try:
            with run_phase(log, "phase 5.5 Thumbnails", "thumbnails", run_id=rid):
                from thumbnail import process_all_thumbnails
                export_dir = str(result.exported[0].parent)
                process_all_thumbnails(export_dir)
        except Exception as e:
            result.failures.append("phase5.5: %s" % e)

    # ── Phase 6: Sync to Google Drive ────────────────────────────────────────
    do_sync = auto_sync and not skip_sync
    if do_sync and result.exported:
        try:
            with run_phase(log, "phase 6 Sync", "sync", run_id=rid):
                from sync import sync_to_drive
                export_folder = str(result.exported[0].parent)
                sync_to_drive(folder_path=export_folder)
        except Exception as e:
            result.failures.append("phase6: %s" % e)
    elif skip_sync:
        log.info("[phase 6 Sync] skipped (--skip-sync)",
                 extra={"stage": "sync", "status": "skipped", "run_id": rid, "duration_ms": 0})

    # ── Phase 7: Upload to YouTube ───────────────────────────────────────────
    if (auto_upload or auto_schedule) and result.exported:
        try:
            with run_phase(log, "phase 7 Upload", "upload", run_id=rid) as ph:
                from upload import upload_video
                privacy = cfg.get("youtube", {}).get("privacy_status", "public")
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

                skipped_no_meta = 0
                for clip_idx, clip_path in enumerate(result.exported):
                    meta_path = clip_path.with_name("%s_metadata.json" % clip_path.stem)
                    if not meta_path.exists():
                        skipped_no_meta += 1
                        log.warning("No metadata for %s — skipping upload", clip_path.name,
                                    extra={"stage": "upload", "status": "skipped",
                                           "run_id": rid, "duration_ms": 0,
                                           "metadata": {"clip": clip_path.name}})
                        continue
                    publish_at = None
                    if auto_schedule and clip_idx > 0:
                        import random
                        slot = slot_map.get(clip_path.stem)
                        if slot:
                            jitter_hours = random.uniform(2, 3)
                            from datetime import timedelta
                            slot = slot + timedelta(hours=jitter_hours * (clip_idx - 1))
                            publish_at = format_for_youtube(slot)
                    try:
                        upload_video(
                            str(clip_path), str(meta_path),
                            privacy=privacy, publish_at=publish_at,
                        )
                        result.uploaded_count += 1
                    except Exception as e:
                        log.error("Upload failed for %s: %s", clip_path.name, e,
                                  exc_info=True,
                                  extra={"stage": "upload", "status": "failed",
                                         "run_id": rid, "error_type": type(e).__name__,
                                         "metadata": {"clip": clip_path.name}})
                        result.failures.append("phase7 %s: %s" % (clip_path.name, e))

                ph.set(uploaded=result.uploaded_count, skipped_no_meta=skipped_no_meta)
        except Exception as e:
            result.failures.append("phase7: %s" % e)

    # ── Phase 8: Analytics & SEO Learning ──────────────────────────────────────
    if auto_upload:
        try:
            with run_phase(log, "phase 8 Analytics", "analytics", run_id=rid):
                from .seo.analytics import generate_daily_insights
                generate_daily_insights()
        except Exception as e:
            result.failures.append("phase8: %s" % e)

    result.total_seconds = time.monotonic() - start
    status = "partial" if result.failures else "ok"
    log.info("[EXIT] pipeline run_id=%s url=%s exported=%d uploaded=%d failures=%d elapsed=%.1fs transcript=%s",
             rid, url, len(result.exported), result.uploaded_count, len(result.failures),
             result.total_seconds, result.transcript_source,
             extra={"stage": "pipeline_total", "status": status, "run_id": rid,
                    "duration_ms": int(result.total_seconds * 1000),
                    "metadata": {
                        "exported": len(result.exported),
                        "uploaded": result.uploaded_count,
                        "failures": len(result.failures),
                        "transcript_source": result.transcript_source,
                    }})
    if result.failures:
        log.warning("[EXIT] %d phase failure(s) — see ERROR records above (run_id=%s)",
                    len(result.failures), rid,
                    extra={"stage": "pipeline_total", "status": "partial", "run_id": rid,
                           "duration_ms": 0, "metadata": {"failures": list(result.failures)}})
    return result
