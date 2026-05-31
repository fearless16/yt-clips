"""orchestrator.py — 9-stage pipeline runner for yt-clips automation.

Stages:
    1. Ingest (git pull / env / download)
    2. Transcribe (ASR + timestamps + speaker chunks)
    3. Chunk into candidates
    4. Parallel LLM scoring (6 dimensions)
    5. Rank + filter (top 10 max)
    6. Generate final cut instructions
    7. SEO (selected clips only)
    8. Analytics writeback
    9. Self-learning updates prompt weights / rules

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
from . import decision_store, provider_health, emit_event, emit_infra_failed
from .memory.event_models import EventType

from utils.logger import get_logger, run_phase, new_run_id
log = get_logger("orchestrator")


@dataclass
class PipelineResult:
    exported: list = field(default_factory=list)
    uploaded_count: int = 0
    failures: list = field(default_factory=list)
    total_seconds: float = 0.0
    transcript_source: str = "none"
    run_id: str = ""
    selected_clips: int = 0
    rejected_clips: int = 0
    seo_generated: int = 0


def run(url: str, skip_download=False, skip_transcribe=False,
        skip_highlight=False, skip_export=False, skip_sync=False,
        skip_seo=False, auto_sync=False, auto_upload=False,
        auto_schedule=False, skip_tests=False, sample_minutes=None,
        sync_from_drive=False, mode=None, learn_only=False) -> PipelineResult:
    """Execute the 9-stage yt-clips pipeline.

    Each phase is a lazy import (zero cost if skipped).
    Error handling policy is hardcoded — see utils/ai_client.py ErrorCategory.

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
        learn_only: Skip all stages except self-learning (stage 9).

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

    # ── Stages 1-8: Skip all when learn_only ──────────────────────────────────
    if not learn_only:

        # ── Stage 1: Ingest (env + download) ──────────────────────────────────
        if not skip_transcribe:
            try:
                with run_phase(log, "stage 1a Transcript fetch", "transcript_fetch", run_id=rid) as ph:
                    from .transcript import fetch as fetch_transcript
                    transcript = fetch_transcript(url, output_path=transcript_path)
                    if transcript.get("segments"):
                        skip_transcribe = True
                        result.transcript_source = transcript.get("source", "api")
                        ph.set(source=result.transcript_source,
                               segments=len(transcript["segments"]))
                        provider_health.record_success("transcript")
                    else:
                        provider_health.record_failure("transcript")
                        ph.set(source="none", segments=0)
            except Exception as e:
                provider_health.record_failure("transcript")
                result.failures.append("stage1a: %s" % e)

        if sync_from_drive:
            try:
                with run_phase(log, "stage 1b Drive pull", "drive_pull", run_id=rid):
                    from sync import download_from_drive
                    download_from_drive(filenames=["video.mp4"], dest_dir=input_dir)
                    download_from_drive(filenames=["%s.json" % stem], dest_dir=transcripts_dir)
                    download_from_drive(filenames=None, dest_dir=shorts_dir)
                    skip_download = True
                    skip_transcribe = True
                provider_health.record_success("drive")
            except Exception as e:
                provider_health.record_failure("drive")
                result.failures.append("stage1b: %s" % e)

        if not skip_download:
            try:
                with run_phase(log, "stage 1c Download", "download", run_id=rid) as ph:
                    from download import download
                    video_path = str(download(url, video_path, sample_minutes=sample_minutes))
                    stem = Path(video_path).stem
                    transcript_path = str(Path(transcripts_dir) / ("%s.json" % stem))
                    highlights_path = str(Path(highlights_dir) / ("%s.yaml" % stem))
                    provider_health.record_success("download")
                    ph.set(video_path=video_path)
            except Exception as e:
                provider_health.record_failure("download")
                result.failures.append("stage1c: %s" % e)

        # ── Stage 2: Transcribe ────────────────────────────────────────────────
        if not skip_transcribe:
            try:
                with run_phase(log, "stage 2 Transcribe", "transcribe", run_id=rid):
                    from transcribe import transcribe
                    transcribe(video_path, transcript_path)
                provider_health.record_success("transcriber")
            except Exception as e:
                provider_health.record_failure("transcriber")
                result.failures.append("stage2: %s" % e)

        # ── Stage 3+4+5: Highlight Detection (chunk + parallel score + rank) ──────
        if not skip_highlight:
            try:
                with run_phase(log, "stage 3-5 Highlight + Score + Rank", "highlight", run_id=rid) as ph:
                    from highlight import detect_highlights
                    highlights = detect_highlights(transcript_path, video_path, highlights_path)
                    result.selected_clips = len(highlights)
                    from automation.scoring.feature_extractor import FeatureExtractor
                    from automation.scoring.scoring import ClipScorer
                    _lock_scorer = ClipScorer(FeatureExtractor())
                    _title = Path(video_path).stem
                    for rank, h in enumerate(highlights, 1):
                        cid = f"{stem}/{h['id']}"
                        emit_event(cid, EventType.candidate_created, {
                            "text": h.get("text", ""),
                            "start": h.get("start"),
                            "end": h.get("end"),
                        })
                        dims = h.get("dimension_scores", {})
                        lock_score = _lock_scorer.score({
                            "clip_id": h["id"],
                            "duration_s": float(h.get("end", 0) or 0) - float(h.get("start", 0) or 0),
                            "transcript": h.get("text", ""),
                            "title": _title,
                        })
                        emit_event(cid, EventType.candidate_scored, {
                            "score": h.get("score", 0.0),
                            "dimension_scores": dims,
                            "ai_score": h.get("ai_score"),
                            "lock_score": lock_score["score"],
                            "lock_features": lock_score["features"],
                        })
                        emit_event(cid, EventType.candidate_ranked, {
                            "rank": rank,
                            "total": len(highlights),
                        })
                    provider_health.record_success("llm")
                    ph.set(selected= result.selected_clips)
            except Exception as e:
                provider_health.record_failure("llm")
                emit_infra_failed(stem, str(e), stage="stage3-5")
                result.failures.append("stage3-5: %s" % e)

        # ── Stage 6: Export (final cut instructions) ──────────────────────────────
        if not skip_export:
            try:
                with run_phase(log, "stage 6 Export", "export", run_id=rid) as ph:
                    from export import export_all
                    result.exported = export_all(
                        highlights_path, video_path,
                        transcript_path=transcript_path,
                        generate_seo=True,
                    )
                    for clip_path in result.exported:
                        emit_event(f"{stem}/{clip_path.stem}", EventType.exported, {
                            "path": str(clip_path),
                        })
                    provider_health.record_success("export")
                    ph.set(exported=len(result.exported))
            except Exception as e:
                provider_health.record_failure("export")
                emit_infra_failed(stem, str(e), stage="stage6")
                result.failures.append("stage6: %s" % e)
        else:
            existing = sorted(Path(shorts_dir).rglob("*.mp4")) if Path(shorts_dir).exists() else []
            existing = [p for p in existing if "test_output" not in p.name]
            if existing:
                latest_dir = max(set(p.parent for p in existing), key=lambda d: d.name)
                existing = sorted(latest_dir.glob("*.mp4"))
            if existing:
                result.exported = existing
                log.info("[stage 6] skipped - using %d existing clips", len(existing))
            else:
                log.warning("[stage 6] skipped - no .mp4 files found in %s", shorts_dir)

        # ── Stage 6b: Enhancement (optional) ──────────────────────────────────────
        if result.exported and mode:
            try:
                with run_phase(log, "stage 6b Enhancement (%s)" % mode, "enhancement", run_id=rid) as ph:
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
                                log.warning("[%s] Color grade failed - keeping original", clip_path.stem)

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
                                log.warning("[%s] Face-mapper failed - keeping original", clip_path.stem)

                    result.exported = enhanced
                    ph.set(enhanced=len(result.exported), mode=mode)
                provider_health.record_success("enhancement")
            except Exception as e:
                provider_health.record_failure("enhancement")
                result.failures.append("stage6b: %s" % e)

        # ── Stage 7: SEO (selected clips ONLY — no token waste on rejected) ──────
        if not skip_seo and result.exported:
            try:
                with run_phase(log, "stage 7 SEO (selected only)", "seo", run_id=rid) as ph:
                    from .seo.seo import process_all_seo, retry_failed_seo
                    export_dir = str(result.exported[0].parent)
                    process_all_seo(highlights_path, export_dir)
                    retry = retry_failed_seo(export_dir)
                    result.seo_generated = len(result.exported) - retry.get("still_failed", 0)
                    ph.set(clips=len(result.exported),
                           seo_generated=result.seo_generated,
                           seo_recovered=retry.get("recovered", 0),
                           seo_still_queued=retry.get("still_failed", 0))
                provider_health.record_success("llm")
            except Exception as e:
                provider_health.record_failure("llm")
                result.failures.append("stage7: %s" % e)

        # ── Stage 7b: Thumbnails ──────────────────────────────────────────────────
        if result.exported:
            try:
                with run_phase(log, "stage 7b Thumbnails", "thumbnails", run_id=rid):
                    from thumbnail import process_all_thumbnails
                    export_dir = str(result.exported[0].parent)
                    process_all_thumbnails(export_dir)
            except Exception as e:
                result.failures.append("stage7b: %s" % e)

        # ── Stage 8: Analytics writeback ──────────────────────────────────────────
        do_sync = auto_sync and not skip_sync
        if do_sync and result.exported:
            try:
                with run_phase(log, "stage 8a Sync", "sync", run_id=rid):
                    from sync import sync_to_drive
                    export_folder = str(result.exported[0].parent)
                    sync_to_drive(folder_path=export_folder)
                provider_health.record_success("drive")
            except Exception as e:
                provider_health.record_failure("drive")
                result.failures.append("stage8a: %s" % e)

        if (auto_upload or auto_schedule) and result.exported:
            try:
                with run_phase(log, "stage 8b Upload", "upload", run_id=rid) as ph:
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
                            log.warning("No metadata for %s - skipping upload", clip_path.name)
                            continue
                        publish_at = None
                        if auto_schedule and clip_idx > 0:
                            import random
                            slot = slot_map.get(clip_path.stem)
                            if slot:
                                jitter_hours = random.uniform(0.5, 2.0)
                                from datetime import timedelta
                                slot = slot + timedelta(hours=jitter_hours)
                                publish_at = format_for_youtube(slot)
                        try:
                            upload_video(str(clip_path), str(meta_path),
                                         privacy=privacy, publish_at=publish_at)
                            result.uploaded_count += 1
                            emit_event(f"{stem}/{clip_path.stem}", EventType.published, {
                                "privacy": privacy,
                                "publish_at": publish_at,
                            })
                        except Exception as e:
                            emit_infra_failed(f"{stem}/{clip_path.stem}", str(e), stage="stage8b_upload")
                            log.error("Upload failed for %s: %s", clip_path.name, e, exc_info=True)
                            result.failures.append("stage8b %s: %s" % (clip_path.name, e))

                    if result.uploaded_count:
                        provider_health.record_success("youtube")
                    ph.set(uploaded=result.uploaded_count, skipped_no_meta=skipped_no_meta)
            except Exception as e:
                provider_health.record_failure("youtube")
                result.failures.append("stage8b: %s" % e)

    # ── Stage 9: Self-learning ────────────────────────────────────────────────
    if learn_only or auto_upload or auto_schedule:
        try:
            with run_phase(log, "stage 9 Analytics + Learning", "analytics", run_id=rid):
                from .seo.analytics import generate_daily_insights
                generate_daily_insights()
        except Exception as e:
            result.failures.append("stage9a: %s" % e)

        try:
            with run_phase(log, "stage 9b Self-Learner", "self_learner", run_id=rid):
                from self_learner import Learner
                learner = Learner()
                learner.observe("pipeline_run", {
                    "duration": result.total_seconds,
                    "exported": len(result.exported),
                    "uploaded": result.uploaded_count,
                    "selected": result.selected_clips,
                    "failures": len(result.failures),
                    "transcript_source": result.transcript_source,
                })
                log.info("[self_learner] observed pipeline_run %s", rid)
                learner.close()
        except Exception as e:
            result.failures.append("stage9b: %s" % e)

        try:
            with run_phase(log, "stage 9c Automation Learner", "automation_learner", run_id=rid):
                from automation import automation_learner, policy_updater, preference_engine, replay_engine
                policy_updater.update_from_events(decision_store.get_all_events())
                if not replay_engine.verify():
                    log.info("[automation_learner] state divergence detected, replaying...")
                    replay_engine.replay()
                prefs = preference_engine.compute_preferences()
                log.info("[automation_learner] processed %d events, prefs=%s",
                         decision_store.count(), prefs)
        except Exception as e:
            result.failures.append("stage9c: %s" % e)

        try:
            with run_phase(log, "stage 9d Provider Health", "provider_health", run_id=rid):
                for provider in ("transcript", "download", "transcriber", "llm",
                                 "export", "enhancement", "drive", "youtube"):
                    stats = provider_health.get_stats(provider)
                    if stats["total_calls"]:
                        log.info("[provider_health] %s: status=%s calls=%d ok=%d fail=%d",
                                 provider, stats["status"].value,
                                 stats["total_calls"], stats["successes"], stats["failures"])
        except Exception as e:
            result.failures.append("stage9d: %s" % e)

    result.total_seconds = time.monotonic() - start
    status = "partial" if result.failures else "ok"
    log.info("[EXIT] pipeline run_id=%s url=%s exported=%d uploaded=%d "
             "selected=%d seo=%d failures=%d elapsed=%.1fs transcript=%s",
             rid, url, len(result.exported), result.uploaded_count,
             result.selected_clips, result.seo_generated,
             len(result.failures), result.total_seconds, result.transcript_source,
             extra={"stage": "pipeline_total", "status": status, "run_id": rid,
                    "duration_ms": int(result.total_seconds * 1000),
                    "metadata": {
                        "exported": len(result.exported),
                        "uploaded": result.uploaded_count,
                        "selected_clips": result.selected_clips,
                        "seo_generated": result.seo_generated,
                        "failures": len(result.failures),
                        "transcript_source": result.transcript_source,
                    }})
    if result.failures:
        log.warning("[EXIT] %d stage failure(s) - see ERROR records above (run_id=%s)",
                    len(result.failures), rid)
    return result
