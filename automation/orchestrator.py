"""orchestrator.py — 9-stage pipeline runner for yt-clips automation.

Stages:
    1. Ingest (env / Drive pull / download)
    2. Transcribe (ASR + timestamps + speaker chunks)
    3. Chunk into candidates
    4. Parallel scoring (6 dimensions)
    5. Rank + filter (top 10 max)
    6. Generate final cut instructions
    7. SEO (selected clips only)
    8. Sync / Upload
    9. Runtime telemetry + canonical Shorts Intelligence persistence

Usage::

    from automation.orchestrator import run

    result = run(url="https://youtu.be/...",
                 auto_sync=True, auto_upload=True)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from automation.memory.event_models import EventType, ClipEvent
from automation.memory.decision_store import DecisionStore
from automation.providers.provider_health import ProviderHealth

from utils.logger import get_logger, run_phase, new_run_id

log = get_logger("orchestrator")

_DECISION_STORE: DecisionStore = DecisionStore()
_PROVIDER_HEALTH: ProviderHealth = ProviderHealth()


def _should_upload(*, auto_upload: bool, has_auth: bool) -> bool:
    """Require both explicit user intent and usable YouTube credentials."""
    return bool(auto_upload and has_auth)


def _abort_required_stage(
    result: PipelineResult,
    started_at: float,
    failure: str,
) -> PipelineResult:
    """Stop before stale artifacts can leak into dependent stages."""
    result.failures.append(failure)
    result.total_seconds = time.monotonic() - started_at
    log.error("[ABORT] required stage failed: %s", failure)
    return result


def _emit_event(
    clip_id: str,
    event_type: EventType,
    payload: dict[str, Any] | None = None,
) -> None:
    event = ClipEvent(
        event_id=f"evt-{uuid.uuid4().hex[:8]}",
        clip_id=clip_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        event_type=event_type,
        payload_json=json.dumps(payload or {}),
    )
    _DECISION_STORE.append_event(event)


def _emit_infra_failed(clip_id: str, error: str, stage: str = "") -> None:
    _emit_event(clip_id, EventType.infra_failed, {
        "error": error, "stage": stage,
    })


@dataclass
class PipelineResult:
    exported: list[Path] = field(default_factory=list)
    uploaded_count: int = 0
    failures: list[str] = field(default_factory=list)
    total_seconds: float = 0.0
    transcript_source: str = "none"
    run_id: str = ""
    selected_clips: int = 0
    rejected_clips: int = 0
    seo_generated: int = 0


# ── Short-hands to keep stage code readable ─────────────────────────────
_CONFIG = None


def _cfg() -> dict:
    global _CONFIG
    if _CONFIG is None:
        from automation.config import load
        _CONFIG = load()
    return _CONFIG


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def _url_match_key(url: str | None) -> str | None:
    """Derive a dedup-history namespace key from a run URL, or None.

    Extracts the YouTube video ID (11 chars) from watch/shorts/live/embed
    URLs; falls back to the bare host+path when no ID is parseable so history
    stays isolated across different sources. None only when the URL is empty
    or missing (callers then rely on video_metadata.json / output stem).
    """
    if not url or not str(url).strip():
        return None
    url = str(url).strip()
    try:
        from automation.clip_selection.pipeline import _extract_youtube_id
        video_id = _extract_youtube_id(url)
        if video_id:
            return video_id
    except Exception:
        pass
    host_path = url.split("?")[0].rstrip("/").replace("https://", "").replace("http://", "")
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in host_path)
    slug = "-".join(part for part in slug.split("-") if part)
    return (slug[:64] or None).lower()


def run(
    url: str,
    skip_download: bool = False,
    skip_transcribe: bool = False,
    skip_highlight: bool = False,
    skip_export: bool = False,
    skip_sync: bool = False,
    skip_seo: bool = False,
    skip_enhancement: bool = False,
    auto_sync: bool = False,
    auto_upload: bool = False,
    auto_schedule: bool = False,
    sample_minutes: int | None = None,
    sync_from_drive: bool = False,
    mode: str | None = None,
    learn_only: bool = False,
) -> PipelineResult:
    """Execute the 9-stage yt-clips pipeline.

    Each phase is a lazy import (zero cost if skipped).
    """
    start = time.monotonic()
    result = PipelineResult()
    result.run_id = new_run_id()
    rid = result.run_id
    log.info("[START] pipeline run_id=%s url=%s", rid, url)

    cfg = _cfg()
    if learn_only:
        try:
            from automation.seo.analytics import sync_clip_performance_from_youtube

            added = sync_clip_performance_from_youtube(config_path="config.yaml")
            log.info("[shorts_intelligence] canonical sync added=%d", added)
        except Exception as exc:
            result.failures.append(f"shorts_intelligence: {exc}")
        result.total_seconds = time.monotonic() - start
        return result

    paths = cfg.get("paths", {})
    dl_cfg = cfg.get("download", {})

    input_dir = paths.get("input", "input")
    transcripts_dir = paths.get("transcripts", "transcripts")
    highlights_dir = paths.get("highlights", "highlights")
    shorts_dir = paths.get("shorts", "shorts")
    output_filename = dl_cfg.get("output_filename", "video.mp4")

    video_path = str(Path(input_dir) / output_filename)
    stem = Path(video_path).stem
    transcript_path = str(Path(transcripts_dir) / f"{stem}.json")
    highlights_path = str(Path(highlights_dir) / f"{stem}.yaml")
    highlights: list[dict] = []

    # Dedup-history namespace: derive a stable per-match key from the run URL
    # so skip-download / drive-sync runs (which never write video_metadata.json)
    # still get isolated history instead of collapsing onto the constant stem.
    match_key = _url_match_key(url)

    # ── Stages 1-8 ──────────────────────────────────────────────────
    if not learn_only:

        # ── Stage 1b: Drive pull ────────────────────────────────
        if sync_from_drive:
            try:
                with run_phase(log, "stage 1b Drive pull", "drive_pull", run_id=rid):
                    from sync import download_from_drive
                    download_from_drive(filenames=["video.mp4"], dest_dir=input_dir)
                    download_from_drive(filenames=[f"{stem}.json"], dest_dir=transcripts_dir)
                    download_from_drive(filenames=None, dest_dir=shorts_dir)
                    skip_download = True
                    skip_transcribe = True
                _PROVIDER_HEALTH.record_success("drive")
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("drive")
                result.failures.append(f"stage1b: {e}")

        # ── Stage 1c: Download ──────────────────────────────────
        if not skip_download:
            try:
                with run_phase(log, "stage 1c Download", "download", run_id=rid) as ph:
                    from download import download
                    video_path = str(download(url, video_path, sample_minutes=sample_minutes))
                    stem = Path(video_path).stem
                    transcript_path = str(Path(transcripts_dir) / f"{stem}.json")
                    highlights_path = str(Path(highlights_dir) / f"{stem}.yaml")
                    _PROVIDER_HEALTH.record_success("download")
                    ph.set(video_path=video_path)
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("download")
                return _abort_required_stage(result, start, f"stage1c: {e}")

        # ── Stage 2: Transcribe ─────────────────────────────────
        if not skip_transcribe:
            try:
                with run_phase(log, "stage 2 Transcribe", "transcribe", run_id=rid):
                    from automation.transcript import fetch
                    import json
                    t_data = fetch(url, output_path=transcript_path, video_path=video_path)
                    if t_data and t_data.get("segments"):
                        with open(transcript_path, "w", encoding="utf-8") as f:
                            json.dump(t_data, f, indent=2, ensure_ascii=False)
                    else:
                        raise Exception("Failed to fetch or generate transcript")
                _PROVIDER_HEALTH.record_success("transcriber")
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("transcriber")
                return _abort_required_stage(result, start, f"stage2: {e}")

        # ── Stage 3-5: Highlight detection + agent scoring + rank ──
        if not skip_highlight:
            try:
                with run_phase(log, "stage 3-5 Highlight + Score + Rank",
                               "highlight", run_id=rid) as ph:
                    from automation.clip_selection.pipeline import detect_highlights
                    highlights = detect_highlights(transcript_path, video_path,
                                                   highlights_path,
                                                   match_key=match_key)
                    result.selected_clips = len(highlights)
                    from automation.scoring.feature_extractor import FeatureExtractor
                    from automation.scoring.scoring import ClipScorer
                    _lock_scorer = ClipScorer(FeatureExtractor())
                    _title = Path(video_path).stem
                    for rank_idx, h in enumerate(highlights, 1):
                        cid = f"{stem}/{h['id']}"
                        _emit_event(cid, EventType.candidate_created, {
                            "text": h.get("text", ""),
                            "start": h.get("start"),
                            "end": h.get("end"),
                        })
                        dims = h.get("dimension_scores", {})
                        agent_scores = h.get("agent_scores", {})
                        lock_score = _lock_scorer.score({
                            "clip_id": h["id"],
                            "duration_s": (
                                float(h.get("end", 0) or 0)
                                - float(h.get("start", 0) or 0)
                            ),
                            "transcript": h.get("text", ""),
                            "title": _title,
                        })
                        _emit_event(cid, EventType.candidate_scored, {
                            "score": h.get("score", 0.0),
                            "final_score": h.get("final_score", h.get("score", 0.0)),
                            "dimension_scores": dims,
                            "agent_scores": agent_scores,
                            "hook_score": h.get("hook_score"),
                            "ai_score": h.get("ai_score"),
                            "lock_score": lock_score["score"],
                            "lock_features": lock_score["features"],
                        })
                        _emit_event(cid, EventType.candidate_ranked, {
                            "rank": rank_idx,
                            "total": len(highlights),
                        })

                    _PROVIDER_HEALTH.record_success("llm")
                    ph.set(selected=result.selected_clips)
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("llm")
                _emit_infra_failed(stem, str(e), stage="stage3-5")
                result.failures.append(f"stage3-5: {e}")

        # ── Stage 6: Export + SEO (merged — SEO runs immediately after export) ──
        if not skip_export:
            try:
                with run_phase(log, "stage 6 Export + SEO", "export", run_id=rid) as ph:
                    from export import export_all
                    result.exported = export_all(
                        highlights_path, video_path,
                        transcript_path=transcript_path,
                        generate_seo=not skip_seo,
                    )
                    for clip_path in result.exported:
                        _emit_event(f"{stem}/{clip_path.stem}", EventType.exported, {
                            "path": str(clip_path),
                        })
                    if result.exported:
                        try:
                            from shorts_intelligence.bridge import record_exported

                            shadow_count = record_exported(cfg, highlights, result.exported)
                            if shadow_count:
                                log.info(
                                    "[shorts_intelligence] shadow-captured %d exports",
                                    shadow_count,
                                )
                        except Exception as e:
                            log.warning(
                                "[shorts_intelligence] export shadow capture failed: %s",
                                e,
                            )
                    # Phase 3 — Hook audit: analyze first 3s of each exported clip
                    if result.exported:
                        try:
                            with run_phase(log, "stage 6a Hook audit",
                                           "hook_audit", run_id=rid) as ph:
                                from automation.clip_selection.hook_auditor import (
                                    HookAuditor,
                                )
                                _auditor = HookAuditor()
                                highlights_by_id = {
                                    h["id"]: h for h in highlights
                                }
                                hook_audited = 0
                                for clip_path in result.exported:
                                    clip_id = clip_path.stem
                                    source_start = highlights_by_id.get(
                                        clip_id, {}
                                    ).get("start", 0.0)
                                    transcript_text = highlights_by_id.get(
                                        clip_id, {}
                                    ).get("text", "")
                                    duration = (
                                        highlights_by_id.get(clip_id, {}).get(
                                            "end", 0.0
                                        )
                                        - source_start
                                    )
                                    if duration <= 0:
                                        duration = 30.0
                                    if not transcript_text:
                                        try:
                                            with open(
                                                clip_path.with_name(
                                                    f"{clip_path.stem}_metadata.json"
                                                )
                                            ) as f:
                                                meta = json.load(f)
                                            transcript_text = meta.get(
                                                "description", ""
                                            )
                                        except Exception:
                                            transcript_text = ""
                                    hook_result = _auditor.analyze_clip_hook(
                                        clip_path=str(clip_path),
                                        transcript_text=transcript_text,
                                        start_sec=source_start,
                                        clip_duration=duration,
                                    )
                                    _emit_event(
                                        f"{stem}/{clip_id}",
                                        EventType.candidate_scored,
                                        {
                                            "hook_score": hook_result["hook_score"],
                                            "hook_type": hook_result["hook_type"],
                                            "swipe_risk": hook_result["swipe_risk"],
                                        },
                                    )
                                    hook_audited += 1
                                ph.set(audited=hook_audited)
                                log.info(
                                    "[hook_audit] %d clips analyzed",
                                    hook_audited,
                                )
                        except Exception as e:
                            log.warning("[hook_audit] failed: %s", e)
                    # SEO is now handled per-clip inside export_all()
                    if not skip_seo and result.exported:
                        result.seo_generated = len(result.exported)
                    _PROVIDER_HEALTH.record_success("export")
                    ph.set(exported=len(result.exported),
                           seo_generated=result.seo_generated)
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("export")
                _emit_infra_failed(stem, str(e), stage="stage6")
                result.failures.append(f"stage6: {e}")
        else:
            existing = sorted(Path(shorts_dir).rglob("*.mp4")) if Path(shorts_dir).exists() else []
            existing = [p for p in existing if "test_output" not in p.name]
            if existing:
                latest_dir = max(set(p.parent for p in existing), key=lambda d: d.name)
                existing = sorted(latest_dir.glob("*.mp4"))
            if existing:
                result.exported = existing
                log.info("[stage 6] skipped — using %d existing clips", len(existing))
            else:
                log.warning("[stage 6] skipped — no .mp4 files found in %s", shorts_dir)

        # ── Stage 6b: Enhancement (optional) ────────────────────
        if result.exported and mode and not skip_enhancement:
            try:
                with run_phase(log, f"stage 6b Enhancement ({mode})",
                               "enhancement", run_id=rid) as ph:
                    import shutil
                    ref_path = cfg.get("enhancement", {}).get(
                        "reference", "expectation.png"
                    )
                    enhanced: list[Path] = []

                    if mode == "ref_grade":
                        from ref_grade import grade_video
                        for clip_path in result.exported:
                            graded = str(
                                Path(paths.get("temp", "temp"))
                                / f"{clip_path.stem}_graded.mp4"
                            )
                            r = grade_video(str(clip_path), ref_path, graded)
                            if r == graded and Path(graded).exists():
                                shutil.move(graded, str(clip_path))
                                enhanced.append(clip_path)
                                log.info("[%s] Color grade applied", clip_path.stem)
                            else:
                                enhanced.append(clip_path)
                                log.warning(
                                    "[%s] Color grade failed — keeping original",
                                    clip_path.stem,
                                )

                    elif mode == "face_mapper":
                        from face_mapper import enhance_video
                        for clip_path in result.exported:
                            enhanced_path = str(
                                Path(paths.get("temp", "temp"))
                                / f"{clip_path.stem}_enhanced.mp4"
                            )
                            r = enhance_video(
                                str(clip_path), ref_path, enhanced_path,
                                use_region_grading=True,
                            )
                            if r == enhanced_path and Path(enhanced_path).exists():
                                shutil.move(enhanced_path, str(clip_path))
                                enhanced.append(clip_path)
                                log.info("[%s] Face-mapper applied", clip_path.stem)
                            else:
                                enhanced.append(clip_path)
                                log.warning(
                                    "[%s] Face-mapper failed — keeping original",
                                    clip_path.stem,
                                )

                    result.exported = enhanced
                    ph.set(enhanced=len(result.exported), mode=mode)
                _PROVIDER_HEALTH.record_success("enhancement")
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("enhancement")
                result.failures.append(f"stage6b: {e}")

        # ── Stage 7b: Thumbnails ────────────────────────────────
        if result.exported:
            try:
                with run_phase(log, "stage 7b Thumbnails", "thumbnails", run_id=rid):
                    from thumbnail import process_all_thumbnails
                    export_dir = str(result.exported[0].parent)
                    process_all_thumbnails(export_dir)
            except Exception as e:
                result.failures.append(f"stage7b: {e}")

        # ── Stage 8a: Sync ──────────────────────────────────────
        do_sync = auto_sync and not skip_sync
        if do_sync and result.exported:
            try:
                with run_phase(log, "stage 8a Sync", "sync", run_id=rid):
                    from sync import sync_to_drive
                    export_folder = str(result.exported[0].parent)
                    sync_to_drive(folder_path=export_folder)
                _PROVIDER_HEALTH.record_success("drive")
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("drive")
                result.failures.append(f"stage8a: {e}")

        # ── Stage 8b: Upload (auto when auth exists) ─────────────
        has_auth = Path("cookies.txt").exists() or Path("yt_channel_token.json").exists()
        if result.exported and _should_upload(
            auto_upload=auto_upload,
            has_auth=has_auth,
        ):
            try:
                with run_phase(log, "stage 8b Upload", "upload",
                               run_id=rid) as ph:
                    from upload import upload_video
                    privacy = cfg.get("youtube", {}).get(
                        "privacy_status", "scheduled"
                    )
                    
                    if privacy == "scheduled" and not auto_schedule:
                        log.info("privacy_status is 'scheduled', auto-enabling scheduling")
                        auto_schedule = True

                    interval = cfg.get("upload_schedule", {}).get(
                        "interval_hours",
                        cfg.get("youtube", {}).get("schedule_interval_hours", 1),
                    )
                    slot_map: dict[str, datetime] = {}
                    if auto_schedule:
                        from scheduler import assign_clips_to_slots, format_for_youtube
                        clip_scores: dict[str, float] = {}
                        for clip_path in result.exported:
                            meta_path = clip_path.with_name(
                                f"{clip_path.stem}_metadata.json"
                            )
                            if meta_path.exists():
                                try:
                                    with open(meta_path) as f:
                                        meta = json.load(f)
                                    score = (
                                        meta.get("quality_score")
                                        or meta.get("seo_score")
                                        or 0.0
                                    )
                                    clip_scores[clip_path.stem] = float(score)
                                except (ValueError, OSError):
                                    pass
                        assignments = assign_clips_to_slots(
                            clips=[p.stem for p in result.exported],
                            interval_hours=interval,
                            clip_scores=clip_scores or None,
                            schedule_config=cfg.get("upload_schedule", {}),
                        )
                        slot_map = {stem: dt for stem, dt in assignments}

                        # Dead-day gating: if any slot lands on a dead day,
                        # shift it to the next suitable upload day
                        from automation.scheduling import is_dead_day, next_upload_day
                        shifted = 0
                        for stem, dt in list(slot_map.items()):
                            day_name = dt.strftime("%A").lower()
                            if is_dead_day(day_name):
                                new_dt = next_upload_day(dt)
                                slot_map[stem] = new_dt
                                shifted += 1
                                log.warning(
                                    "Dead-day gate: %s shifted from %s (%s) → %s (%s)",
                                    stem, day_name, dt.isoformat(),
                                    new_dt.strftime("%A").lower(), new_dt.isoformat(),
                                )
                        if shifted:
                            log.info("Dead-day gate shifted %d/%d clips",
                                     shifted, len(slot_map))

                        log.info("Schedule generated for %d clips",
                                 len(assignments))

                    skipped_no_meta = 0
                    for clip_path in result.exported:
                        meta_path = clip_path.with_name(
                            f"{clip_path.stem}_metadata.json"
                        )
                        if not meta_path.exists():
                            skipped_no_meta += 1
                            log.warning(
                                "No metadata for %s — skipping upload",
                                clip_path.name,
                            )
                            continue
                        publish_at: str | None = None
                        if auto_schedule:
                            slot = slot_map.get(clip_path.stem)
                            if slot:
                                publish_at = format_for_youtube(slot)
                        try:
                            video_id = upload_video(
                                str(clip_path), str(meta_path),
                                privacy=privacy, publish_at=publish_at,
                            )
                            result.uploaded_count += 1
                            _emit_event(
                                f"{stem}/{clip_path.stem}",
                                EventType.published,
                                {"privacy": privacy, "publish_at": publish_at,
                                 "youtube_video_id": video_id},
                            )
                            if video_id:
                                try:
                                    from shorts_intelligence.bridge import record_upload

                                    record_upload(
                                        cfg,
                                        f"{clip_path.parent.name}/{clip_path.stem}",
                                        video_id,
                                    )
                                except Exception as e:
                                    log.warning(
                                        "[shorts_intelligence] upload link failed: %s",
                                        e,
                                    )
                        except Exception as e:
                            _emit_infra_failed(
                                f"{stem}/{clip_path.stem}", str(e),
                                stage="stage8b_upload",
                            )
                            log.error("Upload failed for %s: %s",
                                      clip_path.name, e, exc_info=True)
                            result.failures.append(
                                f"stage8b {clip_path.name}: {e}"
                            )

                    if result.uploaded_count:
                        _PROVIDER_HEALTH.record_success("youtube")
                    ph.set(uploaded=result.uploaded_count,
                           skipped_no_meta=skipped_no_meta)
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("youtube")
                result.failures.append(f"stage8b: {e}")
        elif result.exported:
            reason = (
                "--upload was not requested"
                if not auto_upload
                else "no auth (cookies.txt or yt_channel_token.json)"
            )
            log.info("[stage 8b] Upload skipped — %s", reason)

    # Stage 9: telemetry and canonical persistence
    result.total_seconds = time.monotonic() - start
    if result.exported:
        try:
            with run_phase(log, "stage 9a Analytics", "analytics", run_id=rid):
                from automation.seo.analytics import Analytics
                a = Analytics(_DECISION_STORE)
                summary = a.get_summary()
                log.info("[analytics] summary: %s", summary)
        except Exception as e:
            result.failures.append(f"stage9a: {e}")

        # Runtime provider health is observational only.
        try:
            with run_phase(log, "stage 9d Provider Health",
                           "provider_health", run_id=rid):
                for provider in ("transcript", "download", "transcriber", "llm",
                                 "export", "enhancement", "drive", "youtube"):
                    stats = _PROVIDER_HEALTH.get_stats(provider)
                    if stats["total_calls"]:
                        log.info(
                            "[provider_health] %s: status=%s calls=%d ok=%d fail=%d",
                            provider, stats["status"].value,
                            stats["total_calls"], stats["successes"],
                            stats["failures"],
                        )
        except Exception as e:
            result.failures.append(f"stage9d: {e}")

        try:
            from shorts_intelligence.bridge import shadow_status

            shadow = shadow_status(cfg)
            log.info("[shorts_intelligence] %s", shadow)
        except Exception as e:
            log.warning("[shorts_intelligence] shadow status failed: %s", e)

        # Canonical DB persistence.
        try:
            from sync import sync_db_to_drive
            sync_db_to_drive()
        except Exception as e:
            log.warning("[db_sync] Failed: %s", e)

    status = "partial" if result.failures else "ok"
    log.info(
        "[EXIT] pipeline run_id=%s url=%s exported=%d uploaded=%d "
        "selected=%d seo=%d failures=%d elapsed=%.1fs transcript=%s",
        rid, url, len(result.exported), result.uploaded_count,
        result.selected_clips, result.seo_generated,
        len(result.failures), result.total_seconds, result.transcript_source,
    )
    if result.failures:
        log.warning(
            "[EXIT] %d stage failure(s) — see ERROR records above (run_id=%s)",
            len(result.failures), rid,
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator class — lightweight wrapper for programmatic use
# ═══════════════════════════════════════════════════════════════════════════

_STAGES = [
    "download", "transcribe", "score", "rank",
    "export", "seo", "upload", "cleanup",
]

_STAGE_EVENT_MAP: dict[str, str] = {
    "score": "candidate_scored",
    "rank": "candidate_ranked",
    "export": "exported",
    "upload": "published",
}


class Orchestrator:
    def __init__(
        self,
        decision_store: DecisionStore | None = None,
        clip_scorer: Any = None,
        clip_ranker: Any = None,
    ) -> None:
        self._decision_store: DecisionStore = (
            decision_store if decision_store is not None else DecisionStore()
        )
        self._clip_scorer = clip_scorer
        self._clip_ranker = clip_ranker
        self._pipeline_running: bool = False
        self._last_run: str | None = None
        self._events_emitted: int = 0

    def run_pipeline(
        self,
        url: str,
        clip_data: dict | None = None,
        stages: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        self._pipeline_running = True
        effective_stages = stages if stages is not None else {}
        clip_id = f"clip-{uuid.uuid4().hex[:8]}"
        data: dict[str, Any] = {"clip_id": clip_id, "url": url}
        if clip_data is not None:
            data.update(clip_data)
        self.emit_event(clip_id, "candidate_created", {"url": url})
        stages_completed: list[str] = []
        errors: list[dict[str, str]] = []
        for stage_name in _STAGES:
            if effective_stages.get(stage_name, True):
                try:
                    stage_result = self._run_stage(stage_name, data)
                    stages_completed.append(stage_name)
                    data.update(stage_result)
                    event_type = _STAGE_EVENT_MAP.get(stage_name)
                    if event_type is not None:
                        self.emit_event(clip_id, event_type, {"stage": stage_name})
                except Exception as exc:
                    errors.append({"stage": stage_name, "error": str(exc)})
        self._pipeline_running = False
        self._last_run = datetime.now(timezone.utc).isoformat()
        return {
            "url": url,
            "stages_completed": stages_completed,
            "events_emitted": self._events_emitted,
            "errors": errors,
        }

    def _run_stage(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        if name == "score" and self._clip_scorer is not None:
            return self._clip_scorer.score(data)
        if name == "rank" and self._clip_ranker is not None:
            clips = data.get("clips", [data])
            ranked = self._clip_ranker.rank(clips)
            return {
                "ranked": ranked,
                "rank": ranked[0].get("rank") if ranked else None,
            }
        return {**data, name: True}

    def emit_event(
        self,
        clip_id: str,
        event_type_str: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            raise ValueError(
                f"Invalid event type: '{event_type_str}'. "
                f"Valid types: {[e.value for e in EventType]}"
            )
        event = ClipEvent(
            event_id=f"evt-{uuid.uuid4().hex[:8]}",
            clip_id=clip_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            payload_json=json.dumps(payload or {}),
        )
        self._decision_store.append_event(event)
        self._events_emitted += 1

    def get_pipeline_status(self) -> dict[str, Any]:
        return {
            "pipeline_running": self._pipeline_running,
            "last_run": self._last_run,
            "total_events": self._events_emitted,
        }
