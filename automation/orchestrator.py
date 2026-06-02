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
    9. Self-learning updates prompt weights / rules

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
from automation.memory.decision_store import DecisionStore, LearnedStateStore
from automation.providers.provider_health import ProviderHealth
from automation.learner.learner import Learner as AutomationLearner
from automation.learner.policy_updater import PolicyUpdater
from automation.learner.preference_engine import PreferenceEngine
from automation.learner.replay import ReplayEngine

from utils.logger import get_logger, run_phase, new_run_id

log = get_logger("orchestrator")

_DECISION_STORE: DecisionStore = DecisionStore()
_LEARNED_STATE: LearnedStateStore = LearnedStateStore()
_PROVIDER_HEALTH: ProviderHealth = ProviderHealth()


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

    # ── Stages 1-8 — skipped when learn_only ────────────────────────
    if not learn_only:

        # ── Stage 1a: Transcript fetch ──────────────────────────
        if not skip_transcribe:
            try:
                with run_phase(log, "stage 1a Transcript fetch", "transcript_fetch", run_id=rid) as ph:
                    from automation.transcript import fetch as fetch_transcript
                    transcript = fetch_transcript(url, output_path=transcript_path)
                    if transcript.get("segments"):
                        skip_transcribe = True
                        result.transcript_source = transcript.get("source", "api")
                        ph.set(source=result.transcript_source,
                               segments=len(transcript["segments"]))
                        _PROVIDER_HEALTH.record_success("transcript")
                    else:
                        _PROVIDER_HEALTH.record_failure("transcript")
                        ph.set(source="none", segments=0)
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("transcript")
                result.failures.append(f"stage1a: {e}")

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
                result.failures.append(f"stage1c: {e}")

        # ── Stage 2: Transcribe ─────────────────────────────────
        if not skip_transcribe:
            try:
                with run_phase(log, "stage 2 Transcribe", "transcribe", run_id=rid):
                    from transcribe import transcribe
                    transcribe(video_path, transcript_path)
                _PROVIDER_HEALTH.record_success("transcriber")
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("transcriber")
                result.failures.append(f"stage2: {e}")

        # ── Stage 3-5: Highlight detection + score + rank ───────
        if not skip_highlight:
            try:
                with run_phase(log, "stage 3-5 Highlight + Score + Rank",
                               "highlight", run_id=rid) as ph:
                    from highlight import detect_highlights
                    highlights = detect_highlights(transcript_path, video_path,
                                                   highlights_path)
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
                            "dimension_scores": dims,
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
                    from automation.seo.seo import process_all_seo
                    result.exported = export_all(
                        highlights_path, video_path,
                        transcript_path=transcript_path,
                        generate_seo=False,
                    )
                    for clip_path in result.exported:
                        _emit_event(f"{stem}/{clip_path.stem}", EventType.exported, {
                            "path": str(clip_path),
                        })
                    # SEO runs immediately after export
                    if not skip_seo and result.exported:
                        export_dir = str(result.exported[0].parent)
                        process_all_seo(highlights_path, export_dir)
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
        if result.exported and mode:
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
        if result.exported and not skip_sync and has_auth:
            try:
                with run_phase(log, "stage 8b Upload", "upload",
                               run_id=rid) as ph:
                    from upload import upload_video
                    privacy = cfg.get("youtube", {}).get(
                        "privacy_status", "public"
                    )
                    interval = cfg.get("upload_schedule", {}).get(
                        "interval_hours",
                        cfg.get("youtube", {}).get("schedule_interval_hours", 1),
                    )
                    slot_map: dict[str, datetime] = {}
                    if auto_schedule:
                        from scheduler import (
                            assign_clips_to_slots,
                            format_for_youtube,
                        )
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
                        )
                        slot_map = {stem: dt for stem, dt in assignments}
                        log.info("Schedule generated for %d clips",
                                 len(assignments))

                    skipped_no_meta = 0
                    for clip_idx, clip_path in enumerate(result.exported):
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
                        if auto_schedule and clip_idx > 0:
                            import random
                            slot = slot_map.get(clip_path.stem)
                            if slot:
                                jitter = random.uniform(2, 3)
                                slot += timedelta(hours=jitter * (clip_idx - 1))
                                publish_at = format_for_youtube(slot)
                        try:
                            upload_video(
                                str(clip_path), str(meta_path),
                                privacy=privacy, publish_at=publish_at,
                            )
                            result.uploaded_count += 1
                            _emit_event(
                                f"{stem}/{clip_path.stem}",
                                EventType.published,
                                {"privacy": privacy, "publish_at": publish_at},
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
        elif result.exported and not has_auth:
            log.info("[stage 8b] Upload skipped — no auth (cookies.txt or yt_channel_token.json)")

    # ── Stage 9: Self-learning ──────────────────────────────────────────
    result.total_seconds = time.monotonic() - start
    if result.exported or learn_only:
        try:
            with run_phase(log, "stage 9a Analytics", "analytics", run_id=rid):
                from automation.seo.analytics import Analytics
                a = Analytics(_DECISION_STORE)
                summary = a.get_summary()
                log.info("[analytics] summary: %s", summary)
        except Exception as e:
            result.failures.append(f"stage9a: {e}")

        try:
            with run_phase(log, "stage 9b Self-Learner",
                           "self_learner", run_id=rid):
                from self_learner import (
                    Learner as SelfLearner,
                    SEOLearner,
                    TrendAnalyzer,
                    TrendPoint,
                    RecommendationEngine,
                )

                # Initialize all learners
                learner = SelfLearner()
                seo_learner = SEOLearner()
                trend_analyzer = TrendAnalyzer()
                rec_engine = RecommendationEngine(
                    seo_learner=seo_learner,
                    trend_analyzer=trend_analyzer,
                )

                # Record pipeline observation
                learner.observe("pipeline_run", {
                    "duration": result.total_seconds,
                    "exported": len(result.exported),
                    "uploaded": result.uploaded_count,
                    "selected": result.selected_clips,
                    "failures": len(result.failures),
                    "transcript_source": result.transcript_source,
                })
                log.info("[self_learner] observed pipeline_run %s", rid)

                # Record trends
                trend_analyzer.record_metric(TrendPoint(
                    metric="duration",
                    value=result.total_seconds,
                ))
                trend_analyzer.record_metric(TrendPoint(
                    metric="exported_count",
                    value=float(len(result.exported)),
                ))
                trend_analyzer.record_metric(TrendPoint(
                    metric="failures_count",
                    value=float(len(result.failures)),
                ))
                trend_analyzer.record_metric(TrendPoint(
                    metric="selected_clips",
                    value=float(result.selected_clips),
                ))

                # Analyze trends and log insights
                for trend in trend_analyzer.get_all_trends():
                    if trend.direction != "stable":
                        log.info("[self_learner] trend: %s is %s (slope=%.3f, confidence=%.2f)",
                                 trend.metric, trend.direction, trend.slope, trend.confidence)

                # Detect anomalies
                for metric in ("duration", "failures_count"):
                    anomalies = trend_analyzer.detect_anomalies(metric)
                    if anomalies:
                        log.warning("[self_learner] anomaly in %s: %d unusual values detected",
                                    metric, len(anomalies))

                # Generate and log recommendations
                recommendations = rec_engine.generate_recommendations()
                for rec in recommendations[:3]:  # Log top 3
                    log.info("[self_learner] recommendation [%s/%s]: %s -> %s",
                             rec.category, rec.priority, rec.title, rec.action)

                # Close resources
                rec_engine.close()
                trend_analyzer.close()
                seo_learner.close()
                learner.close()
        except Exception as e:
            result.failures.append(f"stage9b: {e}")

        try:
            with run_phase(log, "stage 9c Automation Learner",
                           "automation_learner", run_id=rid):
                _automation_learner = AutomationLearner(_DECISION_STORE, _LEARNED_STATE)
                updater = PolicyUpdater(_automation_learner)
                updater.update_from_events(_DECISION_STORE.get_all_events())
                replay_engine = ReplayEngine(
                    _DECISION_STORE, _LEARNED_STATE, _automation_learner,
                )
                if not replay_engine.verify():
                    log.info("[automation_learner] state divergence detected, replaying...")
                    replay_engine.replay()
                prefs = PreferenceEngine(
                    _automation_learner, _DECISION_STORE,
                ).compute_preferences()
                log.info("[automation_learner] processed %d events, prefs=%s",
                         _DECISION_STORE.count(), prefs)
        except Exception as e:
            result.failures.append(f"stage9c: {e}")

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
