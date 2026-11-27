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
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any

from automation.memory.event_models import EventType, ClipEvent
from automation.memory.decision_store import DecisionStore
from automation.providers.provider_health import ProviderHealth

from utils.logger import get_logger, run_phase, new_run_id, log_skipped
from automation.artifacts import read_metadata

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
    selection_status: str = "NOT_RUN"
    candidate_count: int = 0
    review_candidates: int = 0
    seo_generated: int = 0
    thumbnails_generated: int = 0
    thumbnails_preserved: int = 0
    release_ready: int = 0
    release_held: int = 0
    stage_results: list[dict] = field(default_factory=list)


def _result_status(result: PipelineResult, *, upload_requested: bool = False) -> str:
    if result.selection_status == "REVIEW_REQUIRED":
        return "review"
    if result.failures:
        return "partial" if result.exported else "failed"
    if result.release_held:
        return "review"
    if upload_requested and result.exported and result.uploaded_count < len(result.exported):
        return "partial"
    return "ok" if result.exported else "empty"


def _report_run(function):
    """Finalize every exit path, including early aborts, after phases close."""
    call_signature = signature(function)

    @wraps(function)
    def reported(*args, **kwargs):
        started = time.monotonic()
        options = call_signature.bind(*args, **kwargs)
        options.apply_defaults()
        events = []

        class StageCollector(logging.Handler):
            def emit(self, record):
                if (getattr(record, "stage", None) not in (None, "pipeline")
                        and getattr(record, "status", None) not in (None, "start")):
                    events.append({key: getattr(record, key, None) for key in
                                   ("run_id", "stage", "phase", "status", "duration_ms", "metadata")})

        collector = StageCollector()
        log.addHandler(collector)
        try:
            result = function(*args, **kwargs)
        except (Exception, KeyboardInterrupt) as exc:
            log.error("[EXIT] pipeline interrupted: %s", exc, extra={
                "stage": "pipeline", "status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                "duration_ms": int((time.monotonic() - started) * 1000),
                "metadata": {"failures": [str(exc) or type(exc).__name__]},
            })
            raise
        finally:
            log.removeHandler(collector)
            collector.close()
        result.total_seconds = time.monotonic() - started
        result.stage_results = [event for event in events if event["run_id"] == result.run_id]
        config = _cfg()
        editorial = config.get("clip_selection", {}).get("editorial", {})
        if (not options.arguments["learn_only"]
                and not options.arguments["skip_highlight"] and not options.arguments["skip_export"]
                and not result.failures):
            target = min(int(editorial.get("target_min", 2)),
                         int(config.get("clip_selection", {}).get("max_selected", 3)))
            count = len(result.exported)
            from automation.cricket_gameplay import GAMEPLAY_COMPOSITIONS
            if (options.arguments.get("composition") or config.get("export", {}).get("composition")) in GAMEPLAY_COMPOSITIONS:
                from automation.cricket_gameplay import count_distinct_exports
                count = count_distinct_exports(result.exported)
            # A shortfall is a quality signal, not a failed run: one good clip still ships.
            # Zero publishable candidates is an honest empty run; drafts that all
            # failed to export is a real failure.
            if count == 0 and result.selected_clips:
                result.failures.append(f"editorial_target: 0 of {result.selected_clips} drafts exported; target {target}")
            elif count < target:
                log.warning("editorial_target: %d/%d exports/distinct moments", count, target)
        status = "ok" if options.arguments["learn_only"] and not result.failures else _result_status(
            result, upload_requested=options.arguments["auto_upload"])
        metadata = {
            "exported": len(result.exported), "uploaded": result.uploaded_count,
            "selected": result.selected_clips, "candidates": result.candidate_count,
            "rejected": result.rejected_clips, "selection_status": result.selection_status,
            "seo_generated": result.seo_generated, "thumbnails_generated": result.thumbnails_generated,
            "thumbnails_preserved": result.thumbnails_preserved,
            "transcript_source": result.transcript_source, "failures": result.failures,
            "output_dirs": sorted({str(clip.parent) for clip in result.exported}),
            "log_file": config.get("logging", {}).get("log_file", "logs/pipeline.log"),
            "review_required": config.get("release_review", {}).get("required", False),
            "upload_requested": options.arguments["auto_upload"],
            "stages": result.stage_results,
        }
        log.log(logging.INFO if status == "ok" else logging.WARNING,
                "[EXIT] pipeline run_id=%s status=%s exported=%d uploaded=%d failures=%d elapsed=%.1fs",
                result.run_id, status, len(result.exported), result.uploaded_count,
                len(result.failures), result.total_seconds, extra={
                    "stage": "pipeline", "phase": "Pipeline", "status": status,
                    "run_id": result.run_id, "duration_ms": int(result.total_seconds * 1000),
                    "metadata": metadata,
                })
        return result
    return reported


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


def _existing_exports(shorts_dir: str, chosen_dir: str | None = None) -> list[Path]:
    """Resume one batch, never recursively discover review renders or guess a batch."""
    root = Path(shorts_dir).resolve()
    if chosen_dir:
        chosen = Path(chosen_dir).resolve()
        if not chosen.is_relative_to(root) or chosen not in (root, root / chosen.name):
            raise ValueError("--export-dir must be the Shorts root or one direct batch directory")
        if not chosen.is_dir():
            raise ValueError("--export-dir does not exist")
        directories = [chosen]
    elif root.exists():
        directories = [root, *sorted(p for p in root.iterdir()
                                     if p.is_dir() and not p.is_symlink()
                                     and not getattr(p, "is_junction", lambda: False)())]
    else:
        return []
    batches = []
    for directory in directories:
        finalization = directory / "finalization.json"
        if finalization.exists():
            try:
                package = json.loads(finalization.read_text(encoding="utf-8"))
                if not isinstance(package, dict):
                    raise ValueError("expected an object")
            except (OSError, ValueError) as exc:
                raise ValueError(f"Final-layout manifest invalid; preserve and review {finalization}: {exc}") from exc
            if package.get("publication_authorized") is not True:
                if chosen_dir:
                    raise ValueError("Final-layout package is not a production/release-approved batch; use cleared exports")
                continue
        clips = sorted(p for p in directory.glob("*.mp4")
                       if p.is_file() and not p.is_symlink() and "test_output" not in p.name)
        if clips:
            batches.append(clips)
    if len(batches) > 1:
        raise ValueError("Multiple export batches found; choose --export-dir explicitly")
    return batches[0] if batches else []


def _validate_enhancement_inputs(clips):
    """Inspect actual media evidence, never the profile requested for a new run."""
    from automation.cricket_gameplay import GAMEPLAY_COMPOSITIONS
    for clip in clips:
        clip = Path(clip)
        if clip.with_name(f"{clip.stem}_gameplay.json").exists():
            raise ValueError(f"{clip.name}: gameplay export cannot receive post-render enhancement")
        edit_path = clip.with_name(f"{clip.stem}_edit.json")
        if not edit_path.exists():
            raise ValueError(f"{clip.name}: enhancement held: missing edit evidence; cannot identify the rendered profile")
        try:
            edit = json.loads(edit_path.read_text(encoding="utf-8"))
            if not isinstance(edit, dict):
                raise ValueError("Invalid edit evidence")
            export_config = edit.get("export_config", {})
            if not isinstance(export_config, dict):
                raise ValueError("Invalid edit evidence")
            # The saved global config can still say stadium when the original
            # export explicitly selected portrait. Prefer the actual edit profile.
            composition = edit.get("composition") or export_config.get("composition")
            if (composition in GAMEPLAY_COMPOSITIONS
                    or edit.get("gameplay_layout") is not None):
                raise ValueError("gameplay export cannot receive post-render enhancement")
            if composition != "portrait-caption-v1":
                raise ValueError("Unknown rendered composition; review or re-export before enhancement")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"{clip.name}: enhancement held: {exc}") from exc


@_report_run
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
    existing_exports_dir: str | None = None,
    composition: str | None = None,
    gameplay_manifest: str | None = None,
) -> PipelineResult:
    """Execute the 9-stage yt-clips pipeline.

    Each phase is a lazy import (zero cost if skipped).
    """
    start = time.monotonic()
    result = PipelineResult()
    result.run_id = new_run_id()
    rid = result.run_id
    cfg = _cfg()
    log.info("[START] pipeline run_id=%s url=%s", rid, url, extra={
        "stage": "pipeline", "phase": "Pipeline", "status": "start", "run_id": rid,
        "duration_ms": 0, "metadata": {"source": url, "mode": "LEARN" if learn_only else
                                      "YOUTUBE UPLOAD REQUESTED" if auto_upload else "LOCAL / NO YOUTUBE UPLOAD"},
    })
    if existing_exports_dir is not None and not skip_export:
        return _abort_required_stage(result, start, "--export-dir requires --skip-export")
    composition = composition or cfg.get("export", {}).get("composition")
    from automation.cricket_gameplay import GAMEPLAY_COMPOSITIONS
    if composition and composition not in (*GAMEPLAY_COMPOSITIONS, "portrait-caption-v1"):
        return _abort_required_stage(result, start, f"Unknown composition: {composition}")
    if composition in GAMEPLAY_COMPOSITIONS and mode and not skip_enhancement and not learn_only and not skip_export:
        return _abort_required_stage(result, start,
            "Gameplay composition does not support post-render enhancement; it changes approved artwork and invalidates hash-bound QC")
    if gameplay_manifest and composition not in GAMEPLAY_COMPOSITIONS:
        return _abort_required_stage(result, start, "--gameplay-manifest requires the gameplay composition")
    if composition in GAMEPLAY_COMPOSITIONS and not learn_only and not skip_export:
        try:
            from automation.cricket_gameplay import load_delivery_library, load_render_library
            from automation.gameplay_export import preflight_library, pipeline_gameplay_seconds
            manifest = gameplay_manifest or cfg.get("export", {}).get("gameplay_manifest")
            if not manifest:
                raise ValueError("No gameplay recording: supply --gameplay-manifest with cleared local media")
            loader = load_delivery_library if auto_upload or (auto_sync and not skip_sync) else load_render_library
            library = loader(manifest)
            preflight_library(library,composition,
                              required_seconds=pipeline_gameplay_seconds(cfg))
        except (ValueError, OSError, TypeError, KeyError, subprocess.SubprocessError) as exc:
            return _abort_required_stage(result, start, f"gameplay preflight: {exc}")
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
    reserve_highlights: list[dict] = []

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
                return _abort_required_stage(result, start, f"stage1b: {e}")

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
        else:
            log_skipped(log, "Download", "download", rid, "--skip-download; using local input", video_path=video_path)

        # ── Stage 2: Transcribe ─────────────────────────────────
        if not skip_transcribe:
            try:
                with run_phase(log, "stage 2 Transcribe", "transcribe", run_id=rid) as ph:
                    from automation.transcript import fetch
                    t_data = fetch(url, output_path=transcript_path, video_path=video_path)
                    if t_data and t_data.get("segments"):
                        result.transcript_source = str(t_data.get("source", "unknown"))
                        with open(transcript_path, "w", encoding="utf-8") as f:
                            json.dump(t_data, f, indent=2, ensure_ascii=False)
                        ph.set(source=result.transcript_source, segments=len(t_data["segments"]))
                    else:
                        raise Exception("Failed to fetch or generate transcript")
                _PROVIDER_HEALTH.record_success("transcriber")
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("transcriber")
                return _abort_required_stage(result, start, f"stage2: {e}")

        elif Path(transcript_path).is_file():
            try:
                cached_transcript = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
                result.transcript_source = "cached:" + str(cached_transcript.get("source", "unknown"))
            except (OSError, ValueError, AttributeError):
                result.transcript_source = "cached:unknown"

        if skip_transcribe:
            log_skipped(log, "Transcribe", "transcribe", rid, "--skip-transcribe", source=result.transcript_source)

        # ── Stage 3-5: Highlight detection + agent scoring + rank ──
        if not skip_highlight:
            try:
                with run_phase(log, "stage 3-5 Highlight + Score + Rank",
                               "highlight", run_id=rid) as ph:
                    from automation.clip_selection.pipeline import detect_highlights
                    diagnostics = {}
                    highlights = detect_highlights(transcript_path, video_path,
                                                   highlights_path,
                                                   match_key=match_key,
                                                   diagnostics=diagnostics)
                    result.selected_clips = len(highlights)
                    reserve_highlights = diagnostics.get("reserves", [])
                    result.selection_status = diagnostics.get("status", "DRAFTS_SELECTED" if highlights else "NO_PUBLISHABLE_CANDIDATES")
                    result.candidate_count = diagnostics.get("candidates", 0)
                    result.review_candidates = diagnostics.get("review_candidates", 0)
                    result.rejected_clips = sum(diagnostics.get(key, 0) for key in
                                                ("hard_rejected", "below_quality", "editorial_rejected"))
                    ph.set(candidates=result.candidate_count, selected=result.selected_clips,
                           rejected=result.rejected_clips, review_candidates=result.review_candidates)
                    if result.selection_status == "REVIEW_REQUIRED" and not highlights:
                        ph.outcome("review", reason="no approved drafts")
                        return _abort_required_stage(result, start,
                                                     "stage3-5: editorial review required; no approved drafts")
                    for rank_idx, h in enumerate(highlights, 1):
                        cid = f"{stem}/{h['id']}"
                        _emit_event(cid, EventType.candidate_created, {
                            "text": h.get("text", ""),
                            "start": h.get("start"),
                            "end": h.get("end"),
                        })
                        _emit_event(cid, EventType.candidate_scored, {
                            "score": h.get("score", 0.0),
                            "dimension_scores": h.get("dimension_scores", {}),
                        })
                        _emit_event(cid, EventType.candidate_ranked, {
                            "rank": rank_idx,
                            "total": len(highlights),
                        })

                    _PROVIDER_HEALTH.record_success("llm")
                    if not highlights:
                        ph.outcome("empty", reason=result.selection_status)
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("llm")
                _emit_infra_failed(stem, str(e), stage="stage3-5")
                return _abort_required_stage(result, start, f"stage3-5: {e}")
        else:
            log_skipped(log, "Discovery + selection", "highlight", rid, "--skip-highlight")

        if not skip_highlight and not highlights and not skip_export:
            result.total_seconds = time.monotonic() - start
            log.info("[EXIT] no clips selected; no cached highlights will be exported (run_id=%s)", rid)
            return result

        # ── Stage 6: Export + SEO (merged — SEO runs immediately after export) ──
        if not skip_export:
            try:
                with run_phase(log, "stage 6 Export + SEO", "export", run_id=rid) as ph:
                    from export import ExportBatchError, export_all
                    try:
                        layout_options = {"composition": composition} if composition else {}
                        if composition in GAMEPLAY_COMPOSITIONS:
                            layout_options["gameplay_manifest"] = gameplay_manifest
                        result.exported = export_all(
                            highlights_path, video_path,
                            transcript_path=transcript_path,
                            generate_seo=not skip_seo,
                            **layout_options,
                        )
                    except ExportBatchError as exc:
                        result.exported = exc.exported
                        result.failures.append(f"stage6: {exc}")
                        ph.outcome("partial" if result.exported else "failed", failures=len(exc.failures))
                        _emit_infra_failed(stem, str(exc), stage="stage6")
                    for clip_path in result.exported:
                        _emit_event(f"{stem}/{clip_path.stem}", EventType.exported, {
                            "path": str(clip_path),
                        })
                    if result.exported:
                        try:
                            from shorts_intelligence.bridge import record_exported

                            shadow_count = record_exported(cfg, highlights + reserve_highlights, result.exported)
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
                    # Remember exported moments so the next run of this match picks new ones.
                    if result.exported and not skip_highlight:
                        try:
                            from automation.clip_selection.pipeline import record_exported_windows
                            by_id = {h["id"]: h for h in highlights + reserve_highlights}
                            record_exported_windows(diagnostics["dedupe_history"],
                                                    [by_id[p.stem] for p in result.exported if p.stem in by_id])
                        except (KeyError, OSError, NameError) as e:
                            log.warning("[dedupe] could not record exported windows: %s", e)
                    # SEO is now handled per-clip inside export_all()
                    if not skip_seo and result.exported:
                        result.seo_generated = sum(read_metadata(p) is not None for p in result.exported)
                    if any(f.startswith("stage6:") for f in result.failures):
                        _PROVIDER_HEALTH.record_failure("export")
                    else:
                        _PROVIDER_HEALTH.record_success("export")
                    ph.set(exported=len(result.exported),
                           seo_generated=result.seo_generated)
                    if not result.exported and not result.failures:
                        ph.outcome("empty", reason="no clips exported")
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("export")
                _emit_infra_failed(stem, str(e), stage="stage6")
                result.failures.append(f"stage6: {e}")
        else:
            try:
                existing = _existing_exports(shorts_dir, existing_exports_dir)
            except ValueError as exc:
                return _abort_required_stage(result, start, f"stage6: {exc}")
            if existing:
                result.exported = existing
                log_skipped(log, "Export", "export", rid, "--skip-export; existing batch", existing=len(existing))
            else:
                log_skipped(log, "Export", "export", rid, "--skip-export; no existing clips", directory=shorts_dir)

        # ── Stage 6b: Enhancement (optional) ────────────────────
        if result.exported and mode and not skip_enhancement:
            try:
                _validate_enhancement_inputs(result.exported)
            except ValueError as exc:
                return _abort_required_stage(result, start, f"stage6b: {exc}")
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

        if cfg.get("delivery_qc", {}).get("enabled", False):
            from automation.delivery_qc import check_delivery
            passed = []
            for clip in result.exported:
                expected_duration = None
                try:
                    edit = json.loads(clip.with_name(f"{clip.stem}_edit.json").read_text(encoding="utf-8"))
                    expected_duration = sum(part["output_end"] - part["output_start"] for part in edit["timeline"])
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                qc = check_delivery(clip, expected_duration)
                if qc["status"] != "PASS":
                    # Quarantine this clip only; siblings continue (cancer register C5).
                    result.failures.append(f"delivery_qc {clip.name}: {'; '.join(qc['errors'])}")
                    result.release_held += 1
                else:
                    passed.append(clip)
            result.exported = passed

        # Upgrade template packaging / legacy failure markers once, before covers,
        # review and scheduling. A marker still invalidates metadata under --skip-seo.
        if result.exported:
            if not skip_seo:
                from automation.seo.seo import retry_failed_seo
                for directory in sorted({p.parent for p in result.exported}):
                    try:
                        # Templates written moments ago by this run would only repeat the outage.
                        retry_failed_seo(str(directory), include_templates=skip_export)
                    except Exception as exc:
                        result.failures.append(f"seo_retry: {type(exc).__name__}: {exc}")
            result.seo_generated = sum(read_metadata(p) is not None for p in result.exported)

        # ── Stage 7b: Thumbnails ────────────────────────────────
        if result.exported and cfg.get("thumbnail", {}).get("enabled", True):
            try:
                with run_phase(log, "stage 7b Thumbnails", "thumbnails", run_id=rid) as ph:
                    from thumbnail import process_all_thumbnails
                    for directory in sorted({clip.parent for clip in result.exported}):
                        batch = process_all_thumbnails(str(directory))
                        result.thumbnails_generated += len(batch.generated)
                        result.thumbnails_preserved += len(batch.preserved)
                        result.failures.extend(f"stage7b: {failure}" for failure in batch.failures)
                        for failure in batch.failures:
                            log.error("[stage7b] %s", failure)
                    ph.set(generated=result.thumbnails_generated, preserved=result.thumbnails_preserved,
                           failures=sum(failure.startswith("stage7b:") for failure in result.failures))
                    if ph.metadata["failures"]:
                        ph.outcome("partial", reason="one or more covers failed")
            except Exception as e:
                result.failures.append(f"stage7b: {e}")
        elif result.exported:
            log_skipped(log, "Thumbnails", "thumbnails", rid, "disabled in config")

        # Build/refresh machine-verified reviews; auto mode approves clean clips.
        review_cfg = cfg.get("release_review", {})
        if review_cfg.get("required", False):
            from automation.release_review import prepare_review
            for clip in result.exported:
                if read_metadata(clip) is not None:
                    try:
                        prepare_review(clip, Path(video_path), mode=review_cfg.get("mode", "manual"))
                    except (OSError, ValueError) as exc:
                        log.warning("Review unavailable for %s: %s", clip.name, exc)

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
        elif result.exported:
            log_skipped(log, "Drive sync", "sync", rid, "sync not requested or explicitly skipped")

        # ── Stage 8b: Upload (auto when auth exists) ─────────────
        # Upload authenticates with the OAuth token only; cookies.txt is for downloads.
        has_auth = Path("yt_channel_token.json").exists()
        if result.exported and _should_upload(
            auto_upload=auto_upload,
            has_auth=has_auth,
        ):
            try:
                with run_phase(log, "stage 8b Upload", "upload",
                               run_id=rid) as ph:
                    from upload import upload_video
                    eligible = []
                    for clip in result.exported:
                        if read_metadata(clip) is not None:
                            if cfg.get("release_review", {}).get("required", False):
                                from automation.release_review import release_status
                                status, reasons = release_status(clip)
                                if status != "READY":
                                    result.release_held += 1
                                    result.failures.append(f"stage8b {clip.name}: {status}: {'; '.join(reasons)}")
                                    continue
                            result.release_ready += 1
                            eligible.append(clip)
                        else:
                            error = "missing, invalid or failed SEO metadata"
                            marker = clip.with_name(f"{clip.stem}_seo_failed.json")
                            try:
                                if marker.exists():
                                    error = json.loads(marker.read_text(encoding="utf-8")).get("error") or error
                            except (OSError, ValueError, AttributeError):
                                pass
                            result.failures.append(f"stage8b {clip.name}: {error}")
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
                        from scheduler import assign_clips_to_slots, format_for_youtube, occupied_publish_times
                        from automation.youtube_delivery import receipt_path
                        # Existing deliveries already have an authorized slot.
                        # Do not feed them a newly jittered time on every retry;
                        # the uploader must validate/reuse (or hold) its receipt.
                        new_deliveries = [clip for clip in eligible if not receipt_path(clip).exists()]
                        clip_scores: dict[str, float] = {}
                        for clip_path in new_deliveries:
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
                            clips=[p.stem for p in new_deliveries],
                            interval_hours=interval,
                            clip_scores=clip_scores or None,
                            schedule_config=cfg.get("upload_schedule", {}),
                            occupied=occupied_publish_times(paths.get("shorts", "shorts")),
                        ) if new_deliveries else []
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
                    for clip_path in eligible:
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
                                privacy="scheduled" if auto_schedule else privacy, publish_at=publish_at,
                            )
                            if not isinstance(video_id, str) or not video_id.strip():
                                raise RuntimeError("upload returned no video ID (see upload log for cause)")
                            result.uploaded_count += 1
                            _emit_event(
                                f"{stem}/{clip_path.stem}",
                                EventType.published,
                                {"privacy": "scheduled" if auto_schedule else privacy, "publish_at": publish_at,
                                 "youtube_video_id": str(video_id) if video_id else None},
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
                            if getattr(e, "stop_batch", False):
                                # Quota/auth refusals repeat for every clip; stop calling the API.
                                pending = eligible[eligible.index(clip_path) + 1:]
                                if pending:
                                    result.failures.append(
                                        "stage8b: upload loop stopped; still pending: "
                                        + ", ".join(p.name for p in pending))
                                break

                    if result.uploaded_count:
                        _PROVIDER_HEALTH.record_success("youtube")
                    ph.set(uploaded=result.uploaded_count,
                           skipped_no_meta=skipped_no_meta)
                    upload_failures = sum(failure.startswith("stage8b") for failure in result.failures)
                    if upload_failures:
                        ph.outcome("partial", failures=upload_failures)
            except Exception as e:
                _PROVIDER_HEALTH.record_failure("youtube")
                result.failures.append(f"stage8b: {e}")
        elif result.exported:
            reason = (
                "--upload was not requested"
                if not auto_upload
                else "no auth (cookies.txt or yt_channel_token.json)"
            )
            log_skipped(log, "YouTube upload", "upload", rid, reason)

    # Stage 9: telemetry and canonical persistence
    if result.exported:
        try:
            with run_phase(log, "stage 9a Analytics", "analytics", run_id=rid):
                from automation.seo.analytics import Analytics
                a = Analytics(_DECISION_STORE)
                summary = a.get_summary()
                log.debug("[analytics] summary: %s", summary)
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
                        log.log(logging.WARNING if stats["failures"] else logging.DEBUG,
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
            log.debug("[shorts_intelligence] %s", shadow)
        except Exception as e:
            log.warning("[shorts_intelligence] shadow status failed: %s", e)

        # Remote DB backup is a sync side effect, not local telemetry.
        if auto_sync and not skip_sync:
            try:
                from sync import sync_db_to_drive
                sync_db_to_drive()
            except Exception as e:
                log.warning("[db_sync] Failed: %s", e)

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
