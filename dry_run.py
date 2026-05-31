"""
dry_run.py — Execute pipeline logic with ALL external calls stubbed.

Runs the same 9-phase logic as automation/orchestrator.py (phases 0,1,2,3,4,
4.5,5,5.5,6,7,8) but replaces every outbound call (Whisper, YouTube API, Drive
API, LLM, ffmpeg, yt-dlp, thumbnail rendering) with no-op mocks or minimal fast
paths. It exercises the REAL observability layer (utils.logger.run_phase +
new_run_id), the REAL transcript correction (utils.transcript_postproc), and the
REAL config — so it validates config, phase ordering, skip flags, structured
logging, the SEO escalate-not-degrade retry queue, and PipelineResult output,
all locally in seconds.

Usage:
    python dry_run.py https://youtu.be/4ylLhtICj1I
    python dry_run.py https://youtu.be/4ylLhtICj1I --skip-download --skip-transcribe
    python dry_run.py https://youtu.be/4ylLhtICj1I --upload --sync --schedule
    python dry_run.py https://youtu.be/4ylLhtICj1I --mode ref_grade
    python dry_run.py --list-external        # list all stubbed external calls
    python dry_run.py --validate-config      # only check required config keys
"""

import argparse
import io
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dry_run")

# Real observability primitives — exercised (not stubbed) so the dry run
# actually validates the structured-logging layer.
from utils.logger import run_phase, new_run_id, JsonStreamHandler

_STUBBED_CALLS: list[str] = []

# Config keys introduced by the pipeline overhaul that MUST exist.
REQUIRED_CONFIG_KEYS = [
    ("ai", "retry_base_delay_seconds"),
    ("ai", "retry_max_delay_seconds"),
    ("ai", "race_tier_timeout_seconds"),
    ("seo", "inject_viral_elements"),
    ("premium", "yolo_device"),
    ("premium", "yolo_batch_size"),
    ("premium", "identity_refs"),
    ("youtube", "upload_chunk_size_mb"),
    ("youtube", "upload_max_retries"),
    ("youtube", "upload_deadline_seconds"),
]

EXPECTED_PHASE_STAGES = [
    "transcript_fetch", "download", "transcribe", "highlight", "export",
    "enhancement", "seo", "thumbnails", "sync", "upload", "analytics",
    "automation_learner", "provider_health", "event_store_validation",
]

# Prompts module must be importable and all templates must format cleanly.
REQUIRED_PROMPTS = [
    "HIGHLIGHT_RANKER_SYSTEM", "HIGHLIGHT_RANKER_USER_TEMPLATE",
    "CANDIDATE_EVAL_SYSTEM", "CANDIDATE_EVAL_USER_TEMPLATE",
    "SEO_SYSTEM", "SEO_USER_TEMPLATE",
    "ANALYTICS_SYSTEM", "ANALYTICS_USER_TEMPLATE",
    "ORCHESTRATOR_SYSTEM",
    "DEFAULT_SCORING_WEIGHTS", "MIN_QUALITY_THRESHOLD",
    "MAX_SELECTED_CLIPS", "MAX_CANDIDATES",
]


@dataclass
class PipelineResult:
    exported: list = field(default_factory=list)
    uploaded_count: int = 0
    failures: list = field(default_factory=list)
    total_seconds: float = 0.0
    transcript_source: str = "none"
    run_id: str = ""
    selected_clips: int = 0
    seo_generated: int = 0


def _stub(name: str, *args, **kwargs):
    _STUBBED_CALLS.append(name)
    log.info("  [STUB] %s%s", name, f" ({args[0]})" if args else "")


def _make_segments():
    # Includes deliberate mishearings ("coaly", "bumra") so the REAL
    # transcript correction layer has something to fix.
    return {
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Hello and welcome to the live stream"},
            {"start": 5.0, "end": 10.0, "text": "Today coaly is batting brilliantly"},
            {"start": 30.0, "end": 35.0, "text": "coaly hits a massive six"},
            {"start": 60.0, "end": 65.0, "text": "bumra bowls a perfect yorker"},
            {"start": 120.0, "end": 125.0, "text": "What a catch in the deep"},
            {"start": 300.0, "end": 305.0, "text": "And that's the end of the over"},
        ],
        "source": "api",
    }


def _make_highlights():
    return [
        {"start_sec": 30, "end_sec": 55, "score": 0.92, "label": "Kohli Six"},
        {"start_sec": 60, "end_sec": 85, "score": 0.88, "label": "Yorker"},
        {"start_sec": 120, "end_sec": 145, "score": 0.85, "label": "Amazing Catch"},
    ]


def _create_fake_video(path: str, size_mb: int = 10):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(b"X" * (size_mb * 1024 * 1024))
        log.info("  [FAKE] Created %s (%d MB)", path, size_mb)


def _create_fake_clips(export_dir: str, stems: list[str]):
    d = Path(export_dir)
    d.mkdir(parents=True, exist_ok=True)
    exported = []
    for s in stems:
        clip = d / f"{s}.mp4"
        if not clip.exists():
            clip.write_bytes(b"C" * 1024 * 1024)
        exported.append(clip)
    return exported


# ─── Stubbed external module replacements ─────────────────────────────────────

def stub_transcript_fetch(url: str, output_path: str = "") -> dict:
    _stub("transcript.fetch", url)
    data = _make_segments()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f)
    return data


def stub_download(url: str, output_path: str, **kwargs) -> str:
    _stub("download.download", url)
    _create_fake_video(output_path)
    return output_path


def stub_transcribe(video_path: str, output_path: str):
    _stub("transcribe.transcribe", video_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(_make_segments(), f)


def stub_detect_highlights(transcript_path: str, video_path: str, output_path: str) -> list:
    _stub("highlight.detect_highlights", transcript_path)
    import yaml
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clips = {f"clip{i+1}": {"start": h["start_sec"], "end": h["end_sec"],
                            "label": h["label"], "text": "coaly six bumra yorker"}
             for i, h in enumerate(_make_highlights())}
    with open(output_path, "w") as f:
        yaml.dump(clips, f)
    return list(clips.keys())


def stub_export_all(highlights_path: str, video_path: str, **kwargs) -> list[Path]:
    _stub("export.export_all", highlights_path)
    import yaml
    with open(highlights_path) as f:
        data = yaml.safe_load(f) or {}
    stems = list(data.keys()) if data else [f"clip{i+1}" for i in range(3)]
    export_dir = str(Path(kwargs.get("output_dir", "shorts")) / "2026-05-29_000000")
    return _create_fake_clips(export_dir, stems)


def stub_grade_video(clip_path: str, ref_path: str, output_path: str) -> str:
    _stub("ref_grade.grade_video", clip_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(Path(clip_path).read_bytes())
    return output_path


def stub_enhance_video(clip_path: str, ref_path: str, output_path: str, **kwargs) -> str:
    _stub("face_mapper.enhance_video", clip_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(Path(clip_path).read_bytes())
    return output_path


def stub_process_all_seo(highlights_path: str, export_dir: str, fail_last: bool = True):
    """Stub SEO. Writes metadata for each clip; to exercise the escalate-not-
    degrade queue, the LAST clip is left WITHOUT metadata and instead gets a
    self-contained *_seo_failed.json marker (as the real code does on failure)."""
    _stub("seo.process_all_seo", export_dir)
    d = Path(export_dir)
    clips = sorted(d.glob("*.mp4"))
    for i, clip in enumerate(clips):
        is_last = (i == len(clips) - 1)
        if fail_last and is_last:
            from datetime import datetime as _dt
            marker = d / f"{clip.stem}_seo_failed.json"
            marker.write_text(json.dumps({
                "clip_id": clip.stem, "reason": "simulated LLM failure after escalation",
                "queued_at": _dt.now().isoformat(),
                "transcript": "coaly six bumra yorker", "video_title": "RCB vs CSK",
                "scorecard": "", "trend_topics": [], "live_stream_url": "",
            }))
            log.info("  [STUB]   %s queued for retry (no generic fallback written)", clip.stem)
            continue
        meta = d / f"{clip.stem}_metadata.json"
        meta.write_text(json.dumps({
            "title": f"LIVE RCB vs CSK | {clip.stem} Hindi Commentary",
            "description": "Kohli six! #Shorts",
            "tags": ["rcb vs csk live", "kohli six", "ipl live hindi"],
            "search_terms": ["rcb vs csk live", "kohli six"],
            "hashtags": ["#Shorts", "#RCBvCSK"],
            "ai_generated": True, "is_shorts": True, "seo_score": 0.85,
        }))


def stub_retry_failed_seo(export_dir: str) -> dict:
    """Stub seo.retry_failed_seo() — recovers queued clips (writes metadata,
    removes the marker), validating the retry-queue flow shape."""
    _stub("seo.retry_failed_seo", export_dir)
    d = Path(export_dir)
    markers = sorted(d.glob("*_seo_failed.json"))
    recovered = 0
    for marker in markers:
        ctx = json.loads(marker.read_text())
        clip_id = ctx["clip_id"]
        (d / f"{clip_id}_metadata.json").write_text(json.dumps({
            "title": f"LIVE {ctx.get('video_title','Cricket')} | {clip_id} (recovered)",
            "description": "Recovered on retry. #Shorts",
            "tags": ["ipl live hindi"], "search_terms": ["ipl live hindi"],
            "hashtags": ["#Shorts"], "ai_generated": True, "is_shorts": True,
        }))
        marker.unlink()
        recovered += 1
        log.info("  [STUB]   %s recovered on retry -> metadata written, dequeued", clip_id)
    return {"retried": len(markers), "recovered": recovered, "still_failed": 0}


def stub_process_all_thumbnails(export_dir: str):
    _stub("thumbnail.process_all_thumbnails", export_dir)
    d = Path(export_dir)
    for clip in sorted(d.glob("*.mp4")):
        thumb = d / f"{clip.stem}_thumb.jpg"
        if not thumb.exists():
            thumb.write_bytes(b"J" * 1024)


def stub_sync_to_drive(folder_path: str):
    _stub("sync.sync_to_drive", folder_path)


def stub_download_from_drive(filenames=None, dest_dir: str = ""):
    _stub("sync.download_from_drive", str(dest_dir))
    d = Path(dest_dir)
    d.mkdir(parents=True, exist_ok=True)
    if filenames is None:
        return
    for name in filenames:
        p = d / name
        if not p.exists():
            p.write_bytes(b"D" * 1024)


def stub_upload_video(video_path: str, meta_path: str, **kwargs):
    _stub("upload.upload_video", video_path)
    log.info("  [STUB]   privacy=%s publish_at=%s (categoryId validated, finite "
             "resumable chunking, bounded retries+deadline)",
             kwargs.get("privacy"), kwargs.get("publish_at"))


def stub_generate_daily_insights():
    _stub("analytics.generate_daily_insights")


def stub_assign_clips_to_slots(clips: list, **kwargs) -> list:
    from datetime import datetime, timedelta
    base = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return [(stem, base + timedelta(hours=i)) for i, stem in enumerate(clips)]


def stub_format_for_youtube(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── Config validation ────────────────────────────────────────────────────────

def validate_config(cfg: dict) -> list[tuple]:
    """Return [(dotted_key, present_bool, value)] for every required key."""
    results = []
    for path in REQUIRED_CONFIG_KEYS:
        node = cfg
        present = True
        for part in path:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                present = False
                break
        results.append((".".join(path), present, node if present else None))
    return results


# ─── Dry-run pipeline ─────────────────────────────────────────────────────────

def dry_run(url: str,
            skip_download=False, skip_transcribe=False,
            skip_highlight=False, skip_export=False,
            skip_sync=False, skip_seo=False,
            auto_sync=False, auto_upload=False,
            auto_schedule=False, sample_minutes=None,
            sync_from_drive=False, mode=None) -> dict:
    """Execute the full pipeline logic with all external calls stubbed,
    mirroring automation/orchestrator.run() including run_phase + retry queue."""

    from automation.config import load as load_config

    # Capturing structured logger so we can VALIDATE the observability layer.
    cap_buf = io.StringIO()
    plog = logging.getLogger("dry_run.pipeline")
    plog.handlers = [JsonStreamHandler(cap_buf)]
    plog.setLevel(logging.DEBUG)
    plog.propagate = False

    start = time.monotonic()
    result = PipelineResult()
    result.run_id = new_run_id()
    rid = result.run_id
    _STUBBED_CALLS.clear()

    cfg = load_config()
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

    log.info("=" * 64)
    log.info("DRY RUN — external calls STUBBED, observability/config/correction REAL")
    log.info("URL: %s   run_id=%s", url, rid)
    log.info("Flags: skip_dl=%s skip_tx=%s skip_hl=%s skip_ex=%s skip_sync=%s "
             "skip_seo=%s sync=%s upload=%s schedule=%s mode=%s",
             skip_download, skip_transcribe, skip_highlight, skip_export,
             skip_sync, skip_seo, auto_sync, auto_upload, auto_schedule, mode)
    log.info("=" * 64)

    from automation import decision_store
    decision_store.clear()
    event_count_before = 0

    validation = {
        "transcript_phase_ran": False,
        "transcript_corrected": False,
        "transcript_before": "", "transcript_after": "",
        "seo_marker_written": False, "seo_recovered": 0,
    }

    # ── Phase 0: YouTube transcript + REAL correction ─────────────────────────
    if not skip_transcribe:
        try:
            with run_phase(plog, "phase 0 Transcript", "transcript_fetch", run_id=rid) as ph:
                validation["transcript_phase_ran"] = True
                transcript = stub_transcript_fetch(url, output_path=transcript_path)
                segs = transcript.get("segments", [])
                if segs:
                    # Exercise the REAL centralized cricket correction layer.
                    from utils.transcript_postproc import correct_segments
                    _stub("transcript_postproc.correct_segments")
                    before = segs[1]["text"]
                    segs, n = correct_segments(segs)
                    after = segs[1]["text"]
                    validation["transcript_before"] = before
                    validation["transcript_after"] = after
                    validation["transcript_corrected"] = ("Kohli" in after and "coaly" not in after.lower())
                    skip_transcribe = True
                    result.transcript_source = transcript.get("source", "api")
                    ph.set(source=result.transcript_source, segments=len(segs), corrections=n)
                else:
                    result.failures.append("phase0: no segments")
        except Exception as e:
            result.failures.append("phase0: %s" % e)

    # ── Phase 1: Drive sync / Download ────────────────────────────────────────
    if sync_from_drive:
        try:
            with run_phase(plog, "phase 1 Drive pull", "drive_pull", run_id=rid):
                stub_download_from_drive(filenames=["video.mp4"], dest_dir=input_dir)
                stub_download_from_drive(filenames=[f"{stem}.json"], dest_dir=transcripts_dir)
                stub_download_from_drive(filenames=None, dest_dir=shorts_dir)
                skip_download = True
                skip_transcribe = True
        except Exception as e:
            result.failures.append("phase1 sync: %s" % e)

    if not skip_download:
        try:
            with run_phase(plog, "phase 1 Download", "download", run_id=rid) as ph:
                video_path = stub_download(url, video_path, sample_minutes=sample_minutes)
                stem = Path(video_path).stem
                transcript_path = str(Path(transcripts_dir) / f"{stem}.json")
                highlights_path = str(Path(highlights_dir) / f"{stem}.yaml")
                ph.set(video_path=video_path)
        except Exception as e:
            result.failures.append("phase1: %s" % e)

    # ── Phase 2: Transcribe ───────────────────────────────────────────────────
    if not skip_transcribe:
        try:
            with run_phase(plog, "phase 2 Transcribe", "transcribe", run_id=rid):
                stub_transcribe(video_path, transcript_path)
        except Exception as e:
            result.failures.append("phase2: %s" % e)

    # ── Phase 3: Highlight detection ──────────────────────────────────────────
    if not skip_highlight:
        try:
            with run_phase(plog, "phase 3 Highlight", "highlight", run_id=rid) as ph:
                clips = stub_detect_highlights(transcript_path, video_path, highlights_path)
                ph.set(clips=len(clips))
        except Exception as e:
            result.failures.append("phase3: %s" % e)

    if not Path(highlights_path).exists():
        log.error("Highlights file missing: %s", highlights_path)
        result.failures.append("phase3: highlights file missing")

    # ── Phase 4: Export ───────────────────────────────────────────────────────
    if not skip_export:
        try:
            with run_phase(plog, "phase 4 Export", "export", run_id=rid) as ph:
                result.exported = stub_export_all(
                    highlights_path, video_path,
                    transcript_path=transcript_path, generate_seo=False,
                )
                ph.set(exported=len(result.exported))
        except Exception as e:
            result.failures.append("phase4: %s" % e)
    else:
        existing = sorted(Path(shorts_dir).rglob("*.mp4")) if Path(shorts_dir).exists() else []
        existing = [p for p in existing if "test_output" not in p.name]
        if existing:
            latest_dir = max(set(p.parent for p in existing), key=lambda d: d.name)
            result.exported = sorted(latest_dir.glob("*.mp4"))
            log.info("[phase 4] skipped — using %d existing clips", len(result.exported))

    # ── Phase 4.5: Enhancement ────────────────────────────────────────────────
    if result.exported and mode:
        try:
            with run_phase(plog, "phase 4.5 Enhancement (%s)" % mode, "enhancement", run_id=rid) as ph:
                import shutil
                ref_path = cfg.get("enhancement", {}).get("reference", "expectation.png")
                enhanced = []
                for clip_path in result.exported:
                    if mode == "ref_grade":
                        out = str(Path(paths.get("temp", "temp")) / f"{clip_path.stem}_graded.mp4")
                        r = stub_grade_video(str(clip_path), ref_path, out)
                    else:
                        out = str(Path(paths.get("temp", "temp")) / f"{clip_path.stem}_enhanced.mp4")
                        r = stub_enhance_video(str(clip_path), ref_path, out, use_region_grading=True)
                    if r == out and Path(out).exists():
                        shutil.move(out, str(clip_path))
                    enhanced.append(clip_path)
                result.exported = enhanced
                ph.set(enhanced=len(result.exported), mode=mode)
        except Exception as e:
            result.failures.append("phase4.5: %s" % e)

    # ── Phase 5: SEO (+ escalate-not-degrade retry queue) ─────────────────────
    if not skip_seo and result.exported:
        try:
            with run_phase(plog, "phase 5 SEO", "seo", run_id=rid) as ph:
                export_dir = str(result.exported[0].parent)
                stub_process_all_seo(highlights_path, export_dir)
                markers = list(Path(export_dir).glob("*_seo_failed.json"))
                validation["seo_marker_written"] = len(markers) > 0
                retry = stub_retry_failed_seo(export_dir)
                validation["seo_recovered"] = retry.get("recovered", 0)
                ph.set(clips=len(result.exported),
                       seo_recovered=retry.get("recovered", 0),
                       seo_still_queued=retry.get("still_failed", 0))
        except Exception as e:
            result.failures.append("phase5: %s" % e)

    # ── Phase 5.5: Thumbnails ─────────────────────────────────────────────────
    if result.exported:
        try:
            with run_phase(plog, "phase 5.5 Thumbnails", "thumbnails", run_id=rid):
                stub_process_all_thumbnails(str(result.exported[0].parent))
        except Exception as e:
            result.failures.append("phase5.5: %s" % e)

    # ── Phase 6: Sync to Drive ────────────────────────────────────────────────
    do_sync = auto_sync and not skip_sync
    if do_sync and result.exported:
        try:
            with run_phase(plog, "phase 6 Sync", "sync", run_id=rid):
                stub_sync_to_drive(str(result.exported[0].parent))
        except Exception as e:
            result.failures.append("phase6: %s" % e)
    elif skip_sync:
        log.info("[phase 6 Sync] skipped (--skip-sync)")

    # ── Phase 7: Upload ───────────────────────────────────────────────────────
    if (auto_upload or auto_schedule) and result.exported:
        try:
            with run_phase(plog, "phase 7 Upload", "upload", run_id=rid) as ph:
                privacy = cfg.get("youtube", {}).get("privacy_status", "private")
                interval = cfg.get("upload_schedule", {}).get(
                    "interval_hours",
                    cfg.get("youtube", {}).get("schedule_interval_hours", 1))
                slot_map = {}
                if auto_schedule:
                    assignments = stub_assign_clips_to_slots(
                        clips=[p.stem for p in result.exported], interval_hours=interval)
                    slot_map = {s: dt for s, dt in assignments}
                    log.info("Schedule generated for %d clips", len(assignments))
                skipped_no_meta = 0
                for clip_path in result.exported:
                    meta_path = clip_path.with_name(f"{clip_path.stem}_metadata.json")
                    if not meta_path.exists():
                        skipped_no_meta += 1
                        log.warning("No metadata for %s — skipping upload (queued)", clip_path.name)
                        continue
                    publish_at = None
                    if auto_schedule and slot_map.get(clip_path.stem):
                        publish_at = stub_format_for_youtube(slot_map[clip_path.stem])
                    stub_upload_video(str(clip_path), str(meta_path),
                                      privacy=privacy, publish_at=publish_at)
                    result.uploaded_count += 1
                ph.set(uploaded=result.uploaded_count, skipped_no_meta=skipped_no_meta)
        except Exception as e:
            result.failures.append("phase7: %s" % e)

    # ── Phase 8: Analytics + Self-learning ───────────────────────────────────
    if auto_upload or auto_schedule:
        try:
            with run_phase(plog, "phase 8 Analytics + Learning", "analytics", run_id=rid):
                stub_generate_daily_insights()
        except Exception as e:
            result.failures.append("phase8: %s" % e)

    # ── Phase 9: Automation Learner + Provider Health + Event validation ─────
    event_count_before = decision_store.count()

    if auto_upload or auto_schedule:
        try:
            with run_phase(plog, "phase 9a Automation Learner", "automation_learner", run_id=rid):
                from automation import policy_updater, preference_engine, replay_engine
                policy_updater.update_from_events(decision_store.get_all_events())
                if not replay_engine.verify():
                    replay_engine.replay()
                prefs = preference_engine.compute_preferences()
                plog.info("[automation_learner] processed %d events, prefs=%s",
                          decision_store.count(), prefs)
        except Exception as e:
            result.failures.append("phase9a: %s" % e)

        try:
            with run_phase(plog, "phase 9b Provider Health", "provider_health", run_id=rid):
                from automation import provider_health
                for provider_name in ("transcript", "download", "transcriber", "llm",
                                      "export", "enhancement", "drive", "youtube"):
                    stats = provider_health.get_stats(provider_name)
                    if stats["total_calls"]:
                        plog.info("[provider_health] %s: status=%s calls=%d ok=%d fail=%d",
                                  provider_name, stats["status"].value,
                                  stats["total_calls"], stats["successes"], stats["failures"])
        except Exception as e:
            result.failures.append("phase9b: %s" % e)

    # Validate event emission directly by calling emit_event and checking the store.
    try:
        with run_phase(plog, "phase 9c Event Store Validation", "event_store_validation", run_id=rid):
            from automation import emit_event, emit_infra_failed, decision_store as _ds
            from automation.memory.event_models import EventType
            emitted = 0
            e1 = emit_event("dry/clip1", EventType.candidate_created, {"text": "test"})
            emitted += 1
            e2 = emit_event("dry/clip1", EventType.candidate_scored, {"score": 0.8})
            emitted += 1
            e3 = emit_infra_failed("dry/clip1", "simulated error", "dry_test")
            emitted += 1
            total = _ds.count()
            plog.info("[event_store] emitted %d events, store has %d total", emitted, total)
            if total >= event_count_before + emitted:
                plog.info("[event_store] event emission + storage: PASS")
            else:
                plog.warning("[event_store] expected %d events, got %d",
                             event_count_before + emitted, total)
    except Exception as e:
        result.failures.append("phase9c: %s" % e)

    result.total_seconds = time.monotonic() - start
    status = "partial" if result.failures else "ok"
    log.info("[EXIT] run_id=%s exported=%d uploaded=%d selected=%d seo=%d "
             "failures=%d elapsed=%.2fs transcript=%s",
             rid, len(result.exported), result.uploaded_count,
             result.selected_clips, result.seo_generated,
             len(result.failures), result.total_seconds, result.transcript_source)

    # Parse the captured structured log to validate observability.
    struct_records = []
    for line in cap_buf.getvalue().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            struct_records.append(json.loads(line))
        except Exception:
            pass
    logged_stages = sorted({r.get("stage") for r in struct_records
                            if r.get("run_id") == rid and r.get("stage")})

    # Validate prompts module
    prompts_ok = True
    try:
        import prompts as _p
        for name in REQUIRED_PROMPTS:
            if not hasattr(_p, name):
                prompts_ok = False
                log.warning("Missing prompt: %s", name)
    except Exception as e:
        prompts_ok = False
        log.warning("Prompts module import failed: %s", e)

    event_count_new = decision_store.count() - event_count_before

    return {
        "run_id": rid,
        "exported": [str(p) for p in result.exported],
        "uploaded_count": result.uploaded_count,
        "selected_clips": result.selected_clips,
        "seo_generated": result.seo_generated,
        "failures": result.failures,
        "total_seconds": result.total_seconds,
        "transcript_source": result.transcript_source,
        "stubbed_calls": _STUBBED_CALLS[:],
        "config_checks": validate_config(cfg),
        "struct_records": len(struct_records),
        "logged_stages": logged_stages,
        "validation": validation,
        "prompts_ok": prompts_ok,
        "event_count": event_count_new,
    }


def list_external_calls():
    calls = [
        ("transcript.fetch", "YouTube Transcript API / yt-dlp subtitle download"),
        ("transcript_postproc.correct_segments", "REAL cricket spelling correction (no network)"),
        ("download.download", "yt-dlp video download"),
        ("transcribe.transcribe", "faster-whisper GPU transcription (BatchedInferencePipeline)"),
        ("highlight.detect_highlights", "ffmpeg audio energy + LLM refinement"),
        ("export.export_all", "ffmpeg clip cutting + encoding (+ YOLO/MediaPipe crop)"),
        ("ref_grade.grade_video", "ffmpeg color grading pipeline"),
        ("face_mapper.enhance_video", "per-frame face enhancement pipeline"),
        ("seo.process_all_seo", "multi-provider LLM SEO (escalate-not-degrade)"),
        ("seo.retry_failed_seo", "retry queue for clips that failed SEO (no generic fallback)"),
        ("thumbnail.process_all_thumbnails", "Pillow image compositing + font rendering"),
        ("sync.sync_to_drive", "Google Drive API upload"),
        ("sync.download_from_drive", "Google Drive API download"),
        ("upload.upload_video", "YouTube Data API v3 (categoryId-validated, resumable)"),
        ("analytics.generate_daily_insights", "YouTube Data + Analytics API (CTR/retention)"),
        ("scheduler.assign_clips_to_slots", "datetime scheduling + prime-time logic"),
    ]
    print("\nStubbed External Calls:")
    print("=" * 70)
    for name, desc in calls:
        print(f"  {name:<38s} {desc}")
    print()


def _print_validation_report(result: dict):
    cfg_checks = result["config_checks"]
    cfg_ok = sum(1 for _, ok, _ in cfg_checks if ok)
    v = result["validation"]
    expected_stages = set(EXPECTED_PHASE_STAGES)
    logged = set(result["logged_stages"])

    print()
    print("=" * 70)
    print("DRY RUN COMPLETE — VALIDATION REPORT")
    print("=" * 70)
    print(f"  run_id:             {result['run_id']}")
    print(f"  Exported:           {len(result['exported'])} clips")
    print(f"  Uploaded:           {result['uploaded_count']}")
    print(f"  Selected clips:     {result.get('selected_clips', 0)}")
    print(f"  SEO generated:      {result.get('seo_generated', 0)}")
    print(f"  Failures:           {len(result['failures'])}")
    print(f"  Elapsed:            {result['total_seconds']:.2f}s")
    print(f"  Transcript source:  {result['transcript_source']}")
    print(f"  Stubbed calls:      {len(result['stubbed_calls'])}")
    print()
    print("  [1] Config tunables (overhaul):")
    for key, ok, val in cfg_checks:
        mark = "PASS" if ok else "FAIL"
        print(f"        [{mark}] {key} = {val}")
    print(f"      -> {cfg_ok}/{len(cfg_checks)} present")
    print()
    print("  [2] Observability (run_phase + run_id):")
    print(f"        run_id generated:        {'PASS' if result['run_id'] else 'FAIL'}")
    print(f"        structured records:      {result['struct_records']}")
    print(f"        stages logged:           {sorted(logged)}")
    print()
    print("  [3] Transcript correction (utils.transcript_postproc):")
    if not v["transcript_phase_ran"]:
        print("        N/A — transcript phase skipped this run")
        transcript_ok = True
    else:
        print(f"        before: {v['transcript_before']!r}")
        print(f"        after:  {v['transcript_after']!r}")
        print(f"        corrected coaly->Kohli:  {'PASS' if v['transcript_corrected'] else 'FAIL'}")
        transcript_ok = v["transcript_corrected"]
    print()
    print("  [4] SEO escalate-not-degrade retry queue:")
    print(f"        failure marker written:  {'PASS' if v['seo_marker_written'] else 'n/a'}")
    print(f"        recovered on retry:      {v['seo_recovered']}")
    print()
    print("  [5] Prompts module:")
    prompts_ok = result.get("prompts_ok", False)
    print(f"        all templates present: {'PASS' if prompts_ok else 'FAIL'}")
    print()
    missing = expected_stages - logged
    print("  [6] Phase coverage:")
    print(f"        expected stages present: {sorted(expected_stages & logged)}")
    if missing:
        print(f"        not run (flag-dependent):{sorted(missing)}")
    print()
    print("  [7] Event store (automation.decision_store):")
    event_count = result.get("event_count", 0)
    print(f"        events emitted:             {event_count}")
    if event_count:
        from automation import decision_store as _ds
        event_types = sorted({e.event_type.value for e in _ds.get_all_events()})
        print(f"        event types:               {event_types}")
    print()
    if result["failures"]:
        print("  Failures detail:")
        for f in result["failures"]:
            print(f"    - {f}")
    # Overall verdict
    ok = (cfg_ok == len(cfg_checks) and result["run_id"] and
          transcript_ok and prompts_ok and not result["failures"])
    print("=" * 70)
    print(f"  OVERALL: {'PASS ✅' if ok else 'CHECK ⚠'}")
    print("=" * 70)
    print()
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Dry-run pipeline — stub external calls, validate phase logic locally.")
    parser.add_argument("url", nargs="?", default="https://youtu.be/4ylLhtICj1I")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-transcribe", action="store_true")
    parser.add_argument("--skip-highlight", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-seo", action="store_true")
    parser.add_argument("--sync", action="store_true", dest="auto_sync")
    parser.add_argument("--upload", action="store_true", dest="auto_upload")
    parser.add_argument("--schedule", action="store_true", dest="auto_schedule")
    parser.add_argument("--sample-minutes", type=int, default=None)
    parser.add_argument("--sync-from-drive", action="store_true")
    parser.add_argument("--mode", choices=["face_mapper", "ref_grade"], default=None)
    parser.add_argument("--list-external", action="store_true")
    parser.add_argument("--validate-config", action="store_true",
                        help="Only check required config keys exist, then exit")
    args = parser.parse_args()

    if args.list_external:
        list_external_calls()
        return

    if args.validate_config:
        from automation.config import load as load_config
        checks = validate_config(load_config())
        print("\nConfig validation:")
        print("=" * 60)
        all_ok = True
        for key, ok, val in checks:
            all_ok = all_ok and ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {key} = {val}")
        print("=" * 60)
        print(f"  {'ALL PRESENT ✅' if all_ok else 'MISSING KEYS ⚠'}\n")
        sys.exit(0 if all_ok else 1)

    result = dry_run(
        url=args.url,
        skip_download=args.skip_download,
        skip_transcribe=args.skip_transcribe,
        skip_highlight=args.skip_highlight,
        skip_export=args.skip_export,
        skip_sync=args.skip_sync,
        skip_seo=args.skip_seo,
        auto_sync=args.auto_sync,
        auto_upload=args.auto_upload,
        auto_schedule=args.auto_schedule,
        sample_minutes=args.sample_minutes,
        sync_from_drive=args.sync_from_drive,
        mode=args.mode,
    )

    ok = _print_validation_report(result)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
