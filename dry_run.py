"""
dry_run.py — Execute pipeline logic with ALL external calls stubbed.

Runs the full 8-phase pipeline (same logic as automation/orchestrator.py)
but replaces every outbound call (Whisper, YouTube API, Drive API, Groq LLM,
ffmpeg, yt-dlp, thumbnail PIL/font rendering) with no-op mocks or minimal
fast paths. Validates config, paths, phase ordering, skip flags, error
handling, and PipelineResult output — all locally, in seconds.

Usage:
    python dry_run.py https://youtu.be/dQw4w9WgXcQ
    python dry_run.py https://youtu.be/dQw4w9WgXcQ --skip-download --skip-transcribe
    python dry_run.py https://youtu.be/dQw4w9WgXcQ --upload --sync
    python dry_run.py https://youtu.be/dQw4w9WgXcQ --mode ref_grade
    python dry_run.py --list-external        # list all stubbed external calls
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dry_run")

_STUBBED_CALLS: list[str] = []


def _stub(name: str, *args, **kwargs):
    _STUBBED_CALLS.append(name)
    log.info("  [STUB] %s%s", name, f" ({args[0]})" if args else "")


def _stubbed(orig_name: str):
    """Decorator that logs the call and returns a canned value."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            _stub(orig_name, *args)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def _make_segments():
    return {
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Hello and welcome to the stream"},
            {"start": 5.0, "end": 10.0, "text": "Today we have an exciting match"},
            {"start": 30.0, "end": 35.0, "text": "Kohli hits a massive six"},
            {"start": 60.0, "end": 65.0, "text": "Crowd goes wild"},
            {"start": 120.0, "end": 125.0, "text": "What a catch in the deep"},
            {"start": 300.0, "end": 305.0, "text": "And that's the end of the over"},
        ],
        "source": "api",
    }


def _make_highlights():
    return [
        {"start_sec": 30, "end_sec": 55, "score": 0.92, "label": "Kohli Six"},
        {"start_sec": 60, "end_sec": 85, "score": 0.88, "label": "Crowd Reaction"},
        {"start_sec": 120, "end_sec": 145, "score": 0.85, "label": "Amazing Catch"},
    ]


def _make_analysis():
    return {
        "summary": {
            "duration_sec": 300,
            "frames_sampled": 150,
            "frames_with_face": 90,
            "face_detection_rate": 60.0,
            "avg_quality": 0.72,
            "max_quality": 0.95,
            "min_quality": 0.30,
            "avg_brightness": 128.0,
            "avg_ref_match": 0.45,
            "reference_photos_used": 2,
            "analysis_time_sec": 0.01,
        },
        "per_second": [{"second": i, "quality": 0.7, "brightness": 128,
                         "ref_match": 0.4, "face_present": True}
                        for i in range(0, 300)],
        "best_segments": _make_highlights(),
    }


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
        meta = d / f"{s}_metadata.json"
        if not meta.exists():
            meta.write_text(json.dumps({
                "title": f"Clip {s}",
                "description": "Auto-generated test clip",
                "tags": ["cricket", "ipl", "shorts"],
                "quality_score": 0.85,
            }))
        thumb = d / f"{s}_thumb.jpg"
        if not thumb.exists():
            thumb.write_bytes(b"J" * 1024)
        exported.append(clip)
    return exported


# ─── Stubbed module replacements ──────────────────────────────────────────────

def stub_transcript_fetch(url: str, output_path: str = "") -> dict:
    """Stub for transcript.fetch() — returns canned segments."""
    _stub("transcript.fetch", url)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(_make_segments(), f)
    return _make_segments()


def stub_download(url: str, output_path: str, **kwargs) -> str:
    """Stub for download.download() — creates a fake video file."""
    _stub("download.download", url)
    _create_fake_video(output_path)
    return output_path


def stub_transcribe(video_path: str, output_path: str):
    """Stub for transcribe.transcribe() — writes canned transcript."""
    _stub("transcribe.transcribe", video_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(_make_segments(), f)


def stub_analyze_video(video_path: str, **kwargs) -> dict:
    """Stub for video_analyzer.analyze_video() — returns canned analysis."""
    _stub("video_analyzer.analyze_video", video_path)
    return _make_analysis()


def stub_detect_highlights(transcript_path: str, video_path: str, output_path: str) -> list:
    """Stub for highlight.detect_highlights() — writes canned YAML."""
    _stub("highlight.detect_highlights", transcript_path)
    import yaml
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clips = [{"start": h["start_sec"], "end": h["end_sec"], "label": h["label"]}
             for h in _make_highlights()]
    with open(output_path, "w") as f:
        yaml.dump({"clips": clips}, f)
    return clips


def stub_export_all(highlights_path: str, video_path: str, **kwargs) -> list[Path]:
    """Stub for export.export_all() — creates fake .mp4 clip files."""
    _stub("export.export_all", highlights_path)
    import yaml
    with open(highlights_path) as f:
        data = yaml.safe_load(f) or {}
    clips = data.get("clips", _make_highlights())
    stems = [f"clip{i+1}" for i in range(len(clips))]
    export_dir = str(Path(kwargs.get("output_dir", "shorts")) / "2026-05-25_000000")
    return _create_fake_clips(export_dir, stems)


def stub_grade_video(clip_path: str, ref_path: str, output_path: str) -> str:
    """Stub for ref_grade.grade_video() — copies input to output."""
    _stub("ref_grade.grade_video", clip_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(Path(clip_path).read_bytes())
    return output_path


def stub_enhance_video(clip_path: str, ref_path: str, output_path: str, **kwargs) -> str:
    """Stub for face_mapper.enhance_video() — copies input to output."""
    _stub("face_mapper.enhance_video", clip_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(Path(clip_path).read_bytes())
    return output_path


def stub_process_all_seo(highlights_path: str, export_dir: str):
    """Stub for seo.process_all_seo() — writes fake metadata."""
    _stub("seo.process_all_seo", export_dir)
    d = Path(export_dir)
    for clip in sorted(d.glob("*.mp4")):
        meta = d / f"{clip.stem}_metadata.json"
        if not meta.exists():
            meta.write_text(json.dumps({
                "title": f"{clip.stem}: Highlight Reel",
                "description": "Amazing cricket moment from the match.",
                "tags": ["cricket", "ipl", "shorts", "highlight"],
                "seo_score": 0.85,
            }))


def stub_process_all_thumbnails(export_dir: str):
    """Stub for thumbnail.process_all_thumbnails() — creates fake JPEGs."""
    _stub("thumbnail.process_all_thumbnails", export_dir)
    d = Path(export_dir)
    for clip in sorted(d.glob("*.mp4")):
        thumb = d / f"{clip.stem}_thumb.jpg"
        if not thumb.exists():
            thumb.write_bytes(b"J" * 1024)


def stub_sync_to_drive(folder_path: str):
    """Stub for sync.sync_to_drive() — logs intent."""
    _stub("sync.sync_to_drive", folder_path)


def stub_download_from_drive(filenames=None, dest_dir: str = ""):
    """Stub for sync.download_from_drive() — creates fake files."""
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
    """Stub for upload.upload_video() — simulates success."""
    _stub("upload.upload_video", video_path)
    log.info("  [STUB]   privacy=%s publish_at=%s",
             kwargs.get("privacy"), kwargs.get("publish_at"))


def stub_generate_daily_insights():
    """Stub for analytics.generate_daily_insights() — logs intent."""
    _stub("analytics.generate_daily_insights")


def stub_assign_clips_to_slots(clips: list, **kwargs) -> list:
    """Stub for scheduler.assign_clips_to_slots() — returns (stem, dt) pairs."""
    from datetime import datetime, timedelta
    interval = kwargs.get("interval_hours", 1)
    base = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return [(stem, base + timedelta(hours=i)) for i, stem in enumerate(clips)]


def stub_format_for_youtube(dt) -> str:
    """Stub for scheduler.format_for_youtube()."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── Dry-run pipeline ─────────────────────────────────────────────────────────

def dry_run(url: str,
            skip_download=False, skip_transcribe=False,
            skip_highlight=False, skip_export=False,
            skip_sync=False, skip_seo=False,
            auto_sync=False, auto_upload=False,
            auto_schedule=False, sample_minutes=None,
            sync_from_drive=False, mode=None) -> dict:
    """Execute the full pipeline logic with all external calls stubbed."""

    from automation.config import load as load_config
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class PipelineResult:
        exported: list = field(default_factory=list)
        uploaded_count: int = 0
        failures: list = field(default_factory=list)
        total_seconds: float = 0.0
        transcript_source: str = "none"

    start = time.monotonic()
    result = PipelineResult()
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

    log.info("=" * 60)
    log.info("DRY RUN — All external calls will be STUBBED")
    log.info("URL: %s", url)
    log.info("Flags: skip_dl=%s skip_tx=%s skip_hl=%s skip_ex=%s skip_sync=%s skip_seo=%s "
             "sync=%s upload=%s schedule=%s mode=%s",
             skip_download, skip_transcribe, skip_highlight, skip_export,
             skip_sync, skip_seo, auto_sync, auto_upload, auto_schedule, mode)
    log.info("=" * 60)

    def _t(start: float) -> str:
        return "%.1fs" % (time.monotonic() - start)

    # ── Phase 0: YouTube transcript ─────────────────────────────────
    if not skip_transcribe:
        t0 = time.monotonic()
        log.info("[phase 0] YouTube transcript")
        transcript = stub_transcript_fetch(url, output_path=transcript_path)
        if transcript.get("segments"):
            skip_transcribe = True
            result.transcript_source = transcript.get("source", "api")
            log.info("[phase 0] done %s source=%s segments=%d",
                     _t(t0), result.transcript_source, len(transcript["segments"]))
        else:
            result.failures.append("phase0: no segments")

    # ── Phase 1: Download / Drive sync ──────────────────────────────
    if sync_from_drive:
        log.info("[phase 1] Pulling from Google Drive (STUBBED)")
        stub_download_from_drive(filenames=["video.mp4"], dest_dir=input_dir)
        stub_download_from_drive(filenames=[f"{stem}.json"], dest_dir=transcripts_dir)
        stub_download_from_drive(filenames=None, dest_dir=shorts_dir)
        skip_download = True
        skip_transcribe = True
        log.info("[phase 1] Drive sync complete (stubbed)")

    if not skip_download:
        t0 = time.monotonic()
        log.info("[phase 1] Download (STUBBED)")
        video_path = stub_download(url, video_path, sample_minutes=sample_minutes)
        stem = Path(video_path).stem
        transcript_path = str(Path(transcripts_dir) / f"{stem}.json")
        highlights_path = str(Path(highlights_dir) / f"{stem}.yaml")
        log.info("[phase 1] done %s", _t(t0))

    # ── Phase 2: Transcribe (Whisper) ──────────────────────────────
    if not skip_transcribe:
        t0 = time.monotonic()
        log.info("[phase 2] Transcribe (STUBBED)")
        stub_transcribe(video_path, transcript_path)
        log.info("[phase 2] done %s", _t(t0))

    # ── Phase 2.5: Video analysis ──────────────────────────────────
    log.info("[phase 2.5] Video analysis (STUBBED)")
    t0 = time.monotonic()
    analysis = stub_analyze_video(video_path)
    log.info("[phase 2.5] done %s face_rate=%.0f%% avg_quality=%.3f",
             _t(t0),
             analysis["summary"]["face_detection_rate"],
             analysis["summary"]["avg_quality"])

    # ── Phase 3: Highlight detection ──────────────────────────────
    if not skip_highlight:
        t0 = time.monotonic()
        log.info("[phase 3] Highlight detection (STUBBED)")
        highlights = stub_detect_highlights(transcript_path, video_path, highlights_path)
        log.info("[phase 3] done %s clips=%d", _t(t0), len(highlights))

    if not Path(highlights_path).exists():
        log.error("Highlights file missing: %s", highlights_path)
        result.failures.append("phase3: highlights file missing")
        result.total_seconds = time.monotonic() - start
        return {"exported": [], "uploaded_count": 0, "failures": result.failures,
                "total_seconds": result.total_seconds,
                "transcript_source": result.transcript_source,
                "stubbed_calls": _STUBBED_CALLS[:]}

    # ── Phase 4: Export shorts ────────────────────────────────────
    if not skip_export:
        t0 = time.monotonic()
        log.info("[phase 4] Export (STUBBED)")
        result.exported = stub_export_all(
            highlights_path, video_path,
            transcript_path=transcript_path, generate_seo=False,
        )
        log.info("[phase 4] done %s exported=%d", _t(t0), len(result.exported))
    else:
        existing = sorted(Path(shorts_dir).rglob("*.mp4")) if Path(shorts_dir).exists() else []
        existing = [p for p in existing if "test_output" not in p.name]
        if existing:
            latest_dir = max(set(p.parent for p in existing), key=lambda d: d.name)
            existing = sorted(latest_dir.glob("*.mp4"))
        if existing:
            result.exported = existing
            log.info("[phase 4] skipped — using %d existing clips", len(existing))
        else:
            log.warning("[phase 4] skipped — no .mp4 files found")

    # ── Phase 4.5: Enhancement ────────────────────────────────────
    if result.exported and mode:
        t0 = time.monotonic()
        log.info("[phase 4.5] Enhancement (%s) (STUBBED)", mode)
        ref_path = cfg.get("enhancement", {}).get("reference", "expectation.png")
        enhanced = []
        for clip_path in result.exported:
            if mode == "ref_grade":
                graded = str(Path(paths.get("temp", "temp")) / f"{clip_path.stem}_graded.mp4")
                r = stub_grade_video(str(clip_path), ref_path, graded)
                if r == graded and Path(graded).exists():
                    import shutil
                    shutil.move(graded, str(clip_path))
                    enhanced.append(clip_path)
                    log.info("[%s] Color grade applied", clip_path.stem)
                else:
                    enhanced.append(clip_path)
            elif mode == "face_mapper":
                enhanced_path = str(Path(paths.get("temp", "temp")) / f"{clip_path.stem}_enhanced.mp4")
                r = stub_enhance_video(str(clip_path), ref_path, enhanced_path, use_region_grading=True)
                if r == enhanced_path and Path(enhanced_path).exists():
                    import shutil
                    shutil.move(enhanced_path, str(clip_path))
                    enhanced.append(clip_path)
                    log.info("[%s] Face-mapper applied", clip_path.stem)
                else:
                    enhanced.append(clip_path)
        result.exported = enhanced
        log.info("[phase 4.5] done %s enhanced=%d", _t(t0), len(result.exported))

    # ── Phase 5: SEO ─────────────────────────────────────────────
    if not skip_seo and result.exported:
        t0 = time.monotonic()
        log.info("[phase 5] SEO (STUBBED)")
        export_dir = str(result.exported[0].parent)
        stub_process_all_seo(highlights_path, export_dir)
        log.info("[phase 5] done %s", _t(t0))

    # ── Phase 5.5: Thumbnails ────────────────────────────────────
    if result.exported:
        t0 = time.monotonic()
        log.info("[phase 5.5] Thumbnails (STUBBED)")
        export_dir = str(result.exported[0].parent)
        stub_process_all_thumbnails(export_dir)
        log.info("[phase 5.5] done %s", _t(t0))

    # ── Phase 6: Sync to Drive ─────────────────────────────────────
    do_sync = auto_sync and not skip_sync
    if do_sync and result.exported:
        t0 = time.monotonic()
        log.info("[phase 6] Sync to Drive (STUBBED)")
        export_folder = str(result.exported[0].parent)
        stub_sync_to_drive(export_folder)
        log.info("[phase 6] done %s", _t(t0))
    elif skip_sync:
        log.info("[phase 6] Skipping sync")

    # ── Phase 7: Upload to YouTube ───────────────────────────────
    if (auto_upload or auto_schedule) and result.exported:
        t0 = time.monotonic()
        log.info("[phase 7] Upload (STUBBED)")
        privacy = cfg.get("youtube", {}).get("privacy_status", "private")
        interval = cfg.get("upload_schedule", {}).get(
            "interval_hours",
            cfg.get("youtube", {}).get("schedule_interval_hours", 1)
        )

        slot_map = {}
        if auto_schedule:
            clip_scores = {}
            for clip_path in result.exported:
                meta_path = clip_path.with_name(f"{clip_path.stem}_metadata.json")
                if meta_path.exists():
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                        score = meta.get("quality_score") or meta.get("seo_score") or 0.0
                        clip_scores[clip_path.stem] = float(score)
                    except (ValueError, OSError):
                        pass
            assignments = stub_assign_clips_to_slots(
                clips=[p.stem for p in result.exported],
                interval_hours=interval,
                clip_scores=clip_scores if clip_scores else None,
            )
            slot_map = {stem: dt for stem, dt in assignments}
            log.info("Schedule generated for %d clips", len(assignments))

        for clip_path in result.exported:
            meta_path = clip_path.with_name(f"{clip_path.stem}_metadata.json")
            if not meta_path.exists():
                log.warning("No metadata for %s — skipping upload", clip_path.name)
                continue
            publish_at = None
            if auto_schedule:
                slot = slot_map.get(clip_path.stem)
                if slot:
                    publish_at = stub_format_for_youtube(slot)
            stub_upload_video(
                str(clip_path), str(meta_path),
                privacy=privacy, publish_at=publish_at,
            )
            result.uploaded_count += 1

        log.info("[phase 7] done %s uploaded=%d", _t(t0), result.uploaded_count)

    # ── Phase 8: Analytics ─────────────────────────────────────────
    if auto_upload:
        t0 = time.monotonic()
        log.info("[phase 8] Analytics (STUBBED)")
        stub_generate_daily_insights()
        log.info("[phase 8] done %s", _t(t0))

    result.total_seconds = time.monotonic() - start
    log.info("[EXIT] pipeline url=%s exported=%d uploaded=%d failures=%d elapsed=%.1fs transcript=%s",
             url, len(result.exported), result.uploaded_count, len(result.failures),
             result.total_seconds, result.transcript_source)
    if result.failures:
        for f in result.failures:
            log.warning("  failure: %s", f)

    return {
        "exported": [str(p) for p in result.exported],
        "uploaded_count": result.uploaded_count,
        "failures": result.failures,
        "total_seconds": result.total_seconds,
        "transcript_source": result.transcript_source,
        "stubbed_calls": _STUBBED_CALLS[:],
    }


def list_external_calls():
    """Print all external calls that dry_run.py stubs."""
    calls = [
        ("transcript.fetch", "YouTube Transcript API / yt-dlp subtitle download"),
        ("download.download", "yt-dlp video download"),
        ("transcribe.transcribe", "faster-whisper GPU transcription"),
        ("video_analyzer.analyze_video", "OpenCV / ffmpeg / face_recognition frame sampling"),
        ("highlight.detect_highlights", "ffmpeg audio energy + AI refinement via Groq"),
        ("export.export_all", "ffmpeg clip cutting + encoding"),
        ("ref_grade.grade_video", "ffmpeg color grading pipeline"),
        ("face_mapper.enhance_video", "per-frame face enhancement pipeline"),
        ("seo.process_all_seo", "Groq LLM title/desc/tag generation"),
        ("thumbnail.process_all_thumbnails", "Pillow image compositing + font rendering"),
        ("sync.sync_to_drive", "Google Drive API upload"),
        ("sync.download_from_drive", "Google Drive API download"),
        ("upload.upload_video", "YouTube Data API v3 upload"),
        ("analytics.generate_daily_insights", "YouTube Data API analytics"),
        ("scheduler.assign_clips_to_slots", "datetime scheduling + prime-time logic"),
    ]
    print("\nStubbed External Calls:")
    print("=" * 60)
    for name, desc in calls:
        print(f"  {name:<35s} {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Dry-run pipeline — stub ALL external calls, test phase logic locally.",
    )
    parser.add_argument("url", nargs="?", default="https://youtu.be/dQw4w9WgXcQ",
                        help="YouTube URL (ignored; stubbed)")
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
    parser.add_argument("--list-external", action="store_true",
                        help="List all stubbed external calls and exit")
    args = parser.parse_args()

    if args.list_external:
        list_external_calls()
        return

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

    print()
    print("=" * 60)
    print("DRY RUN COMPLETE")
    print("=" * 60)
    print(f"  Exported:           {len(result['exported'])} clips")
    print(f"  Uploaded:           {result['uploaded_count']}")
    print(f"  Failures:           {len(result['failures'])}")
    print(f"  Elapsed:            {result['total_seconds']:.1f}s")
    print(f"  Transcript source:  {result['transcript_source']}")
    print(f"  Stubbed calls:      {len(result['stubbed_calls'])}")
    if result["failures"]:
        print(f"  Failures detail:")
        for f in result["failures"]:
            print(f"    - {f}")
    print()


if __name__ == "__main__":
    main()
