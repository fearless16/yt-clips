"""CLI — command-line interface for the orchestration pipeline."""
import _fix_encoding  # noqa: F401 — force UTF-8 on Windows cp1252

import argparse
import json
import sys


def setup_argparse():
    parser = argparse.ArgumentParser(
        description="YouTube clip automation pipeline",
    )
    parser.add_argument("url", nargs="?", default=None, help="YouTube video URL")
    parser.add_argument(
        "--upload", action="store_true", default=False,
        help="Enable YouTube upload after export",
    )
    parser.add_argument(
        "--sync", action="store_true", default=False,
        help="Enable Drive sync after export",
    )
    parser.add_argument(
        "--schedule", action="store_true", default=False,
        help="Enable scheduled upload with time slots",
    )
    parser.add_argument(
        "--learn", action="store_true", default=False,
        help="Sync the channel Shorts shelf, Analytics outcomes, and refit the canonical model",
    )
    parser.add_argument(
        "--skip-download", action="store_true", default=False,
        help="Skip video download (use existing file)",
    )
    parser.add_argument(
        "--skip-transcribe", action="store_true", default=False,
        help="Skip transcript fetch (use cached transcript)",
    )
    parser.add_argument(
        "--skip-highlight", action="store_true", default=False,
        help="Skip highlight detection (use cached highlights)",
    )
    parser.add_argument(
        "--skip-export", action="store_true", default=False,
        help="Skip clip export (use existing exported clips)",
    )
    parser.add_argument(
        "--skip-seo", action="store_true", default=False,
        help="Skip SEO generation (use existing metadata)",
    )
    parser.add_argument(
        "--skip-instagram", action="store_true", default=False,
        help="Skip Instagram Reel publish (sets YT_CLIPS_SKIP_INSTAGRAM=1)",
    )
    parser.add_argument(
        "--sample-minutes", type=int, default=None,
        help="Download only first N minutes of video",
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        help="Enhancement mode: ref_grade or face_mapper",
    )
    parser.add_argument(
        "--skip-enhancement", action="store_true", default=False,
        help="Skip enhancement phase (stage 6b)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print what would be done without executing",
    )
    parser.add_argument(
        "--override", choices=["keep", "reject", "rerank"], default=None,
        help="Human override action",
    )
    parser.add_argument(
        "--override-clip-id", type=str, default=None,
        help="Clip ID for the override",
    )
    parser.add_argument(
        "--memory-report", action="store_true", default=False,
        help="Print memory usage report",
    )
    parser.add_argument(
        "--status", action="store_true", default=False,
        help="Print pipeline status",
    )
    parser.add_argument(
        "--version", action="store_true", default=False,
        help="Print version and exit",
    )
    return parser


def main(args=None):
    parser = setup_argparse()
    try:
        parsed = parser.parse_args(args)
    except SystemExit as e:
        if e.code != 0:
            return 1
        return 0
    if parsed.version:
        from automation import VERSION
        print(VERSION)
        return 0
    if parsed.memory_report:
        print("Memory: OK")
        return 0
    if parsed.status:
        print("Pipeline: idle")
        return 0
    if parsed.dry_run:
        print("Dry run: no pipeline execution")
        return 0
    if parsed.override is not None:
        from automation.memory.decision_store import DecisionStore
        from automation.orchestrator import Orchestrator
        clip_id = parsed.override_clip_id or "unknown"
        orch = Orchestrator(decision_store=DecisionStore())
        orch.emit_event(clip_id, "manual_override", {"override": parsed.override})
        print(f"Override {parsed.override} recorded for clip {clip_id}")
        return 0

    if parsed.learn:
        from automation.seo.analytics import (
            generate_daily_insights,
            sync_clip_performance_from_youtube,
        )
        try:
            added = sync_clip_performance_from_youtube(config_path="config.yaml")
            result = generate_daily_insights(config_path="config.yaml")
            print(json.dumps(
                {"snapshots_added": added, **result},
                ensure_ascii=False,
                separators=(",", ":"),
            ))
            return 0
        except Exception as exc:
            print(json.dumps(
                {"status": "failed", "error": str(exc)},
                separators=(",", ":"),
            ))
            return 1

    if parsed.url is None:
        parser.print_help()
        return 0

    import os
    if parsed.skip_instagram:
        os.environ["YT_CLIPS_SKIP_INSTAGRAM"] = "1"

    from automation.orchestrator import run
    result = run(
        url=parsed.url,
        skip_download=parsed.skip_download,
        skip_transcribe=parsed.skip_transcribe,
        skip_highlight=parsed.skip_highlight,
        skip_export=parsed.skip_export,
        skip_seo=parsed.skip_seo,
        skip_enhancement=parsed.skip_enhancement,
        auto_sync=parsed.sync,
        auto_upload=parsed.upload,
        auto_schedule=parsed.schedule,
        sample_minutes=parsed.sample_minutes,
        mode=parsed.mode,
    )
    n = len(result.exported)
    f = len(result.failures)
    print(
        f"Pipeline done: {n} clips exported, "
        f"{result.uploaded_count} uploaded, "
        f"{f} failure(s) in {result.total_seconds:.1f}s"
    )
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
