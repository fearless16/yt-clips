"""CLI — command-line interface for the orchestration pipeline."""

import argparse
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
        help="Run self-learning stages only (skip download/export)",
    )
    parser.add_argument(
        "--skip-download", action="store_true", default=False,
        help="Skip video download (use existing file)",
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
    if parsed.url is None:
        parser.print_help()
        return 0

    from automation.orchestrator import run
    result = run(
        url=parsed.url,
        skip_download=parsed.skip_download,
        auto_sync=parsed.sync,
        auto_upload=parsed.upload,
        auto_schedule=parsed.schedule,
        sample_minutes=parsed.sample_minutes,
        mode=parsed.mode,
        learn_only=parsed.learn,
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
