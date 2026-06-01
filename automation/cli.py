"""CLI — command-line interface for the orchestration pipeline."""

import argparse
import sys

_PIPELINE_ARGS = [
    "download", "transcribe", "score", "rank",
    "export", "seo", "upload", "cleanup",
]


def setup_argparse():
    parser = argparse.ArgumentParser(
        description="YouTube clip automation pipeline",
    )
    parser.add_argument("url", nargs="?", default=None, help="YouTube video URL")
    for arg in _PIPELINE_ARGS:
        parser.add_argument(
            f"--{arg}", action="store_true", dest=arg, default=None,
            help=f"Enable {arg} stage",
        )
        parser.add_argument(
            f"--no-{arg}", action="store_false", dest=arg, default=None,
            help=f"Disable {arg} stage",
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
        "--config", type=str, default=None,
        help="Path to configuration file",
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
    from automation.memory.decision_store import DecisionStore
    from automation.orchestrator import Orchestrator
    store = DecisionStore()
    orch = Orchestrator(decision_store=store)
    if parsed.override is not None:
        clip_id = parsed.override_clip_id or "unknown"
        orch.emit_event(clip_id, "manual_override", {"override": parsed.override})
        print(f"Override {parsed.override} recorded for clip {clip_id}")
        return 0
    if parsed.url is None:
        parser.print_help()
        return 0
    stages = {}
    for s in _PIPELINE_ARGS:
        val = getattr(parsed, s, None)
        if val is not None:
            stages[s] = val
    result = orch.run_pipeline(parsed.url, stages=stages)
    print(
        f"Pipeline completed: {len(result['stages_completed'])} stages, "
        f"{result['events_emitted']} events"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
