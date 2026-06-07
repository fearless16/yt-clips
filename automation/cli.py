"""CLI — command-line interface for the orchestration pipeline."""

import argparse
import json
import sys
from pathlib import Path


CRED_FILES = {
    "cookies.txt", ".env", "client_secrets.json",
    "yt_channel_token.json", "yt_analytics_token.json",
    "drive_token.json",
}


def _submit_remote(url: str, tunnel_url: str, dry_run: bool = False) -> int:
    """Submit a pipeline job to a remote Colab instance via tunnel."""
    import ssl
    import urllib.request

    ctx = ssl._create_unverified_context()

    job = {"url": url, "flags": []}

    if dry_run:
        for cred_file in CRED_FILES:
            p = Path(cred_file)
            if p.exists():
                job[cred_file] = f"<{p.stat().st_size} bytes>"
        print(f"[DRY RUN] Would POST to {tunnel_url}/job")
        print(f"Payload ({len(job)} keys):")
        for k, v in job.items():
            if k == "url":
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")
        return 0

    for cred_file in CRED_FILES:
        p = Path(cred_file)
        if p.exists():
            job[cred_file] = p.read_text(encoding="utf-8")

    body = json.dumps(job).encode()
    try:
        r = urllib.request.urlopen(
            f"{tunnel_url}/job", data=body, timeout=30, context=ctx,
        )
        if r.status == 202:
            print(f"Job submitted to {tunnel_url}")
            return 0
        resp = json.loads(r.read().decode())
        print(f"Tunnel returned {r.status}: {resp.get('error', 'unknown')}")
        return 1
    except Exception as e:
        print(f"Failed to submit via tunnel: {e}")
        return 1


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
    parser.add_argument(
        "--remote", action="store_true", default=False,
        help="Send job to remote Colab instance via tunnel",
    )
    parser.add_argument(
        "--tunnel-url", type=str, default=None,
        help="Public tunnel URL of the Colab watcher (required with --remote)",
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
    if parsed.remote:
        if not parsed.tunnel_url:
            print("--remote requires --tunnel-url <URL>")
            return 1
        return _submit_remote(parsed.url, parsed.tunnel_url, dry_run=parsed.dry_run)

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
        skip_transcribe=parsed.skip_transcribe,
        skip_highlight=parsed.skip_highlight,
        skip_export=parsed.skip_export,
        skip_seo=parsed.skip_seo,
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
