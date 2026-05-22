"""cli.py — CLI entry point for yt-clips automation.

Modes:
    python -m automation.cli <url>                   # local pipeline
    python -m automation.cli <url> --sync --upload    # with sync + upload
    python -m automation.cli --memory-report         # RAM snapshot
    python -m automation.cli --gpu-info              # GPU info
    python -m automation.cli --tunnel-status         # tunnel health
    python -m automation.cli --fetch-transcript <url> # transcript only
    python -m automation.cli --setup-colab           # Colab setup + tunnel
    python -m automation.cli --sync-only             # sync to Drive
    python -m automation.cli --auto-pilot <url>      # channel watcher
    python -m automation.cli --remote <url>          # beam to Colab
"""

import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("cli")


def main():
    p = argparse.ArgumentParser(description="yt-clips pipeline")
    p.add_argument("url", nargs="?")
    p.add_argument("--memory-report", action="store_true")
    p.add_argument("--gpu-info", action="store_true")
    p.add_argument("--setup-colab", action="store_true")
    p.add_argument("--fetch-transcript", metavar="URL")
    p.add_argument("--tunnel-status", action="store_true")
    p.add_argument("--sync-only", action="store_true")
    p.add_argument("--auto-pilot", metavar="CHANNEL_URL")
    p.add_argument("--remote", metavar="URL", help="Beam job to Colab tunnel")
    p.add_argument("--sync", action="store_true")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--schedule", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-transcribe", action="store_true")
    p.add_argument("--skip-highlight", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-seo", action="store_true")
    p.add_argument("--sample-minutes", type=int)
    p.add_argument("--sync-from-drive", action="store_true")
    p.add_argument("--mode")
    args = p.parse_args()

    actions = [
        args.memory_report, args.gpu_info, args.setup_colab,
        bool(args.fetch_transcript), args.tunnel_status, args.sync_only,
        bool(args.auto_pilot), bool(args.remote),
    ]
    if not any(actions) and not args.url:
        p.print_help()
        sys.exit(1)

    if args.memory_report:
        from .memory import memory_report
        for k, v in memory_report().items():
            print(f"  {k}: {v}")
        return

    if args.gpu_info:
        from .env import gpu_info
        for k, v in gpu_info().items():
            print(f"  {k}: {v}")
        return

    if args.setup_colab:
        from .env import setup
        from .tunnel import start_tunnel
        s = setup()
        u = start_tunnel()
        print(f"Colab: {s['status']} gpu={s['gpu']['name']} tunnel={u}")
        return

    if args.fetch_transcript:
        from .transcript import fetch
        t = fetch(args.fetch_transcript)
        print(f"Transcript: {len(t.get('segments', []))} segs source={t.get('source')}")
        return

    if args.tunnel_status:
        from .tunnel import tunnel_status
        for k, v in tunnel_status().items():
            print(f"  {k}: {v}")
        return

    if args.sync_only:
        from sync import sync_to_drive
        sync_to_drive(folder_path="shorts/")
        print("Sync done")
        return

    if args.auto_pilot:
        from channel_watcher import monitor
        monitor(args.auto_pilot)
        return

    if args.remote:
        from bridge import push_job
        flags = [f"--{f}" for f in ["sync", "upload", "schedule"] if getattr(args, f.replace("-", "_"))]
        push_job(args.remote, flags)
        return

    # ── Local pipeline ──
    from .memory import ensure_free, memory_report
    from .orchestrator import run

    r = memory_report()
    if r.get("free_gb", 0) < 2.0:
        log.warning(f"Low mem: {r['free_gb']}GB free")
        ensure_free(2.0, timeout=30.0)

    result = run(
        url=args.url,
        skip_download=args.skip_download, skip_transcribe=args.skip_transcribe,
        skip_highlight=args.skip_highlight, skip_export=args.skip_export,
        skip_seo=args.skip_seo, auto_sync=args.sync, auto_upload=args.upload,
        auto_schedule=args.schedule, sample_minutes=args.sample_minutes,
        sync_from_drive=args.sync_from_drive, mode=args.mode,
    )
    print(f"Done: exported={len(result.exported)} uploaded={result.uploaded_count}"
          f" failures={len(result.failures)} in {result.total_seconds:.1f}s"
          f" transcript={result.transcript_source}")


if __name__ == "__main__":
    main()
