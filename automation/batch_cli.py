"""CLI entry point: python -m automation.batch <url1> <url2> ..."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Batch pipeline — process multiple YouTube URLs in one Colab session",
    )
    parser.add_argument("urls", nargs="*", help="YouTube URLs to process")
    parser.add_argument("--file", "-f", help="File with URLs (one per line)")
    parser.add_argument("--top", type=int, default=15, help="Top N clips to export (default: 15)")
    parser.add_argument("--upload", action="store_true", help="Upload after export")
    parser.add_argument("--schedule", action="store_true", help="Schedule uploads")
    parser.add_argument("--skip-seo", action="store_true", help="Skip SEO (run on Mac later)")
    parser.add_argument("--checkpoint", help="Checkpoint file path (for resume)")

    args = parser.parse_args()

    urls = list(args.urls or [])
    if args.file:
        from pathlib import Path
        f = Path(args.file)
        if f.exists():
            urls.extend(line.strip() for line in f.read_text().splitlines() if line.strip())

    if not urls:
        parser.print_help()
        return 1

    from automation.batch import run_batch
    result = run_batch(
        urls=urls,
        top_n=args.top,
        auto_upload=args.upload,
        auto_schedule=args.schedule,
        skip_seo=args.skip_seo,
        checkpoint_path=args.checkpoint,
    )

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"  Downloaded:  {result.downloaded}")
    print(f"  Transcribed: {result.transcribed}")
    print(f"  Highlights:  {result.highlighted}")
    print(f"  Exported:    {result.exported}")
    print(f"  Uploaded:    {result.uploaded}")
    print(f"  Failures:    {len(result.failures)}")
    print(f"  Elapsed:     {result.elapsed:.1f}s")
    if result.failures:
        print(f"\nFAILURES:")
        for f in result.failures:
            print(f"  ⚠ {f}")
    print(f"{'='*60}")

    return 1 if result.failures else 0


if __name__ == "__main__":
    sys.exit(main())
