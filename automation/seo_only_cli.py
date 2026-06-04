"""CLI entry point: python -m automation.seo_only <clips_dir>"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate SEO metadata for pre-exported clips (Mac-side, no GPU)",
    )
    parser.add_argument("clips_dir", help="Directory containing exported .mp4 clips")
    parser.add_argument("--highlights", help="Highlights YAML path")
    parser.add_argument("--transcript", help="Transcript JSON path")
    parser.add_argument("--title", default="", help="Video title for context")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-generate SEO even if metadata exists")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Seconds between API calls (default: 2)")

    args = parser.parse_args()

    from automation.seo_only import run_seo_only
    result = run_seo_only(
        clips_dir=args.clips_dir,
        highlights_yaml=args.highlights,
        transcript_json=args.transcript,
        video_title=args.title,
        skip_existing=not args.no_skip,
        inter_clip_sleep=args.sleep,
    )

    print(f"\nSEO COMPLETE: {result['processed']} processed, {result['failed']} failed")
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
