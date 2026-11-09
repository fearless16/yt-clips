"""Bind an explicitly approved JPEG to a delivery MP4, locally and without upload.

python -m tools.stage_thumbnail VIDEO APPROVED_JPEG --sha256 APPROVED_HASH
Preserves any different existing cover; changing covers requires new review.
"""
import argparse

from automation.youtube_delivery import stage_thumbnail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("thumbnail")
    parser.add_argument("--sha256", required=True, help="Digest of the owner-approved artwork")
    args = parser.parse_args()
    try:
        print(stage_thumbnail(args.video, args.thumbnail, expected_sha256=args.sha256))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Thumbnail staging held: {exc}\n")


if __name__ == "__main__":
    main()
