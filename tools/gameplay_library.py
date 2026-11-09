"""Inspect a local gameplay manifest; never download, approve or upload footage.

python -m tools.gameplay_library input/gameplay/library.json --seconds 38
python -m tools.gameplay_library --hash input/gameplay/ea-cricket-07.mp4
"""

import argparse
import json
import subprocess

from automation.cricket_gameplay import load_delivery_library, load_render_library, select_gameplay, sha256
from automation.gameplay_export import preflight_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?")
    parser.add_argument("--seconds", type=float, default=38)
    parser.add_argument("--hash", dest="hash_path")
    parser.add_argument("--local-review", action="store_true",
                        help="Also accept legacy private-review manifests; never grant release approval")
    parser.add_argument("--composition", choices=("cricket-gameplay-split-v1", "cricket-gameplay-stadium-v1"),
                        default="cricket-gameplay-stadium-v1")
    args = parser.parse_args()
    try:
        if args.hash_path:
            print(sha256(args.hash_path))
            return 0
        if not args.manifest:
            parser.error("provide a manifest or --hash FILE")
        loader = load_render_library if args.local_review else load_delivery_library
        library = preflight_library(loader(args.manifest), args.composition,
                                    required_seconds=args.seconds)
        selected = select_gameplay(library, args.seconds, "preflight", preferred_game="EA Cricket 07")
        print(json.dumps({"status": "LOCAL_PREFLIGHT_PASS", "usage": library.get("usage", "PUBLICATION_REVIEW_REQUIRED"),
            "asset_count": len(library["assets"]),
            "required_seconds": args.seconds, "example_asset": selected["asset"]["id"],
            "release_approval": "NOT GRANTED; source and finished media still need review"}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Gameplay held: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
