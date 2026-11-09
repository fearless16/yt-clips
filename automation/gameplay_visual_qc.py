"""Frame-complete dark-transition screening, separate from rights approval.

Decode every frame of the declared crop, at a small spatial resolution. HUD
pixels must not hide a black pitch. This detects darkness, not semantic quality,
logos, frozen action, or permission to expand a manifest's reviewed ranges.
"""

import math
import re
import subprocess

VERSION = 1
GUARD_SECONDS = .3
_BLACK = re.compile(r"black_start:([\d.]+) black_end:([\d.]+) black_duration:([\d.]+)")


def scan_dark_intervals(path, start, end, crop=None):
    if not math.isfinite(start + end) or not 0 <= start < end:
        raise ValueError("Invalid gameplay visual scan interval")
    filters = []
    if crop is not None:
        if (len(crop) != 4 or any(type(v) is not int for v in crop)
                or min(crop[:2]) < 0 or min(crop[2:]) <= 0):
            raise ValueError("Invalid gameplay visual scan crop")
        x, y, width, height = crop
        filters.append(f"crop={width}:{height}:{x}:{y}")
    filters += ["scale=160:90", "blackdetect=d=0:pix_th=0.08:pic_th=0.90"]
    result = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-xerror",
        "-ss", f"{start:.6f}", "-t", f"{end-start:.6f}", "-i", str(path),
        "-map", "0:v:0", "-an", "-vf", ",".join(filters), "-f", "null", "-"],
        check=True, capture_output=True, text=True, timeout=180)
    # A successful command with no decoded frames is not a clean scan.
    if not re.search(r"frame=\s*[1-9]\d*", result.stderr):
        raise ValueError("Gameplay visual scan decoded no frames")
    intervals = []
    for left, right, _duration in _BLACK.findall(result.stderr):
        left, right = max(start, start+float(left)), min(end, start+float(right))
        if left <= right:
            intervals.append([left, right])
    return intervals


def clean_ranges(ranges, dark_intervals, guard=GUARD_SECONDS):
    """Subtract guarded transitions, never add or join reviewed intervals."""
    result = []
    for start, end in ranges:
        cursor = start
        for left, right in sorted(dark_intervals):
            left, right = max(start, left-guard), min(end, right+guard)
            if right <= cursor or left >= end:
                continue
            if left > cursor:
                result.append([cursor, left])
            cursor = max(cursor, right)
        if end > cursor:
            result.append([cursor, end])
    return result


def screen_library(library):
    screens = {}
    for asset in library["assets"]:
        dark = []
        for start, end in asset["approved_ranges"]:
            dark.extend(scan_dark_intervals(asset["path"], start, end, asset.get("crop")))
        screens[asset["id"]] = {"version": VERSION, "method": "every-frame cropped blackdetect",
            "media_sha256": asset["sha256"], "crop": asset.get("crop"),
            "pixel_threshold": .08, "dark_picture_fraction": .90,
            "guard_seconds": GUARD_SECONDS, "dark_intervals": dark,
            "clean_ranges": clean_ranges(asset["approved_ranges"], dark)}
    library["visual_screening"] = screens
    return library
