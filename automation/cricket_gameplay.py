"""Cricket gameplay provenance and seeded continuous stock ranges.

An approval records a human rights decision; it is not a legal certification.
Publisher permission and permission for the recording are separate requirements.
Enabled stock can enter delivery without inventing rights or owner-release approval.
No network, acquisition, looping, or learning database.
"""

import copy
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

COMPOSITION = "cricket-gameplay-split-v1"
STADIUM_COMPOSITION = "cricket-gameplay-stadium-v1"
GAMEPLAY_COMPOSITIONS = (COMPOSITION, STADIUM_COMPOSITION)
RC24_DISCLAIMER = ("This video is subject to the Game’s Terms of Service and was not "
                   "endorsed or sponsored by KRAFTON, Inc.")
EA_DISCLAIMER = "This project is not endorsed by or affiliated with EA or its licensors."


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Gameplay {name} is required")
    return value.strip()


def _hashed_file(base, relative, expected):
    path = (base / _text(relative, "evidence/media path")).resolve()
    if (not isinstance(expected, str) or len(expected) != 64 or not path.is_file()
            or sha256(path) != expected):
        raise ValueError(f"Gameplay file missing or hash changed: {path}")
    return str(path)


def load_library(manifest):
    """Fail closed on missing, withdrawn, malformed or unreviewed local assets."""
    path = Path(manifest).resolve()
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Gameplay manifest unavailable: {path}") from exc
    if (not isinstance(payload, dict) or payload.get("schema_version") != 1
            or not isinstance(payload.get("assets"), list) or not payload["assets"]):
        raise ValueError("Gameplay manifest needs schema_version 1 and approved assets")
    assets, ids = [], set()
    for original in payload["assets"]:
        if not isinstance(original, dict):
            raise ValueError("Gameplay asset must be an object")
        asset = copy.deepcopy(original)
        identifier = _text(asset.get("id"), "asset ID")
        if identifier in ids:
            raise ValueError("Duplicate gameplay asset ID")
        ids.add(identifier)
        for key in ("game", "publisher", "reviewer", "reviewed_at", "attribution"):
            _text(asset.get(key), key)
        date.fromisoformat(asset["reviewed_at"])
        if (asset.get("status") != "APPROVED" or asset.get("modified") is not False
                or asset.get("third_party_visuals_cleared") is not True):
            raise ValueError("Gameplay needs APPROVED unmodified recording and cleared third-party visuals")
        asset["path"] = _hashed_file(path.parent, asset.get("path"), asset.get("sha256"))
        for key, kinds in (("publisher_permission", {"publisher_policy", "written_authorization"}),
                           ("recording_permission", {"self_recorded", "written_authorization", "cc_by"})):
            grant = asset.get(key)
            if (not isinstance(grant, dict) or grant.get("kind") not in kinds
                    or grant.get("commercial_youtube") is not True or grant.get("editing") is not True
                    or grant.get("split_screen_background") is not True):
                raise ValueError(f"Gameplay {key} must explicitly cover editing, commercial YouTube and split-screen background use")
            grant["evidence"] = _hashed_file(path.parent, grant.get("evidence"), grant.get("sha256"))
        publisher = asset["publisher_permission"]
        if publisher["kind"] == "publisher_policy":
            # The only researched general policy in this version. Other games
            # require actual written permission, not a copied 'no copyright' tag.
            policy = urlparse(_text(publisher.get("url"), "publisher policy URL"))
            if (asset["publisher"] != "EA" or asset["game"] != "EA Cricket 07"
                    or policy.scheme != "https" or policy.hostname != "help.ea.com"
                    or policy.path.rstrip("/") != "/en/articles/security-and-rules/ea-content-policy"
                    or publisher.get("no_third_party_promotion") is not True):
                raise ValueError("Unreviewed publisher policy; written authorization required (including RC24)")
        _text(asset["recording_permission"].get("owner"), "recording owner")
        _validate_ranges_and_crop(asset)
        assets.append(asset)
    return {"manifest": str(path), "manifest_sha256": hashlib.sha256(raw).hexdigest(), "assets": assets}


def _validate_ranges_and_crop(asset):
    ranges = asset.get("approved_ranges")
    crop = asset.get("crop")
    if crop is not None:
        if (not isinstance(crop, list) or len(crop) != 4
                or any(type(v) is not int or v % 2 for v in crop)
                or min(crop[:2]) < 0 or min(crop[2:]) <= 0
                or crop[2]*9 != crop[3]*16):
            raise ValueError("Gameplay crop must be even, nonnegative x/y and positive exact 16:9 width/height")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("Gameplay needs approved continuous ranges")
    previous = 0
    for interval in ranges:
        if (not isinstance(interval, list) or len(interval) != 2
                or any(type(t) not in (int, float) or not math.isfinite(t) for t in interval)):
            raise ValueError("Invalid gameplay approved range")
        start, end = interval
        if start < previous or end <= start:
            raise ValueError("Unordered, overlapping or invalid gameplay ranges")
        previous = end


def load_render_library(manifest):
    """Load enabled stock, legacy drafts or explicitly cleared recordings.

    STOCK assets retain their recorded rights status. Owner release review, not
    this loader, records approval of the finished clip. Legacy LOCAL_REVIEW_ONLY
    libraries remain drafts until explicitly enabled. Ranges are visual bounds,
    not a rights grant.
    """
    path = Path(manifest).resolve()
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Gameplay manifest unavailable: {path}") from exc
    if not isinstance(payload, dict) or payload.get("usage") not in ("STOCK", "LOCAL_REVIEW_ONLY"):
        return load_library(manifest)
    usage = payload["usage"]
    expected_status = "AVAILABLE" if usage == "STOCK" else "REVIEW_ONLY"
    if (payload.get("schema_version") != 1 or not isinstance(payload.get("assets"), list)
            or not payload["assets"]):
        raise ValueError("Gameplay stock manifest needs schema_version 1 and assets")
    assets, ids = [], set()
    for original in payload["assets"]:
        if not isinstance(original, dict):
            raise ValueError("Gameplay asset must be an object")
        asset = copy.deepcopy(original)
        identifier = _text(asset.get("id"), "asset ID")
        if identifier in ids:
            raise ValueError("Duplicate gameplay asset ID")
        ids.add(identifier)
        if asset.get("status") != expected_status or asset.get("rights_status") != "UNRESOLVED":
            raise ValueError(f"Gameplay {usage} assets must retain {expected_status} / UNRESOLVED status")
        for key in ("game", "publisher", "reviewer", "reviewed_at", "attribution", "source_video_id", "owner"):
            _text(asset.get(key), key)
        date.fromisoformat(asset["reviewed_at"])
        asset["path"] = _hashed_file(path.parent, asset.get("path"), asset.get("sha256"))
        grant = asset.get("license_evidence")
        if not isinstance(grant, dict):
            raise ValueError("Gameplay stock needs recording license evidence")
        grant["path"] = _hashed_file(path.parent, grant.get("path"), grant.get("sha256"))
        evidence = json.loads(Path(grant["path"]).read_text(encoding="utf-8"))
        if (not isinstance(evidence, dict) or evidence.get("id") != asset["source_video_id"]
                or evidence.get("uploader") != asset["owner"]
                or evidence.get("license") != "Creative Commons Attribution license (reuse allowed)"):
            raise ValueError("Gameplay recording identity or Creative Commons label not verified")
        _validate_ranges_and_crop(asset)
        assets.append(asset)
    return {"manifest": str(path), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "assets": assets, "usage": usage}


def load_delivery_library(manifest):
    """Enabled stock is deliverable; legacy private previews are not promoted."""
    library = load_render_library(manifest)
    if library.get("usage") == "LOCAL_REVIEW_ONLY":
        raise ValueError("Local-review gameplay supports draft rendering only, not upload or Drive sync; "
                         "explicitly enable the stock library first")
    return library


def select_gameplay(library, duration, seed, preferred_game=None):
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid gameplay duration")
    screens = library.get("visual_screening", {})
    options = [(asset, start, end) for asset in sorted(library["assets"], key=lambda a: a["id"])
               for start, end in (screens[asset["id"]]["clean_ranges"] if asset["id"] in screens
                                  else asset["approved_ranges"]) if end-start >= duration]
    if not options:
        raise ValueError(f"No approved continuous gameplay range covers {duration:.3f}s after visual screening; no looping or fallback")
    preferred = [item for item in options if item[0]["game"] == preferred_game]
    if preferred:
        options = preferred
    value = int(hashlib.sha256(str(seed).encode()).hexdigest(), 16)
    asset, start, end = options[value % len(options)]
    # Millisecond offsets are reproducible and remain strictly inside approval.
    available_ms = max(0, math.floor((end-start-duration)*1000))
    offset = (value // len(options)) % (available_ms+1) / 1000
    disclaimer = (RC24_DISCLAIMER if asset["game"] == "Real Cricket 24" else
                  EA_DISCLAIMER if asset["publisher"] == "EA" else asset.get("disclaimer", ""))
    if not isinstance(disclaimer, str):
        raise ValueError("Gameplay disclaimer must be text")
    approved = next(interval for interval in asset["approved_ranges"]
                    if interval[0] <= start and end <= interval[1])
    selected = {"manifest": library["manifest"], "manifest_sha256": library["manifest_sha256"],
            "asset": copy.deepcopy(asset), "approved_range": list(approved),
            "start": start+offset, "end": start+offset+duration, "disclaimer": disclaimer}
    if asset["id"] in screens:
        selected["visual_screening"] = {**copy.deepcopy(screens[asset["id"]]), "selected_clean_range": [start, end]}
    return selected


def source_crop(analysis):
    """Use a valid camera ROI only if it contains every observed face; otherwise fit all."""
    dims = analysis["source_dimensions"]
    width, height = int(dims["width"]), int(dims["height"])
    if width <= 0 or height <= 0:
        raise ValueError("Invalid source dimensions")
    full = [0, 0, width//2*2, height//2*2]
    strategy = analysis.get("export_strategy", {})
    roi = strategy.get("active_crop") or {}
    if strategy.get("guest_cam_on") or strategy.get("guest_cam_off"):
        return full
    values = [roi.get(k) for k in ("x", "y", "width", "height")]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        return full
    x, y, w, h = [int(v)//2*2 for v in values]
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > width or y+h > height:
        return full
    if not any(roi.get(k) is not None for k in ("face_x", "face_y", "face_w", "face_h")):
        return full
    for sample in [roi, *analysis.get("face_samples", [])]:
        face = [sample.get(k) for k in ("face_x", "face_y", "face_w", "face_h")]
        if any(v is not None for v in face):
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in face):
                return full
            fx, fy, fw, fh = face
            if min(fw, fh) <= 0 or fx < x or fy < y or fx+fw > x+w or fy+fh > y+h:
                return full
    return [x, y, w, h]


def credit_text(selection):
    credit = "Illustrative cricket video-game gameplay; not actual match footage.\n" + selection["asset"]["attribution"]
    if selection.get("disclaimer"):
        credit += "\n" + selection["disclaimer"]
    return credit


def append_gameplay_credit(metadata, sidecar):
    """One idempotent attribution seam for first SEO, manual regeneration and retry."""
    sidecar = Path(sidecar)
    if not sidecar.exists():
        return metadata
    selection = json.loads(sidecar.read_text(encoding="utf-8"))["selection"]
    credit = credit_text(selection)
    result = dict(metadata)
    description = result.get("description", "")
    if not isinstance(description, str):
        raise ValueError("Gameplay description must be text")
    if credit not in description:
        description = description.rstrip()+"\n\n"+credit
    # Keep room for the uploader's Shorts marker. Apply this even on the
    # idempotent path: multibyte prose can otherwise silently cut off the credit.
    if len(description.encode("utf-8")) > 4800:
        raise ValueError("Gameplay attribution exceeds reviewed description limit; repackage, do not truncate")
    result["description"] = description
    return result


def count_distinct_exports(clips):
    from automation.clip_selection.dedupe import same_moment
    moments = []
    for clip in clips:
        clip = Path(clip)
        try:
            edit = json.loads(clip.with_name(clip.stem+"_edit.json").read_text(encoding="utf-8"))
            item = {"source": str(Path(edit["source_path"]).resolve()),
                    "start": float(edit["source_start"]), "end": float(edit["source_end"])}
            if not clip.is_file() or not math.isfinite(item["start"]+item["end"]) or not 0 <= item["start"] < item["end"]:
                continue
            if not any(item["source"] == old["source"] and same_moment(item, old) for old in moments):
                moments.append(item)
        except (OSError, ValueError, TypeError, KeyError):
            continue  # missing evidence cannot satisfy the two-moment contract
    return len(moments)
