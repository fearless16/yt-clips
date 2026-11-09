"""Hash-bound, explicit owner review for pilot releases. No publishing side effects.

Usage: python -m automation.release_review init CLIP --source SOURCE
       Edit CLIP_review.json: record evidence and PASS/FAIL for each check.
       python -m automation.release_review approve CLIP --reviewer NAME
       python -m automation.release_review check CLIP

PASS is a recorded human attestation, not a claim of automatic visual/listening QC.
"""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from automation.artifacts import atomic_json, read_metadata


CHECKS = (
    "discovery_coverage", "alternatives_compared", "source_context_listened",
    "complete_meaning", "full_decode", "caption_fidelity_and_timing",
    "framing_and_ui_safety", "audio_and_continuity", "title_cover_promise",
    "phone_playback_with_sound",
)


def required_checks(clip):
    # A missing/changed edit sidecar must not hide extant gameplay obligations.
    gameplay_required = False
    for suffix in ("_edit.json", "_gameplay.json"):
        path = clip.with_name(f"{clip.stem}{suffix}")
        if not path.exists():
            continue
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(evidence, dict):
            raise ValueError(f"Invalid release evidence: {path.name}")
        export_config = evidence.get("export_config", {})
        if not isinstance(export_config, dict):
            raise ValueError(f"Invalid export config evidence: {path.name}")
        composition = evidence.get("composition") or export_config.get("composition")
        from automation.cricket_gameplay import GAMEPLAY_COMPOSITIONS
        # Presence is itself evidence: an emptied or legacy sidecar must never
        # turn a gameplay package into a portrait-only review.
        if (suffix == "_gameplay.json" or composition in GAMEPLAY_COMPOSITIONS
                or evidence.get("gameplay_layout") is not None):
            gameplay_required = True
    if gameplay_required:
        return (*CHECKS, "gameplay_rights_and_attribution", "gameplay_label_and_framing")
    return CHECKS


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256")
    return digest.hexdigest()


def review_path(clip: Path) -> Path:
    return clip.with_name(f"{clip.stem}_review.json")


def _artifacts(clip: Path, source: Path, *, local_draft=False) -> dict:
    paths = {
        "media": clip,
        "metadata": clip.with_name(f"{clip.stem}_metadata.json"),
        "cover": clip.with_name(f"{clip.stem}_thumb.jpg"),
        "source": source,
        "qc": clip.with_name(f"{clip.stem}_qc.json"),
    }
    for name, path in {
        "edit": clip.with_name(f"{clip.stem}_edit.json"),
        "speech": clip.with_name(f"{clip.stem}_speech.json"),
        "cover_plan": clip.with_name(f"{clip.stem}_cover.json"),
        "discovery": clip.parent / "editorial.json",
        "gameplay": clip.with_name(f"{clip.stem}_gameplay.json"),
    }.items():
        if path.exists():
            paths[name] = path
    if "gameplay_rights_and_attribution" in required_checks(clip):
        gameplay = clip.with_name(f"{clip.stem}_gameplay.json")
        evidence = json.loads(gameplay.read_text(encoding="utf-8"))
        from automation.cricket_gameplay import load_delivery_library, load_render_library
        selection = evidence.get("selection")
        if (not isinstance(selection, dict) or not isinstance(selection.get("asset"), dict)
                or not isinstance(selection.get("manifest"), str)
                or not isinstance(selection.get("manifest_sha256"), str)
                or not isinstance(evidence.get("media_sha256"), str)):
            raise ValueError("Gameplay selection/media evidence missing or invalid")
        library = (load_render_library if local_draft else load_delivery_library)(selection["manifest"])
        if (library["manifest_sha256"] != selection["manifest_sha256"]
                or evidence["media_sha256"] != sha256(clip)
                or selection["asset"] not in library["assets"]):
            raise ValueError("Gameplay permissions or media changed after rendering")
        asset = selection["asset"]
        from automation.cricket_gameplay import STADIUM_COMPOSITION
        edit_path = clip.with_name(f"{clip.stem}_edit.json")
        edit = json.loads(edit_path.read_text(encoding="utf-8")) if edit_path.exists() else {}
        if STADIUM_COMPOSITION in (evidence.get("composition"),edit.get("composition")):
            from automation.stadium_layout import preflight_assets
            design = preflight_assets()
            if (evidence.get("composition") != STADIUM_COMPOSITION
                    or evidence.get("design_assets") != design or evidence.get("gameplay_crop") != asset.get("crop")):
                raise ValueError("Approved stadium artwork/font/crop changed after rendering")
            paths.update(stadium_overlay=Path(design["overlay"]["path"]),
                         stadium_font=Path(design["font"]["path"]))
        from automation.cricket_gameplay import credit_text
        metadata = read_metadata(clip) or {}
        if credit_text(selection) not in metadata.get("description", ""):
            raise ValueError("Gameplay attribution/illustrative disclosure missing from reviewed description")
        paths.update(gameplay_manifest=Path(selection["manifest"]), gameplay_recording=Path(asset["path"]))
        if library.get("usage") in ("STOCK", "LOCAL_REVIEW_ONLY"):
            paths["gameplay_license_evidence"] = Path(asset["license_evidence"]["path"])
        else:
            paths.update(gameplay_publisher_permission=Path(asset["publisher_permission"]["evidence"]),
                         gameplay_recording_permission=Path(asset["recording_permission"]["evidence"]))
    return {name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()}


def create_review(clip: Path, source: Path) -> dict:
    clip, source = Path(clip).resolve(), Path(source).resolve()
    path = review_path(clip)
    if path.exists():
        raise FileExistsError(f"Review already exists; preserve or explicitly archive it: {path}")
    if read_metadata(clip) is None:
        raise ValueError("Valid, non-failed metadata is required before review")
    payload = {
        "schema_version": 1,
        "artifacts": _artifacts(clip, source, local_draft=True),
        "checks": {name: {"status": "NOT CHECKED", "evidence": "", "method": "owner review"}
                   for name in required_checks(clip)},
        "selection": {"source_ranges": [], "alternatives": [], "reason": "", "coverage_gaps": []},
        "approval": None,
    }
    edit_path = clip.with_name(f"{clip.stem}_edit.json")
    if edit_path.exists():
        edit = json.loads(edit_path.read_text(encoding="utf-8"))
        payload["selection"]["source_ranges"] = [[edit["source_start"], edit["source_end"]]]
    qc = json.loads(clip.with_name(f"{clip.stem}_qc.json").read_text(encoding="utf-8"))
    if qc.get("status") == "PASS" and qc.get("media_sha256") == sha256(clip):
        payload["checks"]["full_decode"] = {"status": "PASS", "method": "ffmpeg full-file decode + encoded audio measurement",
                                            "evidence": str(clip.with_name(f"{clip.stem}_qc.json"))}
    atomic_json(path, payload)
    return payload


def _review_digest(payload: dict) -> str:
    content = {key: value for key, value in payload.items() if key != "approval"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _validate(clip: Path, payload: dict) -> list[str]:
    errors = []
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return ["Invalid review schema"]
    if read_metadata(clip) is None:
        errors.append("Missing, invalid or failed metadata")
    checks = payload.get("checks", {})
    if not isinstance(checks, dict):
        return ["Invalid review checks"]
    try:
        names = required_checks(clip)
    except (OSError, ValueError, TypeError, AttributeError):
        return ["Invalid edit evidence"]
    for name in names:
        check = checks.get(name)
        if not isinstance(check, dict) or check.get("status") != "PASS":
            errors.append(f"{name}: not PASS")
        elif not all(isinstance(check.get(key), str) and check[key].strip()
                     for key in ("evidence", "method")):
            errors.append(f"{name}: missing evidence/method")
    selection = payload.get("selection", {})
    if not isinstance(selection, dict):
        return errors + ["Invalid selection evidence"]
    if not selection.get("reason") or not selection.get("source_ranges"):
        errors.append("Missing selection reason/source ranges")
    if not selection.get("alternatives"):
        errors.append("Missing strongest-alternative comparison (or documented absence)")
    if selection.get("coverage_gaps") != []:
        errors.append("Unresolved source coverage gaps")
    previous_end = -1.0
    try:
        for start, end in selection.get("source_ranges", []):
            if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in (start, end)):
                raise ValueError("Non-numeric range")
            if start < 0 or start < previous_end or end <= start:
                raise ValueError("Unordered/invalid range")
            previous_end = end
    except (ValueError, TypeError):
        errors.append("Invalid selection source ranges")
    try:
        recorded = payload["artifacts"]
        expected = _artifacts(clip, Path(recorded["source"]["path"]))
        if recorded != expected:
            errors.append("Artifact changed since review was prepared")
        qc = json.loads(clip.with_name(f"{clip.stem}_qc.json").read_text(encoding="utf-8"))
        if qc.get("status") != "PASS" or qc.get("media_sha256") != expected["media"]["sha256"]:
            errors.append("Encoded delivery QC missing, failed or stale")
        if "edit" in expected:
            edit = json.loads(Path(expected["edit"]["path"]).read_text(encoding="utf-8"))
            if selection.get("source_ranges") != [[edit["source_start"], edit["source_end"]]]:
                errors.append("Review source ranges differ from rendered edit")
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        errors.append("Missing or invalid review artifacts")
    return errors


def approve(clip: Path, reviewer: str) -> dict:
    clip = Path(clip).resolve()
    if not reviewer.strip():
        raise ValueError("Named reviewer required")
    payload = json.loads(review_path(clip).read_text(encoding="utf-8"))
    errors = _validate(clip, payload)
    if errors:
        raise ValueError("Review incomplete: " + "; ".join(errors))
    payload["approval"] = {"reviewer": reviewer.strip(), "review_sha256": _review_digest(payload),
                           "approved_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(review_path(clip), payload)
    return payload


def release_status(clip: Path) -> tuple[str, list[str]]:
    clip = Path(clip).resolve()
    try:
        payload = json.loads(review_path(clip).read_text(encoding="utf-8"))
        errors = _validate(clip, payload)
        approval = payload.get("approval") or {}
        if not isinstance(approval, dict) or not approval.get("reviewer") or approval.get("review_sha256") != _review_digest(payload):
            errors.append("Missing approval or review changed after approval")
        checks = payload.get("checks", {})
        failed = isinstance(checks, dict) and any(isinstance(c, dict) and c.get("status") == "FAIL" for c in checks.values())
        return ("REJECT" if failed else "REVIEW", errors) if errors else ("READY", [])
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
        return "REVIEW", [f"Review unavailable or invalid: {exc}"]


def require_ready(clip: Path) -> None:
    status, reasons = release_status(clip)
    if status != "READY":
        raise ValueError(f"Release held ({status}): " + "; ".join(reasons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "approve", "check"))
    parser.add_argument("clip", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reviewer")
    args = parser.parse_args()
    try:
        if args.action == "init":
            if not args.source:
                parser.error("init requires --source")
            create_review(args.clip, args.source)
            print(f"REVIEW: complete evidence in {review_path(args.clip)}")
        elif args.action == "approve":
            if not args.reviewer:
                parser.error("approve requires --reviewer")
            approve(args.clip, args.reviewer)
            print("READY: approval recorded for exact artifacts; nothing uploaded")
        else:
            status, reasons = release_status(args.clip)
            print(json.dumps({"status": status, "reasons": reasons}, indent=2))
            return 0 if status == "READY" else 1
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
