"""Release gate: machine-verified checks plus one owner decision (or a trusted auto tier).

Usage:
  python -m automation.release_review review DIR_OR_CLIP [--reviewer NAME] [--open]
  python -m automation.release_review approve CLIP --reviewer NAME
  python -m automation.release_review reject CLIP --reviewer NAME --reason TEXT
  python -m automation.release_review reset CLIP
  python -m automation.release_review check CLIP

Every check is recomputed from the artifacts on disk; nothing is hand-typed.
Approval binds exact artifact hashes. When an artifact changes, the stale
review is archived under .review-archive/ and rebuilt. Only the auto tier
re-approves automatically. Owner rejections are kept until `reset`.

mode=auto approves a clip only when every machine check is PASS. Clips whose
completeness came from the heuristic fallback (no LLM editor) always need
the owner.
"""

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from automation.artifacts import atomic_json, read_metadata


SCHEMA_VERSION = 2
MACHINE_CHECKS = ("full_decode", "metadata_valid", "complete_meaning", "source_range", "speech_alignment")
GAMEPLAY_CHECKS = ("gameplay_rights_and_attribution",)
# Completeness methods that are not editorial judgement: only a human may release them.
OWNER_ONLY_METHODS = frozenset({"heuristic_fallback"})
MODES = ("manual", "auto")


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256")
    return digest.hexdigest()


def review_path(clip: Path) -> Path:
    return clip.with_name(f"{clip.stem}_review.json")


def _sidecar(clip: Path, suffix: str) -> Path:
    return clip.with_name(f"{clip.stem}{suffix}")


def _read_json(path: Path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_gameplay(clip: Path) -> bool:
    """A missing or emptied sidecar must never hide a gameplay obligation."""
    from automation.cricket_gameplay import GAMEPLAY_COMPOSITIONS
    if _sidecar(clip, "_gameplay.json").exists():
        return True
    edit = _read_json(_sidecar(clip, "_edit.json")) or {}
    export_config = edit.get("export_config") if isinstance(edit.get("export_config"), dict) else {}
    composition = edit.get("composition") or export_config.get("composition")
    return composition in GAMEPLAY_COMPOSITIONS or edit.get("gameplay_layout") is not None


def required_checks(clip: Path) -> tuple:
    return MACHINE_CHECKS + (GAMEPLAY_CHECKS if is_gameplay(clip) else ())


def _gameplay_artifacts(clip: Path, *, local_draft: bool) -> dict:
    """Validate rights/credit/design evidence; raise ValueError when it is missing or stale."""
    evidence = _read_json(_sidecar(clip, "_gameplay.json"))
    if evidence is None:
        raise ValueError("Gameplay evidence sidecar missing or invalid")
    from automation.cricket_gameplay import (STADIUM_COMPOSITION, credit_text,
                                             load_delivery_library, load_render_library)
    selection = evidence.get("selection")
    if (not isinstance(selection, dict) or not isinstance(selection.get("asset"), dict)
            or not isinstance(selection.get("manifest"), str)
            or not isinstance(selection.get("manifest_sha256"), str)
            or not isinstance(evidence.get("media_sha256"), str)):
        raise ValueError("Gameplay selection/media evidence missing or invalid")
    library = (load_render_library if local_draft else load_delivery_library)(selection["manifest"])
    asset = selection["asset"]
    if (library["manifest_sha256"] != selection["manifest_sha256"]
            or evidence["media_sha256"] != sha256(clip) or asset not in library["assets"]):
        raise ValueError("Gameplay permissions or media changed after rendering")
    paths = {}
    edit = _read_json(_sidecar(clip, "_edit.json")) or {}
    if STADIUM_COMPOSITION in (evidence.get("composition"), edit.get("composition")):
        from automation.stadium_layout import preflight_assets
        design = preflight_assets()
        if (evidence.get("composition") != STADIUM_COMPOSITION
                or evidence.get("design_assets") != design or evidence.get("gameplay_crop") != asset.get("crop")):
            raise ValueError("Approved stadium artwork/font/crop changed after rendering")
        paths.update(stadium_overlay=Path(design["overlay"]["path"]), stadium_font=Path(design["font"]["path"]))
    metadata = read_metadata(clip) or {}
    if credit_text(selection) not in metadata.get("description", ""):
        raise ValueError("Gameplay attribution/illustrative disclosure missing from description")
    paths.update(gameplay_manifest=Path(selection["manifest"]), gameplay_recording=Path(asset["path"]))
    if library.get("usage") in ("STOCK", "LOCAL_REVIEW_ONLY"):
        paths["gameplay_license_evidence"] = Path(asset["license_evidence"]["path"])
    else:
        paths.update(gameplay_publisher_permission=Path(asset["publisher_permission"]["evidence"]),
                     gameplay_recording_permission=Path(asset["recording_permission"]["evidence"]))
    return paths


def _check(status: str, evidence: str, method: str) -> dict:
    return {"status": status, "evidence": evidence, "method": method}


def _base_paths(clip: Path, source: Path) -> dict:
    paths = {"media": clip, "metadata": _sidecar(clip, "_metadata.json"),
             "cover": _sidecar(clip, "_thumb.jpg"), "source": source, "qc": _sidecar(clip, "_qc.json")}
    for name, suffix in (("edit", "_edit.json"), ("speech", "_speech.json"),
                         ("cover_plan", "_cover.json"), ("gameplay", "_gameplay.json")):
        if _sidecar(clip, suffix).exists():
            paths[name] = _sidecar(clip, suffix)
    if (clip.parent / "editorial.json").exists():
        paths["discovery"] = clip.parent / "editorial.json"
    return paths


def _hashes(paths: dict) -> dict:
    return {name: {"path": str(Path(p).resolve()), "sha256": sha256(p) if Path(p).is_file() else None}
            for name, p in paths.items()}


def _artifacts(clip: Path, source: Path, *, local_draft: bool = False) -> dict:
    """Strict hash map of every reviewed artifact; raises ValueError on invalid gameplay rights."""
    clip, source = Path(clip).resolve(), Path(source).resolve()
    paths = _base_paths(clip, source)
    if is_gameplay(clip):
        paths.update(_gameplay_artifacts(clip, local_draft=local_draft))
    return _hashes(paths)


def _build(clip: Path, source: Path) -> dict:
    """Recompute artifacts and every check from disk. Deterministic for unchanged inputs."""
    clip, source = Path(clip).resolve(), Path(source).resolve()
    paths = _base_paths(clip, source)
    checks = {}
    if is_gameplay(clip):
        try:
            paths.update(_gameplay_artifacts(clip, local_draft=False))
            checks["gameplay_rights_and_attribution"] = _check(
                "PASS", "delivery library, licence evidence, credit text and design assets hash-verified",
                "automatic")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            checks["gameplay_rights_and_attribution"] = _check("FAIL", str(exc), "automatic")
            try:  # still bind what can be hashed so the owner sees the real reason
                paths.update(_gameplay_artifacts(clip, local_draft=True))
            except (OSError, ValueError, KeyError, TypeError):
                pass
    artifacts = _hashes(paths)

    qc = _read_json(paths["qc"])
    media_hash = artifacts["media"]["sha256"]
    qc_ok = bool(qc and qc.get("status") == "PASS" and qc.get("media_sha256") == media_hash)
    checks["full_decode"] = _check("PASS" if qc_ok else "FAIL",
                                   "encoded file decoded and audio measured" if qc_ok
                                   else "delivery QC missing, failed or stale", "delivery_qc")
    meta = read_metadata(clip)
    if meta is None:
        checks["metadata_valid"] = _check("FAIL", "missing, invalid or failed metadata", "read_metadata")
    elif meta.get("seo_method") == "template":
        # Deterministic packaging (provider down or ungrounded answer): a human signs it off.
        checks["metadata_valid"] = _check("NEEDS_OWNER", "template packaging; confirm title/cover "
                                          f"({meta.get('seo_error', 'provider unavailable')})", "read_metadata")
    else:
        checks["metadata_valid"] = _check("PASS", "SEO metadata valid", "read_metadata")
    edit = _read_json(paths.get("edit", _sidecar(clip, "_edit.json"))) or {}
    completeness = edit.get("completeness") if isinstance(edit.get("completeness"), dict) else {}
    method = str(completeness.get("method") or "unknown")
    if completeness.get("status") == "PASS":
        status = "NEEDS_OWNER" if method in OWNER_ONLY_METHODS else "PASS"
    else:
        status = "FAIL"
    checks["complete_meaning"] = _check(status, str(completeness.get("reason") or "no completeness verdict"),
                                        f"editorial:{method}")
    start, end = edit.get("source_start"), edit.get("source_end")
    range_ok = (all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
                    for t in (start, end)) and 0 <= start < end)
    checks["source_range"] = _check("PASS" if range_ok else "FAIL",
                                    f"{start}-{end}s" if range_ok else "missing or invalid source range",
                                    "edit sidecar")
    flags = []
    speech = _read_json(paths["speech"]) if "speech" in paths else None
    if speech is None:
        checks["speech_alignment"] = _check("PASS", "no word-timed captions rendered", "not applicable")
    else:
        from automation.speech_alignment import alignment_flags, alignment_issues
        segments = speech.get("segments") or []
        rng = speech.get("source_range") or [start, end]
        try:
            issues = alignment_issues(segments, float(rng[0]), float(rng[1]))
            flags = alignment_flags(segments, float(rng[0]), float(rng[1])) if not issues else []
        except (TypeError, ValueError, IndexError) as exc:
            issues = [f"invalid speech evidence: {exc}"]
        checks["speech_alignment"] = _check("FAIL" if issues else "PASS",
                                            "; ".join(issues) or "cut lands on word edges",
                                            "speech_alignment")
    return {"schema_version": SCHEMA_VERSION, "artifacts": artifacts, "checks": checks,
            "selection": {"source_ranges": [[start, end]] if range_ok else []},
            "flags": flags, "approval": None, "rejection": None}


def _review_digest(payload: dict) -> str:
    content = {k: v for k, v in payload.items() if k not in ("approval", "rejection")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _source_of(clip: Path, payload=None) -> Path:
    if isinstance(payload, dict):
        try:
            return Path(payload["artifacts"]["source"]["path"])
        except (KeyError, TypeError):
            pass
    edit = _read_json(_sidecar(clip, "_edit.json")) or {}
    if not edit.get("source_path"):
        raise ValueError(f"Source unknown for {clip.name}; pass --source")
    return Path(edit["source_path"])


def _archive(path: Path) -> None:
    folder = path.parent / ".review-archive"
    folder.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path.replace(folder / f"{path.stem}.{stamp}.json")


def _auto_eligible(payload: dict) -> bool:
    return all(c.get("status") == "PASS" for c in payload["checks"].values())


def prepare_review(clip: Path, source: Path | None = None, *, mode: str = "manual") -> dict:
    """Create or refresh the review for the current artifacts. Idempotent.

    Stale or legacy reviews are archived and rebuilt. Owner rejections are kept.
    In auto mode, a clip whose machine checks all PASS is approved by "auto".
    """
    if mode not in MODES:
        raise ValueError(f"release_review.mode must be one of {MODES}")
    clip = Path(clip).resolve()
    path = review_path(clip)
    existing = _read_json(path) if path.exists() else None
    if existing and existing.get("rejection"):
        return existing
    source = Path(source) if source is not None else _source_of(clip, existing)
    fresh = _build(clip, source)
    if (existing and existing.get("schema_version") == SCHEMA_VERSION
            and _review_digest(existing) == _review_digest(fresh)):
        payload = existing
    else:
        if path.exists():
            _archive(path)
        payload = fresh
        atomic_json(path, payload)
    if mode == "auto" and payload.get("approval") is None and _auto_eligible(payload):
        payload["approval"] = {"reviewer": "auto", "mode": "auto", "review_sha256": _review_digest(payload),
                               "approved_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(path, payload)
    return payload


def create_review(clip: Path, source: Path) -> dict:
    """Write a new manual review draft; refuse to replace an existing file."""
    path = review_path(Path(clip).resolve())
    if path.exists():
        raise FileExistsError(f"Review already exists (use reset to rebuild): {path}")
    if read_metadata(Path(clip)) is None:
        raise ValueError("Valid, non-failed metadata is required before review")
    return prepare_review(clip, source)


def _errors(clip: Path, payload: dict) -> list[str]:
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        return ["Legacy or invalid review; run: python -m automation.release_review reset CLIP"]
    try:
        fresh = _build(clip, _source_of(clip, payload))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"Review artifacts unavailable: {exc}"]
    if _review_digest(fresh) != _review_digest(payload):
        return ["Artifacts changed since review; rerun the pipeline or `review` to rebuild"]
    return [f"{name}: {c['status']} ({c['evidence']})" for name, c in fresh["checks"].items()
            if c["status"] == "FAIL"]


def release_status(clip: Path) -> tuple[str, list[str]]:
    clip = Path(clip).resolve()
    payload = _read_json(review_path(clip))
    if payload is None:
        return "REVIEW", ["No review prepared"]
    rejection = payload.get("rejection")
    if isinstance(rejection, dict):
        return "REJECT", [f"Rejected by {rejection.get('reviewer')}: {rejection.get('reason')}"]
    errors = _errors(clip, payload)
    if errors:
        return "REVIEW", errors
    approval = payload.get("approval")
    if not isinstance(approval, dict) or approval.get("review_sha256") != _review_digest(payload):
        pending = [n for n, c in payload["checks"].items() if c["status"] == "NEEDS_OWNER"]
        return "REVIEW", ["Awaiting owner decision" + (f" ({', '.join(pending)})" if pending else "")]
    if approval.get("mode") == "auto" and not _auto_eligible(payload):
        return "REVIEW", ["Auto tier cannot release owner-only checks"]
    return "READY", []


def require_ready(clip: Path) -> None:
    status, reasons = release_status(clip)
    if status != "READY":
        raise ValueError(f"Release held ({status}): " + "; ".join(reasons))


def approve(clip: Path, reviewer: str) -> dict:
    clip = Path(clip).resolve()
    if not reviewer or not reviewer.strip():
        raise ValueError("Named reviewer required")
    payload = _read_json(review_path(clip))
    if payload is None:
        raise ValueError("No review prepared")
    errors = _errors(clip, payload)
    if errors:
        raise ValueError("Cannot approve: " + "; ".join(errors))
    payload.update(rejection=None, approval={
        "reviewer": reviewer.strip(), "mode": "owner", "review_sha256": _review_digest(payload),
        "approved_at": datetime.now(timezone.utc).isoformat()})
    atomic_json(review_path(clip), payload)
    return payload


def reject(clip: Path, reviewer: str, reason: str) -> dict:
    clip = Path(clip).resolve()
    if not reviewer or not reviewer.strip() or not reason or not reason.strip():
        raise ValueError("Reviewer and reason required")
    payload = _read_json(review_path(clip)) or prepare_review(clip)
    payload.update(approval=None, rejection={"reviewer": reviewer.strip(), "reason": reason.strip(),
                                             "rejected_at": datetime.now(timezone.utc).isoformat()})
    atomic_json(review_path(clip), payload)
    return payload


def reset(clip: Path, source: Path | None = None) -> dict:
    clip = Path(clip).resolve()
    path = review_path(clip)
    existing = _read_json(path) if path.exists() else None
    source = Path(source) if source is not None else _source_of(clip, existing)
    if path.exists():
        _archive(path)
    return prepare_review(clip, source)


def _pending(target: Path) -> list[Path]:
    clips = [target] if target.is_file() else sorted(target.rglob("*.mp4"))
    return [c for c in clips if _sidecar(c, "_metadata.json").exists()
            and release_status(c)[0] not in ("READY", "REJECT")]


def _interactive(target: Path, reviewer: str, open_media: bool) -> int:
    clips = _pending(target)
    print(f"{len(clips)} clip(s) awaiting a decision")
    for clip in clips:
        try:
            payload = prepare_review(clip)
        except (OSError, ValueError) as exc:
            print(f"\n{clip.name}: cannot prepare review: {exc}")
            continue
        meta = read_metadata(clip) or {}
        print(f"\n=== {clip.name}")
        print(f"title : {meta.get('title', '(no metadata)')}")
        print(f"range : {payload['selection']['source_ranges']}")
        for name, check in payload["checks"].items():
            print(f"  {check['status']:<11} {name}: {check['evidence']}")
        for flag in payload.get("flags", []):
            print(f"  FLAG        {flag}")
        if any(c["status"] == "FAIL" for c in payload["checks"].values()):
            print("  -> machine check failed; fix and re-run the pipeline")
            continue
        if open_media and hasattr(os, "startfile"):
            os.startfile(str(clip))  # noqa: S606 - owner-requested local playback
        answer = input("publish? [y]es / [n]o / [s]kip / [q]uit: ").strip().lower()
        if answer == "q":
            break
        if answer == "y":
            approve(clip, reviewer)
            print("  READY")
        elif answer == "n":
            reject(clip, reviewer, input("  reason: ").strip() or "owner rejected")
            print("  REJECTED")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("review", "approve", "reject", "reset", "check", "init"))
    parser.add_argument("clip", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reviewer", default=os.environ.get("USERNAME") or os.environ.get("USER"))
    parser.add_argument("--reason")
    parser.add_argument("--open", action="store_true", help="open each clip in the default player")
    args = parser.parse_args()
    try:
        if args.action == "review":
            return _interactive(args.clip, args.reviewer, args.open)
        if args.action == "approve":
            approve(args.clip, args.reviewer)
            print("READY: approval recorded for exact artifacts; nothing uploaded")
        elif args.action == "reject":
            reject(args.clip, args.reviewer, args.reason or "")
            print("REJECTED")
        elif args.action in ("reset", "init"):
            reset(args.clip, args.source)
            print(json.dumps(dict(zip(("status", "reasons"), release_status(args.clip))), indent=2))
        else:
            status, reasons = release_status(args.clip)
            print(json.dumps({"status": status, "reasons": reasons}, indent=2))
            return 0 if status == "READY" else 1
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
