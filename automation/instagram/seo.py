"""seo.py — Instagram SEO orchestrator (evidence → caption → metadata).

Writes insta_metadata.json per clip and exposes the runner stage hooks
(run_evidence_stage / run_caption_stage / make_graph_client), registering
them into automation.instagram.runner at module import. The evidence pack
comes from automation.instagram.evidence.build_insta_evidence_pack
(teammate I2) — imported lazily so this module stays usable while that
lands; missing evidence module ⇒ None (skip), never a crash.
"""
import json
import logging
import os
from pathlib import Path

from automation.instagram import runner as _runner
from automation.instagram.caption_engine import write_caption

log = logging.getLogger("insta_seo")

PACKAGING_VERSION = "insta_v1_realonly"
METADATA_FILE = "insta_metadata.json"
EVIDENCE_FILE = "insta_evidence.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _build_evidence(clip_dir, transcript, video_title, video_description):
    from automation.instagram.evidence import build_insta_evidence_pack
    return build_insta_evidence_pack(
        video_title=video_title,
        video_description=video_description,
        transcript=transcript,
    )


def _summarize_evidence(pack: dict) -> dict:
    sources = sorted({
        str((item or {}).get("source") or "unknown")
        for item in (pack.get("validated_hashtags") or [])
    })
    return {
        "match_facts": len(pack.get("match_facts") or []),
        "roster": len(pack.get("roster") or []),
        "validated_hashtags": len(pack.get("validated_hashtags") or []),
        "seed_phrases": len(pack.get("seed_phrases") or []),
        "sources": sources,
    }


def run_evidence_stage(clip_dir, transcript, video_title,
                       video_description=""):
    """EVIDENCE stage — build + persist the InstaEvidencePack."""
    try:
        pack = _build_evidence(clip_dir, transcript, video_title,
                               video_description)
    except ImportError:
        log.warning(
            "automation.instagram.evidence unavailable — skipping "
            "Instagram evidence stage for %s", clip_dir)
        return None
    if isinstance(pack, dict):
        _atomic_write_json(Path(clip_dir) / EVIDENCE_FILE, pack)
    return pack


def run_caption_stage(clip_dir, transcript, video_title,
                      video_description=""):
    """CAPTION stage — returns the publishable caption string."""
    clip_dir = Path(clip_dir)
    pack = _read_json(clip_dir / EVIDENCE_FILE)
    if not isinstance(pack, dict):
        pack = run_evidence_stage(clip_dir, transcript, video_title,
                                  video_description)
    if not isinstance(pack, dict):
        return None
    payload = generate_insta_metadata(
        clip_dir, transcript, video_title, video_description)
    if not payload:
        return None
    return payload["caption"]


def make_graph_client():
    """Build FacebookGraphClient from insta_token.json, or None."""
    try:
        from automation.instagram.credential import is_valid, load_token
    except ImportError:
        return None
    token_data = load_token()
    if not is_valid(token_data):
        return None
    access_token = str(token_data.get("access_token") or "").strip()
    ig_user_id = str(token_data.get("ig_user_id") or "").strip()
    if not access_token or not ig_user_id:
        return None
    try:
        from automation.instagram.graph_client import FacebookGraphClient
        return FacebookGraphClient(access_token, ig_user_id)
    except Exception as exc:
        log.warning("FacebookGraphClient construction failed: %s", exc)
        return None


def generate_insta_metadata(clip_dir, transcript, video_title,
                            video_description="", *, force=False):
    """Evidence → grounded caption → insta_metadata.json.

    Returns the payload dict, an already-written payload when resuming,
    or None when the evidence builder is unavailable. CaptionPolicyError
    propagates fail-loud for the failed-marker protocol.
    """
    clip_dir = Path(clip_dir)
    meta_path = clip_dir / METADATA_FILE
    if not force:
        existing = _read_json(meta_path)
        if isinstance(existing, dict) and existing.get("caption"):
            return existing

    try:
        pack = _build_evidence(clip_dir, transcript, video_title,
                               video_description)
    except ImportError:
        log.warning(
            "automation.instagram.evidence unavailable — no Instagram "
            "metadata generated for %s", clip_dir)
        return None
    _atomic_write_json(clip_dir / EVIDENCE_FILE, pack)

    package = write_caption(pack, transcript, video_title)
    payload = {
        "caption": package["caption"],
        "hashtags": package["hashtags"],
        "audio_name": package["audio_name"],
        "evidence_summary": _summarize_evidence(pack),
        "packaging_version": PACKAGING_VERSION,
    }
    _atomic_write_json(meta_path, payload)
    return payload


_runner.register_stage_hook(_runner.HOOK_RUN_EVIDENCE, run_evidence_stage)
_runner.register_stage_hook(_runner.HOOK_RUN_CAPTION, run_caption_stage)
_runner.register_stage_hook(_runner.HOOK_MAKE_CLIENT, make_graph_client)
