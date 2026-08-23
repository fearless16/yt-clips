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

    teams = []
    try:
        from automation.seo.entity_grounding import find_canonical_entities
        merged = " ".join(filter(None, [video_title, video_description,
                                        transcript]))
        teams = find_canonical_entities(merged)["teams"][:2]
    except Exception as exc:
        log.warning("team grounding unavailable: %s", exc)

    match_facts = None
    try:
        from automation.seo.trends import fetch_verified_match_context
        facts = fetch_verified_match_context(
            video_title or transcript[:120] or "cricket")
        if isinstance(facts, dict):
            match_facts = facts
    except Exception as exc:
        log.warning("verified match context unavailable: %s", exc)

    return build_insta_evidence_pack(
        video_title=video_title,
        video_description=video_description,
        transcript=transcript,
        teams=teams,
        match_facts=match_facts,
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


def _insta_config() -> dict:
    try:
        from automation.config import load as load_cfg
        return (load_cfg() or {}).get("instagram") or {}
    except Exception:
        return {}


def make_graph_client():
    """Build FacebookGraphClient from insta_token.json, or None."""
    if str(_insta_config().get("provider") or "graph").casefold() \
            == "upload_post":
        return None
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


def make_upload_post_publish():
    """Publish-hook for provider=upload_post: one REST call per Reel.

    Returns None when the provider isn't upload_post or credentials are
    missing, so the runner falls back to the Graph flow.
    """
    cfg = _insta_config()
    if str(cfg.get("provider") or "graph").casefold() != "upload_post":
        return None
    up_cfg = cfg.get("upload_post") or {}
    api_key = os.environ.get(
        str(up_cfg.get("api_key_env") or "UPLOAD_POST_API_KEY"), "").strip()
    profile = str(up_cfg.get("profile") or "").strip()
    if not api_key or not profile:
        log.warning(
            "instagram.provider=upload_post but %s/profile missing — "
            "falling back to Graph flow",
            up_cfg.get("api_key_env") or "UPLOAD_POST_API_KEY")
        return None

    def _publish(video_path, caption, audio_name=None, clip_dir=None,
                 state=None):
        from automation.instagram.upload_post_client import UploadPostClient
        client = UploadPostClient(api_key, profile)
        result = client.publish_reel(
            video_path, caption, audio_name=audio_name,
            poll_interval_s=10.0, timeout_s=900.0)
        return {
            "media_id": result["media_id"],
            "permalink": result.get("permalink", ""),
            "provider": "upload_post",
        }

    return _publish


def _publish_via_provider(**kwargs):
    fn = make_upload_post_publish()
    return fn(**kwargs) if fn is not None else None


def _register_default_hooks() -> None:
    """Fill only EMPTY hook slots.

    Import side effect must never clobber hooks a caller (or test)
    registered before this module was first imported.
    """
    defaults = {
        _runner.HOOK_RUN_EVIDENCE: run_evidence_stage,
        _runner.HOOK_RUN_CAPTION: run_caption_stage,
        _runner.HOOK_MAKE_CLIENT: make_graph_client,
        _runner.HOOK_PUBLISH: _publish_via_provider,
    }
    for name, fn in defaults.items():
        if name not in _runner._HOOKS:
            _runner.register_stage_hook(name, fn)


_register_default_hooks()
