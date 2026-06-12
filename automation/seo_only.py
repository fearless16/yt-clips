"""seo_only.py — Mac-side SEO generation for pre-exported clips.

Run on your MacBook Air after Colab exports clips to a shared folder
(Google Drive or local). No GPU needed — just AI API calls.

Usage::

    python -m automation.seo_only shorts/2026-06-04/
    python -m automation.seo_only shorts/2026-06-04/ --highlights highlights/video.yaml
    python -m automation.seo_only shorts/2026-06-04/ --skip-existing
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("seo_only", cfg["logging"]["log_file"], cfg["logging"]["level"])


# Lazy AI client — same pattern as seo.py
_ai_instance = None
_ai_lock = __import__("threading").Lock()


def _get_ai():
    """Thread-safe lazy AIClient singleton."""
    global _ai_instance
    if _ai_instance is not None:
        return _ai_instance
    with _ai_lock:
        if _ai_instance is None:
            from utils.ai_client import AIClient
            _ai_instance = AIClient()
    return _ai_instance


# ── Clip discovery ───────────────────────────────────────────────────────────

def discover_clips(
    clips_dir: str,
    skip_existing: bool = True,
) -> List[Dict]:
    """Find .mp4 clips in a directory that need SEO metadata.

    Args:
        clips_dir: Path to directory containing exported clips.
        skip_existing: If True, skip clips that already have *_metadata.json.

    Returns:
        List of dicts with clip_id, path.
    """
    d = Path(clips_dir)
    if not d.exists():
        return []

    clips = []
    for mp4 in sorted(d.glob("*.mp4")):
        clip_id = mp4.stem
        meta_path = mp4.with_name(f"{clip_id}_metadata.json")
        if skip_existing and meta_path.exists():
            log.debug("[seo_only] skip (has metadata): %s", clip_id)
            continue
        clips.append({"clip_id": clip_id, "path": str(mp4)})

    log.info("[seo_only] discovered %d clips needing SEO in %s", len(clips), clips_dir)
    return clips


# ── Transcript loading ───────────────────────────────────────────────────────

def _load_clip_transcript(
    clip_id: str,
    clips_dir: str,
    highlights_yaml: Optional[str] = None,
    transcript_json: Optional[str] = None,
) -> str:
    """Load transcript text for a clip from available sources.

    Priority:
        1. highlights.yaml (has per-clip text)
        2. transcript JSON (full video segments)
        3. Empty string (SEO will work from title/context only)
    """
    # Try highlights YAML
    if highlights_yaml and Path(highlights_yaml).exists():
        try:
            import yaml
            with open(highlights_yaml) as f:
                data = yaml.safe_load(f) or {}
            if clip_id in data and isinstance(data[clip_id], dict):
                return data[clip_id].get("text", "")
            # Try partial match (clip1 in clip1_xyz)
            for key, val in data.items():
                if isinstance(val, dict) and clip_id.startswith(key):
                    return val.get("text", "")
        except Exception as e:
            log.debug("[seo_only] highlights yaml load failed: %s", e)

    # Try transcript JSON
    if transcript_json and Path(transcript_json).exists():
        try:
            with open(transcript_json) as f:
                data = json.load(f)
            segments = data if isinstance(data, list) else data.get("segments", [])
            # Return full transcript — SEO will handle truncation
            return " ".join(s.get("text", "") for s in segments[:50])
        except Exception as e:
            log.debug("[seo_only] transcript json load failed: %s", e)

    # Check for per-clip transcript in clips_dir
    clip_transcript = Path(clips_dir) / f"{clip_id}_transcript.txt"
    if clip_transcript.exists():
        return clip_transcript.read_text(encoding="utf-8").strip()

    return ""


# ── SEO generation ───────────────────────────────────────────────────────────

def _generate_seo_for_clip(
    clip_id: str,
    transcript: str,
    video_title: str = "",
) -> Optional[Dict]:
    """Generate SEO metadata for a single clip using AI.

    Uses the SAME system prompt and quality gate as the main SEO pipeline.
    Generic garbage is rejected — no exceptions.
    """
    from automation.seo.seo import (
        _SYSTEM, _parse_json_response, _enforce_limits, _validate_seo_quality
    )

    prompt = f"""CONTEXT:
  Match: {video_title or 'Cricket Match'}

CLIP TRANSCRIPT: {transcript or 'Cricket highlight clip'}

Generate YouTube Shorts SEO for this clip.
Return ONLY valid JSON:
{{
  "title": "<max 80 chars, Hinglish hook for THIS CLIP>",
  "description": "<English, casual, max 500 chars>",
  "hashtags": ["#Shorts", "<max 4 more>"],
  "search_terms": ["<max 5 search terms>"]
}}"""

    try:
        response = _get_ai().generate_seo_text(
            prompt=prompt,
            system_instruction=_SYSTEM,
        )
        parsed = _parse_json_response(response)
        if not parsed or "title" not in parsed:
            log.warning("[seo_only] AI returned unparseable for %s", clip_id)
            return None

        result = _enforce_limits(parsed, is_shorts=True)

        # Quality gate — reject generic garbage
        if not _validate_seo_quality(result):
            log.warning("[seo_only] Quality gate rejected %s: title='%s'",
                       clip_id, result.get('title', '')[:60])
            return None

        return result
    except Exception as e:
        log.error("[seo_only] AI failed for %s: %s", clip_id, e)
        return None


# ── Main runner ──────────────────────────────────────────────────────────────

def run_seo_only(
    clips_dir: str,
    highlights_yaml: Optional[str] = None,
    transcript_json: Optional[str] = None,
    video_title: str = "",
    skip_existing: bool = True,
    inter_clip_sleep: float = 2.0,
) -> Dict:
    """Generate SEO metadata for all clips in a directory.

    Returns dict with processed/failed/skipped counts.
    """
    clips = discover_clips(clips_dir, skip_existing=skip_existing)
    if not clips:
        log.info("[seo_only] no clips to process")
        return {"processed": 0, "failed": 0, "skipped": 0}

    # Auto-discover highlights yaml if not provided
    if not highlights_yaml:
        for candidate in Path(clips_dir).parent.glob("*.yaml"):
            highlights_yaml = str(candidate)
            break

    # Auto-discover video title from metadata
    if not video_title:
        meta_file = Path(clips_dir).parent / "input" / "video_metadata.json"
        if not meta_file.exists():
            meta_file = Path("input") / "video_metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    video_title = json.load(f).get("title", "")
            except Exception:
                pass

    processed = 0
    failed = 0

    for idx, clip in enumerate(clips):
        clip_id = clip["clip_id"]
        log.info("[seo_only] [%d/%d] %s", idx + 1, len(clips), clip_id)

        transcript = _load_clip_transcript(
            clip_id, clips_dir, highlights_yaml, transcript_json
        )

        seo = _generate_seo_for_clip(clip_id, transcript, video_title)
        if seo:
            meta_path = Path(clips_dir) / f"{clip_id}_metadata.json"
            meta = {
                "clip_id": clip_id,
                "title": seo.get("title", clip_id),
                "description": seo.get("description", ""),
                "hashtags": seo.get("hashtags", ["#Shorts"]),
                "search_terms": seo.get("search_terms", []),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": "seo_only",
            }
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            processed += 1
            log.info("[seo_only] ✅ %s → %s", clip_id, seo.get("title", "?")[:60])
        else:
            failed += 1
            log.warning("[seo_only] ❌ %s — AI generation failed", clip_id)

        # Breathing room between API calls
        if idx < len(clips) - 1:
            time.sleep(inter_clip_sleep)

    result = {"processed": processed, "failed": failed, "skipped": 0}
    log.info("[seo_only] DONE: %d processed, %d failed", processed, failed)
    return result
