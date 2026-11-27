"""Per-clip YouTube Shorts packaging: title, cover words, description, tags.

Design (cancer register C4):
- One grounded LLM call per clip, at most one corrective retry. Each call has
  its own timeout, so a clip's packaging time is bounded per clip, never by
  a batch-wide deadline.
- Validation REPAIRS (trim, dedupe, cap) instead of blocking. The only
  rejection is a fact problem (a player/team named that nobody said), and
  that rejection routes to a deterministic template for owner review.
- Packaging never decides whether a clip is cricket: the discovery gate did
  that once, upstream.
- ``build_metadata`` is pure (no disk); ``write_metadata`` owns persistence,
  so a caller that gave up waiting can simply not write a late result.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from automation.artifacts import atomic_json, valid_metadata, validate_clip_id
from automation.seo.cricket_context import correct_cricket_spelling, find_canonical_entities
from utils.config import load_config
from utils.devanagari import to_roman
from utils.logger import get_logger

log = get_logger("seo")
cfg = load_config()

PACKAGING_VERSION = "grounded_v5"
TITLE_LIMIT = 100           # YouTube hard limit
DESCRIPTION_BYTES = 4800    # margin under YouTube's 5000 bytes
TAG_CHARS = 500             # YouTube tag-field limit including quoting
_EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿️]")
_HASHTAG = re.compile(r"#[^\s#]+")
_SLOP = ("you won't believe", "must watch", "gone wrong", "shocking", "watch till the end",
         "jaw-dropping", "mind-blowing", "insane moment")

_ai = None


class SEOGenerationError(Exception):
    """Kept for callers that catch it; build_metadata itself never raises it."""


def _get_ai():
    global _ai
    if _ai is None:
        from utils.ai_client import AIClient
        _ai = AIClient()
    return _ai


def _settings() -> dict:
    seo = cfg.get("seo", {}) or {}
    return {
        "title_max": max(20, min(TITLE_LIMIT, int(seo.get("title_max_chars", 60)))),
        "timeout": max(10.0, float(seo.get("clip_timeout_seconds", 90))),
        "max_hashtags": max(1, int(seo.get("max_hashtags", 3))),
        "max_tags": max(3, int(seo.get("max_tags", 12))),
        "description_target": max(80, int(seo.get("description_target_chars", 350))),
    }


SYSTEM = """You package one YouTube Short from an independent Hindi/Hinglish cricket
live-commentary channel. You only describe what the clip transcript actually says.

Return one JSON object:
{"title": str, "cover_text": str, "description": str, "tags": [str], "hashtags": [str]}

title: the clip's specific premise in <= {title_max} characters, written the way
  the speaker talks (Roman Hinglish or English, never Devanagari). Front-load the
  player, team or moment; for banter or a live moment, front-load what happens
  and do not force cricket into it. It must be a promise the clip keeps. No emojis, no
  ALL CAPS sentences, no generic phrases ("cricket highlights", "must watch").
cover_text: 1 to 3 punchy words for the thumbnail (e.g. "Pehli Ball Out").
description: 2-3 short sentences (~{description_target} chars): what happens or
  what is argued in the clip, then the match/stream context. Say it is independent live
  commentary/reaction. No hashtags inside the description text, no links.
tags: 5-{max_tags} specific search phrases (players, teams, match, moment type,
  Hinglish phrasing viewers would type). No single generic words like "video".
hashtags: up to {max_hashtags}, specific (e.g. "#INDvsNZ"); "#Shorts" is added for you.

Hard rule: name a player or team ONLY if it appears in the transcript or in the
source stream title/description. Never invent scores, results or quotes."""


def _system_prompt(settings: dict) -> str:
    # str.format would trip over the literal JSON braces in SYSTEM.
    text = SYSTEM
    for key in ("title_max", "description_target", "max_tags", "max_hashtags"):
        text = text.replace("{" + key + "}", str(settings[key]))
    return text


# ── grounding ────────────────────────────────────────────────────────────────

def _roman(text: object) -> str:
    return correct_cricket_spelling(to_roman(str(text or "")))


def _entities(text: str) -> set[str]:
    found = find_canonical_entities(_roman(text))
    return {name.casefold() for key in ("players", "teams") for name in found.get(key, [])}


def _ungrounded(meta: dict, evidence: str) -> list[str]:
    """Names in the packaging that nobody said. A name is grounded when the entity
    finder sees it in the evidence OR any of its name parts was actually spoken
    (the finder alone misses context such as "Arre Kohli ne")."""
    allowed = _entities(evidence)
    spoken = set(re.findall(r"\w+", _roman(evidence).casefold()))
    claimed = _entities(" ".join([meta.get("title", ""), meta.get("cover_text", ""), meta.get("description", "")]))
    return sorted(name for name in claimed - allowed
                  if not any(len(part) >= 3 and part in spoken for part in name.split()))


# ── normalisation: repair, never block ──────────────────────────────────────

def _clean(text: object) -> str:
    text = _EMOJI.sub("", to_roman(str(text or "")))
    text = re.sub(r"[*_`]+", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit + 1].rsplit(" ", 1)[0].rstrip(" ,;:-|")
    return head or text[:limit]


def _cut_bytes(text: str, limit: int) -> str:
    raw = text.encode("utf-8")
    return text if len(raw) <= limit else raw[:limit].decode("utf-8", "ignore").rstrip()


def _hashtag(value: object) -> str:
    word = re.sub(r"[^0-9A-Za-z_]", "", to_roman(str(value or "")).lstrip("#"))
    return f"#{word}" if word else ""


def _pack_tags(values, max_tags: int) -> list[str]:
    tags, used, seen = [], 0, set()
    for value in values or []:
        tag = _clean(value).strip(",#")
        tag = re.sub(r"[<>\"]", "", tag)
        key = tag.casefold()
        if not tag or key in seen or len(tag) > 100:
            continue
        cost = len(tag) + (2 if " " in tag else 0) + (1 if tags else 0)
        if used + cost > TAG_CHARS or len(tags) >= max_tags:
            break
        tags.append(tag)
        seen.add(key)
        used += cost
    return tags


def normalise(raw: dict, settings: dict | None = None) -> dict:
    """Coerce an LLM answer into upload-safe metadata. Never raises on style."""
    settings = settings or _settings()
    title = _cut(_HASHTAG.sub("", _clean(raw.get("title"))).strip(" -|:"), settings["title_max"])
    words = _clean(raw.get("cover_text")).split()[:3] or title.split()[:2]
    hashtags = []
    for value in ["#Shorts", *(raw.get("hashtags") or [])]:
        tag = _hashtag(value)
        if tag and tag.casefold() not in {h.casefold() for h in hashtags}:
            hashtags.append(tag)
    hashtags = hashtags[:settings["max_hashtags"]]
    body = _HASHTAG.sub("", _clean(raw.get("description"))).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    description = _cut_bytes(body, DESCRIPTION_BYTES - 200)
    description = (description + "\n\n" + " ".join(hashtags)).strip()
    return {"title": title, "cover_text": " ".join(words), "description": description,
            "tags": _pack_tags(raw.get("tags"), settings["max_tags"]), "hashtags": hashtags}


def _problems(meta: dict, evidence: str) -> list[str]:
    """Fact problems only. Style is already repaired by normalise()."""
    problems = []
    if len(meta["title"]) < 8:
        problems.append("title is empty or too short")
    if not meta["cover_text"]:
        problems.append("cover_text is empty")
    if len(_HASHTAG.sub("", meta["description"]).strip()) < 30:
        problems.append("description is empty")
    lowered = meta["title"].casefold()
    if any(phrase in lowered for phrase in _SLOP):
        problems.append("title uses clickbait filler instead of the clip's premise")
    missing = _ungrounded(meta, evidence)
    if missing:
        problems.append("names not said in the clip or source: " + ", ".join(missing))
    return problems


# ── deterministic fallback ──────────────────────────────────────────────────

_MOMENT_WORDS = ("wicket", "out", "six", "chhakka", "four", "chauka", "catch", "century",
                 "fifty", "drop", "review", "drs", "no ball", "run out", "bowled", "lbw")


def _match_label(video_title: str) -> str:
    teams = find_canonical_entities(_roman(video_title)).get("teams", [])
    return " vs ".join(teams[:2]) if teams else "Cricket"


def template_metadata(transcript: str, video_title: str = "", video_description: str = "",
                      settings: dict | None = None) -> dict:
    """Honest, grounded packaging built only from what was said. Owner reviews it."""
    settings = settings or _settings()
    text = _clean(_roman(transcript))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\s{2,}", text) if len(s.split()) >= 3]
    hook = sentences[0] if sentences else (text or "Live cricket reaction")
    label = _match_label(video_title)
    title = _cut(f"{label}: {hook}" if label != "Cricket" else hook, settings["title_max"])
    low = text.casefold()
    moment = next((w for w in _MOMENT_WORDS if re.search(rf"\b{re.escape(w)}\b", low)), "")
    cover = moment.title() if moment else " ".join(title.split()[:2])
    found = find_canonical_entities(_roman(" ".join([transcript, video_title])))
    description = (f"{_cut(text, 260)}\n\nIndependent live commentary reaction"
                   + (f" from {label}." if label != "Cricket" else "."))
    tags = [*found.get("players", []), *found.get("teams", []),
            f"{label} reaction" if label != "Cricket" else "cricket reaction",
            "cricket shorts", "hindi cricket commentary"]
    hashtags = ["#Shorts", _hashtag(label.replace(" vs ", "vs")) if label != "Cricket" else "#Cricket"]
    return normalise({"title": title, "cover_text": cover, "description": description,
                      "tags": tags, "hashtags": hashtags}, settings)


# ── public API ──────────────────────────────────────────────────────────────

def _prompt(transcript: str, video_title: str, video_description: str, feedback: list[str]) -> str:
    payload = {"clip_transcript": _roman(transcript)[:6000],
               "source_stream_title": _roman(video_title)[:300],
               "source_stream_description": _roman(video_description)[:1500]}
    if feedback:
        payload["fix_these_problems_from_your_last_answer"] = feedback
    return json.dumps(payload, ensure_ascii=False)


def build_metadata(clip_id: str, transcript: str, video_title: str = "",
                   video_description: str = "") -> dict:
    """Return complete metadata. Never raises for provider or quality problems."""
    settings = _settings()
    evidence = " ".join([transcript, video_title, video_description])
    system = _system_prompt(settings)
    feedback: list[str] = []
    error = ""
    for _attempt in range(2):
        try:
            ai = _get_ai()
            raw = json.loads(ai.generate_json(_prompt(transcript, video_title, video_description, feedback),
                                              system, max_tokens=4096, timeout=settings["timeout"]))
            if not isinstance(raw, dict):
                raise ValueError("response is not a JSON object")
            meta = normalise(raw, settings)
            feedback = _problems(meta, evidence)
            if not feedback:
                meta.update(seo_method="llm", provider=getattr(ai, "_last_provider", None),
                            model=getattr(ai, "_last_model", None))
                return _finish(meta, transcript, video_title, video_description)
            error = "; ".join(feedback)
            log.warning("[%s] SEO answer needs correction: %s", clip_id, error)
        except Exception as exc:  # provider/parse failure: carry the real reason (A6/A13)
            error = f"{type(exc).__name__}: {str(exc) or repr(exc)}"
            log.warning("[%s] SEO provider attempt failed: %s", clip_id, error)
            feedback = []
            break
    log.warning("[%s] SEO fell back to template packaging (owner review): %s", clip_id, error)
    meta = template_metadata(transcript, video_title, video_description, settings)
    meta.update(seo_method="template", seo_error=error)
    return _finish(meta, transcript, video_title, video_description)


def _finish(meta: dict, transcript: str, video_title: str, video_description: str) -> dict:
    meta.update(is_shorts=True, ai_generated=meta.get("seo_method") == "llm",
                packaging_version=PACKAGING_VERSION, metadata_valid=True,
                contains_synthetic_media=False,
                seo_inputs={"transcript": transcript, "video_title": video_title,
                            "video_description": video_description})
    return meta


def write_metadata(output_dir: str | Path, clip_id: str, meta: dict) -> dict:
    """Persist metadata (with gameplay credit) and clear any legacy failure marker."""
    from automation.cricket_gameplay import append_gameplay_credit

    validate_clip_id(clip_id)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = append_gameplay_credit(dict(meta), out / f"{clip_id}_gameplay.json")
    if not valid_metadata(meta):
        raise ValueError(f"{clip_id}: refusing to write structurally invalid metadata")
    atomic_json(out / f"{clip_id}_metadata.json", meta)
    (out / f"{clip_id}_seo_failed.json").unlink(missing_ok=True)
    return meta


def generate_seo_for_exported_clip(clip_id: str, transcript: str, output_dir: str,
                                   video_title: str = "", video_description: str = "", **_legacy) -> dict:
    """Build and write metadata for one exported clip. Only disk errors raise."""
    meta = build_metadata(clip_id, transcript, video_title, video_description)
    return write_metadata(output_dir, clip_id, meta)


def retry_failed_seo(output_dir: str, include_templates: bool = True) -> dict:
    """Upgrade template packaging and legacy failure markers once the provider is back.

    Clips that already have a delivery receipt are never repackaged.
    """
    out = Path(output_dir)
    total = recovered = 0
    if not out.is_dir():
        return {"recovered": 0, "total": 0}
    stems = {p.name.removesuffix("_seo_failed.json") for p in out.glob("*_seo_failed.json")}
    if include_templates:
        stems |= {p.name.removesuffix("_metadata.json") for p in out.glob("*_metadata.json")}
    for stem in sorted(stems):
        try:
            validate_clip_id(stem)  # the filename is the identity, never the payload
        except ValueError:
            continue
        clip = out / f"{stem}.mp4"
        if clip.with_name(f"{stem}_delivery.json").exists():
            continue
        marker = clip.with_name(f"{stem}_seo_failed.json")
        meta_path = clip.with_name(f"{stem}_metadata.json")
        source = None
        try:
            if marker.exists():
                source = json.loads(marker.read_text(encoding="utf-8"))
            elif include_templates and meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("seo_method") == "template":
                    source = meta.get("seo_inputs") or {}
        except (OSError, ValueError) as exc:
            log.warning("[retry] %s unreadable: %s", stem, exc)
            continue
        if source is None:
            continue
        total += 1
        meta = build_metadata(stem, str(source.get("transcript", "")),
                              str(source.get("video_title", "")), str(source.get("video_description", "")))
        if meta["seo_method"] == "template" and not marker.exists():
            continue  # still no provider: keep the existing template untouched
        try:
            # A legacy marker always resolves: LLM packaging, or a template for owner review (A4).
            write_metadata(out, stem, meta)
        except (OSError, ValueError) as exc:
            log.error("[retry] %s could not be written: %s", stem, exc)
            continue
        recovered += meta["seo_method"] == "llm"
    return {"recovered": recovered, "total": total}
