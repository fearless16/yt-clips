"""caption_engine.py — Evidence-grounded Instagram Reel caption writer.

REAL-ONLY policy: every entity and hashtag in the produced package must
trace to the supplied InstaEvidencePack. Post-generation audit reuses the
YT entity-grounding scrub pattern; violations get ONE corrective LLM repair
pass; a still-dirty result raises CaptionPolicyError (fail loud -> failed
marker upstream).
"""
import json
import re
import threading
from datetime import datetime, timezone

from automation.seo.entity_grounding import audit_written_copy_llm

HOOK_MAX_CHARS = 55
CAPTION_TARGET_MIN = 150
CAPTION_TARGET_MAX = 250
CAPTION_HARD_MAX = 2200
HASHTAG_COUNT = 5
BRAND_SUFFIX = " – CricketWithPrajjwal"

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_HASHTAG_TOKEN = re.compile(r"#[A-Za-z_][A-Za-z0-9_]*")
_TAG_SAFE_RE = re.compile(r"[^a-z0-9]")

_SYSTEM = (
    "You are the Instagram Reels caption engine for @cricketwithprajjwal2.0 "
    "cricket clips. REAL-ONLY policy: every player, team, score, venue, "
    "series, moment and hashtag MUST come from the supplied evidence pack — "
    "never invent or embellish an entity. Write ROMANIZED Hindi/Hinglish "
    "only; Devanagari script is forbidden. No engagement bait (never ask "
    "for likes, shares, comments, follows, tags). Return ONLY valid JSON."
)

_USER_TMPL = """INSTAGRAM REELS CAPTION TASK — EVIDENCE PACK (the ONLY allowed source):
{evidence_json}

COMPLETE CLIP TRANSCRIPT:
{transcript}

Source video title: {video_title}

Return ONLY this valid JSON object:
{{{{
  "caption": "<hook line>\\n<body>",
  "hashtags": ["#Tag1", "#Tag2", "#Tag3", "#Tag4", "#Tag5"],
  "audio_name": "{{Moment}} – CricketWithPrajjwal"
}}}}

HARD CONTRACT:
- LANGUAGE: FULL ENGLISH ONLY. No Hindi, no Hinglish, no romanized
  Hindi words anywhere in caption or audio_name.
- HOOK LINE (first line): max {hook_max} chars — this is the Reels-tab fold.
  Pattern: "{{Player/moment}} {{outcome}}! {{Series}}". The primary
  keyword (main player/moment from the evidence) must sit inside those
  first {hook_max} chars.
- CAPTION BODY: total caption target {target_min}-{target_max} chars
  (hard API max {hard_max}, hashtags excluded).
- EXACTLY {tag_count} hashtags, taken ONLY from validated_hashtags/seeds
- VARY the tag selection across posts: start your tiered pick at index
  {rotation_start} (mod pool size) of the allowed list, keeping tiers intact
  above. Tier mix: 1 broad + 2 mid series/team + 1 long-tail moment +
  1 rotating matchday tag. Never #reels #viral #explore or any generic tag.
- Exactly ONE genuine reply-driving question in the body (a real cricket
  question fans answer). No engagement bait anywhere.
- Only entities present in the evidence pack. No Devanagari characters.
"""


class CaptionPolicyError(Exception):
    """Raised when caption output violates the REAL-ONLY policy after the
    single corrective repair pass."""


_ai_instance = None
_ai_lock = threading.Lock()


def _get_ai():
    global _ai_instance
    if _ai_instance is not None:
        return _ai_instance
    with _ai_lock:
        if _ai_instance is None:
            from utils.ai_client import AIClient
            _ai_instance = AIClient()
    return _ai_instance


def _parse_json_response(raw):
    text = str(raw or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(text[start:], start):
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(text[start:i + 1])
                            if isinstance(data, dict):
                                return data
                        except json.JSONDecodeError:
                            return None
                        break
    try:
        data = json.loads(re.sub(r"'", '"', text))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _norm_tag(value):
    return _TAG_SAFE_RE.sub("", str(value or "").lstrip("#").casefold())


def _seed_text(seed):
    if isinstance(seed, dict):
        seed = seed.get("phrase") or ""
    return str(seed or "").strip()


def _allowed_tag_set(evidence_pack):
    allowed = {}
    for item in evidence_pack.get("validated_hashtags") or []:
        tag = str((item or {}).get("tag") or "").strip().lstrip("#")
        if tag:
            allowed.setdefault(_norm_tag(tag), "#" + tag)
    for seed in evidence_pack.get("seed_phrases") or []:
        words = [w for w in _TAG_SAFE_RE.split(_seed_text(seed)) if w]
        if words:
            display = "".join(w.capitalize() for w in words)[:30]
            allowed.setdefault(_norm_tag(display), "#" + display)
    return allowed


def _entity_corpus(evidence_pack):
    corpus = set()
    for name in evidence_pack.get("roster") or []:
        folded = str(name).strip().casefold()
        if folded:
            corpus.add(folded)
            corpus.update(t for t in _TAG_SAFE_RE.split(folded) if len(t) >= 3)
    for seed in evidence_pack.get("seed_phrases") or []:
        folded = _seed_text(seed).casefold()
        if folded:
            corpus.add(folded)
            corpus.update(t for t in _TAG_SAFE_RE.split(folded) if len(t) >= 4)
    for item in evidence_pack.get("validated_hashtags") or []:
        stem = _norm_tag((item or {}).get("tag"))
        if stem:
            corpus.add(stem)
    return corpus


def _primary_keyword(evidence_pack, transcript, video_title):
    context = f"{transcript or ''}\n{video_title or ''}".casefold()
    for name in evidence_pack.get("roster") or []:
        if str(name).strip().casefold() in context:
            return str(name).strip()
    for seed in evidence_pack.get("seed_phrases") or []:
        text = _seed_text(seed)
        if text.casefold() in context:
            return text
    roster = evidence_pack.get("roster") or []
    if roster:
        return str(roster[0]).strip()
    seeds = evidence_pack.get("seed_phrases") or []
    if seeds:
        return _seed_text(seeds[0])
    return "Cricket Moment"


def _build_audio_name(evidence_pack, transcript, video_title):
    moment = _primary_keyword(evidence_pack, transcript, video_title)
    moment = re.sub(r"\s+", " ", moment).strip()
    if len(moment) > 32:
        moment = moment[:32].rstrip()
    return f"{moment}{BRAND_SUFFIX}"


def _build_prompt(evidence_pack, transcript, video_title,
                  rotation_offset: int = 0):
    return _USER_TMPL.format(
        rotation_start=int(rotation_offset),
        evidence_json=json.dumps(evidence_pack, ensure_ascii=False),
        transcript=str(transcript or "")[:4000],
        video_title=video_title or "(unknown)",
        hook_max=HOOK_MAX_CHARS,
        target_min=CAPTION_TARGET_MIN,
        target_max=CAPTION_TARGET_MAX,
        hard_max=CAPTION_HARD_MAX,
        tag_count=HASHTAG_COUNT,
    )


def _attempt(prompt):
    try:
        raw = _get_ai().generate_text(prompt, system_instruction=_SYSTEM)
    except Exception:
        return None
    return _parse_json_response(raw)


def _scrub_names(text, unsupported_names):
    for name in unsupported_names:
        pattern = re.compile(r"\b" + re.escape(str(name)) + r"\b",
                             re.IGNORECASE)
        text = pattern.sub("", text)
    lines = [re.sub(r"[ \t]{2,}", " ", ln).strip(" -–—:;")
             for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def _validate_and_scrub(candidate, evidence_pack, transcript, video_title,
                        clip_id):
    """Enforce the caption contract; scrub what can be salvaged locally.

    Returns (package_dict_or_None, violations_list).
    """
    if not isinstance(candidate, dict):
        return None, ["model returned no parsable JSON object"]

    violations = []

    caption_raw = str(candidate.get("caption") or "")
    had_deva = bool(_DEVANAGARI_RE.search(caption_raw))
    caption = _DEVANAGARI_RE.sub("", caption_raw)

    inline_tags = _HASHTAG_TOKEN.findall(caption)
    caption = _HASHTAG_TOKEN.sub("", caption)

    caption = _scrub_names(caption, _audit_unsupported(
        candidate, caption, transcript, video_title, clip_id))
    if had_deva:
        violations.append(
            "Devanagari script is forbidden — romanized Hindi/Hinglish only")

    lines = [ln.strip() for ln in caption.split("\n") if ln.strip()]
    hook = lines[0] if lines else ""
    body_lines = lines[1:]
    body_text = " ".join(body_lines).strip()

    if not hook:
        violations.append("caption is empty")
    else:
        if len(hook) > HOOK_MAX_CHARS:
            violations.append(
                f"hook line must be <= {HOOK_MAX_CHARS} chars "
                f"(got {len(hook)})")
        corpus = _entity_corpus(evidence_pack)
        hook_fold = hook.casefold()
        if not any(entity in hook_fold for entity in corpus):
            violations.append(
                "primary keyword (player/moment from the evidence pack) "
                f"missing within the first {HOOK_MAX_CHARS} chars")

    if "?" not in body_text:
        violations.append(
            "exactly one genuine reply-driving question required in body")
    elif body_text.count("?") > 3:
        violations.append(
            "multiple questions read as engagement bait; keep one")

    allowed = _allowed_tag_set(evidence_pack)
    valid_tags = []
    seen_tags = set()
    for tag in list(candidate.get("hashtags") or []) + inline_tags:
        norm = _norm_tag(tag)
        if norm and norm in allowed and norm not in seen_tags:
            seen_tags.add(norm)
            valid_tags.append(allowed[norm])
    topped_up = []
    if len(valid_tags) < HASHTAG_COUNT:
        # Deterministic REAL-ONLY top-up: fill the shortfall from the
        # allowed pool (rotation order) instead of failing the whole
        # caption over one slipped tag. Invented tags are still dropped.
        rotation = getattr(_validate_and_scrub, "_rotation", 0)
        pool = list(allowed.items())
        start = (int(rotation) % len(pool)) if pool else 0
        ordered = pool[start:] + pool[:start]
        for norm, display in ordered:
            if len(valid_tags) >= HASHTAG_COUNT:
                break
            if norm not in seen_tags:
                seen_tags.add(norm)
                valid_tags.append(display)
                topped_up.append(display)
    if len(valid_tags) != HASHTAG_COUNT:
        violations.append(
            f"exactly {HASHTAG_COUNT} hashtags from validated_hashtags/"
            f"seeds required (got {len(valid_tags)}); invented tags are "
            "rejected")

    body_len = len(f"{hook}\n{body_text}".strip())
    if body_len < CAPTION_TARGET_MIN or body_len > CAPTION_TARGET_MAX:
        violations.append(
            f"caption length must be {CAPTION_TARGET_MIN}-"
            f"{CAPTION_TARGET_MAX} chars excluding hashtags "
            f"(got {body_len})")

    tag_block = " ".join(valid_tags)
    full_caption = f"{hook}\n" + "\n".join(body_lines) + \
        ("\n\n" + tag_block if tag_block else "")
    if len(full_caption) > CAPTION_HARD_MAX:
        violations.append(
            f"caption exceeds hard max {CAPTION_HARD_MAX} chars")

    package = {
        "caption": full_caption.strip(),
        "hashtags": valid_tags,
        "audio_name": _build_audio_name(evidence_pack, transcript,
                                        video_title),
        "tags_topped_up": topped_up,
    }
    return package, violations


def _audit_unsupported(candidate, caption_text, transcript, video_title,
                       clip_id):
    lines = [ln for ln in caption_text.split("\n") if ln.strip()]
    hook = lines[0] if lines else ""
    description = " ".join(lines[1:])
    audit = audit_written_copy_llm(
        clip_id, transcript,
        title=hook,
        description=description,
        video_title=video_title,
    )
    return [str(n) for n in (audit.get("unsupported_entities") or [])]


def _repair_prompt(user_prompt, previous, violations):
    return (
        user_prompt
        + "\n\nCORRECTION REQUIRED — your previous JSON violated these rules:\n"
        + "".join(f"- {problem}\n" for problem in violations)
        + "\nPrevious JSON (reference only — do not copy blindly):\n"
        + json.dumps(previous, ensure_ascii=False)[:1200]
        + "\nRegenerate the COMPLETE corrected metadata now, following "
        "every original rule (hook <=55 chars with primary keyword inside, "
        "150-250 chars total, exactly 5 hashtags only from the pack's "
        "validated_hashtags/seeds, one reply-driving question, no "
        "Devanagari). Return ONLY the JSON object."
    )


class InsufficientEvidenceError(RuntimeError):
    """Pack cannot yield the required distinct hashtags — no LLM spend."""


def _rotation_offset(evidence_pack) -> int:
    """Stable per-pack offset so identical pools rotate across posts."""
    import hashlib
    raw = json.dumps(
        [str((i or {}).get("tag") or "") for i in
         evidence_pack.get("validated_hashtags") or []],
        sort_keys=True) + datetime.now(timezone.utc).strftime("%Y%m%d")
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16)


def write_caption(evidence_pack, transcript, video_title) -> dict:
    """Write one grounded IG package: {caption, hashtags, audio_name}.

    Raises InsufficientEvidenceError BEFORE any LLM spend when the pack
    cannot possibly yield HASHTAG_COUNT distinct tags. Raises
    CaptionPolicyError when even the single corrective repair pass still
    violates the REAL-ONLY contract.
    """
    evidence_pack = dict(evidence_pack or {})
    allowed = _allowed_tag_set(evidence_pack)
    if len(allowed) < HASHTAG_COUNT:
        raise InsufficientEvidenceError(
            f"evidence pack yields only {len(allowed)} distinct hashtags "
            f"but {HASHTAG_COUNT} are required; refusing to invent tags")
    clip_id = re.sub(r"\W+", "-",
                     str(video_title or "insta-caption"))[:40] or "insta"
    rotation_offset = _rotation_offset(evidence_pack)
    user_prompt = _build_prompt(
        evidence_pack, transcript, video_title,
        rotation_offset=rotation_offset)

    _validate_and_scrub._rotation = int(rotation_offset)

    candidate = _attempt(user_prompt)
    package, violations = _validate_and_scrub(
        candidate, evidence_pack, transcript, video_title, clip_id)

    if violations:
        repaired = _attempt(
            _repair_prompt(user_prompt, candidate or {}, violations))
        package, violations = _validate_and_scrub(
            repaired, evidence_pack, transcript, video_title, clip_id)
        if violations:
            raise CaptionPolicyError(
                "Instagram caption failed REAL-ONLY policy after repair: "
                + "; ".join(violations))

    return package
