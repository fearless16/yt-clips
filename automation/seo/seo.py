"""seo.py — Per-clip SEO generation for Indian cricket Shorts.

Uses restricted SEO models with one strict escalation attempt. Every title is
clip-specific and must promise the same thought that the clip actually opens
with; failed attempts retain their complete research evidence for retry.
"""
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

from utils.config import load_config
from utils.logger import get_logger
from utils.ai_client import AIClient
from .trends import get_trending_context
from automation._cache import TTLCache
from utils.ocr import extract_ocr_entities
from automation.seo.entity_grounding import (
    audit_written_copy_llm,
    extract_grounded_entities_llm,
    name_vouched_by_topics,
)
from .cricket_context import (
    correct_cricket_spelling,
    discover_grounded_player_names,
    find_canonical_entities,
    is_cricket_content,
)
from .context_engine import (
    build_cricket_evidence_pack,
    build_grounded_search_queries,
)

SUGGEST_CACHE = TTLCache(maxsize=16, ttl=600)
TREND_CACHE = TTLCache(maxsize=4, ttl=300)

cfg = load_config()
log = get_logger("seo", cfg["logging"]["log_file"], cfg["logging"]["level"])

# Lazy singleton — do NOT call AIClient() at module level (violates no-side-effects rule)
_ai_instance = None
_ai_lock = __import__("threading").Lock()


def _get_ai() -> "AIClient":
    """Thread-safe lazy AIClient singleton."""
    global _ai_instance
    if _ai_instance is not None:
        return _ai_instance
    with _ai_lock:
        if _ai_instance is None:
            _ai_instance = AIClient()
    return _ai_instance


class SEOGenerationError(Exception):
    """Raised when all AI providers fail during SEO generation."""


def _get_learner_context() -> str:
    """Build evidence-only channel context from the canonical learner."""
    try:
        from shorts_intelligence.bridge import recommendation_context

        return recommendation_context(cfg)
    except Exception as exc:
        log.warning("Shorts Intelligence context unavailable: %s", exc)
        return ""


STOP_WORDS = {
    "i","me","my","you","your","we","our","they","their","this","that","these","those",
    "am","is","are","was","were","be","been","have","has","had","do","does","did",
    "a","an","the","and","or","but","if","as","of","to","in","on","at","for","from",
    "with","by","about","into","over","under","again","then","here","there","when",
    "where","why","how","all","any","more","most","some","such","no","nor","not",
    "only","very",
}

GENERIC_TAGS = {
    "cricket","shorts","viral","trending","youtube","video","sports",
    "highlight","highlights","amazing","awesome","incredible","wow",
}

# Generic search terms that kill channel performance — NEVER let these through
GENERIC_POISON_TERMS = {
    "cricket highlights", "cricket live match", "ipl match video",
    "t20 cricket live", "best cricket moments", "cricket video",
    "sports video", "sports highlights", "cricket live",
    "ipl highlights", "cricket match", "live cricket",
    "cricket best moments", "ipl live",
}

# Generic titles that signal low-effort SEO
GENERIC_TITLES = {
    "cricket highlights", "highlights", "cricket match",
    "ipl highlights", "live cricket", "cricket video",
    "sports highlights", "match highlights",
}

# Vague LLM adjectives that dilute SEO quality — fans never search these and
# the algorithm buries them. Strip from titles so the named entity (what
# people actually type) leads. Kept OUT of the word-count; we raise QUALITY,
# not reduce volume.
_VAGUE_FILLER_RE = re.compile(
    r"\b(?:epic|thrilling|incredible|amazing|awesome|best|shocking|"
    r"stunning|mind[-\s]?blowing|unbelievable|fantastic|sensational|"
    r"breathtaking|jaw[-\s]?dropping|must[-\s]?watch|crazy|insane|"
    r"phenomenal|spectacular|greatest|top|superb|brilliant|classic)\b",
    re.IGNORECASE,
)

PACKAGING_VERSION = "promise_v4_longtail"

# ── Anti-AI-generic gate ────────────────────────────────────────────────────
# Narration slop that LLMs default to. Any hit in public copy fails the
# quality gate; the prompt also bans them so the model self-corrects first.
AI_SLOP_PHRASES = (
    "stay tuned", "is video mein", "dekhte hain", "welcome back",
    "cricket lovers", "dil jhoom", "dhamakedaar", "aapko pasand aayega",
    "toh chaliye", "aaj ke match mein", "hello guys",
    "in this video we will", "in this video, we", "video ko like karo",
    "subscribe karna na bhoolein", "channel ko subscribe",
)

_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

_PROMISE_STOP_WORDS = STOP_WORDS | GENERIC_TAGS | {
    "ka", "ki", "ke", "ko", "ne", "hai", "hain", "tha", "thi", "aur",
    "kyun", "kyo", "bana", "banana", "ban", "sakte", "sakta", "chahiye",
    "short", "clip", "match", "india", "indian", "explained", "analysis",
    "debate", "discussion", "opinion",
}

# Ordinary title words that the two-word person heuristic keeps mistaking
# for player names ('Case Made', 'Pressure Explained', 'Captaincy Shock').
_COMMON_TITLE_WORDS = frozenset({
    "case", "made", "call", "big", "pressure", "explained", "debate",
    "shock", "verdict", "magic", "breakdown", "reaction", "highlights",
    "update", "review", "moment", "moments", "six", "four", "run", "runs",
    "wicket", "wickets", "target", "chase", "collapses", "collapse",
    "innings", "test", "tests", "match", "today", "news", "story", "truth",
    "best", "first", "record", "world", "series", "tour", "squad", "team",
    "cricket", "live", "score", "stream", "watch", "full", "final", "over",
    "powerplay", "session", "spell", "comeback", "turning", "point",
    "captaincy", "coach", "coaching", "masterclass", "lessons", "plan",
    "plans", "tactics", "battle", "rivalry", "preview", "recap",
})


_PROMISE_SYNONYMS = {
    "chhakka": "six", "chakka": "six", "sixer": "six",
    "coaching": "coach", "coached": "coach", "bowled": "wicket",
    "wickets": "wicket", "yorkers": "yorker", "sixes": "six",
}

# ── SEO Model Restrictions ─────────────────────────────────────────────────────
# Only these models are trusted for SEO generation.
# nvidia (nemotron/llama) and groq (llama/grok) produce generic, low-quality SEO.
# qwen3.7-max excluded: returns 401 "not supported for format oa-compat"
SEO_PREFERRED_MODELS = [
    ("opencode", "mimo-v2.5-pro"),
    ("opencode", "deepseek-v4-pro"),
]
SEO_BLOCKED_PROVIDERS = {"nvidia", "groq"}

# ── Prompt ─────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an elite cricket Shorts SEO engine for @cricketwithprajjwal2.0. "
    "The description is written for the YouTube ALGORITHM, not for humans \u2014 "
    "nobody reads it. Your job is maximum keyword surface area that still "
    "reads as natural sentences: pack every verified entity, player name, "
    "team name, match detail, event, and search phrase into 3000-4500 "
    "characters of flowing English prose with emoji section markers. "
    "Evidence boundary: source video title/description, clip transcript, "
    "OCR, live autocomplete suggestions, recent YouTube search titles, and "
    "explicitly verified scorecard facts. Never invent a player, team, "
    "score, venue, event, date, or record. All public copy is simple "
    "ENGLISH only: no Hindi in Roman script, no Devanagari, no Hinglish. "
    "Title = one specific premise (canonical player/team + exact event), "
    "max 60 chars, exactly 1-2 emojis, never 'LIVE' or '#Shorts'. Never "
    "narrate ('in this video', 'stay tuned'), never greet. Return ONLY "
    "valid JSON."
)

_CRICKET_ONLY_PROMPT_TMPL = """CRICKET SHORT \u2014 FULL MATCH CONTEXT:
  Source video title: {video_title}
  Source video description: {video_description}
  Verified scorecard / match facts (live Cricbuzz data): {match_facts}
  Teams in this match: {teams}
  Player roster (verified names you may use): {roster}
  Live YouTube autocomplete (real searches happening now): {trend_topics}
  Recent YouTube search result titles: {research_sources}

COMPLETE CLIP TRANSCRIPT: {transcript}

SEARCH SEEDS (grounded anchors from research \u2014 expand these into long-tail):
{approved_search_queries}

TASK \u2014 build metadata that wins YouTube search for THIS exact clip.

Return ONLY this valid JSON object:
{{
  "title": "<specific English title, max 60 chars, player/team + event>",
  "description": "<3000-4500 chars of keyword-rich English prose>",
  "hashtags": ["#Shorts", "<#SeriesTag like #INDvSL>", "<1 topic tag>"],
  "search_terms": ["<exactly 25 long-tail viewer searches>"],
  "primary_search_terms": ["<4-8 of those terms>"],
  "tags": ["<the same 25 phrases, trimmed to fit the 500-char API field>"]
}}

STRICT RULES:
- TITLE: max {title_max_chars} chars. Lead with the canonical name + the
  exact moment ("Prasidh Krishna Strikes Twice: Sri Lanka Collapse"). One
  clear premise, 1-2 emojis. No LIVE/#Shorts/pipe segments.
- DESCRIPTION ({description_target_chars} chars target, hard range
  {description_min_chars}-{description_max_chars}): written for the
  algorithm, not a human reader.
    * First 125 characters: primary keyword + clip's exact moment.
    * Then flowing English sentences organized with emoji markers
      (\U0001F3CF \u26A1 \U0001F525 \U0001F4CA) into sections: what happens in this clip, verified
      match situation from the scorecard facts, both teams' position,
      every named player's role in this clip, series/match context,
      what happens next in the match.
    * Weave EVERY selected search term into sentences across the body,
      preserving the exact query wording at least once when grammar allows.
      Use the available character budget aggressively; do not stop early just
      because the core event has already been explained. Repetition of key
      entities is good; robotic lists are not.
    * End with the 3 hashtags on one line.
- SEARCH TERMS: choose and RANK exactly 25 long-tail phrases from the
  grounded SEARCH SEEDS above. Prefer phrases supported by live YouTube
  autocomplete and by wording/themes visible in recent YouTube result titles.
  Do not invent a new entity. Mix patterns:
    "{teams_lower} highlights", "<player> bowling today",
    "<player> wickets", "{series_guess} day 4", "cricket shorts",
    "<team> collapse", "<team> target chase". No invented players.
- TAGS: same 25 phrases; drop/shorten as needed to stay under 450 total
  characters (YouTube's API budget). Most specific first.
- HASHTAGS: exactly 3. #Shorts + series tag (#INDvSL pattern) + topic.
- LANGUAGE: English only everywhere. No Hinglish, no Devanagari.
- BANNED anywhere: "in this video", "stay tuned", "welcome back",
  "cricket lovers", "hello guys".
- Never invent a player, team, score, venue, or event not supported by
  the evidence blocks above.
"""

_CRICKET_ONLY_SALVAGE_TMPL = """Generate grounded, keyword-rich metadata for this cricket Short.
Clip transcript: {transcript}
Source title: {video_title}

Return only JSON with title, description, hashtags, search_terms,
primary_search_terms, and tags.
- LANGUAGE: English only everywhere. No Hinglish, no Devanagari.
- Title: max 60 characters; one specific premise plus 1-2 emojis;
  no LIVE/#Shorts.
- Description: 3000-4500 characters of algorithm-facing keyword-rich
  English prose; first 125 chars carry the primary keyword; emoji
  section markers allowed; weave search phrases into sentences.
- Hashtags: exactly 3 including #Shorts.
- Search terms: exactly 25 long-tail grounded cricket phrases.
- Tags: the same phrases trimmed under 450 total characters.
- Never invent a player, team, score, or event. No narration
  ("in this video", "stay tuned").
"""


def _clean_dict_from_description(raw: str) -> str:
    """Extract dict or JSON object from a mixed LLM response string.

    Handles: markdown-wrapped JSON, dict() representation,
    stray backticks, key-only extractions.
    """
    text = raw.strip()

    # Remove markdown code fence markers
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    # If it's a Python dict representation (single-quoted keys), convert to JSON.
    # Only replace single quotes when the first value after { uses single quotes,
    # so we don't corrupt valid double-quoted JSON that starts with "{ ".
    stripped = text.lstrip()
    if stripped.startswith("{'") or stripped.startswith("{ '"):
        text = re.sub(r"'", '"', text)

    return text.strip()


def _seo_config_int(key: str, default: int, low: int, high: int) -> int:
    try:
        value = cfg.get("seo", {}).get(key, default)
        if isinstance(value, bool):
            return default
        value = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(value, low), high)


def _seo_config_float(key: str, default: float, low: float, high: float) -> float:
    try:
        value = cfg.get("seo", {}).get(key, default)
        if isinstance(value, bool):
            return default
        value = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(value, low), high)


def _description_min_chars() -> int:
    return _seo_config_int("description_min_chars", 3000, 2000, 4500)


def _description_target_chars() -> int:
    return _seo_config_int("description_target_chars", 3500, 2500, 4800)


def _description_max_chars() -> int:
    return _seo_config_int("description_max_chars", 4500, 3000, 4950)


def _shorts_hashtag_cap() -> int:
    """Read seo.max_hashtags from config, crash-proof and clamped to [1, 15].

    Null/float/string config values or an out-of-range number silently fall
    back to the default of 3 instead of raising (a misconfig must never take
    down the whole SEO path or zero out every Short's hashtags).
    """
    try:
        cap = int(cfg.get("seo", {}).get("max_hashtags", 3))
    except (TypeError, ValueError):
        cap = 3
    return min(max(cap, 1), 15)


_HASHTAG_TOKEN = re.compile(r"#[A-Za-z_\u0900-\u097F][A-Za-z0-9_\u0900-\u097F]*")

# Live-framing detection — this channel uploads on-demand Shorts clips, never
# live streams. Any tag/term that frames the clip as live ('live score', 'live
# stream', 'livestream', Devanagari 'लाइव') misleads the algorithm and
# suppresses CTR.
_LIVE_FRAMING_RE = re.compile(r"\blive\b", re.IGNORECASE)
_DEVA_LIVE_TERMS = ("लाइव", "लाईव")

# Words that form live-framing compounds with 'live' as a standalone segment:
# livestream / livecricket / live_score / liveshorts (prefix), cricketlive /
# matchlive / shorts (suffix), or multi-token cricketlivematch. Only these
# combine; '#Liverpool', '#Lively', '#Relive', '#Alive' are plain words and
# must survive.
_LIVE_FRAMING_WORDS = (
    "stream", "streaming", "score", "scoring", "cricket", "match",
    "football", "commentary", "updates", "coverage", "tv", "watch",
    "ball", "batting", "ipl", "t20", "blog", "shorts", "short",
)
# Longest-first so multi-token compounds group cleanly without backtracking
# surprises (stream/streaming, score/scoring, short/shorts).
_LIVE_FRAMING_WORDS_OR = "|".join(
    sorted(_LIVE_FRAMING_WORDS, key=len, reverse=True))
_LIVE_COMPOUND_RE = re.compile(
    r"\blive(?:_|-)?(?:%s)+\b"           # livescore, liveshorts, livecricketmatch
    r"|\b(?:%s)+live\b"                  # cricketlive, matchlive, scorelive
    r"|\b(?:%s)+live(?:_|-)?(?:%s)+\b"   # cricketlivematch, scorelivetv
    % (_LIVE_FRAMING_WORDS_OR, _LIVE_FRAMING_WORDS_OR,
       _LIVE_FRAMING_WORDS_OR, _LIVE_FRAMING_WORDS_OR),
    re.IGNORECASE,
)


def _is_live_framed_search_term(term: str) -> bool:
    """True if a search term frames the clip as a live stream.

    Catches: standalone 'live' word ('ipl 2026 live'), separator forms
    ('live-stream', 'live_score'), single-token compounds ('livestream',
    'livecricket', 'cricketlive'), and Devanagari 'लाइव'/'लाईव'. On-demand
    words that merely contain 'live' as a substring ('alive and kicking',
    'olive', 'deliver', 'lively') survive untouched.
    """
    t = term.lower()
    if any(deva in t for deva in _DEVA_LIVE_TERMS):
        return True
    # Separator-normalized: 'live-score'/'live_score' → 'live score'
    norm = re.sub(r"[_\-]+", " ", t)
    if _LIVE_FRAMING_RE.search(norm):
        return True
    return bool(_LIVE_COMPOUND_RE.search(t))


def _is_live_framed_hashtag(tag: str) -> bool:
    """True if a hashtag is a live-framing tag (#LiveCricket, #CricketLive).

    Live-framing tags are compound words where 'live' is a standalone framing
    segment (#LiveScore, #CricketLive, #live, #लाइव). Plain words that merely
    contain 'live' (#Alive, #Relive, #Liverpool, #Lively) survive.
    """
    name = tag.lstrip("#").lower()
    if any(deva in name for deva in _DEVA_LIVE_TERMS):
        return True
    if _LIVE_FRAMING_RE.search(re.sub(r"[_\-]+", " ", name)):
        return True
    return bool(_LIVE_COMPOUND_RE.search(name))


def _cap_description_hashtags(description: str, hashtags: List[str]) -> str:
    """Strip every hashtag token from the description body, then re-append a
    canonical capped block.

    YouTube renders the first 3 #-prefixed tokens in the description as chips
    above the title, and the rest as plain body text. To guarantee the VISIBLE
    hashtags on an uploaded Short are exactly the capped, #Shorts-first set,
    we remove all stray ``#Word`` tokens from the body and append the final
    ``hashtags`` list as one clean block at the end.
    """
    body = _HASHTAG_TOKEN.sub("", description or "")
    body = re.sub(r"[ \t]{2,}", " ", body).rstrip()
    block = " ".join(hashtags)
    max_chars = _description_max_chars()
    if not block:
        return _truncate_at_word(body, max_chars)
    suffix = "\n\n" + block
    return _truncate_at_word(body, max(0, max_chars - len(suffix))) + suffix


def _clean_title(title: object, cap: int) -> str:
    """Remove vague filler/format labels and truncate at a word boundary."""
    text = str(title or "")
    # Search-first titles should spend their tiny character budget on the
    # named entity + exact moment, not generic LLM adjectives.
    text = _VAGUE_FILLER_RE.sub("", text)
    text = _LIVE_FRAMING_RE.sub("", text)
    text = _LIVE_COMPOUND_RE.sub("", text)
    text = re.sub(r"(?i)(?:^|\s)#shorts\b", " ", text)
    text = re.sub(r"^[\s|:;,.!\-–—🔴🟢⚫]+", "", text)
    text = re.sub(r"\s*\|\s*\|+", " | ", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip(" |:;,.!-–—")
    if len(text) <= cap:
        return text
    cut = text[:cap].rstrip()
    if len(text) > cap and not text[cap].isspace() and " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" |:;,.!-–—")


def _promise_tokens(text: object) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", str(text or "").casefold()):
        token = _PROMISE_SYNONYMS.get(token, token)
        if len(token) >= 3 and token not in _PROMISE_STOP_WORDS:
            tokens.add(token)
    return tokens


def _promise_alignment_score(title: str, transcript: str, description: str) -> float:
    """Score whether the public promise matches the spoken clip and opening copy."""
    title_tokens = _promise_tokens(title)
    if not title_tokens:
        return 0.0
    clip_overlap = len(title_tokens & _promise_tokens(transcript))
    opening_overlap = len(title_tokens & _promise_tokens(description[:320]))
    clip_score = min(1.0, clip_overlap / 2.0)
    opening_score = min(1.0, opening_overlap / 2.0)
    return round((clip_score + opening_score) / 2.0, 3)


def _llm_repair_seo(
    clip_id: str,
    user_prompt: str,
    previous_result: Dict,
    problems: List[str],
    allowed_people_note: str,
    transcript: str,
    video_title: str,
    is_shorts: bool,
) -> Optional[Dict]:
    """One corrective regeneration pass through the writer LLM.

    The original context prompt plus an explicit violation list goes back to
    the model; the repaired JSON is re-enforced. Returns None on any failure
    so callers can fail loudly instead of shipping template junk.
    """
    repair_prompt = (
        str(user_prompt)
        + "\n\nCORRECTION REQUIRED — your previous JSON violated these rules:\n"
        + "".join(f"- {problem}\n" for problem in problems)
        + "\nPrevious JSON (reference only — do not copy blindly):\n"
        + json.dumps(previous_result, ensure_ascii=False)[:1200]
        + f"\n\nAllowed people whitelist: {allowed_people_note}\n"
        + "Regenerate the COMPLETE corrected metadata now, following every "
        "original rule (length budgets, 25 long-tail search terms, emoji "
        "title). Return ONLY the JSON object."
    )
    try:
        repaired = _attempt_seo_generation(
            clip_id, repair_prompt, transcript, video_title, is_shorts,
            sys_instruction=_SYSTEM,
            salvage_tmpl=_CRICKET_ONLY_SALVAGE_TMPL.replace(
                "{description_target_chars}", str(_description_target_chars())
            ).replace("{description_max_chars}", str(_description_max_chars())),
        )
    except Exception as exc:  # noqa: BLE001 — repair is best-effort
        log.warning("[%s] LLM repair unavailable: %s", clip_id, exc)
        return None
    if not isinstance(repaired, dict) or not str(repaired.get("title") or "").strip():
        return None
    return _enforce_limits(repaired, is_shorts=is_shorts)


def _enforce_limits(item: Dict, fallback_terms: List[str] = None, is_shorts: bool = True) -> Dict:
    """Ensure title length, description length, hashtag count, search term count.

    Enforces config caps: title≤100, description below YouTube's 5000-char
    ceiling, hashtags≤15 (Shorts:
    seo.max_hashtags, default 3), terms≤30.
    Strips generic poison terms from search_terms.
    For Shorts, the description hashtag block is scrubbed to the same cap so
    the visible hashtags match the metadata (upload uses description text).
    YouTube API limits: title=100 chars, description=5000 bytes, tags=500 chars.
    We stay under API limits with margin for safety.
    """
    out = dict(item)
    out["is_shorts"] = is_shorts
    try:
        title_cap = int(cfg.get("seo", {}).get("title_max_chars", 70))
    except (TypeError, ValueError):
        title_cap = 70
    title_cap = max(30, min(100, title_cap))
    out["title"] = _ensure_title_emoji(
        _clean_title(out.get("title"), title_cap), title_cap
    )
    out["description"] = _truncate_at_word(
        str(out.get("description") or ""), _description_max_chars()
    )

    htags = out.get("hashtags") or []
    if isinstance(htags, str):
        htags = [htags]
    elif not isinstance(htags, list):
        htags = []
    seen = set()
    deduped = []
    for t in htags:
        if not isinstance(t, str):
            continue
        t_clean = t.lstrip("#").strip()
        if _is_live_framed_hashtag(t_clean):
            continue
        if t_clean.lower() not in seen:
            seen.add(t_clean.lower())
            deduped.append(f"#{t_clean}")

    if is_shorts:
        # Shorts get 2-3 hashtags max; #Shorts always leads the list.
        cap = _shorts_hashtag_cap()
        shorts_hashtags = [t for t in deduped if t.lstrip("#").lower() == "shorts"]
        others = [t for t in deduped if t.lstrip("#").lower() != "shorts"]
        if not shorts_hashtags:
            shorts_hashtags = ["#Shorts"]
        deduped = shorts_hashtags + others
    else:
        cap = 15
    out["hashtags"] = deduped[:cap]
    if is_shorts:
        out["description"] = _cap_description_hashtags(out["description"], out["hashtags"])

    terms = out.get("search_terms") or []
    if isinstance(terms, str):
        terms = [terms]
    elif not isinstance(terms, list):
        terms = []
    seen = set()
    deduped_t = []
    for st in terms:
        if not isinstance(st, str):
            continue
        st_clean = st.strip()
        # Strip generic poison terms that kill channel performance
        if st_clean.lower() in GENERIC_POISON_TERMS:
            continue
        # Strip live-framing terms — on-demand Shorts must not be tagged live
        if _is_live_framed_search_term(st_clean):
            continue
        if st_clean.lower() not in seen:
            seen.add(st_clean.lower())
            deduped_t.append(st_clean)
    try:
        term_cap = int(cfg.get("seo", {}).get("max_search_terms", 26))
    except (TypeError, ValueError):
        term_cap = 15
    term_cap = max(5, min(30, term_cap)) if is_shorts else 30
    out["search_terms"] = deduped_t[:term_cap]

    primary = out.get("primary_search_terms") or []
    if isinstance(primary, str):
        primary = [primary]
    elif not isinstance(primary, list):
        primary = []
    allowed = {term.casefold(): term for term in out["search_terms"]}
    selected = []
    for value in primary:
        clean = str(value or "").strip()
        canonical = allowed.get(clean.casefold())
        if canonical and canonical not in selected:
            selected.append(canonical)
    # Older valid responses did not name primary phrases. Promote only phrases
    # already present in prose; never stuff missing research queries into copy.
    if not selected:
        description_key = re.sub(r"\s+", " ", out["description"]).casefold()
        selected = [
            term for term in out["search_terms"]
            if re.sub(r"\s+", " ", term).casefold() in description_key
        ]
    primary_cap = _seo_config_int("max_primary_search_terms", 8, 1, 12)
    out["primary_search_terms"] = selected[:primary_cap]

    # Defense-in-depth: models sometimes emit a 'tags' key outside the JSON
    # schema; upload.py merges it straight into the YouTube API tags. Filter it
    # through the same poison + live-framing rules as search_terms.
    tags = out.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    elif not isinstance(tags, list):
        # Malformed (dict/int) tags must never leak keys/values as API tags.
        tags = []
    deduped_tags = []
    seen_tags = set()
    for tg in tags:
        if not isinstance(tg, str):
            continue
        tg_clean = tg.strip().lstrip("#")
        if tg_clean.lower() in GENERIC_POISON_TERMS:
            continue
        if _is_live_framed_search_term(tg_clean):
            continue
        if tg_clean.lower() not in seen_tags:
            seen_tags.add(tg_clean.lower())
            deduped_tags.append(tg_clean)
    tag_budget = _seo_config_int("max_tag_chars", 450, 50, 500)
    budgeted_tags = []
    used_chars = 0
    for tag in deduped_tags:
        added = len(tag) + (1 if budgeted_tags else 0)
        if used_chars + added > tag_budget:
            continue
        budgeted_tags.append(tag)
        used_chars += added
    out["tags"] = budgeted_tags

    return out


def _ensure_title_emoji(title: str, cap: int) -> str:
    """Shorts feed titles carry at least one emoji (mobile-first packaging).

    Appends the channel-neutral cricket ball when the model forgot one.
    Deterministic — never invents content, only completes the format.
    """
    if not title or _EMOJI_RE.search(title):
        return title
    if len(title) + 2 <= cap:
        return title.rstrip() + " 🏏"
    return title


def _truncate_at_word(text: str, limit: int) -> str:
    """Cut prose at a whitespace boundary so no half-word survives.

    Hard slicing once produced 'explained wi' from 'explained with'; a
    downstream \\bwi\\b match then hallucinated 'West Indies' as an entity.
    """
    text = str(text or "")
    if len(text) <= limit:
        return text
    cut = text[:limit]
    nxt = text[limit:limit + 1]
    if nxt and not nxt.isspace():
        space = cut.rfind(" ")
        if space >= int(limit * 0.5):
            cut = cut[:space]
    return cut.rstrip()


def _has_ai_slop(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", str(text or "").lower())
    return any(phrase in lowered for phrase in AI_SLOP_PHRASES)


def _validate_seo_quality(item: Dict) -> bool:

    """Quality gate: reject SEO that would hurt channel performance.

    Returns True if the SEO output is clip-specific and worth uploading.
    Returns False if it's generic garbage that should be dropped.
    """
    title = (item.get("title") or "").strip()
    description = (item.get("description") or "").strip()

    # 1. Title must exist and be meaningful
    if not title or len(title) < 10:
        return False

    # 2. Title must not be a known generic pattern
    if title.lower().rstrip("!.?") in GENERIC_TITLES:
        return False

    # 3. Description must use the configured short-form evidence budget.
    if len(description) < _description_min_chars():
        return False

    # 4. No Devanagari script anywhere in public copy (kills discoverability)
    if _DEVANAGARI_RE.search(title):
        return False
    if _DEVANAGARI_RE.search(description):
        return False
    hashtags_text = " ".join(
        str(tag) for tag in (item.get("hashtags") or []) if isinstance(tag, str)
    )
    if _DEVANAGARI_RE.search(hashtags_text):
        return False

    # 5. AI narration slop is banned in public copy — it reads as template
    #    output and erodes trust in a feed where every word earns swipes.
    if _has_ai_slop(title) or _has_ai_slop(description):
        return False

    # 6. Natural embedding check — reject if keyword list/tag block is
    #    appended at the end of description (signals lazy SEO).
    #    Look for patterns like:
    #      - "Tags:", "Keywords:", "Search terms:" at end (label + colon)
    #      - Comma-separated single-word dump on last line
    last_200 = description[-200:].lower()
    if re.search(r'\b(?:search[_ ]terms?|tags?|keywords?)\s*:', last_200):
        return False
    # Check last line for high comma density (keyword dump indicator)
    last_line = description.split('\n')[-1].strip().lower()
    words = last_line.split(',')
    if len(words) >= 6 and all(len(w.strip().split()) <= 2 for w in words) \
       and not last_line.rstrip().endswith(('.', '!', '?')):
        return False

    return True


def _parse_json_response(text: str) -> Optional[Dict]:
    """Parse LLM JSON output, handling markdown, truncation, and Python dict quirks.

    Attempts: direct json.loads, then markdown code fence stripping,
    then truncation repair (add closing braces, trim incomplete values),
    then Python dict-to-JSON conversion. Returns None on total failure.
    """
    if not text or not text.strip():
        return None

    text = _clean_dict_from_description(text)

    # Helper: try to parse, optionally repairing truncation
    def _try_parse(s: str) -> Optional[Dict]:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    # 1. Direct parse
    result = _try_parse(text)
    if result is not None:
        return result

    # 2. Extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        result = _try_parse(json_match.group(1))
        if result is not None:
            return result

    # 3. Look for { onwards — first try full string from first brace,
    #    then try to find a balanced {…} object (handles prose wrappers like
    #    "Here is the result: {...} — done.")
    brace_start = text.find("{")
    if brace_start >= 0:
        # 3a. Full candidate from first brace
        candidate = text[brace_start:]
        result = _try_parse(candidate)
        if result is not None:
            return result

        # 3b. Try to find balanced brace span (handles trailing prose)
        depth = 0
        in_str = False
        esc = False
        end_pos = None
        for i, ch in enumerate(candidate):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"' and not esc:
                in_str = not in_str
            if not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break
        if end_pos is not None:
            balanced = candidate[:end_pos]
            result = _try_parse(balanced)
            if result is not None:
                return result

        # 3c. Truncation repair on full candidate
        fixed = _repair_truncated_json(candidate)
        if fixed is not None:
            return fixed

    # 4. Single quotes fallback
    try:
        single_quoted = re.sub(r"'", '"', text)
        result = _try_parse(single_quoted)
        if result is not None:
            return result
        # 4b. With truncation repair
        brace_start = single_quoted.find("{")
        if brace_start >= 0:
            fixed = _repair_truncated_json(single_quoted[brace_start:])
            if fixed is not None:
                return fixed
    except Exception:
        pass

    return None


def _repair_truncated_json(s: str) -> Optional[Dict]:
    """Attempt to repair truncated JSON by closing open braces/brackets
    and trimming incomplete trailing values."""
    if not s:
        return None
    # Find the last complete key-value pair before truncation
    # Strategy: try progressively shorter suffixes
    for _ in range(min(5, len(s) // 10 + 1)):
        # Remove trailing whitespace
        s = s.rstrip()
        if not s:
            break
        # Count unclosed braces/brackets
        opens = s.count("{") + s.count("[")
        closes = s.count("}") + s.count("]")
        if opens == closes:
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
        # Trim last line/partial value
        last_newline = s.rfind("\n")
        if last_newline > s.rfind("{"):
            s = s[:last_newline]
        else:
            # Try closing unclosed braces
            needed = opens - closes
            if needed > 0:
                try:
                    return json.loads(s + "}" * needed)
                except json.JSONDecodeError:
                    pass
            break
    return None


# ── Main SEO generation ─────────────────────────────────────────────────────────

def generate_clip_seo(
    clip_id: str,
    transcript: str,
    video_title: str = "",
    scorecard: str = "",
    trend_topics: Optional[List[str]] = None,
    live_stream_url: str = "",
    teams: Optional[List[str]] = None,
    is_shorts: bool = True,
    fallback_terms: Optional[List[str]] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    video_path: str = "",
    video_description: str = "",
    approved_search_queries: Optional[List[str]] = None,
    match_facts: Optional[List[str]] = None,
    grounded_players: Optional[List[str]] = None,
    grounded_aliases: Optional[Dict[str, str]] = None,
    research_sources: Optional[List[Dict]] = None,
) -> Dict:
    """Generate grounded, promise-aligned metadata for one cricket clip."""
    trend_topics = trend_topics or []
    teams = teams or []
    match_facts = match_facts or ([scorecard] if scorecard else [])
    grounded_players = grounded_players or []
    grounded_aliases = grounded_aliases or {}
    research_sources = research_sources or []

    # Magic anchor: if the caller supplied no real trend/query evidence, pull
    # LIVE high-search-volume cricket entities (YouTube autocomplete, Google
    # Trends IN, verified match facts) so the LLM seeds its keywords with
    # QUALITY entities fans actually search (Vaibhav Sooryavanshi, RCB, "why
    # Jadeja sad IPL") instead of inventing generic filler ("epic moment").
    # Network misses never block SEO — we degrade to the empty anchors we had.
    if not trend_topics and not approved_search_queries:
        try:
            _ctx = get_trending_context(
                domain="cricket", region="IN",
                video_title=video_title, video_description=video_description,
                transcript=transcript,
            )
            if not trend_topics:
                trend_topics = list(_ctx.get("topics") or [])[:12]
            if not approved_search_queries:
                approved_search_queries = list(_ctx.get("search_queries") or [])[:30]
            if not research_sources:
                research_sources = list(_ctx.get("sources") or [])
        except Exception as _exc:  # pragma: no cover - defensive
            log.warning("live trend context unavailable: %s", _exc)

    if not transcript:
        raise SEOGenerationError(f"SEO blocked for {clip_id}: empty/non-cricket transcript")
    transcript = correct_cricket_spelling(
        transcript, grounded_players, grounded_aliases
    )
    video_title = correct_cricket_spelling(
        video_title, grounded_players, grounded_aliases
    )
    video_description = correct_cricket_spelling(
        video_description, grounded_players, grounded_aliases
    )
    scorecard = correct_cricket_spelling(
        scorecard, grounded_players, grounded_aliases
    )
    grounding_context = " ".join((video_title, video_description, scorecard, transcript))
    # Reasoning-model grounding reads the (possibly transliterated) evidence
    # before the static gate: when it vouches for real cricket entities, the
    # keyword-based gate must not veto a genuine clip.
    llm_ground = extract_grounded_entities_llm(
        clip_id, transcript, video_title=video_title,
        video_description=video_description,
    )
    vouched_topics = list(llm_ground.get("topic_phrases", []))
    if not is_cricket_content(transcript, grounding_context) and not (
        llm_ground.get("players") or llm_ground.get("teams")
    ):
        raise SEOGenerationError(f"SEO blocked for {clip_id}: non-cricket content")

    teams_str = ", ".join(teams)
    # Keep substantially more live evidence in the model context.  The old
    # [:5] cap silently threw away most autocomplete/SERP signals.
    trend_str = "\n".join(f"- {topic}" for topic in trend_topics[:20]) if trend_topics else ""

    # This channel is cricket-only. Football/general prompt routes remain
    # unavailable even if a mixed-niche source reaches this function.
    sys_instruction = _SYSTEM
    prompt_tmpl = _CRICKET_ONLY_PROMPT_TMPL
    salvage_tmpl = (
        _CRICKET_ONLY_SALVAGE_TMPL
        .replace("{description_target_chars}", str(_description_target_chars()))
        .replace("{description_max_chars}", str(_description_max_chars()))
    )
    default_title = "Cricket Match"
    default_teams = "N/A"

    # OPTIONAL OCR entity extraction (degraded gracefully if easyocr unavailable)
    ocr_entities = extract_ocr_entities(video_path) if video_path else {}
    ocr_text = ""
    if ocr_entities:
        sb = "; ".join(ocr_entities.get("scoreboard", []))
        pn = "; ".join(ocr_entities.get("player_names", []))
        os = "; ".join(ocr_entities.get("on_screen_text", []))
        ocr_text = f"\n\nON-SCREEN TEXT (OCR from video):\n  Scoreboard: {sb or '(none)'}\n  Player names: {pn or '(none)'}\n  Raw text: {os or '(none)'}"
    research_context = {
        "match_facts": match_facts,
        "topics": trend_topics,
        "search_queries": approved_search_queries or [],
        "sources": research_sources,
        "player_names": grounded_players,
        "player_aliases": grounded_aliases,
    }
    evidence_pack = build_cricket_evidence_pack(
        video_title=video_title,
        video_description=video_description,
        clip_transcript=transcript,
        ocr_entities=ocr_entities,
        research_context=research_context,
    )
    approved_queries = evidence_pack["approved_search_queries"]
    grounded_entities = evidence_pack["grounded_entities"]

    # Entity allow-lists must exist BEFORE any repair path runs: the
    # corrective LLM prompt needs them, so compute them right after the
    # evidence pack instead of inside the late validation block.
    player_catalog = list(dict.fromkeys([
        *grounded_entities["players"], *grounded_players,
        *llm_ground.get("players", []),
    ]))
    clip_players = set(find_canonical_entities(transcript, player_catalog)["players"])
    allowed_people = {str(name).casefold() for name in clip_players}
    allowed_people.update(str(p).casefold() for p in llm_ground.get("players", []))

    grounding_context = " ".join((
        video_title,
        video_description,
        transcript,
        " ".join(evidence_pack["match_facts"]),
        ocr_text,
    ))
    user_prompt = prompt_tmpl.format(
        video_title=video_title or default_title,
        video_description=video_description or "N/A",
        match_facts="\n".join(evidence_pack["match_facts"]) or "N/A",
        trend_topics=trend_str or "N/A",
        research_sources="\n".join(
            f"- {title}"
            for source in evidence_pack["sources"]
            if isinstance(source, dict) and source.get("kind") == "youtube_search"
            for title in (source.get("titles") or [])
            if str(title).strip()
        ) or "N/A",
        teams=teams_str or default_teams,
        roster=", ".join(dict.fromkeys([
            *grounded_players,
            *grounded_entities["players"],
            *llm_ground.get("players", []),
        ])) or "none verified — use team names only",
        teams_lower=teams_str.lower() if teams_str else "india vs sri lanka",
        series_guess=" vs ".join(teams[:2]).lower() if len(teams) >= 2 else "today cricket match",
        transcript=transcript,
        approved_search_queries="\n".join(f"- {query}" for query in approved_queries),
        title_max_chars=_seo_config_int("title_max_chars", 60, 30, 100),
        description_target_chars=_description_target_chars(),
        description_min_chars=_description_min_chars(),
        description_max_chars=_description_max_chars(),
    )
    if ocr_text:
        user_prompt += ocr_text

    user_prompt += (
        "\n\nENTITY GROUNDING (canonical cricket knowledge):\n"
        f"Grounded players: {', '.join(grounded_entities['players']) or 'none'}\n"
        f"Grounded teams: {', '.join(grounded_entities['teams']) or 'none'}\n"
        "Resolve aliases exactly as listed above. Never reinterpret an alias as "
        "a technical acronym and never invent another player."
    )

    # Inject learner intelligence into the prompt
    learner_ctx = _get_learner_context()
    if learner_ctx:
        user_prompt += (
            "\n\nCHANNEL INTELLIGENCE (measured Shorts outcomes only):\n"
            f"{learner_ctx}\nUse only supported priors; never override grounded facts."
        )

    # Call AI with parallel fastest-first
    result = _attempt_seo_generation(clip_id, user_prompt, transcript, video_title,
                                     is_shorts,
                                     provider_override=provider_override,
                                     model_override=model_override,
                                     sys_instruction=sys_instruction,
                                     salvage_tmpl=salvage_tmpl)
    result = _enforce_limits(result, is_shorts=is_shorts)

    # Copy audit: the writer model can hallucinate celebrity names the clip
    # never discusses (e.g. Ravindra Jadeja in an IND-SL gloves debate) and
    # capitalize ordinary phrases ('Massive Target'). One reasoning call
    # splits the copy into unsupported names (scrubbed below) and evidence-
    # backed topics (treated as vouched by the validators).
    copy_audit = audit_written_copy_llm(
        clip_id, transcript,
        title=str(result.get("title") or ""),
        description=str(result.get("description") or ""),
        video_title=video_title, video_description=video_description,
    )
    supported_topics = vouched_topics + [
        t for t in copy_audit.get("supported_topics", []) if t not in vouched_topics
    ]
    unsupported_patterns = []
    for name in copy_audit.get("unsupported_entities", []):
        pattern = re.compile(
            r"\b" + re.escape(str(name)) + r"\b", re.IGNORECASE
        )
        unsupported_patterns.append(pattern)
        for key in ("title", "description"):
            if key in result and isinstance(result[key], str):
                cleaned = pattern.sub("", result[key])
                result[key] = re.sub(r"[ \t]{2,}", " ",
                                     cleaned).replace(" ,", ",").strip(" -–—:;")
        for key in ("tags", "search_terms", "hashtags"):
            items = result.get(key) or []
            result[key] = [
                item for item in items if not pattern.search(str(item))
            ]
        log.warning("[%s] Scrubbed unsupported entity from copy: %s",
                    clip_id, name)
    if unsupported_patterns:
        # Upstream research can bake a hallucinated name into the approved
        # queries themselves; those must never reach search_terms or the
        # rebuilt fallback copy.
        before = len(approved_queries)
        approved_queries = [
            query for query in approved_queries
            if not any(p.search(str(query)) for p in unsupported_patterns)
        ]
        if len(approved_queries) != before:
            log.warning(
                "[%s] Dropped %d contaminated search queries",
                clip_id, before - len(approved_queries),
            )
        floor = _seo_config_int("min_search_terms", 24, 1, 30)
        if len(approved_queries) < floor:
            # The dedupe budget in the evidence pack was consumed by the
            # dirty queries, so deterministic local combos never made it in.
            # Rebuild them now from the same grounded evidence and top up.
            rebuilt = build_grounded_search_queries(
                video_title, video_description, transcript,
                suggestions=[],
                player_names=grounded_players,
                player_aliases=grounded_aliases or {},
            )
            existing = {str(q).casefold() for q in approved_queries}
            added = 0
            for query in rebuilt:
                if len(approved_queries) >= floor:
                    break
                if any(p.search(str(query)) for p in unsupported_patterns):
                    continue
                if str(query).casefold() in existing:
                    continue
                approved_queries.append(query)
                existing.add(str(query).casefold())
                added += 1
            log.warning(
                "[%s] Rebuilt %d grounded search queries after scrub",
                clip_id, added,
            )
        # Scrubbing the hallucinated name can gut the title's whole promise
        # (e.g. 'Ravindra Jadeja Six Magic' loses every token). Swap in an
        # evidence-built title instead of letting the alignment gate fail.
        min_alignment = _seo_config_float(
            "min_promise_alignment_score", 0.5, 0.0, 1.0
        )
        if _promise_alignment_score(
            str(result.get("title") or ""),
            transcript,
            str(result.get("description") or ""),
        ) < min_alignment:
            repaired = _llm_repair_seo(
                clip_id, user_prompt, result,
                ["Scrubbing the ungrounded name gutted the title's core "
                 "promise tokens. Rebuild the title around the exact moment "
                 "in the transcript using only verified entities."],
                ", ".join(sorted(allowed_people)) or "none yet — use teams only",
                transcript, video_title, is_shorts,
            )
            if not repaired:
                raise SEOGenerationError(
                    f"SEO blocked for {clip_id}: scrubbed title lost promise "
                    "alignment and LLM repair failed"
                )
            if _promise_alignment_score(
                str(repaired.get("title") or ""),
                transcript,
                str(repaired.get("description") or ""),
            ) < min_alignment:
                raise SEOGenerationError(
                    f"SEO blocked for {clip_id}: LLM repair still failed "
                    "promise alignment"
                )
            result = repaired
            log.warning(
                "[%s] Title promise rebuilt via LLM repair", clip_id,
            )

    # Title attribution check is CATALOG-based: a real canonical player named
    # in the title who the clip transcript never mentions is exactly the
    # 'Southee -> Saud Shakeel' hallucination class. Ordinary capitalized
    # phrases ('Straight Talk') are not names and must never trigger here;
    # creative hallucinations outside any catalog are the copy-audit's job.
    unknown_title_people = [
        name for name in find_canonical_entities(
            str(result.get("title") or ""), player_catalog
        )["players"]
        if name.casefold() not in allowed_people
    ]
    if unknown_title_people:
        repaired = _llm_repair_seo(
            clip_id, user_prompt, result,
            [f"Title names ungrounded player(s): "
             f"{', '.join(unknown_title_people)}. Rewrite the title around "
             "the transcript moment using ONLY verified roster/team names."],
            ", ".join(sorted(allowed_people)) or "none yet — use teams only",
            transcript, video_title, is_shorts,
        )
        if not repaired:
            raise SEOGenerationError(
                f"SEO blocked for {clip_id}: ungrounded title name(s) "
                f"{', '.join(unknown_title_people)} and LLM repair failed"
            )
        still_unknown = [
            name for name in find_canonical_entities(
                str(repaired.get("title") or ""), player_catalog
            )["players"]
            if name.casefold() not in allowed_people
        ]
        if still_unknown:
            raise SEOGenerationError(
                f"SEO blocked for {clip_id}: LLM repair kept ungrounded "
                f"title name(s) {', '.join(still_unknown)}"
            )
        result = repaired
        log.warning(
            "[%s] Title rewritten via LLM repair after ungrounded name(s): %s",
            clip_id, ", ".join(unknown_title_people),
        )

    # Hard entity-presence gate for search-first packaging. The old checks only
    # rejected WRONG named entities; a generic but transcript-aligned title like
    # "Huge Six in Death Overs" could still ship. If verified player/team
    # evidence exists, require at least one canonical entity in the title and
    # give the writer exactly one repair pass before failing closed.
    grounded_teams = set(grounded_entities["teams"]) | {
        str(t) for t in llm_ground.get("teams", [])
    }

    def _title_has_grounded_entity(candidate: object) -> bool:
        entities = find_canonical_entities(str(candidate or ""), player_catalog)
        title_people = {str(p).casefold() for p in entities.get("players", [])}
        title_teams = {str(t).casefold() for t in entities.get("teams", [])}
        valid_people = set(allowed_people)
        valid_teams = {str(t).casefold() for t in grounded_teams}
        return bool((title_people & valid_people) or (title_teams & valid_teams))

    # Only enforce a grounded entity when the title is otherwise promise-aligned
    # (alignment OK). If alignment is off the hard promise gate (further down)
    # rejects loudly, and an ungrounded player name is owned by its own repair
    # pass — so this stays a single corrective pass that never masks a mismatch.
    _entity_min_alignment = _seo_config_float(
        "min_promise_alignment_score", 0.5, 0.0, 1.0
    )
    if (allowed_people or grounded_teams) and _promise_alignment_score(
        str(result.get("title") or ""), transcript,
        str(result.get("description") or ""),
    ) >= _entity_min_alignment and not _title_has_grounded_entity(result.get("title")):
        repaired = _llm_repair_seo(
            clip_id, user_prompt, result,
            [
                "Title is generic/entity-less. Lead with at least one VERIFIED "
                "canonical player or team name that is supported by the clip "
                "evidence, then state the exact searchable moment. Do not use "
                "vague filler adjectives."
            ],
            ", ".join(sorted(allowed_people)) or "none verified — use teams only",
            transcript, video_title, is_shorts,
        )
        if not repaired:
            raise SEOGenerationError(
                f"SEO blocked for {clip_id}: title has no grounded entity and "
                "LLM repair failed"
            )
        if not _title_has_grounded_entity(repaired.get("title")):
            raise SEOGenerationError(
                f"SEO blocked for {clip_id}: repaired title still has no "
                "grounded player/team entity"
            )
        result = repaired
        log.warning(
            "[%s] Title rewritten via LLM repair to enforce grounded entity lead",
            clip_id,
        )

    public_copy_text = " ".join([
        str(result.get("title", "")),
        str(result.get("description", "")),
        " ".join(str(item) for item in result.get("hashtags", []) or []),
        " ".join(str(item) for item in result.get("search_terms", []) or []),
    ])
    api_tag_text = " ".join(str(item) for item in result.get("tags", []) or [])
    public_copy_players = set(
        find_canonical_entities(public_copy_text, player_catalog)["players"]
    ) - clip_players
    api_tag_players = set(
        find_canonical_entities(api_tag_text, player_catalog)["players"]
    ) - clip_players
    if api_tag_players and not public_copy_players:
        raise SEOGenerationError(
            f"SEO blocked for {clip_id}: ungrounded entities "
            + ", ".join(sorted(api_tag_players))
        )

    rendered_text = " ".join([
        public_copy_text,
        " ".join(str(item) for item in result.get("tags", []) or []),
    ])
    rendered_entities = find_canonical_entities(
        rendered_text, player_catalog
    )

    def _title_vouched(name: str) -> bool:
        return name_vouched_by_topics(name, supported_topics)

    extra_players = {
        player for player in set(rendered_entities["players"]) - clip_players
        if not _title_vouched(player)
    }
    # grounded_teams was computed earlier for the title entity-presence gate.
    extra_teams = set(rendered_entities["teams"]) - grounded_teams
    if extra_players:
        repaired = _llm_repair_seo(
            clip_id, user_prompt, result,
            [f"Copy/tags mention ungrounded entities: "
             f"{', '.join(sorted(extra_players))}. Remove every unverified "
             "name from title, description, tags and search terms; use only "
             "verified roster/team entities."],
            ", ".join(sorted(allowed_people)) or "none yet — use teams only",
            transcript, video_title, is_shorts,
        )
        if not repaired:
            raise SEOGenerationError(
                f"SEO blocked for {clip_id}: ungrounded entities "
                f"{', '.join(sorted(extra_players))} and LLM repair failed"
            )
        result = repaired
        log.warning(
            "[%s] Copy regenerated via LLM repair after ungrounded entity(ies)",
            clip_id,
        )
        rendered_text = " ".join([
            str(result.get("title", "")), str(result.get("description", "")),
            " ".join(str(item) for item in result.get("hashtags", []) or []),
            " ".join(str(item) for item in result.get("search_terms", []) or []),
            " ".join(str(item) for item in result.get("tags", []) or []),
        ])
        rendered_entities = find_canonical_entities(
            rendered_text, player_catalog
        )
        extra_players = {
            player for player in set(rendered_entities["players"]) - clip_players
            if not _title_vouched(player)
        }
        extra_teams = set(rendered_entities["teams"]) - grounded_teams
    if extra_players or extra_teams:
        extras = sorted(extra_players | extra_teams)
        raise SEOGenerationError(
            f"SEO blocked for {clip_id}: ungrounded entities {', '.join(extras)}"
        )

    min_queries = _seo_config_int("min_search_terms", 8, 1, 15)
    max_queries = _seo_config_int("max_search_terms", 26, min_queries, 40)
    # Respect the model as a RANKER while keeping grounding deterministic.
    # Previously we discarded every LLM-selected search term and replaced the
    # list with approved_queries in fixed order, making the model's search-term
    # reasoning pointless.
    approved_clean = list(dict.fromkeys(
        str(query).strip() for query in approved_queries if str(query).strip()
    ))
    approved_by_key = {query.casefold(): query for query in approved_clean}
    ranked = []
    for query in result.get("search_terms") or []:
        clean = str(query).strip()
        canonical = approved_by_key.get(clean.casefold())
        if canonical and canonical not in ranked:
            ranked.append(canonical)
    for query in approved_clean:
        if query not in ranked:
            ranked.append(query)
    output_queries = ranked[:max_queries]
    result["search_terms"] = output_queries
    if not min_queries <= len(output_queries) <= max_queries:
        raise SEOGenerationError(
            f"SEO blocked for {clip_id}: expected {min_queries}-{max_queries} "
            f"grounded search queries, got {len(output_queries)}"
        )
    min_primary = _seo_config_int("min_primary_search_terms", 4, 1, 10)
    max_primary = _seo_config_int("max_primary_search_terms", 8, min_primary, 14)
    output_keys = {str(query).strip().casefold() for query in output_queries}
    primary_queries = []
    for query in result.get("primary_search_terms") or []:
        clean = str(query).strip()
        if clean.casefold() in output_keys and clean not in primary_queries:
            primary_queries.append(clean)
        if len(primary_queries) >= max_primary:
            break
    for query in output_queries:
        if len(primary_queries) >= min_primary:
            break
        if query not in primary_queries:
            primary_queries.append(query)
    result["primary_search_terms"] = primary_queries
    description_key = re.sub(r"\s+", " ", str(result.get("description") or "")).casefold()
    missing_queries = [
        str(query) for query in primary_queries
        if re.sub(r"\s+", " ", str(query)).strip().casefold() not in description_key
    ]
    if missing_queries:
        # Preserve exact primary-query coverage without wasting budget on an
        # opaque pipe-delimited debug-looking block.
        suffix = "\n\nAlso relevant to searches for " + "; ".join(missing_queries) + "."
        max_chars = _description_max_chars()
        base = str(result.get("description") or "").rstrip()
        result["description"] = (
            _truncate_at_word(base, max(0, max_chars - len(suffix))) + suffix
        )

    alignment = _promise_alignment_score(
        str(result.get("title") or ""), transcript, str(result.get("description") or "")
    )
    minimum_alignment = _seo_config_float(
        "min_promise_alignment_score", 0.5, 0.0, 1.0
    )
    if alignment < minimum_alignment:
        raise SEOGenerationError(
            f"SEO blocked for {clip_id}: title promise does not match clip/opening "
            f"(alignment={alignment:.3f})"
        )
    result["packaging_version"] = PACKAGING_VERSION
    result["promise_alignment_score"] = alignment

    return result


def _attempt_seo_generation(
    clip_id: str,
    user_prompt: str,
    transcript: str,
    video_title: str,
    is_shorts: bool,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    sys_instruction: str = _SYSTEM,
    salvage_tmpl: str = _CRICKET_ONLY_SALVAGE_TMPL,
) -> Dict:
    """Attempt AI SEO with two-tier escalation.

    Tier 1: parallel fastest-first AI racing.
    Tier 2: single-provider escalation with stricter prompt.

    Never degrades to keyword fallback — raises on total failure.
    """
    ai_result = _generate_ai_seo(clip_id, user_prompt, transcript, is_shorts,
                                  provider_override=provider_override,
                                  model_override=model_override,
                                  sys_instruction=sys_instruction)
    if ai_result:
        ai_result["ai_generated"] = True
        return ai_result

    esc_result = _escalation_seo(clip_id, user_prompt, transcript, video_title, is_shorts,
                                  provider_override=provider_override,
                                  model_override=model_override,
                                  sys_instruction=sys_instruction,
                                  salvage_tmpl=salvage_tmpl)
    if esc_result:
        esc_result["ai_generated"] = True
        return esc_result

    raise SEOGenerationError(
        f"AI SEO failed for {clip_id} — all providers exhausted"
    )


def _generate_ai_seo(clip_id: str, user_prompt: str,
                     transcript: str, is_shorts: bool,
                     provider_override: Optional[str] = None,
                     model_override: Optional[str] = None,
                     sys_instruction: str = _SYSTEM) -> Optional[Dict]:
    """Parallel fastest-first AI generation.

    Fires available models concurrently, returns first valid JSON.
    """
    try:
        ai = _get_ai()
        if model_override or provider_override:
            response = ai.generate_text(
                prompt=user_prompt,
                system_instruction=sys_instruction,
                prefer_model=model_override,
                prefer_provider=provider_override,
            )
        else:
            # Use SEO-restricted models (OpenCode Go only)
            response = ai.generate_seo_text(
                prompt=user_prompt,
                system_instruction=sys_instruction,
            )
        if not response or not response.strip():
            log.warning("[%s] AI returned empty response", clip_id)
            return None

        parsed = _parse_json_response(response)
        if not parsed:
            log.warning("[%s] AI returned unparseable: %.100s", clip_id, response)
            return None

        if "title" not in parsed or "description" not in parsed:
            log.warning("[%s] AI response missing required keys: %s",
                       clip_id, list(parsed.keys()))
            return None

        result = _enforce_limits(parsed, is_shorts=is_shorts)
        provider = ai.get_used_provider()
        model = ai.get_used_model()
        if isinstance(provider, str) and provider:
            result["provider"] = provider
        if isinstance(model, str) and model:
            result["model"] = model

        # Quality gate: reject generic garbage from AI
        if not _validate_seo_quality(result):
            log.warning("[%s] AI SEO failed quality gate — dropping: title='%s'",
                       clip_id, result.get('title', '')[:60])
            return None

        return result
    except Exception as e:
        log.warning("[%s] AI generation failed: %s", clip_id, e)
        return None


def _escalation_seo(clip_id: str, user_prompt: str,
                    transcript: str, video_title: str,
                    is_shorts: bool,
                    provider_override: Optional[str] = None,
                    model_override: Optional[str] = None,
                    sys_instruction: str = _SYSTEM,
                    salvage_tmpl: str = _CRICKET_ONLY_SALVAGE_TMPL) -> Optional[Dict]:
    """Escalation SEO: stricter prompt with more context.

    Called when Tier 1 fails. Uses a different model/provider if available.
    """
    # Try single-provider generation with a more constrained prompt
    salvage_rules = salvage_tmpl.format(
        video_title=video_title or "Video Match",
        transcript=transcript,
    )
    salvage_prompt = user_prompt.rstrip() + "\n\nESCALATION RULES:\n" + salvage_rules
    try:
        ai = _get_ai()
        if model_override or provider_override:
            response = ai.generate_text(
                prompt=salvage_prompt,
                system_instruction=sys_instruction,
                prefer_model=model_override,
                prefer_provider=provider_override,
            )
        else:
            # Use SEO-restricted models (OpenCode Go only)
            response = ai.generate_seo_text(
                prompt=salvage_prompt,
                system_instruction=sys_instruction,
            )
        if not response:
            return None
        parsed = _parse_json_response(response)
        if parsed and "title" in parsed:
            result = _enforce_limits(parsed, is_shorts=is_shorts)
            provider = ai.get_used_provider()
            model = ai.get_used_model()
            if isinstance(provider, str) and provider:
                result["provider"] = provider
            if isinstance(model, str) and model:
                result["model"] = model
            # Quality gate on escalation too — no generic garbage
            if not _validate_seo_quality(result):
                log.warning("[%s] Escalation SEO failed quality gate", clip_id)
                return None
            return result
        return None

    except Exception as e:
        log.warning("[%s] Escalation SEO failed: %s", clip_id, e)
        return None


# ── High-level export integration ───────────────────────────────────────────────

def generate_seo_for_exported_clip(
    clip_id: str,
    transcript: str,
    output_dir: str,
    video_title: str = "",
    scorecard: str = "",
    trend_topics: Optional[List[str]] = None,
    live_stream_url: str = "",
    teams: Optional[List[str]] = None,
    is_shorts: bool = True,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    video_path: str = "",
    video_description: str = "",
    approved_search_queries: Optional[List[str]] = None,
    match_facts: Optional[List[str]] = None,
    grounded_players: Optional[List[str]] = None,
    grounded_aliases: Optional[Dict[str, str]] = None,
    research_sources: Optional[List[Dict]] = None,
) -> Dict:
    """Generate SEO for an already-exported clip and write metadata to disk.

    If AI generation fails after escalation, writes a ``*_seo_failed.json``
    marker so the retry queue can pick it up later. Never emits generic SEO.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metadata_path = Path(output_dir) / f"{clip_id}_metadata.json"

    # Research at the clip seam. Source-level research misses the exact spoken
    # opinion/event and was the main cause of generic or mismatched metadata.
    if approved_search_queries is None:
        try:
            research = get_trending_context(
                domain="cricket",
                region="IN",
                video_title=video_title,
                video_description=video_description,
                transcript=transcript,
                include_live_stream_url=False,
            )
        except Exception as exc:
            log.warning("[%s] Per-clip research unavailable: %s", clip_id, exc)
            research = {}
        scorecard = scorecard or research.get("scorecard", "")
        trend_topics = trend_topics or research.get("topics", [])
        teams = teams or research.get("teams", [])
        approved_search_queries = research.get("search_queries", [])
        match_facts = match_facts or research.get("match_facts", [])
        grounded_players = grounded_players or research.get("player_names", [])
        grounded_aliases = grounded_aliases or research.get("player_aliases", {})
        research_sources = research_sources or research.get("sources", [])

    retry_payload = {
        "clip_id": clip_id,
        "transcript": transcript,
        "video_title": video_title,
        "video_description": video_description,
        "scorecard": scorecard,
        "trend_topics": trend_topics or [],
        "teams": teams or [],
        "is_shorts": is_shorts,
        "approved_search_queries": approved_search_queries or [],
        "match_facts": match_facts or [],
        "grounded_players": grounded_players or [],
        "grounded_aliases": grounded_aliases or {},
        "research_sources": research_sources or [],
        "video_path": video_path,
    }

    try:
        result = generate_clip_seo(
            clip_id=clip_id,
            transcript=transcript,
            video_title=video_title,
            scorecard=scorecard,
            trend_topics=trend_topics,
            live_stream_url=live_stream_url,
            teams=teams,
            is_shorts=is_shorts,
            provider_override=provider_override,
            model_override=model_override,
            video_path=video_path,
            video_description=video_description,
            approved_search_queries=approved_search_queries,
            match_facts=match_facts,
            grounded_players=grounded_players,
            grounded_aliases=grounded_aliases,
            research_sources=research_sources,
        )
        if result.get("ai_generated") is False:
            log.warning("[%s] AI SEO failed — writing failure marker", clip_id)
            marker_path = Path(output_dir) / f"{clip_id}_seo_failed.json"
            with open(marker_path, "w", encoding="utf-8") as f:
                json.dump(retry_payload, f, ensure_ascii=True)
            if metadata_path.exists():
                metadata_path.unlink()
            result["_seo_failed"] = True
        else:
            tmp_path = metadata_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, metadata_path)
            marker_path = Path(output_dir) / f"{clip_id}_seo_failed.json"
            marker_path.unlink(missing_ok=True)
        return result
    except Exception as e:
        log.error("[%s] SEO generation failed: %s", clip_id, e)
        marker_path = Path(output_dir) / f"{clip_id}_seo_failed.json"
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(retry_payload, f, ensure_ascii=True)
        return {"_seo_failed": True, "error": str(e)}


def process_all_seo(highlights_path: str, output_dir: str,
                    video_path: str = "") -> str:
    """
    Sequential per-clip SEO. Loads highlights YAML, fetches trend context once,
    then generates SEO for each clip one at a time.
    """
    from .trends import get_trending_context
    import yaml

    h_path = Path(highlights_path)
    if not h_path.exists():
        log.error("Highlights not found: %s", h_path)
        return ""

    with open(h_path, "r", encoding="utf-8") as f:
        highlights = yaml.safe_load(f) or {}

    # Load video metadata once
    video_title = ""
    video_description = ""
    live_stream_url = ""
    meta_file = Path(cfg["paths"]["input"]) / "video_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                video_title = meta.get("title", "")
                video_description = meta.get("description", "")
                live_stream_url = meta.get("live_stream_url", "")
        except Exception as exc:
            log.warning("Could not load source video metadata for SEO: %s", exc)

    if not video_path:
        dl_fn = cfg.get("download", {}).get("output_filename", "video.mp4")
        video_path = str(Path(cfg["paths"]["input"]) / dl_fn)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    clips = list(highlights.items())
    failures = []
    for idx, (clip_id, info) in enumerate(clips, start=1):
        transcript = info.get("text", "")
        log.info("SEO [%d/%d]: %s", idx, len(clips), clip_id)

        try:
            trend_cache_key = f"trend:{video_title[:60]}:{transcript[:80]}"
            trend = TREND_CACHE.get(trend_cache_key)
            if trend is None:
                trend = get_trending_context(
                    domain="cricket",
                    region="IN",
                    video_title=video_title,
                    video_description=video_description,
                    transcript=transcript,
                )
                TREND_CACHE.set(trend_cache_key, trend)
            result = generate_clip_seo(
                clip_id=clip_id,
                transcript=transcript,
                video_title=video_title,
                video_description=video_description,
                scorecard=trend.get("scorecard", ""),
                trend_topics=trend.get("topics", []),
                live_stream_url=(live_stream_url or trend.get("live_stream_url", "")),
                teams=trend.get("teams", []),
                approved_search_queries=trend.get("search_queries", []),
                match_facts=trend.get("match_facts", []),
                grounded_players=trend.get("player_names", []),
                grounded_aliases=trend.get("player_aliases", {}),
                research_sources=trend.get("sources", []),
                video_path=video_path,
            )
            all_results.append(result)

            # Save individual file immediately (atomic: temp + replace so an
            # interrupted run never leaves a truncated metadata file behind)
            per_clip_path = Path(output_dir) / f"{clip_id}_metadata.json"
            tmp_path = per_clip_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, per_clip_path)
        except SEOGenerationError as e:
            log.warning("[%s] SEO failed — writing failure marker: %s", clip_id, e)
            failures.append(clip_id)
            marker_data = {
                "clip_id": clip_id,
                "transcript": transcript,
                "video_title": video_title,
                "video_description": video_description,
                "scorecard": trend.get("scorecard", ""),
                "trend_topics": trend.get("topics", []),
                "teams": trend.get("teams", []),
                "approved_search_queries": trend.get("search_queries", []),
                "match_facts": trend.get("match_facts", []),
                "grounded_players": trend.get("player_names", []),
                "grounded_aliases": trend.get("player_aliases", {}),
                "research_sources": trend.get("sources", []),
                "video_path": video_path,
                "is_shorts": True,
            }
            marker_path = Path(output_dir) / f"{clip_id}_seo_failed.json"
            with open(marker_path, "w", encoding="utf-8") as f:
                json.dump(marker_data, f)
            all_results.append({"_seo_failed": True, "clip_id": clip_id})

        # Breathing room between clips — configurable, default 30s
        if idx < len(clips):
            sleep_s = cfg.get("seo", {}).get("inter_clip_sleep_s", 30)
            log.info("Sleeping %.1fs before next SEO call...", sleep_s)
            time.sleep(sleep_s)

    if failures:
        log.warning("SEO failures for %d clip(s): %s", len(failures), failures)

    # Also write a combined results file
    combined_path = Path(output_dir) / "seo_results.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    log.info("SEO complete: %d clips → %s", len(all_results), output_dir)
    return str(combined_path)


def retry_failed_seo(output_dir: str) -> dict:
    """Retry SEO generation for clips with ``*_seo_failed.json`` markers.

    Scans *output_dir* for failure markers, re-generates SEO via
    ``generate_seo_for_exported_clip``, and removes the marker on success.

    Returns:
        Dict with ``recovered`` count and ``total`` markers found.
    """
    from pathlib import Path as _Path
    out = _Path(output_dir)
    if not out.is_dir():
        log.warning("retry_failed_seo: output_dir %s not found", output_dir)
        return {"recovered": 0, "total": 0}

    markers = sorted(out.glob("*_seo_failed.json"))
    total = len(markers)
    if not total:
        return {"recovered": 0, "total": 0}

    recovered = 0
    for m in markers:
        try:
            data = json.loads(m.read_text(encoding="utf-8"))
            clip_id = data.get("clip_id", m.stem.replace("_seo_failed", ""))
            retry_keys = (
                "transcript", "video_title", "video_description", "scorecard",
                "trend_topics", "teams", "is_shorts", "video_path",
                "approved_search_queries", "match_facts", "grounded_players",
                "grounded_aliases", "research_sources",
            )
            kwargs = {key: data[key] for key in retry_keys if key in data}
            result = generate_clip_seo(clip_id=clip_id, **kwargs)
            if result and not result.get("_seo_failed"):
                meta_path = out / f"{clip_id}_metadata.json"
                tmp_path = meta_path.with_suffix(".tmp")
                tmp_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(tmp_path, meta_path)
                m.unlink()
                recovered += 1
                log.info("[retry] %s recovered", clip_id)
            else:
                log.warning("[retry] %s still failing", clip_id)
        except Exception as e:
            log.error("[retry] %s error: %s", m.name, e)
    return {"recovered": recovered, "total": total}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    process_all_seo("highlights/video.yaml", "shorts/test")
