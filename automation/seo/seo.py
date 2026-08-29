"""seo.py — Per-clip SEO generation for Indian cricket Shorts.

Uses restricted SEO models with one strict escalation attempt. Every title is
clip-specific and must promise the same thought that the clip actually opens
with; failed attempts retain their complete research evidence for retry.
"""
import json
import math
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
from .vidiq import VidiqClient

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
    "Write metadata for BOTH viewers and YouTube Search. The first 2-3 description lines "
    "must clearly explain the exact clip promise using the strongest grounded search phrase. "
    "Then write a detailed, journalist-style article covering match context, player analysis, "
    "pitch conditions, head-to-head stats, and predictions. Embed vidIQ keywords naturally "
    "into flowing prose — NEVER dump a 'Popular Searches:' section or comma-separated keyword "
    "list anywhere in the description. YouTube actively suppresses keyword-stuffed descriptions. "
    "Evidence boundary: source title/description, clip transcript, OCR, live autocomplete, "
    "recent YouTube search titles, verified scorecard facts, and explicitly supplied vidIQ "
    "ranking guidance. Never invent a player, team, score, venue, date, series, or event. "
    "All public copy is simple ENGLISH only: no Hinglish, no Devanagari. "
    "Title = one specific premise, max 60 chars, 1-2 emojis, never LIVE or #Shorts. "
    "Return ONLY valid JSON."
)

_CRICKET_ONLY_PROMPT_TMPL = """CRICKET SHORT — VERIFIED CONTEXT:
  Source video title: {video_title}
  Source video description: {video_description}
  Verified scorecard / match facts: {match_facts}
  Teams in this match: {teams}
  Verified player roster/entities: {roster}
  Current YouTube autocomplete/search signals: {trend_topics}
  Recent YouTube result titles: {research_sources}

COMPLETE CLIP TRANSCRIPT: {transcript}

APPROVED SEARCH SEEDS (grounded; rank and phrase naturally):
{approved_search_queries}

TASK — create unique metadata for THIS exact Short.

Return ONLY this valid JSON object:
{{
  "title": "<specific English title, max 60 chars, grounded player/team + exact moment>",
  "description": "<unique, human-readable, keyword-rich description written as a natural article>",
  "hashtags": ["#Shorts", "<#specific team/series tag>", "<#specific topic/player tag>"],
  "search_terms": ["<ranked long-tail viewer searches; prefer 12-25 strong phrases over filler>"],
  "primary_search_terms": ["<4-8 strongest phrases>"],
  "tags": ["<grounded phrases, packed under the API limit>" ]
}}

DESCRIPTION RULES:
1. Aim for {description_target_chars} chars; never below {description_min_chars} when evidence is rich and never above {description_max_chars}. Write an extremely long, detailed, natural-language article to maximize SEO surface area.
2. NEVER include a "Popular Searches:" section or any comma-separated keyword list. YouTube flags this as keyword stuffing and suppresses reach. Instead, weave all keywords naturally into flowing prose paragraphs.
3. Structure the description exactly like this:

SECTION 1 — HOOK (first 150 chars, main keyword front-loaded):
"[Match short name] — [exact clip moment in one punchy sentence]. Watch [specific player] [specific action] in this cricket Short."

SECTION 2 — 🏏 MATCH DETAILS:
"Match: [Full match name]
Venue: [Venue]
Tournament: [Tournament name]
Date: [Date if known]
Teams: [Team 1] vs [Team 2]"

SECTION 3 — 🔥 DETAILED CLIP ANALYSIS (2-4 long paragraphs):
Write like a cricket journalist. Detail the exact action in the transcript, the players involved, the match situation, the stakes, the strategy, head-to-head stats, pitch conditions, and why this moment matters. Naturally embed vidIQ keywords and search phrases into sentences — never as lists.

SECTION 4 — 🏟️ SQUAD & PLAYING XI CONTEXT:
"Squad players for [Team 1]: [names]. Squad players for [Team 2]: [names]."
Include player analysis, form discussion, and predictions woven around the squad data.

SECTION 5 — ⚠️ DISCLAIMER:
"This clip is intended for independent cricket commentary, updates, analysis and fan discussion. No official match broadcast footage or copyrighted television audio is being rebroadcast. Team names, player names, league names, logos and trademarks belong to their respective owners."

SECTION 6 — CALL TO ACTION:
"Like the video, subscribe to Cricket With Prajjwal, and drop your prediction for [Teams] in the comments."

4. Do NOT fabricate match facts; use ONLY the provided entities or omit the specific detail if entirely unknown.
5. CRITICAL: Every keyword from the APPROVED SEARCH SEEDS must appear naturally inside prose sentences — never dumped as a comma-separated block.

TITLE RULES:
- Max {title_max_chars} chars, 1-2 emojis, no LIVE/#Shorts.
- Lead with the most searchable grounded entity when it fits naturally.
- Promise the exact clip moment; avoid generic adjectives and fake urgency.
"""

_CRICKET_ONLY_SALVAGE_TMPL = """Generate grounded, keyword-rich metadata for this cricket Short.
Clip transcript: {transcript}
Source title: {video_title}

Return only JSON with title, description, hashtags, search_terms, primary_search_terms, and tags.
- Description MUST be a natural article-style essay (3000-4000 chars) with keywords woven into prose.
- NEVER include a "Popular Searches:" section or any comma-separated keyword list — YouTube suppresses this as keyword stuffing.
- Structure: Hook sentence → Match Details → 2-4 detailed analysis paragraphs → Squad context → Disclaimer → CTA.
- Title: max 60 characters; one specific premise plus 1-2 emojis. No LIVE/#Shorts.
- Search terms: 12-25 ranked long-tail grounded cricket phrases; no filler.
- Tags: only the strongest 12-15 exact-match grounded phrases; pack within 480 chars.
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
    return _seo_config_int("description_min_chars", 1800, 100, 4800)


def _description_target_chars() -> int:
    return _seo_config_int("description_target_chars", 3200, 150, 4800)


def _description_max_chars() -> int:
    return _seo_config_int("description_max_chars", 4500, 200, 4800)


def _vidiq_required_keyword_floor() -> int:
    """Single source of truth for the grounded vidIQ keyword minimum."""
    vidiq_cfg = cfg.get("seo", {}).get("vidiq", {})
    try:
        vidiq_min = int(vidiq_cfg.get("min_keywords", 12))
    except (TypeError, ValueError):
        vidiq_min = 12
    vidiq_min = min(max(vidiq_min, 4), 25)
    search_min = _seo_config_int("min_search_terms", 12, 1, 25)
    return max(vidiq_min, search_min)


def _youtube_tag_chars(tags: List[str]) -> int:
    """Return YouTube's effective tag-field size, including separators/quotes."""
    return sum(len(tag) + (2 if " " in tag else 0) for tag in tags) + max(0, len(tags) - 1)


def _pack_youtube_tags(tags: List[str], max_chars: int) -> List[str]:
    packed = []
    for tag in tags:
        candidate = [*packed, tag]
        if _youtube_tag_chars(candidate) <= max_chars:
            packed.append(tag)
    return packed


def _build_vidiq_description(
    title: str,
    transcript: str,
    video_title: str,
    video_description: str,
    match_facts: List[str],
    keywords: List[str],
    teams: Optional[List[str]] = None,
) -> str:
    """Grounded fallback description used only when the writer model is unavailable.

    Builds a natural article-style description. vidIQ phrases are woven into
    prose — never dumped as a comma-separated "Popular Searches" block.
    """
    target = min(_description_target_chars(), _description_max_chars())
    teams = [str(t).strip() for t in (teams or []) if str(t).strip()]
    facts = list(dict.fromkeys(
        str(fact).strip() for fact in match_facts if str(fact).strip()
    ))
    ranked = list(dict.fromkeys(
        str(keyword).strip() for keyword in keywords if str(keyword).strip()
    ))[:15]

    # SECTION 1 — Hook
    if len(teams) >= 2:
        opening = (
            f"{teams[0]} vs {teams[1]} — {title}. Watch the exact cricket moment "
            "discussed in this Short, with analysis and commentary from Cricket With Prajjwal."
        )
    else:
        opening = (
            f"{title}. This Short covers a key cricket moment with detailed analysis "
            "and commentary from Cricket With Prajjwal."
        )
    sections = [opening]

    # SECTION 2 — Match Details
    if teams or facts:
        details = ["🏏 MATCH DETAILS"]
        if len(teams) >= 2:
            details.append(f"Match: {teams[0]} vs {teams[1]}")
        elif teams:
            details.append(f"Team context: {teams[0]}")
        details.extend(f"• {fact}" for fact in facts[:10])
        sections.append("\n".join(details))

    # SECTION 3 — Detailed Analysis (natural prose with keywords embedded)
    clip_subject = "the verified cricket discussion in this Short"
    if ranked:
        clip_subject = ranked[0]

    analysis_lines = [
        "🔥 CLIP ANALYSIS",
        f"This clip centers on {clip_subject}. The metadata stays tied to the spoken "
        "segment and verified match evidence, without adding an unsupported player, score, "
        "result, venue, or event.",
    ]
    # Weave keywords into natural sentences instead of dumping them
    if ranked:
        keyword_chunks = [ranked[i:i+3] for i in range(0, len(ranked), 3)]
        for chunk in keyword_chunks:
            if len(chunk) == 3:
                analysis_lines.append(
                    f"Viewers searching for {chunk[0]} will find detailed coverage here. "
                    f"This Short also covers aspects related to {chunk[1]} and {chunk[2]}, "
                    "providing context from the actual match discussion."
                )
            elif len(chunk) == 2:
                analysis_lines.append(
                    f"Coverage in this clip extends to {chunk[0]} and {chunk[1]}, "
                    "grounded in the actual transcript evidence."
                )
            else:
                analysis_lines.append(
                    f"This moment is particularly relevant for fans following {chunk[0]}."
                )
    sections.append("\n".join(analysis_lines))

    # SECTION 4 — Source context
    if video_title.strip():
        sections.append(f"Source program context: {video_title.strip()[:300]}")

    # SECTION 5 — Disclaimer
    sections.append(
        "⚠️ DISCLAIMER\n"
        "This clip is intended for independent cricket commentary, updates, analysis and "
        "fan discussion. No official match broadcast footage or copyrighted television audio "
        "is being rebroadcast. Team names, player names, league names, logos and trademarks "
        "belong to their respective owners."
    )

    # SECTION 6 — CTA
    if len(teams) >= 2:
        sections.append(
            f"Like the video, subscribe to Cricket With Prajjwal, and drop your "
            f"prediction for {teams[0]} vs {teams[1]} in the comments."
        )
    else:
        sections.append(
            "Like the video, subscribe to Cricket With Prajjwal, and share your "
            "thoughts in the comments."
        )

    description = "\n\n".join(sections)
    return _truncate_at_word(description, target)


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
        "original rule (length budgets, 12-25 strong long-tail search terms, emoji "
        "title, natural article-style description with NO 'Popular Searches' keyword "
        "lists). Return ONLY the JSON object."
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
    tag_budget = _seo_config_int("max_tag_chars", 480, 50, 500)
    out["tags"] = _pack_youtube_tags(deduped_tags[:15], tag_budget)

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
        log.warning("Quality Gate Failed: Title too short")
        return False

    # 2. Title must not be a known generic pattern
    if title.lower().rstrip("!.?") in GENERIC_TITLES:
        log.warning("Quality Gate Failed: Generic title")
        return False

    # 3. Description must use the configured short-form evidence budget.
    if len(description) < _description_min_chars():
        log.warning("Quality Gate Failed: Description too short (%d < %d)", len(description), _description_min_chars())
        return False

    # 4. No Devanagari script anywhere in public copy (kills discoverability)
    if _DEVANAGARI_RE.search(title):
        log.warning("Quality Gate Failed: Devanagari in title")
        return False
    if _DEVANAGARI_RE.search(description):
        log.warning("Quality Gate Failed: Devanagari in description")
        return False
    hashtags_text = " ".join(
        str(tag) for tag in (item.get("hashtags") or []) if isinstance(tag, str)
    )
    if _DEVANAGARI_RE.search(hashtags_text):
        log.warning("Quality Gate Failed: Devanagari in hashtags")
        return False

    # 5. AI narration slop is banned in public copy — it reads as template
    #    output and erodes trust in a feed where every word earns swipes.
    if _has_ai_slop(title) or _has_ai_slop(description):
        log.warning("Quality Gate Failed: AI slop detected")
        return False

    # 6. Natural embedding check — reject if keyword list/tag block is
    #    appended at the end of description (signals lazy SEO).
    #    Look for patterns like:
    #      - "Tags:", "Keywords:", "Search terms:" at end (label + colon)
    #      - Comma-separated single-word dump on last line
    last_200 = description[-200:].lower()
    if re.search(r'\b(?:search[_ ]terms?|tags?|keywords?|popular[_ ]searches?)\s*:', last_200):
        log.warning("Quality Gate Failed: Lazy SEO keyword dump at end")
        return False
    # Also reject "Popular Searches:" ANYWHERE in description (keyword stuffing)
    if re.search(r'(?i)\bpopular\s+searches?\s*:', description):
        log.warning("Quality Gate Failed: 'Popular Searches' keyword stuffing block detected")
        return False
    # Check last line for high comma density (keyword dump indicator)
    last_line = description.split('\n')[-1].strip().lower()
    words = last_line.split(',')
    if len(words) >= 6 and all(len(w.strip().split()) <= 2 for w in words) \
       and not last_line.rstrip().endswith(('.', '!', '?')):
        log.warning("Quality Gate Failed: High comma density on last line")
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


def _get_vidiq_context(approved_queries: List[str], grounding_text: str, player_names: Optional[List[str]] = None):
    """Return grounded vidIQ intelligence with a strict per-request credit budget.

    High-value order:
      1) cached recent own-channel Shorts history (optional, for title novelty),
      2) one deep keyword research call,
      3) one India broad-search refinement only when needed,
      4) one title-generation call.

    This replaces the old "up to six near-duplicate keyword calls per clip" pattern.
    """
    vidiq_cfg = cfg.get("seo", {}).get("vidiq", {})
    try:
        max_calls = int(vidiq_cfg.get("max_calls_per_clip", 4))
    except (TypeError, ValueError):
        max_calls = 4
    client = VidiqClient(
        enabled=vidiq_cfg.get("enabled", False),
        timeout_seconds=vidiq_cfg.get("timeout_seconds", 8),
        endpoint=vidiq_cfg.get("endpoint", "https://mcp.vidiq.com/mcp"),
        max_calls=max_calls,
        default_cache_ttl_seconds=vidiq_cfg.get("cache_ttl_seconds", 1800),
    )
    seed = next((str(query).strip() for query in approved_queries if str(query).strip()), "")
    if not seed:
        return "", client.audit, {"keywords": [], "titles": [], "previous_titles": []}

    source_tokens = set(re.findall(r"[a-z0-9]+", grounding_text.casefold())) - STOP_WORDS
    player_names = list(player_names or [])
    grounded_source_entities = find_canonical_entities(grounding_text, player_names)
    grounded_source_players = set(grounded_source_entities["players"])
    grounded_source_teams = set(grounded_source_entities["teams"])
    entity_tokens = set()
    for name in [*grounded_source_players, *grounded_source_teams]:
        entity_tokens.update(re.findall(r"[a-z0-9]+", name.casefold()))
    seed_tokens = set(re.findall(r"[a-z0-9]+", seed.casefold())) - STOP_WORDS
    anchors = entity_tokens or seed_tokens or source_tokens
    safe_modifiers = {
        "cricket", "shorts", "short", "match", "moment", "reaction", "analysis",
        "explained", "debate", "discussion", "highlight", "highlights", "innings",
        "batting", "bowling", "bowler", "batter", "wicket", "wickets", "six", "sixes",
        "four", "fours", "yorker", "captain", "captaincy", "team", "series", "t20",
        "odi", "test", "ipl", "world", "cup", "news", "update", "viral", "today",
        "over", "overs", "spell", "pace", "fast", "death", "powerplay", "reaction",
        "swing", "reverse", "seam", "spin", "ball", "delivery", "deliveries",
    }

    def topic_signals(words: set[str]) -> set[str]:
        """Normalize factual cricket-event concepts so topic drift is rejected."""
        signals = set()
        for word in words:
            if word.startswith("bowl") or word in {"pace", "pacer"}:
                signals.add("bowling")
            elif word.startswith("bat"):
                signals.add("batting")
            elif word.startswith("wicket") or word in {"bowled", "lbw", "stumped", "catch", "caught"}:
                signals.add("wicket")
            elif word.startswith("yorker"):
                signals.add("yorker")
            elif word in {"six", "sixes", "sixer", "chhakka", "chakka"}:
                signals.add("six")
            elif word in {"four", "fours", "boundary", "boundaries", "chauka"}:
                signals.add("four")
            elif word.startswith("captain"):
                signals.add("captaincy")
            elif word.startswith("coach"):
                signals.add("coach")
            elif word.startswith("retir"):
                signals.add("retirement")
            elif word.startswith("injur"):
                signals.add("injury")
            elif word.startswith("record"):
                signals.add("record")
            elif word.startswith("centur"):
                signals.add("century")
            elif word in {"run", "runs", "score", "scored"}:
                signals.add("runs")
            elif word in {"chase", "target"}:
                signals.add("chase")
            elif word in {"over", "overs", "powerplay", "spell", "death"}:
                signals.add("phase")
            elif word.startswith("select"):
                signals.add("selection")
            elif word.startswith("comeback") or word == "return":
                signals.add("comeback")
        return signals

    source_topic_signals = topic_signals(source_tokens)
    allowed_topic_signals = set(source_topic_signals)
    bowling_family = {"bowling", "wicket", "yorker", "phase"}
    batting_family = {"batting", "six", "four", "runs", "chase", "century", "phase"}
    leadership_family = {"captaincy", "coach", "selection"}
    if source_topic_signals & bowling_family:
        allowed_topic_signals.update(bowling_family)
    if source_topic_signals & batting_family:
        allowed_topic_signals.update(batting_family)
    if source_topic_signals & leadership_family:
        allowed_topic_signals.update(leadership_family)

    def clean_text(value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()[:160]
        return re.sub(r"[^A-Za-z0-9 #&'?!:,.-]", "", text).strip()

    def clean_score(value: object) -> Optional[float]:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return score if math.isfinite(score) else None

    def grounded(value: object) -> bool:
        text = clean_text(value)
        if not text or _is_live_framed_search_term(text):
            return False
        words = set(re.findall(r"[a-z0-9]+", text.casefold())) - STOP_WORDS
        if not words or not (words & anchors):
            return False
        entities = find_canonical_entities(text, player_names)
        if set(entities["players"]) - grounded_source_players:
            return False
        if set(entities["teams"]) - grounded_source_teams:
            return False
        # A famous player's name alone is not topic grounding. vidIQ can return
        # high-volume but wrong-intent phrases ("Bumrah retirement news" for a
        # yorker clip). Reject any factual cricket-event signal not supported by
        # the source, and require topic overlap whenever the source has one.
        candidate_topic_signals = topic_signals(words)
        if candidate_topic_signals - allowed_topic_signals:
            return False
        if source_topic_signals and not (candidate_topic_signals & allowed_topic_signals):
            return False
        # Catch out-of-catalog names such as "John Smith" even when the static
        # cricket catalog cannot identify them as entities. Generic title-cased
        # SEO words are exempt through safe_modifiers/source/entity tokens.
        unknown_proper = [
            token for token in re.findall(r"\b[A-Z][A-Za-z'-]{3,}\b", text)
            if token.casefold() not in source_tokens
            and token.casefold() not in safe_modifiers
            and token.casefold() not in entity_tokens
        ]
        if unknown_proper:
            return False
        unexplained = words - source_tokens - safe_modifiers - entity_tokens
        return len(unexplained) <= 1

    def grounded_title(value: object) -> bool:
        text = _clean_title(clean_text(value), _seo_config_int("title_max_chars", 60, 30, 100))
        if not text:
            return False
        entities = find_canonical_entities(text, player_names)
        players = set(entities["players"])
        teams = set(entities["teams"])
        if not (players or teams):
            return False
        if players - grounded_source_players or teams - grounded_source_teams:
            return False
        title_tokens = _promise_tokens(text)
        evidence_tokens = _promise_tokens(grounding_text)
        return bool(title_tokens & evidence_tokens)

    keyword_candidates = {}
    keyword_order = 0

    def keyword_rank(value):
        # India is the target market. Prefer in-country demand first, then vidIQ's
        # overall opportunity score, global demand, and lower competition.
        return (
            value["country_volume"] if value["country_volume"] is not None else -1.0,
            value["overall"] if value["overall"] is not None else -1.0,
            value["volume"] if value["volume"] is not None else -1.0,
            -(value["competition"] if value["competition"] is not None else 101.0),
            -value["order"],
        )

    def absorb_keyword_payload(keyword_data: object) -> None:
        nonlocal keyword_order
        if not isinstance(keyword_data, dict):
            return
        rows = []
        if isinstance(keyword_data.get("seedKeyword"), dict):
            rows.append(keyword_data["seedKeyword"])
        for key in ("relatedKeywords", "risingKeywords"):
            if isinstance(keyword_data.get(key), list):
                rows.extend(keyword_data[key])
        for item in rows:
            if not isinstance(item, dict):
                continue
            text = clean_text(item.get("keyword"))
            if not text or not grounded(text):
                continue
            candidate = {
                "text": text,
                "overall": clean_score(item.get("overall")),
                "country_volume": clean_score(item.get("countryVolume")),
                "volume": clean_score(item.get("volume")),
                "competition": clean_score(item.get("competition")),
                "order": keyword_order,
            }
            keyword_order += 1
            key = text.casefold()
            previous = keyword_candidates.get(key)
            if previous is None or keyword_rank(candidate) > keyword_rank(previous):
                keyword_candidates[key] = candidate

    # Optional recent own-channel history. It is cached for six hours and filtered
    # back to cricket. vidIQ previousTitles is a novelty/repetition signal, so use
    # recent uploads here; winning-pattern learning belongs to channel analytics.
    previous_titles = []
    channel_ref = (
        vidiq_cfg.get("channel_id")
        or cfg.get("youtube", {}).get("channel_id")
        or vidiq_cfg.get("channel_handle")
    )
    # With a smaller custom budget, preserve one slot for keyword research and
    # one for title generation. History is optional; title generation is not.
    if vidiq_cfg.get("use_channel_history", True) and channel_ref and max_calls >= 4:
        history = client.call_tool(
            vidiq_cfg.get("channel_videos_tool", "vidiq_channel_videos"),
            {"channelId": channel_ref, "videoFormat": "short", "popular": False},
            cache_ttl_seconds=21600,
        )
        if isinstance(history, dict) and isinstance(history.get("videos"), list):
            for row in history["videos"]:
                if not isinstance(row, dict):
                    continue
                title = clean_text(row.get("title"))
                title_entities = find_canonical_entities(title, player_names)
                cricket_specific = bool(re.search(
                    r"\b(?:cricket|ipl|t20|odi|test|wicket|yorker|bowling|batting|"
                    r"innings|powerplay|over|runs?|century|six|four)\b",
                    title, re.IGNORECASE,
                ))
                # Mixed-niche channels can have football country-v-country titles
                # that the generic cricket gate mistakes for cricket. Require an
                # actual cricket signal or a canonical cricket player.
                if title and (cricket_specific or title_entities["players"]):
                    previous_titles.append(title)
                if len(previous_titles) >= 20:
                    break

    primary = client.call_tool(
        vidiq_cfg.get("keyword_tool", "vidiq_keyword_research"),
        {"mode": "research", "keyword": seed, "country": "IN", "includeRelated": True},
        cache_ttl_seconds=1800,
    )
    absorb_keyword_payload(primary)

    min_keywords = _vidiq_required_keyword_floor()

    # One broad India refinement is enough. Do it only when the deep call did not
    # produce a healthy grounded set; this is the main credit-saving gate.
    if len(keyword_candidates) < min_keywords and client.audit.get("calls", 0) < max_calls - 1:
        broad = client.call_tool(
            vidiq_cfg.get("keyword_tool", "vidiq_keyword_research"),
            {
                "mode": "country_search",
                "keyword": seed[:120],
                "country": "IN",
                "limit": 30,
                "broad": True,
            },
            cache_ttl_seconds=1800,
        )
        absorb_keyword_payload(broad)

    ranked_keyword_rows = sorted(keyword_candidates.values(), key=keyword_rank, reverse=True)
    accepted_keywords = [item["text"] for item in ranked_keyword_rows][:_seo_config_int(
        "max_search_terms", 25, 8, 30
    )]
    ranked_keywords = []
    for item in ranked_keyword_rows[:10]:
        metrics = []
        if item["overall"] is not None:
            metrics.append(f"overall {item['overall']}")
        if item["country_volume"] is not None:
            metrics.append(f"India searches {item['country_volume']}")
        if item["volume"] is not None:
            metrics.append(f"volume {item['volume']}")
        if item["competition"] is not None:
            metrics.append(f"competition {item['competition']}")
        ranked_keywords.append(
            f"- {item['text']}" + (f" ({', '.join(metrics)})" if metrics else "")
        )

    title_args = {
        "title": seed[:500],
        "description": grounding_text[:5000],
        "analysisSummary": grounding_text[:4000],
        "numTitles": 5,
        "type": "short",
        "language": "en",
        "regionCode": "IN",
    }
    if previous_titles:
        title_args["previousTitles"] = previous_titles[:20]
    title_data = client.call_tool(
        vidiq_cfg.get("title_tool", "vidiq_generate_titles"),
        title_args,
        cache_ttl_seconds=3600,
    )

    title_candidates = {}
    title_order = 0
    if isinstance(title_data, dict):
        title_rows = title_data.get("titles")
        if isinstance(title_rows, list):
            for item in title_rows:
                if not isinstance(item, dict):
                    continue
                text = clean_text(item.get("title"))
                text = re.sub(r"(?:^|\s)#[A-Za-z0-9_]+", "", text).strip()
                text = _clean_title(
                    text, _seo_config_int("title_max_chars", 60, 30, 100)
                )
                if text and grounded_title(text):
                    score = clean_score(item.get("score"))
                    key = text.casefold()
                    candidate = {"text": text, "score": score, "order": title_order}
                    title_order += 1
                    previous = title_candidates.get(key)
                    if previous is None or (score if score is not None else -1.0) > (
                        previous["score"] if previous["score"] is not None else -1.0
                    ):
                        title_candidates[key] = candidate

    ranked_title_rows = sorted(
        title_candidates.values(),
        key=lambda item: (
            item["score"] if item["score"] is not None else -1.0,
            -item["order"],
        ),
        reverse=True,
    )
    accepted_titles = [item["text"] for item in ranked_title_rows]
    ranked_titles = [
        f"- {item['text']}" + (f" (score {item['score']})" if item["score"] is not None else "")
        for item in ranked_title_rows[:5]
    ]

    sections = []
    if ranked_keywords:
        sections.append("Grounded India keyword opportunities:\n" + "\n".join(ranked_keywords))
    if ranked_titles:
        sections.append("Grounded title scoring references:\n" + "\n".join(ranked_titles))
    if previous_titles:
        sections.append("Avoid repeating these recent cricket title patterns:\n" + "\n".join(
            f"- {title}" for title in previous_titles[:8]
        ))
    return "\n".join(sections), client.audit, {
        "keywords": accepted_keywords,
        "titles": accepted_titles,
        "previous_titles": previous_titles,
    }


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
    vidiq_required = bool(cfg.get("seo", {}).get("vidiq", {}).get("required", False))

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
                transcript=transcript, include_live_stream_url=False,
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
    # Reasoning grounding may normalize ugly transliteration, but it must not
    # become a license to invent a player outside verified runtime evidence.
    static_ground = find_canonical_entities(grounding_context, grounded_players)
    if grounded_players:
        verified_people = {str(name).casefold() for name in grounded_players}
        verified_people.update(str(name).casefold() for name in static_ground["players"])
        llm_ground["players"] = [
            name for name in llm_ground.get("players", [])
            if str(name).casefold() in verified_people
        ]
    if teams:
        verified_teams = {str(name).casefold() for name in teams}
        verified_teams.update(str(name).casefold() for name in static_ground["teams"])
        llm_ground["teams"] = [
            name for name in llm_ground.get("teams", [])
            if str(name).casefold() in verified_teams
        ]
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
    vidiq_context, vidiq_audit, vidiq_recommendations = _get_vidiq_context(
        approved_queries,
        " ".join((video_title, video_description, transcript, scorecard, *match_facts)),
        player_names=list(dict.fromkeys([*grounded_players, *evidence_pack.get("player_names", [])])),
    )
    if vidiq_required:
        vidiq_titles = list(vidiq_recommendations.get("titles") or [])
        vidiq_keywords = list(vidiq_recommendations.get("keywords") or [])
        if not vidiq_audit.get("used") or not vidiq_titles or not vidiq_keywords:
            raise SEOGenerationError(
                f"SEO blocked for {clip_id}: vidIQ required but returned no grounded title/keywords"
            )
        approved_queries = vidiq_keywords

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
    if vidiq_context:
        user_prompt += (
            "\n\nVIDIQ INDIA SEO CONTEXT (ranking guidance only):\n"
            f"{vidiq_context}\nUse it to rank grounded wording. It is not factual "
            "evidence and cannot introduce names, events, or promises."
        )
    if vidiq_required:
        user_prompt += (
            "\n\nEXCLUSIVE SEO SOURCE CONTRACT:\n"
            f"Use this exact vidIQ title: {vidiq_titles[0]}\n"
            "Choose search terms/tags only from these vidIQ-ranked phrases:\n"
            + "\n".join(f"- {term}" for term in vidiq_keywords)
            + "\nYou are formatting the evidence-backed description only. Do not "
            "invent or substitute any title, keyword, or tag. Hashtags may be derived only "
            "from verified teams/players/topics already present in the evidence."
        )

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

    if vidiq_required:
        try:
            result = _attempt_seo_generation(
                clip_id, user_prompt, transcript, video_title, is_shorts,
                provider_override=provider_override,
                model_override=model_override,
                sys_instruction=sys_instruction,
                salvage_tmpl=salvage_tmpl,
            )
            result["description_source"] = "grounded_writer"
        except SEOGenerationError:
            fallback_description = _build_vidiq_description(
                vidiq_titles[0], transcript, video_title, video_description,
                evidence_pack["match_facts"], vidiq_keywords, teams=teams,
            )
            if len(fallback_description) < _description_min_chars():
                raise SEOGenerationError(
                    f"SEO blocked for {clip_id}: writer unavailable and deterministic "
                    f"description was too short ({len(fallback_description)} chars)"
                )
            result = {
                "title": vidiq_titles[0],
                "description": fallback_description,
                "hashtags": ["#Shorts"],
                "ai_generated": False,
                "metadata_valid": True,
                "description_source": "deterministic_grounded_fallback",
            }
        # vidIQ owns ranking choices; the writer owns readable evidence-backed prose.
        result["title"] = vidiq_titles[0]
        result["search_terms"] = vidiq_keywords
        result["primary_search_terms"] = vidiq_keywords[:_seo_config_int(
            "min_primary_search_terms", 4, 1, 10
        )]
        result["tags"] = vidiq_keywords[:_seo_config_int("max_tag_terms", 12, 3, 20)]
        result["metadata_source"] = "vidiq_ranked_grounded_writer"
    else:
        result = _attempt_seo_generation(
            clip_id, user_prompt, transcript, video_title, is_shorts,
            provider_override=provider_override,
            model_override=model_override,
            sys_instruction=sys_instruction,
            salvage_tmpl=salvage_tmpl,
        )
    result = _enforce_limits(result, is_shorts=is_shorts)

    if vidiq_required:
        min_vidiq_keywords = _vidiq_required_keyword_floor()
        if len(vidiq_keywords) < min_vidiq_keywords:
            raise SEOGenerationError(
                f"SEO blocked for {clip_id}: vidIQ returned only {len(vidiq_keywords)} "
                f"grounded keywords; need {min_vidiq_keywords}"
            )

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
        floor = _seo_config_int("min_search_terms", 12, 1, 25)
        if len(approved_queries) < floor and vidiq_required:
            raise SEOGenerationError(
                f"SEO blocked for {clip_id}: vidIQ terms fell below the grounded minimum after audit"
            )
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
            if vidiq_required:
                raise SEOGenerationError(
                    f"SEO blocked for {clip_id}: vidIQ title lost grounding during copy audit"
                )
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
        and not name_vouched_by_topics(name, supported_topics)
    ]
    if unknown_title_people:
        if vidiq_required:
            raise SEOGenerationError(
                f"SEO blocked for {clip_id}: vidIQ title contains ungrounded player(s) "
                f"{', '.join(unknown_title_people)}"
            )
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
        return any(name.casefold() in allowed_people for name in entities["players"]) or any(
            name in grounded_teams for name in entities["teams"]
        )

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
        if vidiq_required:
            raise SEOGenerationError(
                f"SEO blocked for {clip_id}: vidIQ title has no grounded player/team entity"
            )
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
                f"SEO blocked for {clip_id}: repaired title remains ungrounded "
                "and has no verified player/team entity"
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
    api_tag_players = {
        name for name in find_canonical_entities(api_tag_text, player_catalog)["players"]
        if name.casefold() not in allowed_people
        and not name_vouched_by_topics(name, supported_topics)
    }
    if api_tag_players:
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
        name for name in rendered_entities["players"]
        if name.casefold() not in allowed_people and not _title_vouched(name)
    }
    extra_teams = set(rendered_entities["teams"]) - grounded_teams
    if extra_players or extra_teams:
        extras = sorted(extra_players | extra_teams)
        raise SEOGenerationError(
            f"SEO blocked for {clip_id}: ungrounded entities {', '.join(extras)}"
        )

    min_queries = _seo_config_int("min_search_terms", 12, 1, 25)
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
        # Preserve exact primary-query coverage in readable prose. Re-canonicalize
        # the hashtag block afterwards so hashtags remain the final description line.
        suffix = "\n\nViewer search context also includes " + "; ".join(missing_queries) + "."
        max_chars = _description_max_chars()
        base = _HASHTAG_TOKEN.sub("", str(result.get("description") or "")).rstrip()
        result["description"] = (
            _truncate_at_word(base, max(0, max_chars - len(suffix))) + suffix
        )
    if is_shorts:
        result["description"] = _cap_description_hashtags(
            str(result.get("description") or ""), result.get("hashtags") or []
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
    if not _validate_seo_quality(result):
        raise SEOGenerationError(
            f"SEO blocked for {clip_id}: final post-processed metadata failed quality gate"
        )
    result["packaging_version"] = PACKAGING_VERSION
    result["promise_alignment_score"] = alignment
    result["vidiq_audit"] = vidiq_audit
    result["metadata_valid"] = True

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
                reasoning_effort="high"
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
                reasoning_effort="high"
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
        # Successful return from generate_clip_seo means metadata passed all
        # gates, regardless of whether prose came from AI or deterministic fallback.
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
                    include_live_stream_url=False,
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
        except Exception as e:  # one bad clip must never abort the whole batch
            if isinstance(e, SEOGenerationError):
                log.warning("[%s] SEO failed — writing failure marker: %s", clip_id, e)
            else:
                log.exception("[%s] Unexpected SEO failure — isolating clip", clip_id)
            failures.append(clip_id)
            trend_data = trend if isinstance(locals().get("trend"), dict) else {}
            marker_data = {
                "clip_id": clip_id,
                "transcript": transcript,
                "video_title": video_title,
                "video_description": video_description,
                "scorecard": trend_data.get("scorecard", ""),
                "trend_topics": trend_data.get("topics", []),
                "teams": trend_data.get("teams", []),
                "approved_search_queries": trend_data.get("search_queries", []),
                "match_facts": trend_data.get("match_facts", []),
                "grounded_players": trend_data.get("player_names", []),
                "grounded_aliases": trend_data.get("player_aliases", {}),
                "research_sources": trend_data.get("sources", []),
                "video_path": video_path,
                "is_shorts": True,
            }
            marker_path = Path(output_dir) / f"{clip_id}_seo_failed.json"
            with open(marker_path, "w", encoding="utf-8") as f:
                json.dump(marker_data, f)
            all_results.append({"_seo_failed": True, "clip_id": clip_id, "error": str(e)})

        # Breathing room between clips — misconfigured YAML must not crash the batch.
        if idx < len(clips):
            try:
                sleep_s = float(cfg.get("seo", {}).get("inter_clip_sleep_s", 30))
            except (TypeError, ValueError):
                sleep_s = 30.0
            sleep_s = min(max(sleep_s, 0.0), 300.0)
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
