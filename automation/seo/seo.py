"""seo.py — Per-clip SEO generation for Indian cricket Shorts.

Uses parallel fastest-first model racing: fires the fastest available models
concurrently and takes the first valid JSON response. No backoff — on failure
the next tier of models is tried immediately. Three-tier fallback: AI → salvage
→ transcript-aware dynamic generation. Every title is clip-specific — no
generic templates or prefixes.
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
from .trends import TEAM_MAPPINGS
from automation._cache import TTLCache
from utils.ocr import extract_ocr_entities

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


def _maybe_auto_benchmark():
    """Lazy auto-benchmark: runs once on first SEO call if enabled in config."""
    if not getattr(_maybe_auto_benchmark, "_done", False):
        _maybe_auto_benchmark._done = True
        if cfg.get("ai", {}).get("auto_benchmark", False):
            try:
                from .seo_learner import run_auto_benchmark, get_best_model
                log.info("Auto-benchmark enabled — discovering best model...")
                run_auto_benchmark()
                best_provider, best_model = get_best_model()
                if best_provider and best_model:
                    log.info("Applying best model: %s/%s", best_provider, best_model)
                    _get_ai()._provider = best_provider
                    _get_ai()._model = best_model
            except Exception as e:
                log.warning("Auto-benchmark failed: %s", e)

class SEOGenerationError(Exception):
    """Raised when all AI providers fail during SEO generation."""


def _load_learner_state() -> Optional[Dict]:
    """Load learned entity/format scores from self_learner.db.

    Returns:
        Dict with entity_scores and format_scores JSON, or None on failure.
    """
    try:
        import sqlite3
        db_path = Path("self_learner.db")
        if not db_path.exists():
            return None
        conn = sqlite3.connect(str(db_path))
        try:
            c = conn.cursor()
            result = {}
            for key in ("entity_scores", "format_scores"):
                row = c.execute(
                    "SELECT value_json FROM learned_state WHERE state_key=?", (key,)
                ).fetchone()
                if row:
                    result[key] = row[0]
            return result if result else None
        finally:
            conn.close()
    except Exception:
        return None


def _get_learner_context() -> str:
    """Build a context string from learner data for SEO prompt injection.

    Gives the AI knowledge of what entities and formats perform best
    on THIS channel, so titles/tags are biased toward proven winners.

    Returns:
        Multi-line context string, or empty string if no data.
    """
    state = _load_learner_state()
    if not state:
        return ""

    lines = []
    try:
        if "entity_scores" in state:
            data = json.loads(state["entity_scores"])
            players = data.get("players", {})
            if players:
                top = sorted(players.items(), key=lambda x: x[1].get("score", 0), reverse=True)[:5]
                avoid = [name for name, info in players.items()
                         if info.get("n", 0) > 10 and info.get("avg_views", 0) < 150]
                player_strs = [f"{name}({info['avg_views']:.0f} avg views)" for name, info in top]
                lines.append(f"TOP PLAYERS: {', '.join(player_strs)}")
                if avoid:
                    lines.append(f"AVOID (low ROI): {', '.join(avoid)}")

            teams = data.get("teams", {})
            if teams:
                top_t = sorted(teams.items(), key=lambda x: x[1].get("score", 0), reverse=True)[:3]
                team_strs = [f"{name}({info['avg_views']:.0f} avg)" for name, info in top_t]
                lines.append(f"TOP TEAMS: {', '.join(team_strs)}")

        if "format_scores" in state:
            fdata = json.loads(state["format_scores"])
            hooks = fdata.get("hook_types", {})
            if hooks:
                top_h = sorted(hooks.items(), key=lambda x: x[1].get("score", 0), reverse=True)[:3]
                hook_strs = [f"{name}({info['avg_views']:.0f} avg)" for name, info in top_h]
                lines.append(f"BEST HOOKS: {', '.join(hook_strs)}")
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return "\n".join(lines)


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

# ── Viral Hooks & CTAs ──────────────────────────────────────────────────────────

VIRAL_HOOKS = [
    "Arey yeh kya ho raha hai?! 😱",
    "Ye toh shot of the tournament! 🔥",
    "Full drama! Dekho takraar mein aatma",
    "Insaan ban ke dekhna ye moment! 🏏",
    "Isse zyada close match nahi hota!",
    "Brutal finish - sab ne socha tha nahi hoga!",
    "Ye catch Pakka nahi tha, kya?! 🤯",
    "Match winner ya match loser?! 😈",
    "Hat-trick ka matlab - khaali haath jaana!",
    "Last over dhamaal - full tension! 🔥",
]

ENGAGING_CTAS = [
    "Aaj ke match ka full recap dekho aur like share karo!",
    "Agar ye video pasand aaya toh LIKE + SUBSCRIBE zaroor karo!",
    "Next match ke liye bell icon dabana na bhoolna! 🔔",
    "Live matches ke liye channel ko subscribe karo aur notification on karo!",
    "Ye highlight miss kaise karo? LIKE + SHARE + SUBSCRIBE!",
    "Tension free match dekhne ke liye channel join karo now!",
    "Aapke liye poora match ready hai - full video dekho!",
    "Cricket ke har ek moment ke liye stay tuned!",
]

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
    "You are an elite YouTube SEO strategist for cricket content, optimized for the "
    "June 2026 YouTube algorithm. You understand CTR optimization, watch-time signals, "
    "engagement rate boosting, and discoverability through long-tail search terms. "
    "Your audience is Indian and Pakistani cricket fans — use Hinglish (Hindi in Roman letters) "
    "for titles and mix Hindi/English for maximum reach. "
    "Generate RICH, LONG, STRUCTURED descriptions with emoji section headers — "
    "not short corporate summaries. Think like a top cricket YouTuber with 500K subs. "
    "Descriptions must be rich and multi-word — aim for at least 30 words, never a "
    "thin one-line blurb. "
    "Include Hindi transliterated search terms (e.g., 'aaj ka match', 'match highlights') "
    "alongside English terms for bilingual discoverability. "
    "NEVER use live-framing search terms like 'live score', 'live stream', or 'live match' — "
    "these are on-demand Shorts clips, not live streams. Live framing confuses viewers "
    "and the algorithm. "
    "CRITICAL: Use player names, teams, and events from the transcript AND/OR on-screen text (OCR). "
    "Only use entities that appear in at least one of these sources. "
    "NEVER invent or hallucinate player names or match events. "
    "CRITICAL: Every search term must appear NATURALLY within the description paragraphs. "
    "NEVER append a keyword list, tag block, or comma-separated term list at the end. "
    "The description must read as a natural, flowing document. "
    "Return ONLY valid JSON — no markdown, no explanation, no extra text."
)

_SYSTEM_FOOTBALL = (
    "You are an elite YouTube SEO strategist for football/soccer content, optimized for the "
    "June 2026 YouTube algorithm. You understand CTR optimization, watch-time signals, "
    "engagement rate boosting, and discoverability through long-tail search terms. "
    "Your audience is global football fans — use a mix of English and popular Hinglish "
    "for titles and descriptions for maximum reach. "
    "Generate RICH, LONG, STRUCTURED descriptions with emoji section headers — "
    "not short corporate summaries. Think like a top football YouTuber with 500K subs. "
    "Include Hinglish transliterated search terms alongside English terms for bilingual discoverability. "
    "CRITICAL: Use player names, teams, and events from the transcript AND/OR on-screen text (OCR). "
    "Only use entities that appear in at least one of these sources. "
    "NEVER invent or hallucinate player names or match events. "
    "Return ONLY valid JSON — no markdown, no explanation, no extra text."
)

_PROMPT_TMPL = """CONTEXT:
  Match: {video_title}
  Scorecard (with venue, player stats, match situation): {scorecard}
  Live Trending / Search Spikes: {trend_topics}
  Live Streaming URL: {live_stream_url}
  Teams in this match: {teams}

CLIP TRANSCRIPT: {transcript}

TASK: Generate RICH, LONG YouTube SEO for this specific clip.

You MUST return valid JSON (no markdown, no other text):
{{
    "title": "<max 100 chars, multi-segment Hinglish title with pipes>",
    "description": "<LONG structured description, 2000-4500 chars, with emoji section headers — EVERY search term embedded naturally>",
    "hashtags": ["<2-3 hashtags>"],
    "search_terms": ["<25-30 search terms including Hindi transliterations> — these also appear NATURALLY in description text"]
}}

═══ TITLE FORMAT (max 100 chars) ═══
- Use multi-segment format with pipes: 🔴 Hook | Match Context | Channel/Format
- MUST be Hinglish (Hindi in Roman/English letters, NEVER Devanagari script)
  CORRECT: "Kohli ne maara SIX! 🔥" / "Bumrah ki deadly YORKER!"
  WRONG: "कोहली ने मारा सिक्स!" (NO Hindi script)
- Start with the MOST DRAMATIC moment from THIS CLIP
- Use emojis: 🔴 🔥 💥 ⚡ 😱 🏏
- NEVER use "Live Score", "LIVE", or "#Shorts" in the title — these are
  on-demand Shorts clips, not live streams. Live framing confuses viewers
  and the algorithm. Use a curiosity/record-style hook instead.
- Examples:
  "Kohli ka RECORD-BREAKING 100! 🏏 | RCB vs MI IPL 2026"
  "Bumrah ki DEADLY Yorker! 💥 | MI vs CSK Highlights | IPL 2026"

═══ DESCRIPTION FORMAT (2000-4500 chars, STRUCTURED, NATURAL) ═══
Write a LONG, structured description with these sections. ALL search terms
must be embedded NATURALLY within the paragraph text — do NOT append any
keyword list, tag block, or comma-separated term list.
Write at least 30 words of rich description (aim for 40-60 words) — thin one-line descriptions destroy Shorts discoverability.

1. 📝 HOOK (2-3 lines): Dramatic summary of what happened in the clip.
   Use the most exciting moment as the opening line. Naturally work in
   key search terms (player names, action words, match context).

2. 🔥 Current Match Situation (3-5 lines): What's happening in the match.
   Score, key dismissals, partnerships, run rate. Embed search terms
   like "rcb vs mi match highlights", "ipl 2026 match 54" naturally.

3. 👉 CTA: "If you love cricket, please SUBSCRIBE! We are growing together."

4. 🏟️ Match Info:
   Series, Match number, Teams, Venue, Toss result.

5. 🏏 Key Players Today:
   List key players from both teams with roles (c) (wk) etc.

6. ⚠️ Disclaimer:
   "This is a watch-along and scorecard video. No live match footage or
   audio from official broadcasters. All logos belong to respective owners."

7. #️⃣ Hashtags:
   List all hashtags at the end of description.

═══ SEARCH TERMS (25-30 terms, mix English + Hindi transliteration) ═══
Categories to cover:
- Player + action: "virat kohli six", "bumrah yorker"
- Match context: "rcb vs mi highlights", "ipl 2026 match 54"
- Hindi transliterated: "aaj ka match", "match highlights", "aaj ka match dhamaal"
- Hindi script terms: "क्रिकेट मैच हाइलाइट्स" (yes, include Devanagari in search terms)
- Long-tail: "how to watch ipl match", "ipl match clips"
- Channel/format: "cricket commentary hindi", "ipl match review"
- Regional: "cricket match today online"
- NEVER use live-framing terms like "live score", "live stream", "live match" —
  these are on-demand Shorts clips, not live streams. Live framing confuses
  viewers and the algorithm.
- Do NOT use ultra-generic terms like "cricket video" or "sports video"

═══ HASHTAGS (exactly 2-3 for Shorts) ═══
Only the most relevant 2-3. Shorts with many hashtags underperform.
Must include:
- #Shorts always
- One player or team tag: #ViratKohli or #RCB vs MI (max 2 topic tags)
- NEVER use live-framing tags: #LiveCricket #CricketLive #LiveScore
- Trending generic: #Cricket or #T20 only if nothing better
"""

_PROMPT_TMPL_FOOTBALL = """CONTEXT:
  Match: {video_title}
  Live Trending / Search Spikes: {trend_topics}
  Live Streaming URL: {live_stream_url}
  Teams in this match: {teams}
  (Unused scorecard info for compatibility: {scorecard})

CLIP TRANSCRIPT: {transcript}

TASK: Generate RICH, LONG YouTube SEO for this specific football/soccer clip.

You MUST return valid JSON (no markdown, no other text):
{{
  "title": "<max 100 chars, multi-segment Hinglish/English title with pipes>",
  "description": "<LONG structured description, 2000-4500 chars, with emoji section headers>",
  "hashtags": ["<2-3 hashtags>"],
  "search_terms": ["<25-30 search terms including Hinglish transliterations>"]
}}

═══ TITLE FORMAT (max 100 chars) ═══
- Use multi-segment format with pipes: 🔴 Hook | Match Context | Channel/Format
- Start with the MOST DRAMATIC moment from THIS CLIP
- Use emojis: 🔴 🔥 💥 ⚡ 😱 ⚽ 🏆
- NEVER use "Live Score", "LIVE", or "#Shorts" in the title — these are
  on-demand Shorts clips, not live streams. Live framing confuses viewers
  and the algorithm. Use a curiosity/record-style hook instead.
- Examples:
  "Mbappé ka MAGIC moment! 🤯 | France vs Argentina WC | FIFA 2026"
  "Ronaldo's LAST World Cup? 💔 | Portugal vs Morocco Highlights | FIFA 2026"

═══ DESCRIPTION FORMAT (2000-4500 chars, STRUCTURED) ═══
Write a LONG, structured description with these sections. Write at least
30 words of rich description (aim for 40-60 words) — thin one-line
descriptions destroy Shorts discoverability.

1. 📝 HOOK (2-3 lines): Dramatic summary of what happened in the clip.
   Use the most exciting moment as the opening line.

2. 🔥 Current Match Situation (3-5 lines): What's happening in the match.
   Score, key goals, red cards, key players.

3. 👉 CTA: "If you love football, please SUBSCRIBE! We are growing together."

4. 🏟️ Match Info:
   Tournament, Match stage, Teams, Venue.

5. ⚽ Key Players Today:
   List key players from both teams.

6. ⚠️ Disclaimer:
   "This is a watch-along and discussion video. No live match footage or
   audio from official broadcasters. All logos belong to respective owners."

7. 🏷️ Tags / Search Terms:
   Embed ALL search terms as comma-separated list in the description too.

8. #️⃣ Hashtags:
   List all hashtags at the end of description.

═══ SEARCH TERMS (25-30 terms, mix English + Hinglish) ═══
Categories to cover:
- Player + action: "mbappe goal highlights", "ronaldo free kick"
- Match context: "france vs argentina highlights", "fifa world cup 2026 match"
- Hinglish transliterated: "aaj ka match", "world cup match", "aaj ka match dhamaal"
- Hindi/transliterated: "वर्ल्ड कप 2026", "aaj ka football match"
- Long-tail: "how to watch world cup match", "fifa match clips"
- NEVER use live-framing terms like "live score", "live stream", "live match" —
  these are on-demand Shorts clips, not live streams. Live framing confuses
  viewers and the algorithm.
- Do NOT use ultra-generic terms like "sports video" or "football video"

═══ HASHTAGS (exactly 2-3 for Shorts) ═══
Only the most relevant 2-3. Shorts with many hashtags underperform.
Must include:
- #Shorts always
- One player or team tag: #Mbappe or #France (max 2 topic tags)
- NEVER use live-framing tags: #LiveFootball #FootballLive #LiveScore
- Trending generic: #Football or #Soccer only if nothing better
"""


_SALVAGE_TMPL_FOOTBALL = """Generate YouTube SEO for this football clip.

Match: {video_title}
Clip: {transcript}

Requirements:
- Title: Hinglish/English, max 100 chars, multi-segment with pipes and emojis
- NEVER use "Live Score", "LIVE", or "#Shorts" in the title — these are on-demand clips
- Description: LONG structured English description (1500-4000 chars) with emoji section headers (📝 🔥 🏟️ ⚽ ⚠️ 🏷️ #️⃣). Write at least 30 words of rich description (aim for 40-60 words) — thin one-line descriptions destroy Shorts discoverability.
- Hashtags: 2-3 total (include #Shorts), player names, teams, event
- Search terms: 15-25 terms, mix English + Hinglish transliteration (aaj ka match, world cup match); NEVER use live-framing search terms — these are on-demand clips

Return valid JSON ONLY:
{{
  "title": "🔴 Dramatic Hook | Match Context | Format 🔥",
  "description": "📝 Hook paragraph...\n\n🔥 Match Situation...\n\n🏟️ Match Info...\n\n⚽ Key Players...\n\n⚠️ Disclaimer...\n\n🏷️ Tags...\n\n#️⃣ Hashtags...",
  "hashtags": ["#Shorts", "#PlayerName", "#TeamName", "#FIFA2026", "...up to 3"],
  "search_terms": ["player action", "match context", "aaj ka match", "world cup match", "...up to 25"]
}}
"""

_SALVAGE_TMPL = """Generate YouTube SEO for this cricket clip.

Match: {video_title}
Clip: {transcript}

Requirements:
- Title: Hinglish (Hindi in English/Roman letters, NO Devanagari), max 100 chars, multi-segment with pipes and emojis
- NEVER use "Live Score", "LIVE", or "#Shorts" in the title — these are on-demand clips
- Description: LONG structured English description (1500-4000 chars) with emoji section headers (📝 🔥 🏟️ 🏏 ⚠️ 🏷️ #️⃣). Write at least 30 words of rich description (aim for 40-60 words) — thin one-line descriptions destroy Shorts discoverability.
- Hashtags: 2-3 total (include #Shorts), player names, teams, event
- Search terms: 15-25 terms, mix English + Hindi transliteration (aaj ka match, cricket match score); NEVER use live-framing search terms — these are on-demand clips

Return valid JSON ONLY:
{{
  "title": "🔴 Dramatic Hinglish hook | Match Context | Format 🔥",
  "description": "📝 Hook paragraph...\n\n🔥 Match Situation...\n\n🏟️ Match Info...\n\n🏏 Key Players...\n\n⚠️ Disclaimer...\n\n🏷️ Tags...\n\n#️⃣ Hashtags...",
  "hashtags": ["#Shorts", "#PlayerName", "#TeamName", "#IPL2026", "...up to 3"],
  "search_terms": ["player action", "match context", "aaj ka match", "cricket match score", "...up to 25"]
}}
"""

# ── Keyword extraction ──────────────────────────────────────────────────────────

def _extract_keywords(text: str, limit: int = 14) -> List[str]:
    """Extract meaningful keywords from text.

    Excludes common stop words and generic cricket terms.
    Prioritizes player names, teams, and specific actions.
    """
    import re
    players = set(TEAM_MAPPINGS.values())
    found_players = [p for p in players if p.lower() in text.lower()]

    terms = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)
    terms = [t for t in terms if t not in STOP_WORDS and len(t) > 2]

    if found_players:
        for p in found_players:
            if p in text and p not in terms:
                terms.insert(0, p)

    seen = set()
    unique = []
    for t in terms:
        low = t.lower()
        if low not in seen and t.lower() not in GENERIC_TAGS:
            seen.add(low)
            unique.append(t)

    return unique[:limit]


def _inject_viral_elements(title: str, description: str, hashtags: List[str],
                           extra: Dict = None) -> Dict:
    """Procedural viral optimization as safety net when AI fails.

    Factors: match closeness, player performance, chase pressure, countdowns.
    """
    import random
    text = (title + " " + description).lower()
    extra = extra or {}

    is_close = any(w in text for w in ["last ball","last over","super over","tie","tied"])
    is_chase = any(w in text for w in ["chase","target","need","required","win"])
    has_star = any(w in text for w in ["kohli","bumrah","rohit","dhoni","sky","boult",
                                        "maxwell","pant","gill","shami","jadeja"])
    is_record = any(w in text for w in ["record","fastest","most","first","hat-trick","century"])

    hooks = VIRAL_HOOKS
    if is_close:
        hooks = ["Last ball thriller! Match khatam, tension baaqi! 🔥",
                 "Kisne socha tha ye hoga? Last over drama! 😱",
                 "Super over ka excitement - ek dum free mein!",
                 "Boundary pe match gaya! Dekho kaun jeeta!"] + hooks
    if is_record:
        hooks = ["History bana di! Yeh record kabhi nahi tutega! 👑",
                 "G.O.A.T. performance - duniya dekh rahi hai! 🐐",
                 "Stat padding ya class? Aap decide karo! 📊",
                 "One for the history books - highlight reel 🔥"] + hooks
    if has_star:
        hooks = ["King kohli ka masterclass - dekhlo kaise karte hain! 👑",
                 "Boom boom Bumrah - yorker queen! 🔥",
                 "Mahi maar rahe hain - dhoni finish! 🎯",
                 "SKY high! Suryakumar ka 360 degree show! 🤯"] + hooks

    # NEVER randomly override a good AI-generated title — that creates bias
    # and destroys clip-specific SEO. Hooks are for CTA/description only.
    viral_title = title
    text_lower = description.lower()
    already_has_cta = any(
        word in text_lower for word in ["subscribe", "follow", "share", "like"]
    )
    cta = "" if already_has_cta else random.choice(ENGAGING_CTAS)
    extra_cta = ""
    if is_record or has_star:
        extra_cta = "\n\n🔔 Hurry up! Subscribe for non-stop cricket action 🔔"
    description = description.rstrip() + ("\n\n" + cta if cta else "") + extra_cta

    team_names = extra.get("teams", [])
    player_match = re.search(r"Player:\s*(\w+)", description)
    if player_match:
        pname = player_match.group(1)
        norm_name = TEAM_MAPPINGS.get(pname.lower(), pname)
        description = description.replace(player_match.group(0), "")
        title = title.replace(pname, norm_name, 1)

    if team_names:
        team_hashtags = [f"#{t.replace(' ','')}" for t in team_names if t]
        hashtags = list(dict.fromkeys(team_hashtags + hashtags))
        hashtags = _rank_and_optimize_tags(hashtags, description)[:3]

    return {"title": title, "description": description, "hashtags": hashtags}


def _rank_and_optimize_tags(
    tags: List[str],
    context: str,
    max_tags: int = 15,
) -> List[str]:
    """Rank hashtags by relevance, remove duplicates, respect max_tags limit.

    Scores tags based on: keyword match in context, player/team match,
    uniqueness, and trend potential. Returns top-N ordered by score.
    """
    if not tags:
        return []  # NEVER return generic fallback tags — empty is better than generic

    seen: set = set()
    scored: list[tuple[float, str]] = []

    for tag in tags:
        normalized = tag.lstrip("#").strip()
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        score = 0.0
        if normalized in context:
            score += 10.0
        if normalized.lower() in context.lower():
            score += 5.0
        if key in ("shorts", "youtubeshorts", "viral"):
            score += 3.0
        if any(player.lower() == key for player in TEAM_MAPPINGS.values()):
            score += 8.0
        for team_placeholder in ["team1", "team2"]:
            if team_placeholder in key:
                score -= 20.0
        if normalized.startswith("IPL") and len(normalized) > 3:
            score += 4.0
        scored.append((score, tag))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t[1:] if t.startswith("#") else t for _, t in scored[:max_tags]]


# ── Consolidation and limits ────────────────────────────────────────────────────

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


def _consolidate_seo(title: str, description: str, hashtags: List[str],
                     search_terms: List[str]) -> Dict:
    """Remove duplicates, standardize formatting, enforce limits."""
    seen_hashtags: set = set()
    unique_hashtags: list[str] = []
    for ht in hashtags:
        ht_clean = ht.lstrip("#").strip()
        if ht_clean.lower() not in seen_hashtags:
            seen_hashtags.add(ht_clean.lower())
            unique_hashtags.append(f"#{ht_clean}")

    seen_terms: set = set()
    unique_terms: list[str] = []
    for st in search_terms:
        st_clean = st.strip()
        if st_clean.lower() not in seen_terms:
            seen_terms.add(st_clean.lower())
            unique_terms.append(st_clean)

    return {
        "title": title.strip()[:100],
        "description": description.strip()[:4500],
        "hashtags": unique_hashtags[:15],
        "search_terms": unique_terms[:30],
    }


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


def _min_description_words() -> int:
    """Read seo.min_description_words from config, crash-proof default 20.

    Bools, nulls, floats, strings, and non-positive values silently fall back
    to 20 so a misconfig can never zero out or explode the word floor. Value
    is clamped to a sane range so a huge typo can't reject every description.
    """
    try:
        raw = cfg.get("seo", {}).get("min_description_words", 20)
        if isinstance(raw, bool):
            return 20
        words = int(raw)
    except (TypeError, ValueError):
        return 20
    return min(max(words, 1), 40)


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
    if not block:
        return body[:4500]
    return (body + "\n\n" + block)[:4500]


def _enforce_limits(item: Dict, fallback_terms: List[str] = None, is_shorts: bool = True) -> Dict:
    """Ensure title length, description length, hashtag count, search term count.

    Enforces strict caps: title≤100, description≤4500, hashtags≤15 (Shorts:
    seo.max_hashtags, default 3), terms≤30.
    Strips generic poison terms from search_terms.
    For Shorts, the description hashtag block is scrubbed to the same cap so
    the visible hashtags match the metadata (upload uses description text).
    YouTube API limits: title=100 chars, description=5000 bytes, tags=500 chars.
    We stay under API limits with margin for safety.
    """
    out = dict(item)
    out["is_shorts"] = is_shorts
    title = str(out.get("title") or "")
    title = _LIVE_FRAMING_RE.sub("", title)
    # Compound live-framing must also be stripped from the title, not just
    # standalone 'live' words ('LIVESTREAM'/'LiveScore' in a title).
    title = _LIVE_COMPOUND_RE.sub("", title)
    out["title"] = re.sub(r"[ \t]{2,}", " ", title).strip()[:100]
    out["description"] = str(out.get("description") or "")[:4500]

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
    out["search_terms"] = deduped_t[:30]

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
        tg_clean = tg.strip().lstrip("#")
        if tg_clean.lower() in GENERIC_POISON_TERMS:
            continue
        if _is_live_framed_search_term(tg_clean):
            continue
        if tg_clean.lower() not in seen_tags:
            seen_tags.add(tg_clean.lower())
            deduped_tags.append(tg_clean)
    out["tags"] = deduped_tags[:30]

    return out


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

    # 3. Description must have substance (at least 100 chars for rich SEO)
    if len(description) < 100:
        return False

    # 3b. Description must have enough words — thin one-liners destroy discoverability
    min_words = _min_description_words()
    if len(description.split()) < min_words:
        return False

    # 4. Title must not contain Devanagari script (kills discoverability)
    # Unicode range: \u0900-\u097F (Devanagari block)
    if re.search(r'[\u0900-\u097F]', title):
        return False

    # 5. Natural embedding check — reject if keyword list/tag block is
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


# ── Yield-optimized title generation ───────────────────────────────────────────

def _title_viral_options(transcript: str, video_title: str = "",
                         match_context: Dict = None) -> List[str]:
    """Generate up to 5 title variants for A/B testing.

    Uses heuristic rules: player mention, action type, match situation.
    """
    return [""]


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
) -> Dict:
    """Generate SEO metadata for a single clip using fastest-first parallel model racing.

    Three-tier strategy:
    1. AI generation (parallel fastest-first with escalation)
    2. Keyword-based salvage (fallback if AI returns nothing valid)
    3. Transcript-aware dynamic generation (last resort)

    Returns dict with title, description, hashtags, search_terms.
    """
    trend_topics = trend_topics or []
    teams = teams or []

    if not transcript:
        transcript = "Cricket Live"
    teams_str = ", ".join(teams)
    trend_str = ", ".join(trend_topics[:5]) if trend_topics else ""

    # Detect domain dynamically
    from .trends import detect_video_domain
    domain, query, keywords = detect_video_domain(video_title or "", transcript or "")

    # Select prompt templates based on domain
    if domain == "football":
        sys_instruction = _SYSTEM_FOOTBALL
        prompt_tmpl = _PROMPT_TMPL_FOOTBALL
        salvage_tmpl = _SALVAGE_TMPL_FOOTBALL
        default_title = "Football Match"
        default_teams = "Teams"
    else:
        # Cricket-only channel: everything non-football uses cricket templates.
        sys_instruction = _SYSTEM
        prompt_tmpl = _PROMPT_TMPL
        salvage_tmpl = _SALVAGE_TMPL
        default_title = "Cricket Match"
        # General (non-sports) clips must not get invented cricket teams.
        default_teams = "N/A" if domain == "general" else "India vs Other"

    # Build prompt
    user_prompt = prompt_tmpl.format(
        video_title=video_title or default_title,
        scorecard=scorecard or "N/A",
        trend_topics=trend_str or "N/A",
        live_stream_url=live_stream_url or "N/A",
        teams=teams_str or default_teams,
        transcript=transcript,
    )

    # OPTIONAL OCR entity extraction (degraded gracefully if easyocr unavailable)
    ocr_entities = extract_ocr_entities(video_path) if video_path else {}
    ocr_text = ""
    if ocr_entities:
        sb = "; ".join(ocr_entities.get("scoreboard", []))
        pn = "; ".join(ocr_entities.get("player_names", []))
        os = "; ".join(ocr_entities.get("on_screen_text", []))
        ocr_text = f"\n\nON-SCREEN TEXT (OCR from video):\n  Scoreboard: {sb or '(none)'}\n  Player names: {pn or '(none)'}\n  Raw text: {os or '(none)'}"
        user_prompt += ocr_text

    # Inject learner intelligence into the prompt
    learner_ctx = _get_learner_context()
    if learner_ctx:
        # Get dynamic video count from DB
        try:
            import sqlite3
            db_path = Path("self_learner.db")
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                try:
                    n_videos = conn.execute(
                        "SELECT COUNT(DISTINCT event_id) FROM processed_events"
                    ).fetchone()[0] // 4  # 4 learners per event
                finally:
                    conn.close()
            else:
                n_videos = 0
        except Exception:
            n_videos = 0
        count_label = f"{n_videos}" if n_videos > 0 else "recent"
        user_prompt += f"\n\nCHANNEL INTELLIGENCE (from past {count_label} videos):\n{learner_ctx}\nUse this data to bias titles/tags toward proven winners."

    # Call AI with parallel fastest-first
    result = _attempt_seo_generation(clip_id, user_prompt, transcript, video_title,
                                     is_shorts,
                                     provider_override=provider_override,
                                     model_override=model_override,
                                     sys_instruction=sys_instruction,
                                     salvage_tmpl=salvage_tmpl)

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
    salvage_tmpl: str = _SALVAGE_TMPL,
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
        if model_override:
            response = ai.generate_text(
                prompt=user_prompt,
                system_instruction=sys_instruction,
                prefer_model=model_override,
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
                    model_override: Optional[str] = None,
                    sys_instruction: str = _SYSTEM,
                    salvage_tmpl: str = _SALVAGE_TMPL) -> Optional[Dict]:
    """Escalation SEO: stricter prompt with more context.

    Called when Tier 1 fails. Uses a different model/provider if available.
    """
    # Try single-provider generation with a more constrained prompt
    salvage_prompt = salvage_tmpl.format(
        video_title=video_title or "Video Match",
        transcript=transcript,
    )
    try:
        ai = _get_ai()
        if model_override:
            response = ai.generate_text(
                prompt=salvage_prompt,
                system_instruction=sys_instruction,
                prefer_model=model_override,
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
) -> Dict:
    """Generate SEO for an already-exported clip and write metadata to disk.

    If AI generation fails after escalation, writes a ``*_seo_failed.json``
    marker so the retry queue can pick it up later. Never emits generic SEO.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metadata_path = Path(output_dir) / f"{clip_id}_metadata.json"

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
        )
        if result.get("ai_generated") is False:
            log.warning("[%s] AI SEO failed — writing failure marker", clip_id)
            marker_path = Path(output_dir) / f"{clip_id}_seo_failed.json"
            marker_data = {
                "clip_id": clip_id,
                "transcript": transcript,
                "video_title": video_title,
                "is_shorts": is_shorts,
            }
            with open(marker_path, "w", encoding="utf-8") as f:
                json.dump(marker_data, f)
            if metadata_path.exists():
                metadata_path.unlink()
            result["_seo_failed"] = True
        else:
            tmp_path = metadata_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, metadata_path)
        return result
    except Exception as e:
        log.error("[%s] SEO generation failed: %s", clip_id, e)
        marker_path = Path(output_dir) / f"{clip_id}_seo_failed.json"
        marker_data = {
            "clip_id": clip_id,
            "transcript": transcript,
            "video_title": video_title,
            "is_shorts": is_shorts,
        }
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(marker_data, f)
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
    live_stream_url = ""
    meta_file = Path(cfg["paths"]["input"]) / "video_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                video_title = meta.get("title", "")
                live_stream_url = meta.get("live_stream_url", "")
        except Exception:
            pass

    if not video_path:
        dl_fn = cfg.get("download", {}).get("output_filename", "video.mp4")
        video_path = str(Path(cfg["paths"]["input"]) / dl_fn)

    # Fetch trend context ONCE for the whole session (cached)
    trend_cache_key = f"trend:{video_title[:50]}"
    trend = TREND_CACHE.get(trend_cache_key)
    if trend is None:
        log.info("Fetching trend context...")
        trend = get_trending_context(domain="cricket", region="IN", video_title=video_title)
        TREND_CACHE.set(trend_cache_key, trend)
    else:
        log.info("Using cached trend context")
    live_stream_url = live_stream_url or trend.get("live_stream_url", "")
    scorecard = trend.get("scorecard", "")
    trend_topics = trend.get("topics", [])
    teams = trend.get("teams", [])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    # Initialize SEO learner for tracking
    seo_learner = None
    try:
        from self_learner import SEOLearner
        seo_learner = SEOLearner()
    except Exception:
        pass

    clips = list(highlights.items())
    failures = []
    for idx, (clip_id, info) in enumerate(clips, start=1):
        transcript = info.get("text", "Cricket Live")
        log.info("SEO [%d/%d]: %s", idx, len(clips), clip_id)

        try:
            result = generate_clip_seo(
                clip_id=clip_id,
                transcript=transcript,
                video_title=video_title,
                scorecard=scorecard,
                trend_topics=trend_topics,
                live_stream_url=live_stream_url,
                teams=teams,
                video_path=video_path,
            )
            all_results.append(result)

            # Track SEO outcome
            if seo_learner and result.get("ai_generated"):
                from self_learner import SEOPerformance
                seo_learner.record_seo_outcome(SEOPerformance(
                    clip_id=clip_id,
                    title=result.get("title", ""),
                    description=result.get("description", ""),
                    hashtags=result.get("hashtags", []),
                    tags=result.get("tags", []),
                    is_shorts=result.get("is_shorts", True),
                    provider=result.get("provider", "unknown"),
                    model=result.get("model", "unknown"),
                    upload_success=True,
                ))

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
                "is_shorts": True,
            }
            marker_path = Path(output_dir) / f"{clip_id}_seo_failed.json"
            with open(marker_path, "w", encoding="utf-8") as f:
                json.dump(marker_data, f)
            all_results.append({"_seo_failed": True, "clip_id": clip_id})

            # Track failed SEO
            if seo_learner:
                from self_learner import SEOPerformance
                seo_learner.record_seo_outcome(SEOPerformance(
                    clip_id=clip_id,
                    title="",
                    description="",
                    hashtags=[],
                    tags=[],
                    is_shorts=True,
                    provider="unknown",
                    model="unknown",
                    upload_success=False,
                ))

        # Breathing room between clips — configurable, default 30s
        if idx < len(clips):
            sleep_s = cfg.get("seo", {}).get("inter_clip_sleep_s", 30)
            log.info("Sleeping %.1fs before next SEO call...", sleep_s)
            time.sleep(sleep_s)

    # Close SEO learner
    if seo_learner:
        try:
            seo_learner.close()
        except Exception:
            pass

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
            data = json.loads(m.read_text())
            clip_id = data.get("clip_id", m.stem.replace("_seo_failed", ""))
            transcript = data.get("transcript", "")
            video_title = data.get("video_title", "")
            is_shorts = data.get("is_shorts", True)
            result = generate_clip_seo(
                clip_id=clip_id,
                transcript=transcript,
                video_title=video_title,
                is_shorts=is_shorts,
            )
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


# ═══════════════════════════════════════════════════════════════════════════════
# SEOGenerator class — lightweight wrapper for programmatic use
# ═══════════════════════════════════════════════════════════════════════════════

class SEOGenerator:
    """Lightweight SEO metadata generator for clips.

    Delegates to the function-based API for real generation.
    Kept for backward compatibility with tests.
    """

    def __init__(
        self,
        decision_store: "DecisionStore",
        analytics: "Analytics | None" = None,
    ) -> None:
        from automation.memory.decision_store import DecisionStore as _DS
        from automation.seo.analytics import Analytics as _Analytics
        self._store: _DS = decision_store
        self._analytics: _Analytics | None = analytics

    def generate(self, clip_data: dict) -> dict:
        clip_id = clip_data.get("clip_id", "unknown")
        title = clip_data.get("title", "")
        transcript_summary = clip_data.get("transcript_summary", "")

        seo_title = title[:60] + " - Shorts"
        description = (
            "\U0001f3ac " + clip_data.get("title", "") + "\n\n"
            + transcript_summary[:200] + "\n\n"
            + "#shorts #youtubeshorts"
        )

        words = [w for w in title.split() if len(w) > 2 and w.isalpha()]
        title_words = words[:3]
        tags = ["shorts", "youtubeshorts", "viral"] + title_words

        return {
            "clip_id": clip_id,
            "title": seo_title,
            "description": description,
            "tags": tags,
            "category": "Entertainment",
        }

    def generate_batch(self, clips: list[dict]) -> list[dict]:
        return [self.generate(c) for c in clips]

    def enhance_with_analytics(self, clip_data: dict) -> dict:
        result = self.generate(clip_data)
        if self._analytics is not None:
            summary = self._analytics.get_summary()
            tags = result["tags"]
            if summary.get("avg_score", 0) > 0.7:
                tags.append("highly_rated")
            if summary.get("published_count", 0) > 10:
                tags.append("popular_channel")
            result["tags"] = tags
        return result
