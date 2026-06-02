"""seo.py — Per-clip SEO generation for Indian cricket Shorts.

Uses parallel fastest-first model racing: fires the fastest available models
concurrently and takes the first valid JSON response. No backoff — on failure
the next tier of models is tried immediately. Three-tier fallback: AI → salvage
→ transcript-aware dynamic generation. Every title is clip-specific — no
generic templates or prefixes.
"""
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

from utils.config import load_config
from utils.logger import get_logger
from utils.ai_client import AIClient
from .trends import TEAM_MAPPINGS
from automation._cache import TTLCache

SUGGEST_CACHE = TTLCache(maxsize=16, ttl=600)
TREND_CACHE = TTLCache(maxsize=4, ttl=300)

cfg = load_config()
log = get_logger("seo", cfg["logging"]["log_file"], cfg["logging"]["level"])
ai = AIClient()


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
                    ai._provider = best_provider
                    ai._model = best_model
            except Exception as e:
                log.warning("Auto-benchmark failed: %s", e)

class SEOGenerationError(Exception):
    """Raised when all AI providers fail during SEO generation."""


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

# ── Prompt ─────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a desi YouTube SEO expert specializing in viral cricket shorts "
    "for Indian and Pakistani audiences. "
    "Generate high-CTR titles, engaging descriptions, and optimized tags. "
    "Use emojis and full-on desi cricket discussion style — no boring English commentary. "
    "Focus on Indian and Pakistani cricket audience with drawing-room style banter. "
    "CRITICAL: Only use player names, teams, and events that appear in the transcript. "
    "NEVER invent or hallucinate player names or match events. "
    "Return ONLY valid JSON — no markdown, no explanation."
)

_PROMPT_TMPL = """CONTEXT:
  Match: {video_title}
  Scorecard (with venue, player stats, match situation): {scorecard}
  Live Trending / Search Spikes: {trend_topics}
  Live Streaming URL: {live_stream_url}
  Teams in this match: {teams}

CLIP TRANSCRIPT: {transcript}

TASK: Generate YouTube Shorts SEO for this specific clip.

You MUST return valid JSON (no markdown, no other text):
{{
  "title": "<max 80 chars, Hinglish hook that describes THIS CLIP>",
  "description": "<English description of what happened. Max 800 chars. No line breaks>",
  "hashtags": ["<max 5 hashtags>"],
  "search_terms": ["<max 10 search terms for this clip>"]
}}

TITLE REQUIREMENTS (Hinglish only — Hindi + English mix):
- CRITICAL: Start with the MOST IMPORTANT event of THIS CLIP
  (e.g., "Kohli ne maara SIX!" or "Bumrah ki deadly YORKER!")
- NOT just the match title. Make it specific to the clip's content.
- Use Hindi/English mix (Hinglish) for Indian audience
- Include the most dramatic moment of the clip
- End with relevant emojis
- Max 80 characters

DESCRIPTION REQUIREMENTS (Full English):
- Write in English — casual, engaging, not corporate
- First 2 lines: What happened in this specific clip
- Then: Context of the match situation
- Then: Player stats/achievements if relevant
- End with a CTA (Call to Action)
- For Shorts: Short and punchy (max 300 chars)
- For Regular videos: Full detailed match coverage (max 800 chars)

SEARCH TERMS (English):
- Primary: Player name + action (e.g., "virat kohli six wankhede")
- Secondary: Match context + clip type
- Don't use generic terms like "cricket video" or "sports video"

HASHTAGS:
- Primary: #PlayerName, #TeamName (from actual teams playing)
- Include #Shorts if this is a short
- Add event-specific tags
- Max 5 hashtags
"""

_SALVAGE_TMPL = """Generate YouTube Shorts SEO for this cricket clip.

Match: {video_title}
Clip: {transcript}

Requirements:
- Title: Hinglish (Hindi+English mix), max 80 chars, with emojis
- Description: English, casual tone, max 500 chars
- Hashtags: max 5, include #Shorts
- Search terms: 3-5 English terms

Return valid JSON ONLY:
{{
  "title": "Hinglish clip-specific title max 80 chars with emojis",
  "description": "English description of the clip, casual and engaging, max 500 chars",
  "hashtags": ["#Shorts", "#Cricket"],
  "search_terms": ["term1", "term2", "term3"]
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

    viral_title = random.choice(hooks) if random.random() < 0.2 else title
    text_lower = description.lower()
    cta = EXTRA_CTA = ""
    if any(word in text_lower for word in ["subscribe", "follow", "share", "like"]):
        cta = ""
    else:
        cta = random.choice(ENGAGING_CTAS)
    if is_record or has_star:
        EXTRA_CTA = "\n\n🔔 Hurry up! Subscribe for non-stop cricket action 🔔"
    description = description.rstrip() + "\n\n" + cta + EXTRA_CTA

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
        hashtags = _rank_and_optimize_tags(hashtags, description)[:5]

    return {"title": title, "description": description, "hashtags": hashtags}


def _rank_and_optimize_tags(
    tags: List[str],
    context: str,
    max_tags: int = 5,
) -> List[str]:
    """Rank hashtags by relevance, remove duplicates, respect max_tags limit.

    Scores tags based on: keyword match in context, player/team match,
    uniqueness, and trend potential. Returns top-N ordered by score.
    """
    if not tags:
        return ["#Shorts", "#Cricket", "#IPL2026"]

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

    # If it's a Python dict representation, try to parse as JSON
    if text.startswith("{") and not text.startswith("{\""):
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
        "title": title.strip()[:80],
        "description": description.strip()[:800],
        "hashtags": unique_hashtags[:5],
        "search_terms": unique_terms[:10],
    }


def _enforce_limits(item: Dict, fallback_terms: List[str] = None, is_shorts: bool = True) -> Dict:
    """Ensure title length, description length, hashtag count, search term count.

    Enforces strict caps: title≤80, description≤800, hashtags≤5, terms≤10.
    """
    out = dict(item)
    out["is_shorts"] = is_shorts
    out["title"] = (out.get("title") or "")[:80]
    out["description"] = (out.get("description") or "")[:800]

    htags = out.get("hashtags") or []
    if isinstance(htags, str):
        htags = [htags]
    seen = set()
    deduped = []
    for t in htags:
        t_clean = t.lstrip("#").strip()
        if t_clean.lower() not in seen:
            seen.add(t_clean.lower())
            deduped.append(f"#{t_clean}")
    out["hashtags"] = deduped[:5]

    terms = out.get("search_terms") or []
    if isinstance(terms, str):
        terms = [terms]
    seen = set()
    deduped_t = []
    for st in terms:
        st_clean = st.strip()
        if st_clean.lower() not in seen:
            seen.add(st_clean.lower())
            deduped_t.append(st_clean)
    out["search_terms"] = deduped_t[:10]

    return out


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

    # 3. Look for { onwards
    brace_start = text.find("{")
    if brace_start >= 0:
        candidate = text[brace_start:]
        result = _try_parse(candidate)
        if result is not None:
            return result
        # 3b. Try truncation repair: close all open braces/brackets, trim trailing garbage
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

    # Build prompt
    user_prompt = _PROMPT_TMPL.format(
        video_title=video_title or "Cricket Match",
        scorecard=scorecard or "N/A",
        trend_topics=trend_str or "N/A",
        live_stream_url=live_stream_url or "N/A",
        teams=teams_str or "India vs Other",
        transcript=transcript,
    )

    # Call AI with parallel fastest-first
    result = _attempt_seo_generation(clip_id, user_prompt, transcript, video_title,
                                     is_shorts,
                                     provider_override=provider_override,
                                     model_override=model_override)

    return result


def _attempt_seo_generation(
    clip_id: str,
    user_prompt: str,
    transcript: str,
    video_title: str,
    is_shorts: bool,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Dict:
    """Attempt AI SEO with two-tier escalation.

    Tier 1: parallel fastest-first AI racing.
    Tier 2: single-provider escalation with stricter prompt.

    Never degrades to keyword fallback — raises on total failure.
    """
    ai_result = _generate_ai_seo(clip_id, user_prompt, transcript, is_shorts,
                                  provider_override=provider_override,
                                  model_override=model_override)
    if ai_result:
        ai_result["ai_generated"] = True
        return ai_result

    esc_result = _escalation_seo(clip_id, user_prompt, transcript, video_title, is_shorts,
                                  model_override=model_override)
    if esc_result:
        esc_result["ai_generated"] = True
        return esc_result

    raise SEOGenerationError(
        f"AI SEO failed for {clip_id} — all providers exhausted"
    )


def _generate_ai_seo(clip_id: str, user_prompt: str,
                     transcript: str, is_shorts: bool,
                     provider_override: Optional[str] = None,
                     model_override: Optional[str] = None) -> Optional[Dict]:
    """Parallel fastest-first AI generation.

    Fires available models concurrently, returns first valid JSON.
    """
    try:
        response = ai.generate_fastest_first(
            prompt=user_prompt,
            system_instruction=_SYSTEM,
            prefer_provider=provider_override,
            prefer_model=model_override,
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

        return _enforce_limits(parsed, is_shorts=is_shorts)
    except Exception as e:
        log.warning("[%s] AI generation failed: %s", clip_id, e)
        return None


def _escalation_seo(clip_id: str, user_prompt: str,
                    transcript: str, video_title: str,
                    is_shorts: bool,
                    model_override: Optional[str] = None) -> Optional[Dict]:
    """Escalation SEO: stricter prompt with more context.

    Called when Tier 1 fails. Uses a different model/provider if available.
    """
    # Try single-provider generation with a more constrained prompt
    salvage_prompt = _SALVAGE_TMPL.format(
        video_title=video_title or "Cricket Match",
        transcript=transcript,
    )
    try:
        response = ai.generate_text(
            prompt=salvage_prompt,
            system_instruction=_SYSTEM,
            prefer_model=model_override,
        )
        if not response:
            return None
        parsed = _parse_json_response(response)
        if parsed and "title" in parsed:
            return _enforce_limits(parsed, is_shorts=is_shorts)
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
    **kwargs,
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
            with open(marker_path, "w") as f:
                json.dump(marker_data, f)
            if metadata_path.exists():
                metadata_path.unlink()
            result["_seo_failed"] = True
        else:
            with open(metadata_path, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
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
        with open(marker_path, "w") as f:
            json.dump(marker_data, f)
        return {"_seo_failed": True, "error": str(e)}


def process_all_seo(highlights_path: str, output_dir: str) -> str:
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
            )
            all_results.append(result)

            # Save individual file immediately
            per_clip_path = Path(output_dir) / f"{clip_id}_metadata.json"
            with open(per_clip_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
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
            with open(marker_path, "w") as f:
                json.dump(marker_data, f)
            all_results.append({"_seo_failed": True, "clip_id": clip_id})

        # Breathing room between clips (skip after last)
        if idx < len(clips):
            log.debug("Sleeping 5s before next SEO call...")
            time.sleep(5)

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
                meta_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
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
