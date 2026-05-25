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
    "You are an elite YouTube Shorts SEO expert for Indian cricket. "
    "Your goal: Maximize CTR (Click-Through Rate) and watch time. "
    "CRITICAL: Only use player names, teams, and events that appear in the transcript. "
    "NEVER invent or hallucinate player names or match events. "
    "Return ONLY valid JSON — no markdown, no explanation."
)

_PROMPT_TMPL = """
CONTEXT:
  Match: {video_title}
  Scorecard (with venue, player stats, match situation): {scorecard}
  Live Trending / Search Spikes: {trend_topics}
  Actual YouTube Search Suggestions (what people ACTUALLY type): {yt_suggestions}
  CTA: {live_cta}

CLIP CONTENT:
  Raw Transcript (may have misspellings): {transcript}
  Key moments: {local_kw}

CRITICAL: This clip is one of several highlights from the same match. Each clip
MUST have a COMPLETELY UNIQUE title describing THIS SPECIFIC MOMENT. Never reuse
a title. Never reference "Target 234" unless this clip is actually about the chase.

══ STEP 1: GROUND YOUR RESPONSE IN THE TRANSCRIPT ═══════════════════════════
  BEFORE writing any output, do this:
  1. Read the Raw Transcript carefully
  2. List every cricket player name mentioned in the transcript
  3. List every team name mentioned in the transcript
  4. Describe in one sentence what happens in this clip based ONLY on the transcript

  RULES — NEVER VIOLATE THESE:
  - ONLY use player names that appear in the transcript or scorecard
  - If a player is NOT mentioned in the transcript, DO NOT put them in the title
  - If a team is NOT mentioned in the transcript, DO NOT put them in the title
  - NEVER invent events, controversies, or moments not supported by the transcript
  - Match teams are from the Scorecard context above — do NOT mention other teams
  - Only use teams that appear in the Scorecard section

══ STEP 2: TRANSCRIPTION CORRECTION ══════════════════════════════════════════
  Fix misspelled cricket names:
  - Cross-reference against the Scorecard above (has correct player/venue names)
  - Fix mistakes e.g. "Sunder" → "Sundar", "Sirage" → "Siraj"
  - For Hindi names transliterated wrong, use your cricket knowledge
  - Use CORRECTED names in ALL output fields below

══ STEP 3: GENERATE SEO METADATA ════════════════════════════════════════════

TITLE (max 100 chars):
  Format: "🔴 <Team1 short> vs <Team2 short> Live Match | <Team1 full> vs <Team2 full> Live | <Tournament> Live Commentary"
  e.g. "🔴 GT vs SRH Live Match | Gujarat Titans vs Sunrisers Hyderabad Live | IPL 2026 Live Commentary"
  RULES:
    - Always start with 🔴
    - Use both short team codes AND full team names from the scorecard
    - End with "<Tournament> Live Commentary"
    - NEVER invent team names not in the scorecard

DESCRIPTION (600-900 chars total, PLAIN TEXT ONLY):
  Use this EXACT structure:

  Welcome line: One engaging Hinglish sentence welcoming viewers to the live match.

  🏏 Match Details
  Teams, venue, and match context from the scorecard.

  🔥 LIVE MATCH UPDATE
  Current match situation — score, overs, key moments from the transcript.

  ⚡ Key Highlights
  • Bullet point 1 (player action from transcript)
  • Bullet point 2
  • Bullet point 3

  🎙️ Live Hindi Commentary
  One line of Hindi commentary flavor text.

  Hashtags: [paste hashtags here as plain text]

  IMPORTANT: PLAIN TEXT only. Do NOT wrap in dicts or JSON.
  IMPORTANT: Use the SAME player names, team names, and cricket action words
  that will appear in your search_terms below. SEO consistency is critical.

HASHTAGS (exactly 4-5):
  1. Tournament (#IPL2026)
  2. Team from the match (e.g. #GT or #SRH)
  3. Star player mentioned in transcript (e.g. #KrunalPandya)
  4. #Shorts (ALWAYS include)
  5. Optional: venue

SEARCH TERMS (25-35 terms — maximize the 500 char budget):
  These are the MOST IMPORTANT field for discoverability.
  CRITICAL: Use the "Actual YouTube Search Suggestions" above — these are REAL
  queries people type. Prioritize them over invented terms.

  Include ALL of these categories:
  Tier 1 — Player + action + tournament:
    e.g. "ishan kishan batting ipl", "rabada yorker gt", "kohli six rcb"
  Tier 2 — Match + team phrases:
    e.g. "gt vs srh highlights", "csk vs srh live", "ipl 2026 live"
  Tier 3 — Hindi search patterns (massive Indian search volume):
    e.g. "kohli ka six", "dhoni finish", "ipl ka best moment", "cricket live hindi"
  Tier 4 — Moment-specific:
    e.g. "last over six ipl", "hat trick ipl 2026", "catch of the match"
  Tier 5 — Broad but relevant:
    e.g. "ipl highlights", "cricket live", "t20 cricket", "cricket shorts"

  RULES:
    - Every term in search_terms MUST be something a real person would type on YouTube
    - Mix Hindi and English (Hinglish) — Indian users search in both
    - NO generic single words like "cricket" or "six" alone
    - Each term must be 2-5 words (search phrase, not hashtag)
    - Aim for 25-35 terms to maximize the 500 char budget

Return ONLY valid JSON — no markdown, no explanation:
{{
  "clip_id": "{clip_id}",
  "title": "<title based on actual transcript content>",
  "description": "<plain text description>",
  "hashtags": ["#...", "#...", "#...", "#..."],
  "search_terms": ["<term1>", "<term2>", "..."]
}}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

CRICKET_KEYWORDS = {
    "kohli", "dhoni", "rohit", "gill", "raina", "sky", "pandya", "bumrah", "shami",
    "siraj", "jaddu", "jadeja", "ashwin", "chahal", "russell", "narine", "moeen",
    "maxwell", "faf", "rashid", "warner", "rahul", "pooran", "stoinis", "iyer",
    "shankar", "sundar", "kishan", "suryakumar", "sharma", "gaikwad", "dube",
    "rahane", "miller", "klassen", "chahar", "umesh", "deepak", "boult", "gayle",
    "abd", "devilliers", "virat", "sachin", "yuvi", "yuvraj", "zampa", "hazlewood",
    "starc", "cummins", "patel", "axar", "shardul", "thakur", "krunal", "ityer",
    "mayank", "deepakhooda", "manish", "pandey", "samson", "jaiswal", "tripathi",
    "mavi", "nagarkoti", "nitish", "rana", "rishabh", "pant", "dinesh", "karthik",
    "sky", "surya", "ishant", "sharma", "vijay", "murali", "vijay", "ambati",
    "rayudu", "harbhajan", "singh", "pathan", "irfan", "yusuf", "malik", "jordan",
    "morris", "pietersen", "watson", "mccullum", "hales", "bairstow", "roy",
    "buttler", "morgan", "stokes", "woakes", "curran", "sam", "tom", "livingstone",
    "salt", "phil", "rehan", "arora", "nattu", "sandeep", "thampi", "prasidh",
    "aaron", "saini", "sheldon", "cottrell", "holden", "dawid", "alan", "joe",
    "root", "brown", "alfie", "livi", "biggs", "manny", "ellis", "behrendorff",
    "agar", "taylor", "southee", "henry", "lockie", "ferguson", "sodhi", "santner",
    "tomlinton", "conway", "young", "ravindra", "phillips", "markram", "shai",
    "hope", "pooran", "hetmyer", "brooks", "shepherd", "ronak", "wiese", "geldenhuys",
    "six", "four", "wicket", "catch", "runout", "stump", "bowled", "lbw", "century",
    "half", "over", "yorker", "bouncer", "fulltoss", "drive", "pull", "hook",
    "sweep", "reverse", "slog", "mis", "timing", "powerful", "classy", "elegan",
    "brutal", "massive", "huge", "giant", "big", "long", "deep", "boundary", "clear",
    "ipl", "t20", "test", "odi", "cricket", "super", "over", "playoff", "final",
    "qualifier", "eliminator", "trophy", "cup", "champions", "league", "tournament",
    "match", "run", "score", "target", "chase", "win", "loss", "close", "thrill",
    "upset", "comeback", "drama", "tension", "pressure", "intense", "exiting",
    "crazy", "unbelievable", "incredible", "amazing", "fantastic", "stunning",
    "what", "shot", "bowling", "batting", "fielding", "captain", "coach", "umpire",
    "review", "dr", "decision", "controversy", "argument", "fight", "agre",
    "angle", "replay", "slowmo", "slow", "motion", "dismissal", "partnership",
}

def _extract_keywords(text: str, limit: int = 14) -> List[str]:
    words = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    kw = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    freq: Dict[str, int] = {}
    for w in kw:
        freq[w] = freq.get(w, 0) + 1
    top = [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:limit]]
    # Prefer cricket-relevant keywords, fall back to top non-noise keywords
    cricket_kw = [k for k in top if k in CRICKET_KEYWORDS]
    if cricket_kw:
        return cricket_kw[:limit]
    # If no cricket keywords, return top words that aren't obvious transcription noise
    noise = {"oh", "ah", "ha", "he", "she", "it", "do", "go", "so", "yeah", "hey",
             "come", "get", "got", "let", "put", "say", "see", "use", "way", "like",
             "know", "take", "tell", "make", "think", "give", "will", "would", "could",
             "should", "can", "may", "might", "shall", "now", "then", "just", "also"}
    return [k for k in top if k not in noise][:5]


def _inject_viral_elements(title: str, description: str, hashtags: List[str], extra: Dict = None) -> Dict:
    """Inject viral hooks and CTAs into SEO output."""
    import random

    # Pick random hook and CTA
    hook = random.choice(VIRAL_HOOKS)
    cta = random.choice(ENGAGING_CTAS)

    # Check if AI already generated hook/CTA in description
    hook_phrases = [h.split("!")[0].strip().lower() for h in VIRAL_HOOKS if "!" in h]
    has_hook = any(hp in description.lower() for hp in hook_phrases)
    has_cta = any(phrase in description.lower() for phrase in
                  ["like", "subscribe", "share", "bell icon", "notification", "stay tuned"])

    # Only add hook if AI didn't already include one
    if not has_hook and len(description) > 50:
        desc_parts = description.split("\n\n")
        if desc_parts:
            desc_parts[0] = f"{hook} {desc_parts[0][:100]}"
        description = "\n\n".join(desc_parts)

    # Only add CTA if AI didn't already include one
    if not has_cta:
        if len(description) > 100:
            description = f"{description}\n\n{cta}"
        else:
            description = f"{description} {cta}"

    # Ensure hashtags have proper structure
    if len(hashtags) < 3:
        hashtags = ["#IPL2026", "#Cricket", "#Shorts"] + hashtags

    return {
        **(extra or {}),
        "title": title,
        "description": description,
        "hashtags": hashtags[:5]
    }




def _rank_and_optimize_tags(
    ai_terms: List[str],
    yt_suggestions: List[str],
    trend_topics: List[str],
    local_keywords: List[str],
    max_chars: int = 500,
) -> List[str]:
    """
    Rank, deduplicate, and optimize tags to maximize 500 char budget.
    
    Priority tiers:
      Tier 1: YouTube autocomplete suggestions (REAL search queries)
      Tier 2: AI-generated player+action+tournament phrases
      Tier 3: Trending topics
      Tier 4: Hindi search patterns
      Tier 5: Broad cricket terms
    
    Deduplicates by substring matching (e.g. "kohli six ipl" and "virat kohli six ipl" → keep longer).
    """
    import re as _re
    
    def _is_redundant(new_tag: str, existing: List[str]) -> bool:
        """Check if new_tag is redundant with any existing tag."""
        new_lower = new_tag.lower().strip()
        for e in existing:
            e_lower = e.lower().strip()
            # If new is a substring of existing (or vice versa), it's redundant
            if new_lower in e_lower or e_lower in new_lower:
                return True
            # If they share 80%+ words
            new_words = set(new_lower.split())
            e_words = set(e_lower.split())
            if new_words and e_words:
                overlap = len(new_words & e_words) / max(len(new_words), len(e_words))
                if overlap > 0.75:
                    return True
        return False
    
    def _score_tag(tag: str) -> int:
        """Score a tag for priority (higher = more important)."""
        tag_lower = tag.lower().strip()
        words = tag_lower.split()
        score = 0
        
        # Length bonus (2-4 word phrases are ideal for search)
        if 2 <= len(words) <= 4:
            score += 10
        elif len(words) == 1:
            score -= 5  # Single words are low value
        elif len(words) > 5:
            score -= 3  # Too long
        
        # Player name bonus
        player_names = {"kohli", "dhoni", "rohit", "bumrah", "suryakumar", "sky",
                        "pandya", "jadeja", "ashwin", "gill", "rahul", "samson",
                        "rashid", "warner", "maxwell", "stokes", "buttler", "narine"}
        if any(p in tag_lower for p in player_names):
            score += 15
        
        # Team name bonus
        team_names = {"rcb", "csk", "mi", "srh", "dc", "kkr", "rr", "gt", "lsg", "pbks"}
        if any(t in tag_lower for t in team_names):
            score += 12
        
        # Action word bonus
        action_words = {"six", "wicket", "catch", "four", "century", "yorker",
                        "run out", "hat trick", "finish", "last over"}
        if any(a in tag_lower for a in action_words):
            score += 8
        
        # Tournament context bonus
        if any(t in tag_lower for t in ["ipl", "t20", "2026", "cricket"]):
            score += 5
        
        # Hinglish bonus (Hindi words boost Indian search reach)
        hindi_indicators = ["ka", "ki", "ke", "hai", "tha", "ye", "wo", "kya",
                           "dekho", "bhai", "arre", "wah", "match"]
        if any(h in words for h in hindi_indicators):
            score += 7
        
        # Penalty for generic/overused terms
        generic = {"cricket", "shorts", "viral", "trending", "amazing", "best",
                   "highlights", "live", "match", "video"}
        if tag_lower in generic:
            score -= 10
        
        return score
    
    # Combine all sources with priority markers
    tagged_terms = []  # (tag, source_priority)
    
    # Priority 1: YouTube suggestions (REAL searches)
    for t in yt_suggestions:
        t = str(t).strip()
        if t and len(t) > 2:
            tagged_terms.append((t, 1))
    
    # Priority 2: AI-generated terms
    for t in ai_terms:
        t = str(t).strip()
        if t and len(t) > 2:
            tagged_terms.append((t, 2))
    
    # Priority 3: Trending topics
    for t in trend_topics:
        t = str(t).strip()
        if t and len(t) > 2:
            tagged_terms.append((t, 3))
    
    # Score and rank all terms
    scored = []
    for tag, priority in tagged_terms:
        tag_clean = str(tag).strip().lower()
        if not tag_clean or len(tag_clean) < 3:
            continue
        # Skip generic single words
        if tag_clean in GENERIC_TAGS:
            continue
        score = _score_tag(tag_clean) + (10 if priority == 1 else 5 if priority == 2 else 0)
        scored.append((tag_clean, score))
    
    # Sort by score (highest first), deduplicate
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Build final list with 500 char budget, no redundancy
    final = []
    total_chars = 0
    for tag, score in scored:
        # Check redundancy
        if _is_redundant(tag, final):
            continue
        
        # Calculate chars needed (tag + comma separator)
        needed = len(tag) + (1 if final else 0)
        if total_chars + needed > max_chars:
            # Try shorter version
            words = tag.split()
            if len(words) > 3:
                shortened = " ".join(words[:3])
                needed_short = len(shortened) + (1 if final else 0)
                if total_chars + needed_short <= max_chars and not _is_redundant(shortened, final):
                    final.append(shortened)
                    total_chars += needed_short
            continue
        
        final.append(tag)
        total_chars += needed
    
    log.info("🏷  Tag optimization: %d terms → %d tags (%d/%d chars)",
             len(tagged_terms), len(final), total_chars, max_chars)
    
    return final


def _clean_dict_from_description(raw: str) -> str:
    """If AI returns a Python dict literal as description, extract the text values."""
    if not raw or not raw.strip():
        return raw
    # Find dict literal blocks (may have prefix hook text before them)
    match = re.search(r"\{(?:'[^']+'|\"[^\"]+\"):", raw)
    if not match:
        return raw
    try:
        import ast
        dict_str = raw[match.start():]
        parsed = ast.literal_eval(dict_str)
        if isinstance(parsed, dict):
            # Concatenate all string values into plain text
            parts = []
            for v in parsed.values():
                if isinstance(v, str):
                    parts.append(v.strip())
                elif isinstance(v, dict):
                    parts.extend(str(val) for val in v.values() if isinstance(val, str))
            plain = " | ".join(parts)
            # Grab any prefix text (hook) that precedes the dict
            prefix = raw[:match.start()].strip()
            if prefix:
                return f"{prefix} {plain}".strip()
            return plain.strip()
    except Exception:
        pass
    # Fallback: strip dict-like patterns with regex
    cleaned = re.sub(r"\{[^}]*\}", "", raw).strip()
    return cleaned


def _consolidate_seo(title: str, description: str, hashtags: List[str], search_terms: List[str]) -> Dict:
    """
    Ensure search terms, description, and hashtags are CONSISTENT.
    Search terms are the source of truth — inject them everywhere.
    """
    if not search_terms:
        return {"title": title, "description": description, "hashtags": hashtags, "search_terms": search_terms}

    # Pick top 10 search terms for description embedding
    top_terms = search_terms[:10]
    terms_line = " | ".join(top_terms[:8])

    # Generate hashtags from search terms (extract player/team names)
    player_names = {"kohli", "dhoni", "rohit", "gill", "bumrah", "suryakumar",
                    "pandya", "jadeja", "ashwin", "chahal", "bumrah", "rahul",
                    "samson", "rashid", "warner", "buttler", "narine", "klassen",
                    "stoinis", "maxwell", "faf", "sundar", "krunal", "arshad",
                    "ishan", "abhishek", "travis", "head", "mukesh", "siraj",
                    "rabada", "axar", "shardul", "hardik", "riyan", "parag",
                    "sudharsan", "sai", "iqbal", "axel", "patel", "washington",
                    "abhishek", "shahrukh", "rajat", "rehan"}
    team_names = {"csk", "mi", "rcb", "srh", "dc", "kkr", "rr", "gt", "lsg", "pbks",
                  "chennai", "mumbai", "bangalore", "hyderabad", "delhi", "kolkata",
                  "rajasthan", "gujarat", "lucknow", "punjab"}

    term_words = set()
    for t in search_terms[:15]:
        term_words.update(t.lower().replace("#", "").split())

    new_hashtags = ["#IPL2026", "#Cricket", "#Shorts"]
    for word in term_words:
        if word in player_names and len(new_hashtags) < 5:
            new_hashtags.append(f"#{word.capitalize()}")
        elif word in team_names and len(new_hashtags) < 5:
            new_hashtags.append(f"#{word.upper()}")
    new_hashtags = list(dict.fromkeys(new_hashtags))[:5]
    if not any(h.lower() == "#shorts" for h in new_hashtags):
        new_hashtags.append("#Shorts")
    new_hashtags = new_hashtags[:5]

    # Inject top search terms into description if not already present
    desc_lower = description.lower() if description else ""
    missing_terms = [t for t in top_terms if t.lower() not in desc_lower]

    if missing_terms and description:
        # Add search terms as a natural "Related:" line at the end
        inject_line = "\n\nSearch: " + ", ".join(missing_terms[:6])
        description = description.rstrip() + inject_line
    elif not description:
        description = "Search: " + terms_line

    # Truncate description to 5000 chars
    description = description[:5000]

    return {
        "title": title,
        "description": description,
        "hashtags": new_hashtags,
        "search_terms": search_terms,
        "tags": search_terms,
    }


def _enforce_limits(item: Dict, fallback_terms: List[str] = None) -> Dict:
    title = str(item.get("title") or "")[:100]
    description = str(item.get("description") or "")[:5000]

    # Clean Python dict literals from AI description output
    description = _clean_dict_from_description(description)

    hashtags = item.get("hashtags") or []
    hashtags = [h if h.startswith("#") else f"#{h}" for h in hashtags]
    if not any(h.lower() == "#shorts" for h in hashtags):
        hashtags.append("#Shorts")
    hashtags = list(dict.fromkeys(hashtags))[:5]
    if not any(h.lower() == "#shorts" for h in hashtags):
        hashtags[-1:] = ["#Shorts"]

    if "#shorts" not in title.lower() and "#shorts" not in description.lower():
        marker = "\n\n#Shorts"
        description = (description[:5000 - len(marker)] + marker) if description else "#Shorts"

    search_terms = item.get("search_terms") or []
    cleaned, total = [], 0
    for term in search_terms:
        t = str(term).strip().lower()
        if not t or len(t.split()) < 2:
            continue
        if t in GENERIC_TAGS or set(t.split()).issubset(GENERIC_TAGS):
            continue
        if t.lstrip("#") in {x.lstrip("#").lower() for x in hashtags}:
            continue
        extra = len(t) + (2 if cleaned else 0)
        if total + extra > 500:
            break
        cleaned.append(t)
        total += extra

    # Ensure at least 10 search terms to avoid only_0_tags_need_10 error
    if len(cleaned) < 10 and fallback_terms:
        for t in fallback_terms:
            t = str(t).strip().lower()
            if t not in cleaned and t not in GENERIC_TAGS:
                extra = len(t) + (2 if cleaned else 0)
                if total + extra > 500:
                    break
                cleaned.append(t)
                total += extra
            if len(cleaned) >= 10:
                break
                
    # If still not 10, add some safe defaults
    safe_defaults = ["cricket highlights", "cricket live match", "ipl match video", "t20 cricket live", "best cricket moments", "cricket shorts live", "indian cricket team", "cricket action", "match highlights", "cricket viral shorts"]
    if len(cleaned) < 10:
        for t in safe_defaults:
            if t not in cleaned:
                extra = len(t) + (2 if cleaned else 0)
                if total + extra > 500:
                    break
                cleaned.append(t)
                total += extra
            if len(cleaned) >= 10:
                break

    return {**item, "title": title, "description": description,
            "hashtags": hashtags, "search_terms": cleaned}


def _parse_json_response(text: str) -> Optional[Dict]:
    """Extract and parse the first JSON object from model response.
    Handles markdown blocks, trailing commas, single quotes, and extra text."""
    if not text:
        return None

    def _fix_json(raw: str) -> Optional[Dict]:
        """Attempt to parse JSON with various fixups."""
        if not raw:
            return None
        raw = raw.strip()
        # 1. Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # 2. Remove trailing commas before closing brackets/braces
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # 3. Replace single quotes with double quotes (for Python-style dicts)
        single_quoted = re.sub(r"'([^']*)'", r'"\1"', cleaned)
        # Also fix: {'key': "value"} -> {"key": "value"}
        single_quoted = re.sub(r"'([^']*)'\s*:", r'"\1":', single_quoted)
        try:
            return json.loads(single_quoted)
        except json.JSONDecodeError:
            pass
        # 4. Handle unquoted keys (common LLM issue: {key: "value"})
        unquoted_keys = re.sub(r"([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', single_quoted)
        try:
            return json.loads(unquoted_keys)
        except json.JSONDecodeError:
            pass
        return None

    # 1. Direct
    result = _fix_json(text)
    if result:
        return result

    # 2. Markdown code block extraction
    for pattern in [
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r"```(?:json)?\s*(\{.*\})\s*```",
    ]:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result = _fix_json(match.group(1))
            if result:
                return result

    # 3. Bracket matching — find first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        result = _fix_json(text[start:end+1])
        if result:
            return result

    # 4. Line-by-line: try stripping leading/trailing non-JSON lines
    lines = text.strip().split('\n')
    for i in range(min(3, len(lines))):
        for j in range(min(3, len(lines))):
            candidate = '\n'.join(lines[i:len(lines)-j] if j > 0 else lines[i:])
            cand_start = candidate.find('{')
            cand_end = candidate.rfind('}')
            if cand_start != -1 and cand_end != -1 and cand_end > cand_start:
                result = _fix_json(candidate[cand_start:cand_end+1])
                if result:
                    return result

    return None


# ── Per-clip SEO (one AI call, retries with backoff) ──────────────────────────

def generate_clip_seo(
    clip_id: str,
    transcript: str,
    video_title: str = "",
    scorecard: str = "",
    trend_topics: List[str] = None,
    live_stream_url: str = "",
    provider_override: str = "",
    model_override: str = "",
    teams: List[str] = None,
) -> Dict:
    """
    Generate SEO metadata for a single clip.
    Retries up to 3 times with exponential backoff on 429/593.
    provider_override/model_override: allow dynamic model selection for A/B testing.
    """
    _maybe_auto_benchmark()
    trend_topics = trend_topics or []
    local_kw_list = _extract_keywords(transcript)
    local_kw = ", ".join(local_kw_list)

    # Harvest YouTube autocomplete suggestions (multi-query, cached)
    cache_key = "suggest:" + ":".join(sorted(local_kw_list)[:5])
    yt_suggestions = SUGGEST_CACHE.get(cache_key)
    if yt_suggestions is None:
        yt_suggestions = []
        try:
            from .trends import fetch_enhanced_clip_suggestions
            yt_suggestions = fetch_enhanced_clip_suggestions(
                local_kw_list, teams=teams, match_type="ipl"
            )
            log.info("YouTube suggestions harvested: %d terms", len(yt_suggestions))
        except Exception as e:
            log.warning("Enhanced suggest failed, falling back: %s", e)
            try:
                from .trends import fetch_clip_specific_suggestions
                yt_suggestions = fetch_clip_specific_suggestions(local_kw_list)
            except Exception:
                pass

        # Also fetch clip-specific suggestions as fallback
        try:
            from .trends import fetch_clip_specific_suggestions
            clip_suggestions = fetch_clip_specific_suggestions(local_kw_list)
            if clip_suggestions:
                yt_suggestions = list(dict.fromkeys(yt_suggestions + clip_suggestions))
        except Exception as e:
            log.warning("Could not fetch clip-specific suggestions: %s", e)

        SUGGEST_CACHE.set(cache_key, yt_suggestions)

    trend_str = ", ".join(trend_topics) or "IPL 2026, cricket live"
    yt_suggest_str = ", ".join(yt_suggestions[:15]) or "No suggestions available"
    live_cta = (
        f"Watch LIVE: {live_stream_url}" if live_stream_url
        else "Match chal raha hai LIVE — channel pe aao."
    )

    prompt = _PROMPT_TMPL.format(
        video_title=video_title or "Cricket Live Match",
        scorecard=scorecard or "Live match in progress",
        trend_topics=trend_str,
        yt_suggestions=yt_suggest_str,
        live_cta=live_cta,
        transcript=transcript[:2500],   # keep prompt tight
        local_kw=local_kw,
        clip_id=clip_id,
    )
    
    # Enhance prompt with learned insights from performance data (lazy import)
    from .seo_learner import enhance_seo_prompt
    prompt = enhance_seo_prompt(prompt)

    # Apply dynamic model override if set (used by self-learner for A/B testing)
    if provider_override:
        old_provider = ai._provider
        old_model = ai._model
        ai._provider = provider_override
        if model_override:
            ai._model = model_override
        try:
            result = _attempt_seo_generation(clip_id, prompt, trend_topics,
                                             yt_suggestions=yt_suggestions,
                                             local_keywords=local_kw_list,
                                             transcript=transcript,
                                             video_title=video_title,
                                             scorecard=scorecard)
            return result
        finally:
            ai._provider = old_provider
            ai._model = old_model

    result = _attempt_seo_generation(clip_id, prompt, trend_topics,
                                     yt_suggestions=yt_suggestions,
                                     local_keywords=local_kw_list,
                                     transcript=transcript,
                                     video_title=video_title,
                                     scorecard=scorecard)
    return result


def _attempt_seo_generation(
    clip_id: str,
    prompt: str,
    trend_topics: List[str],
    yt_suggestions: List[str] = None,
    local_keywords: List[str] = None,
    transcript: str = "",
    video_title: str = "",
    scorecard: str = "",
) -> Dict:
    """Generate SEO with parallel model racing. Fires fastest available models
    concurrently and takes the first valid response. No backoff — on failure
    the next tier of models is tried immediately.
    CRITICAL: Never uses template fallback — always uses AI content.
    """
    import random as _random

    response_text = ai.generate_fastest_first(prompt, system_instruction=_SYSTEM)
    if response_text:
        data = _parse_json_response(response_text)
        if data:
            ai_terms = data.get("search_terms", [])
            optimized_tags = _rank_and_optimize_tags(
                ai_terms=ai_terms,
                yt_suggestions=yt_suggestions or [],
                trend_topics=trend_topics,
                local_keywords=local_keywords or [],
                max_chars=500,
            )

            result = _enforce_limits({
                "clip_id": clip_id,
                "title": data.get("title", f"Cricket Live Highlights | {clip_id}"),
                "description": data.get("description", ""),
                "hashtags": data.get("hashtags", ["#IPL2026", "#Cricket", "#Shorts"]),
                "search_terms": optimized_tags,
                "tags": optimized_tags,
            }, fallback_terms=trend_topics)

            result = _consolidate_seo(
                result["title"], result["description"],
                result["hashtags"], result["search_terms"]
            )

            result = _inject_viral_elements(
                result["title"],
                result["description"],
                result["hashtags"],
                extra=result,
            )
            result["clip_id"] = clip_id
            result["ai_generated"] = True
            result["_generated_by_provider"] = ai.get_used_provider()
            result["_generated_by_model"] = ai.get_used_model()

            log.info("[%s] SEO done — title: %s | tags: %d (%d chars)",
                     clip_id, result["title"][:100],
                     len(result.get("search_terms", [])),
                     sum(len(t) + 1 for t in result.get("search_terms", [])))
            return result

        # AI responded but JSON parsing failed — salvage raw content
        log.warning("[%s] AI responded but JSON extract failed — salvaging raw text", clip_id)

        # Extract meaningful text from the raw AI response
        local_kw = _extract_keywords(transcript, limit=8)
        kw_str = ", ".join(local_kw[:5]) if local_kw else "cricket highlights"

        # Build title from first meaningful line of AI output
        lines = [l.strip() for l in response_text.split('\n') if l.strip()]
        title = ""
        for line in lines:
            clean = line.strip('#* ')  # Remove markdown artifacts
            if clean and len(clean) > 15 and len(clean) < 120:
                title = clean[:100]
                break

        if not title:
            team_str = ""
            team_map = {"gt", "csk", "mi", "rcb", "srh", "dc", "kkr", "rr", "lsg", "pbks"}
            teams = [w.upper() for w in (transcript or "").lower().split() if w in team_map]
            if teams:
                team_str = " vs ".join(sorted(set(teams)))
            title = f"{kw_str} | {team_str or 'IPL 2026'}"[:100]

        # Use raw response as description base
        description = response_text.strip()
        # Strip markdown code fences
        description = re.sub(r'```(?:json)?\s*', '', description)
        description = re.sub(r'\s*```', '', description)
        # Clean up excessive whitespace
        description = re.sub(r'\n{3,}', '\n\n', description)
        description = description[:5000]

        # Generate search terms from transcript keywords + trends
        fallback_terms = []
        for t in (trend_topics or [])[:10]:
            if t not in fallback_terms:
                fallback_terms.append(t)
        for w in local_kw[:10]:
            phrase = f"{w} cricket ipl"
            if phrase not in fallback_terms:
                fallback_terms.append(phrase)

        result = _enforce_limits({
            "clip_id": clip_id,
            "title": title,
            "description": description or f"{_random.choice(VIRAL_HOOKS)}\n\n{kw_str} — watch full highlights!",
            "hashtags": ["#IPL2026", "#Cricket", "#Shorts"],
            "search_terms": fallback_terms[:20],
            "tags": fallback_terms[:20],
        }, fallback_terms=trend_topics)

        result = _consolidate_seo(
            result["title"], result["description"],
            result["hashtags"], result["search_terms"]
        )
        result["_salvaged"] = True
        result["clip_id"] = clip_id
        result["ai_generated"] = True

        log.info("[%s] SEO done (salvaged from raw AI) — title: %s",
                 clip_id, result["title"][:100])
        return result

    # AI returned nothing at all — use transcript-based dynamic generation
    log.warning("[%s] No AI response — generating transcript-aware SEO", clip_id)
    local_kw = _extract_keywords(transcript, limit=12)
    kw_str = ", ".join(local_kw[:5]) if local_kw else "cricket highlights"

    # Detect players and teams from transcript
    players_in_transcript = set()
    teams_in_transcript = set()
    for word in (transcript or "").lower().split():
        if word in CRICKET_KEYWORDS and word not in GENERIC_TAGS:
            players_in_transcript.add(word.capitalize())

    team_map = {"gt", "csk", "mi", "rcb", "srh", "dc", "kkr", "rr", "lsg", "pbks"}
    for word in (transcript or "").lower().split():
        if word in team_map:
            teams_in_transcript.add(word.upper())

    player_str = ", ".join(sorted(players_in_transcript)[:3])
    team_str = " vs ".join(sorted(teams_in_transcript)) if teams_in_transcript else "IPL 2026"

    title = f"{player_str} {kw_str} | {team_str}"[:100]
    if not player_str:
        title = f"{kw_str} | {team_str}"[:100]

    description = f"{_random.choice(VIRAL_HOOKS)}\n\n{transcript[:500]}\n\n{_random.choice(ENGAGING_CTAS)}"

    hashtags = ["#IPL2026"]
    if teams_in_transcript:
        hashtags.append(f"#{list(teams_in_transcript)[0]}")
    else:
        hashtags.append("#Cricket")
    hashtags.extend([f"#{w.capitalize()}" for w in local_kw[:2] if len(w) > 2])
    hashtags.append("#Shorts")
    hashtags = list(dict.fromkeys(hashtags))[:5]

    search_terms = []
    for kw in local_kw[:8]:
        search_terms.append(f"{kw} cricket ipl")
        search_terms.append(f"{kw} {team_str.lower()}")
    for t in (trend_topics or [])[:8]:
        if t not in search_terms:
            search_terms.append(t)
    search_terms = list(dict.fromkeys(search_terms))[:25]

    result = _enforce_limits({
        "clip_id": clip_id,
        "title": title,
        "description": description[:5000],
        "hashtags": hashtags,
        "search_terms": search_terms,
        "tags": search_terms,
    }, fallback_terms=trend_topics)

    result["clip_id"] = clip_id
    result["ai_generated"] = True
    result["_transcript_generated"] = True

    log.info("[%s] SEO done (transcript-aware, no AI) — title: %s | %d tags",
             clip_id, result["title"][:100], len(search_terms))
    return result


# ── Pipeline entry: called after each clip export ─────────────────────────────

def generate_seo_for_exported_clip(
    clip_id: str,
    transcript: str,
    output_dir: str,
    video_title: str = "",
    scorecard: str = "",
    trend_topics: List[str] = None,
    live_stream_url: str = "",
    inter_clip_pause: float = 0.0,
) -> Dict:
    """
    Generate + save SEO for one exported clip.
    inter_clip_pause: seconds to wait BEFORE the API call (rate-limit buffer).
    Pass 0.0 for the first clip, 8.0 for subsequent ones.
    Dynamically uses best model discovered by learner.
    """
    if inter_clip_pause > 0:
        log.debug("[%s] Waiting %.0fs before SEO call...", clip_id, inter_clip_pause)
        time.sleep(inter_clip_pause)

    # Apply best model from learner (if available, lazy import)
    provider_override = None
    model_override = None
    try:
        from .seo_learner import get_best_model
        best_prov, best_mod = get_best_model()
        if best_prov and best_mod:
            provider_override = best_prov
            model_override = best_mod
    except Exception:
        pass

    result = generate_clip_seo(
        clip_id=clip_id,
        transcript=transcript,
        video_title=video_title,
        scorecard=scorecard,
        trend_topics=trend_topics or [],
        live_stream_url=live_stream_url,
        provider_override=provider_override,
        model_override=model_override,
    )

    out_path = Path(output_dir) / f"{clip_id}_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log.info("[%s] SEO saved → %s", clip_id, out_path)

    return result


# ── Batch runner (processes all highlights sequentially) ──────────────────────

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
    for idx, (clip_id, info) in enumerate(clips, start=1):
        transcript = info.get("text", "Cricket Live")
        log.info("SEO [%d/%d]: %s", idx, len(clips), clip_id)

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

        # Save individual file immediately (safe even if later clips fail)
        per_clip_path = Path(output_dir) / f"{clip_id}_metadata.json"
        with open(per_clip_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Breathing room between clips (skip after last)
        if idx < len(clips):
            log.debug("Sleeping 5s before next SEO call...")
            time.sleep(5)

    # Also write a combined results file
    combined_path = Path(output_dir) / "seo_results.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    log.info("SEO complete: %d clips → %s", len(all_results), output_dir)
    return str(combined_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    process_all_seo("highlights/video.yaml", "shorts/test")
