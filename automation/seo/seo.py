"""seo.py — Per-clip SEO generation for Indian cricket Shorts.

Uses parallel fastest-first model racing: fires the fastest available models
concurrently and takes the first valid JSON response. On failure it ESCALATES
(stricter JSON-only prompt via the full provider failover chain), and if that
still fails it raises SEOGenerationError so the clip is SKIPPED + QUEUED for
retry — it never emits generic/template SEO. Every title is clip-specific.
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


class SEOGenerationError(RuntimeError):
    """Raised when LLM SEO generation fails after escalation.

    The product contract forbids generic/template fallback SEO. When all
    providers/models fail (even after a stricter-prompt retry), we raise this
    instead of degrading — callers SKIP + QUEUE the clip for retry rather than
    upload generic metadata.
    """


def _current_season_year() -> int:
    """Derive the current cricket/IPL season year from the clock.

    Avoids hardcoding a season (e.g. '2026'). IPL runs Mar–May; before March we
    still reference the upcoming season's calendar year, which matches how the
    audience searches.
    """
    from datetime import datetime
    return datetime.now().year


def _season_tag() -> str:
    return "IPL %d" % _current_season_year()


def _season_hashtag() -> str:
    return "#IPL%d" % _current_season_year()


def _default_hashtags() -> List[str]:
    """Season-aware default hashtag seed (no hardcoded year)."""
    return [_season_hashtag(), "#Cricket", "#Shorts"]


def _maybe_auto_benchmark():
    """Lazy auto-benchmark: runs once on first SEO call if enabled in config.
    
    NOTE: Does NOT mutate ai._provider/_model (the racer uses FASTEST_TIERS with
    shuffle + prefer_provider, not the singleton's fields). The benchmark result
    is persisted in seo_performance.json and read via get_best_model() -> 
    prefer_provider on each clip call.
    """
    if not getattr(_maybe_auto_benchmark, "_done", False):
        _maybe_auto_benchmark._done = True
        if cfg.get("ai", {}).get("auto_benchmark", False):
            try:
                from .seo_learner import run_auto_benchmark, get_best_model
                log.info("Auto-benchmark enabled — discovering best model...")
                run_auto_benchmark()
                best_provider, best_model = get_best_model()
                if best_provider and best_model:
                    log.info("Auto-benchmark result: best=%s/%s (used as prefer_provider)",
                             best_provider, best_model)
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
    "CRITICAL: Generate title, description, and tags matching the EXACT format requested. "
    "Only use player names, teams, and events that appear in the transcript or scorecard. "
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

══ REQUIRED JSON FORMAT ═════════════════════════════════════════════════════
Return a JSON object with these EXACT keys:
{{
  "title": "<LIVE Team1 vs Team2 Score | Team1 vs Team2 Tournament Match No | Aaj Ka Match Hindi Commentary>",
  "description": "<entire finished description string matching the format rules below>",
  "search_terms": [
    "<term1>",
    "<term2>",
    ...
  ],
  "hashtags": [
    "#...",
    "#..."
  ]
}}

══ TITLE RULES ══════════════════════════════════════════════════════════════
- Always start with "LIVE "
- Format: "LIVE <Team1 short> vs <Team2 short> <Score> | <Team1 city/full> vs <Team2 city/full> <Tournament> Match <MatchNo> | Aaj Ka Match Hindi Commentary"
- Example: "LIVE GT vs CSK 189/1 | Gujarat vs Chennai IPL 2026 Match 66 | Aaj Ka Match Hindi Commentary"
- Never use 🔴 in the title.
- If score is not available, use current match situation/runs/wickets from transcript.

══ DESCRIPTION FORMAT RULES ═════════════════════════════════════════════════
Inside the "description" JSON string, you must generate the entire text matching this EXACT layout (use actual newlines `\\n` inside the JSON string value):

LIVE: <Team1 full> vs <Team2 full> – <Tournament> Match <MatchNo>, <Venue>
<Toss winner> won toss and chose to <bowl/bat> first
Current: <Team short> <Score> (<Overs>) – <Batsman1> <runs>(<balls>) <wicket details if any>, <Batsman2> <runs>*, ...

🇮🇳 India: JioHotstar, Star Sports
🇵🇰 Pakistan: Yupp TV

<Engaging Hinglish/Hindi flavor sentence about the match situation, commentary, pitch/dew factor, etc.>

CHAPTERS
00:00 Live Start & Toss
02:15 <Moment 1 description>
12:20 <Moment 2 description>
17:00 <Moment 3 description>
20:00 <Moment 4 description>

Search: <comma-separated list of 7 key search terms from your search_terms field>

#<Space-separated list of hashtags from your hashtags field>

Disclaimer: Live score updates and commentary only. For official broadcast watch JioHotstar (India) or Yupp TV (Pakistan).

══ EXAMPLE DESCRIPTION VALUE ════════════════════════════════════════════════
"LIVE: Gujarat Titans vs Chennai Super Kings – IPL 2026 Match 66, Narendra Modi Stadium Ahmedabad\\nCSK won toss and chose to bowl first\\nCurrent: GT 189/1 (17.2) – Shubman Gill 64(37) c Dube b Johnson, Sai Sudharsan 68*, Jos Buttler 34*\\n\\n🇮🇳 India: JioHotstar, Star Sports\\n🇵🇰 Pakistan: Yupp TV\\n\\nAaj ka match live score, Hindi commentary, ball by ball updates. Dew factor active in Ahmedabad second innings.\\n\\nCHAPTERS\\n00:00 Live Start & Toss\\n02:15 GT Powerplay 62/0\\n12:20 WICKET – Gill 64\\n17:00 Buttler Acceleration\\n20:00 CSK Chase Starts\\n\\nSearch: gt vs csk live, csk vs gt live score today, ipl 2026 live hindi, aaj ka match live, live cricket match today online, ipl live kaise dekhe, ipl live pakistan\\n\\n#GTvsCSK #IPL2026Live #LiveCricket #AajKaMatch #CSKvsGT #IPLPakistan #HindiCommentary\\n\\nDisclaimer: Live score updates and commentary only. For official broadcast watch JioHotstar (India) or Yupp TV (Pakistan)."

Return ONLY valid JSON.
"""

_SYSTEM_SHORTS = (
    "You are an elite YouTube Shorts SEO expert specializing in viral cricket shorts for Indian and Pakistani audiences. "
    "Your goal: Maximize click-through rate (CTR), engagement, and viral potential. "
    "Use highly engaging, emotional, and catchy titles (under 60 characters) with emojis and relevant hashtags. "
    "Only use players and events that appear in the transcript or scorecard. Do NOT invent events. "
    "Return ONLY valid JSON — no markdown, no explanation."
)

_PROMPT_TMPL_SHORTS = """
CONTEXT:
  Match: {video_title}
  Scorecard: {scorecard}
  Live Trending / Search Spikes: {trend_topics}
  Actual YouTube Search Suggestions: {yt_suggestions}
  CTA: {live_cta}

CLIP CONTENT:
  Raw Transcript: {transcript}
  Key moments: {local_kw}

══ REQUIRED JSON FORMAT ═════════════════════════════════════════════════════
Return a JSON object with these EXACT keys:
{{
  "title": "<Catchy clickbait title under 60 characters with emojis and #Shorts>",
  "description": "<Short viral description under 400 characters, targeting Indian/Pakistani viewers, with engaging hook, CTA to subscribe, and key search terms/hashtags>",
  "search_terms": [
    "<term1>",
    "<term2>",
    ...
  ],
  "hashtags": [
    "#Shorts",
    "#...",
    "#..."
  ]
}}

══ SHORTS TITLE RULES ═══════════════════════════════════════════════════════
- Keep it under 60 characters. Must be extremely punchy, capitalizing key words.
- Always include 1-2 relevant emojis (e.g. 😱, 🔥, 💥, 🤯) and #Shorts.
- Target Indian/Pakistani emotions (e.g. "KOHLI DESTROYS PAKISTAN! 😱🔥 #Shorts" or "BABAR AZAM CLASS CLASS CLASS! 🤯🔥 #Shorts").

══ SHORTS DESCRIPTION RULES ═════════════════════════════════════════════════
- Keep it short and sweet (under 400 characters).
- Start with a viral hook in Hindi/Hinglish (e.g., "Kohli ne phir se kar dikhaya!").
- Include a CTA to like/subscribe (e.g., "Subscribe and like for more IPL updates!").
- Embed hashtags and a few top search terms naturally at the end.
- Target audience context: subcontinent viewers (India/Pakistan).

Return ONLY valid JSON.
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

FULL_TEAM_NAMES = {
    "CSK": "Chennai Super Kings",
    "MI": "Mumbai Indians",
    "RCB": "Royal Challengers Bengaluru",
    "KKR": "Kolkata Knight Riders",
    "SRH": "Sunrisers Hyderabad",
    "DC": "Delhi Capitals",
    "PBKS": "Punjab Kings",
    "RR": "Rajasthan Royals",
    "LSG": "Lucknow Super Giants",
    "GT": "Gujarat Titans"
}

CITY_TEAM_NAMES = {
    "CSK": "Chennai",
    "MI": "Mumbai",
    "RCB": "Bengaluru",
    "KKR": "Kolkata",
    "SRH": "Hyderabad",
    "DC": "Delhi",
    "PBKS": "Punjab",
    "RR": "Rajasthan",
    "LSG": "Lucknow",
    "GT": "Gujarat"
}

def assemble_description(
    data: Dict,
    scorecard: str = "",
    video_title: str = "",
    transcript: str = "",
    fallback_search_terms: List[str] = None,
    fallback_hashtags: List[str] = None
) -> str:
    """
    Assemble description in the exact requested format:
    LIVE: <Team1 full> vs <Team2 full> – <Tournament> Match <MatchNo>, <Venue>
    <Toss details>
    Current: <Current Score (Overs)> – <Batsman1> <runs>(<balls>) <wicket details>, ...

    🇮🇳 India: JioHotstar, Star Sports
    🇵🇰 Pakistan: Yupp TV

    <Flavor text commentary / match updates / pitch / dew factor>

    CHAPTERS
    00:00 Live Start & Toss
    ...

    Search: <comma-separated list of search queries>

    #Hashtags

    Disclaimer: Live score updates and commentary only. For official broadcast watch JioHotstar (India) or Yupp TV (Pakistan).
    """
    # If the description is already fully formatted and has the new style, keep it
    desc_str = data.get("description", "")
    if desc_str and "Disclaimer:" in desc_str and "CHAPTERS" in desc_str and "Search:" in desc_str:
        return desc_str

    # Extract team names
    from .trends import extract_match_teams, TEAM_MAPPINGS
    teams_list, match_type = [], "ipl"
    if video_title:
        teams_list, match_type = extract_match_teams(video_title)
    
    # Try scorecard if video_title didn't yield enough
    if len(teams_list) < 2 and scorecard:
        for abbr, name in TEAM_MAPPINGS.items():
            if abbr in scorecard.lower() and name not in teams_list:
                teams_list.append(name)

    t1_full, t2_full = "Team 1", "Team 2"
    t1_short, t2_short = "T1", "T2"
    if len(teams_list) >= 2:
        t1_short, t2_short = teams_list[0], teams_list[1]
        t1_full = FULL_TEAM_NAMES.get(t1_short, t1_short)
        t2_full = FULL_TEAM_NAMES.get(t2_short, t2_short)
    elif len(teams_list) == 1:
        t1_short = teams_list[0]
        t1_full = FULL_TEAM_NAMES.get(t1_short, t1_short)

    # 1. Match Details
    match_info = data.get("match_info")
    if not match_info:
        # Match No
        match_no = ""
        if video_title:
            m = re.search(r"match\s*(\d+)", video_title, re.IGNORECASE)
            if m:
                match_no = f" Match {m.group(1)}"
            else:
                match_no = ""
        else:
            match_no = ""
            
        # Venue
        venue = ""
        if scorecard:
            m = re.search(r"Stadium|Ahmedabad|Chennai|Mumbai|Bangalore|Kolkata|Delhi|Mohali|Jaipur|Hyderabad|Pune", scorecard, re.IGNORECASE)
            if m:
                # simple heuristic to find stadium/city
                lines = scorecard.split("\n")
                for line in lines[:3]:
                    if "stadium" in line.lower() or "ground" in line.lower() or "park" in line.lower():
                        venue = line.strip()
                        break
        
        from datetime import datetime
        current_year = str(datetime.now().year)
        match_info = f"LIVE: {t1_full} vs {t2_full} – {match_type.upper() or 'IPL'} {current_year}{match_no}"
        if venue:
            match_info += f", {venue}"

    # 2. Toss Info
    toss_info = data.get("toss_info")
    if not toss_info:
        if scorecard:
            # Look for toss statement in scorecard
            for line in scorecard.split("\n"):
                if "toss" in line.lower() or "opted" in line.lower() or "chose" in line.lower():
                    toss_info = line.strip()
                    break
        if not toss_info:
            toss_info = "Toss details not available"

    # 3. Current Score
    current_score = data.get("current_score")
    if not current_score:
        if scorecard:
            # Construct a dynamic current score string from Cricbuzz parsed scorecard
            score_line = ""
            batsmen = []
            for line in scorecard.split("\n"):
                if "/" in line and any(char.isdigit() for char in line):
                    score_line = line.strip()
                elif "Top:" in line:
                    batsmen_text = line.replace("Top:", "").strip()
                    batsmen = [b.strip() for b in batsmen_text.split(",") if b.strip()]
            if score_line:
                current_score = f"Current: {score_line}"
                if batsmen:
                    current_score += " – " + ", ".join(batsmen[:3])
        if not current_score:
            current_score = f"Current: {t1_short} score updates in progress"

    # Assemble sections
    parts = [match_info, toss_info, current_score]

    # Broadcasters
    parts.append("\n🇮🇳 India: JioHotstar, Star Sports\n🇵🇰 Pakistan: Yupp TV")

    # Flavor text
    flavor_text = data.get("flavor_text")
    if not flavor_text:
        # Generate some Hinglish flavor text from transcript/keywords
        flavor_text = "Aaj ka match live score, Hindi commentary, ball by ball updates."
        if "dew" in (transcript or "").lower() or "dew" in scorecard.lower():
            flavor_text += " Dew factor active in second innings."
    parts.append("\n" + flavor_text)

    # Chapters
    chapters = data.get("chapters")
    ch_lines = ["\nCHAPTERS"]
    if chapters:
        if isinstance(chapters, list):
            for ch in chapters:
                if isinstance(ch, dict):
                    ch_lines.append(f"{ch.get('time', '00:00')} {ch.get('event', '')}")
                else:
                    ch_lines.append(str(ch))
        else:
            ch_lines.append(str(chapters))
    else:
        # Default dynamic chapters based on video duration / clip
        ch_lines.extend([
            "00:00 Live Start & Toss",
            f"02:15 {t1_short} Powerplay batting",
            "12:20 Key Wicket moment",
            "17:00 Innings Acceleration",
            f"20:00 {t2_short} Chase Starts"
        ])
    parts.append("\n".join(ch_lines))

    # Search Terms
    search_list = data.get("search_terms") or fallback_search_terms or []
    if search_list:
        # Select first 7 search terms for "Search:" section in description
        search_line = ", ".join(search_list[:7])
        parts.append(f"\nSearch: {search_line}")

    # Hashtags
    hashtags_list = data.get("hashtags") or fallback_hashtags or []
    if hashtags_list:
        # Format as space separated hashtags
        ht_line = " ".join([h if h.startswith("#") else f"#{h}" for h in hashtags_list[:8]])
        parts.append(f"\n{ht_line}")

    # Disclaimer
    parts.append("\nDisclaimer: Live score updates and commentary only. For official broadcast watch JioHotstar (India) or Yupp TV (Pakistan).")

    return "\n".join(parts)


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
    """Optionally inject viral hooks and CTAs into SEO output.

    DISABLED by default (config ``seo.inject_viral_elements``). The random
    hardcoded hooks/CTAs were generic, non-clip-specific templated copy — the
    product contract prefers per-clip LLM-generated hooks. When disabled this
    function is a pass-through that only normalizes hashtags (no generic year).
    """
    if not cfg.get("seo", {}).get("inject_viral_elements", False):
        norm = hashtags[:8] if hashtags else _default_hashtags()
        return {
            **(extra or {}),
            "title": title,
            "description": description,
            "hashtags": norm,
        }

    if description and "Disclaimer:" in description and "CHAPTERS" in description:
        return {
            **(extra or {}),
            "title": title,
            "description": description,
            "hashtags": hashtags[:8] if hashtags else _default_hashtags()
        }

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
        hashtags = _default_hashtags() + hashtags

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
    if description and "Disclaimer:" in description and "CHAPTERS" in description:
        return {
            "title": title,
            "description": description,
            "hashtags": hashtags,
            "search_terms": search_terms,
            "tags": search_terms,
        }

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

    new_hashtags = _default_hashtags()
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

    # Backfill from real, clip-derived fallback terms (trends/transcript), if any.
    # NOTE: we deliberately do NOT pad with generic phrases — the product
    # contract forbids generic tag generation. Fewer specific tags beats
    # generic ones, and total SEO failure is escalated/queued upstream.
    if len(cleaned) < 10 and fallback_terms:
        for t in fallback_terms:
            t = str(t).strip().lower()
            if t not in cleaned and t not in GENERIC_TAGS and len(t.split()) >= 2:
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
    is_shorts: bool = True,
) -> Dict:
    """
    Generate SEO metadata for a single clip via LLM reasoning.

    Escalates on failure (fastest-first race -> stricter-prompt failover retry)
    and raises SEOGenerationError if no valid AI output is produced — never
    emits generic/template SEO. provider_override/model_override allow dynamic
    model selection for A/B testing.
    """
    _maybe_auto_benchmark()
    trend_topics = trend_topics or []

    # Pre-SEO factual correction — applied to ALL context fed to the LLM
    # (transcript + title + scorecard), so the model reasons over corrected
    # player/team/venue/tournament names BEFORE generation.
    from .cricket_context import correct_cricket_spelling, find_canonical_entities
    corrected_transcript = correct_cricket_spelling(transcript)
    video_title = correct_cricket_spelling(video_title) if video_title else video_title
    scorecard = correct_cricket_spelling(scorecard) if scorecard else scorecard

    local_kw_list = _extract_keywords(corrected_transcript)

    # Ground keywords in verified cricket entities (wires the canonical sets):
    # canonical players/teams found in the corrected context are prepended as
    # high-value, factually-correct keywords for tag generation + the prompt.
    entities = find_canonical_entities(corrected_transcript + " " + (video_title or ""))
    canonical_kw = [e.lower() for e in (entities["players"] + entities["teams"])]
    for kw in canonical_kw:
        if kw not in local_kw_list:
            local_kw_list.insert(0, kw)
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

    trend_str = ", ".join(trend_topics[:5]) or ("%s, cricket live" % _season_tag())
    yt_suggest_str = ", ".join(yt_suggestions[:5]) or "No suggestions available"
    live_cta = (
        f"Watch LIVE: {live_stream_url}" if live_stream_url
        else "Match chal raha hai LIVE — channel pe aao."
    )

    # Truncate aggressively to stay within Groq free-tier ~32KB limit
    scorecard_trimmed = (scorecard or "Live match in progress")[:800]
    transcript_trimmed = corrected_transcript[:1200]
    
    if is_shorts:
        system_instruction = _SYSTEM_SHORTS
        prompt = _PROMPT_TMPL_SHORTS.format(
            video_title=video_title or "Cricket Live Match",
            scorecard=scorecard_trimmed,
            trend_topics=trend_str,
            yt_suggestions=yt_suggest_str,
            live_cta=live_cta,
            transcript=transcript_trimmed,
            local_kw=local_kw,
        )
    else:
        system_instruction = _SYSTEM
        prompt = _PROMPT_TMPL.format(
            video_title=video_title or "Cricket Live Match",
            scorecard=scorecard_trimmed,
            trend_topics=trend_str,
            yt_suggestions=yt_suggest_str,
            live_cta=live_cta,
            transcript=transcript_trimmed,
            local_kw=local_kw,
            clip_id=clip_id,
        )
    
    # Enhance prompt with learned insights from performance data (lazy import)
    from .seo_learner import enhance_seo_prompt
    prompt = enhance_seo_prompt(prompt)

    # Apply dynamic model override if set (used by self-learner for A/B testing)
    # Instead of mutating the shared ai._provider/_model (which the racer ignores
    # anyway), pass it as prefer_provider so the racer BOOSTS that provider's
    # models to front-of-tier while still racing others for resilience.
    result = _attempt_seo_generation(clip_id, prompt, trend_topics,
                                     yt_suggestions=yt_suggestions,
                                     local_keywords=local_kw_list,
                                     transcript=corrected_transcript,
                                     video_title=video_title,
                                     scorecard=scorecard,
                                     system_instruction=system_instruction,
                                     is_shorts=is_shorts,
                                     prefer_provider=provider_override or None)
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
    system_instruction: str = _SYSTEM,
    is_shorts: bool = True,
    prefer_provider: Optional[str] = None,
) -> Dict:
    """Generate SEO via LLM, ESCALATING on failure (never degrading).

    Strategy (escalation, not degradation):
      1. Race the fastest available models (``generate_fastest_first``), with
         *prefer_provider* boosted to front-of-tier and models shuffled within
         tiers for diversity across clips.
      2. If the response is missing/unparseable, ESCALATE: retry once via the
         full failover chain (``generate_text``) with a stricter JSON-only
         prompt + JSON-repair parsing.
      3. If still no valid AI JSON, raise :class:`SEOGenerationError` so the
         caller SKIPS + QUEUES the clip — we never emit generic/template SEO.

    For Shorts the LLM's short (<400 char) description is preserved; only
    long-form clips get the structured LIVE/CHAPTERS/Disclaimer layout.
    """
    # ── 1. Primary: fastest-first racing (shuffled + provider-boosted) ────────
    try:
        response_text = ai.generate_fastest_first(
            prompt, system_instruction=system_instruction,
            prefer_provider=prefer_provider)
    except Exception as e:
        log.warning("[%s] generate_fastest_first failed: %s", clip_id, e)
        response_text = ""

    data = _parse_json_response(response_text) if response_text else None

    # ── 2. Escalation: stricter JSON-only prompt via full failover chain ──────
    if not data:
        log.warning("[%s] SEO attempt-1 unusable — escalating (stricter prompt + failover chain)", clip_id)
        strict_system = (system_instruction or _SYSTEM) + (
            " You MUST return ONLY a single minified JSON object. No markdown, no commentary."
        )
        strict_prompt = prompt + (
            "\n\nSTRICT OUTPUT: Return ONLY one JSON object with keys "
            '"title", "description", "search_terms", "hashtags". '
            "No code fences, no explanation, no surrounding text."
        )
        try:
            response_text = ai.generate_text(strict_prompt, system_instruction=strict_system)
        except Exception as e:
            log.warning("[%s] SEO escalation attempt failed: %s", clip_id, e)
            response_text = ""
        data = _parse_json_response(response_text) if response_text else None

    # ── 3. No degradation: signal queue-for-retry ─────────────────────────────
    if not data:
        raise SEOGenerationError(
            "[%s] SEO generation failed after escalation — no valid AI JSON" % clip_id
        )

    # ── Build result from AI data ─────────────────────────────────────────────
    ai_terms = data.get("search_terms", [])
    optimized_tags = _rank_and_optimize_tags(
        ai_terms=ai_terms,
        yt_suggestions=yt_suggestions or [],
        trend_topics=trend_topics,
        local_keywords=local_keywords or [],
        max_chars=500,
    )

    ai_hashtags = data.get("hashtags") or _default_hashtags()

    if is_shorts:
        # Preserve the LLM's punchy short description; do NOT overwrite it with
        # the long LIVE/CHAPTERS/Disclaimer template (that was a Shorts bug).
        description = str(data.get("description", "")).strip()
    else:
        description = assemble_description(
            data,
            scorecard=scorecard,
            video_title=video_title,
            transcript=transcript,
            fallback_search_terms=optimized_tags,
            fallback_hashtags=ai_hashtags,
        )

    title = data.get("title") or ("LIVE %s" % (video_title or "Cricket Highlights"))

    result = _enforce_limits({
        "clip_id": clip_id,
        "title": title,
        "description": description,
        "hashtags": ai_hashtags,
        "search_terms": optimized_tags,
        "tags": optimized_tags,
    }, fallback_terms=trend_topics)

    result = _consolidate_seo(
        result["title"], result["description"],
        result["hashtags"], result["search_terms"]
    )

    result = _inject_viral_elements(
        result["title"], result["description"], result["hashtags"], extra=result,
    )
    result["clip_id"] = clip_id
    result["ai_generated"] = True
    result["is_shorts"] = is_shorts
    result["_generated_by_provider"] = ai.get_used_provider()
    result["_generated_by_model"] = ai.get_used_model()

    log.info("[%s] SEO done — title: %s | tags: %d (%d chars)",
             clip_id, result["title"][:100],
             len(result.get("search_terms", [])),
             sum(len(t) + 1 for t in result.get("search_terms", [])))
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

    try:
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
    except SEOGenerationError as e:
        # No generic fallback: write a queue marker and return WITHOUT a
        # *_metadata.json so the upload phase skips this clip (queued for retry).
        log.error("[%s] SEO failed — queued for retry (no generic fallback): %s",
                  clip_id, e,
                  extra={"stage": "seo", "status": "failed",
                         "error_type": type(e).__name__,
                         "metadata": {"clip_id": clip_id}})
        from datetime import datetime as _dt
        marker = Path(output_dir) / f"{clip_id}_seo_failed.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        # Self-contained payload so retry_failed_seo() can re-run without the
        # original highlights file or pipeline context.
        payload = {
            "clip_id": clip_id, "reason": str(e), "queued_at": _dt.now().isoformat(),
            "transcript": transcript, "video_title": video_title,
            "scorecard": scorecard, "trend_topics": trend_topics or [],
            "live_stream_url": live_stream_url,
        }
        with open(marker, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return {"clip_id": clip_id, "_seo_failed": True, "_queued": True, "reason": str(e)}

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
    queued = []
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
        except SEOGenerationError as e:
            # No generic fallback: queue the clip for retry instead of writing
            # generic metadata. A *_seo_failed.json marker is written; NO
            # *_metadata.json is produced, so the upload phase skips this clip.
            log.error("[%s] SEO failed — queued for retry (no generic fallback): %s",
                      clip_id, e,
                      extra={"stage": "seo", "status": "failed",
                             "error_type": type(e).__name__,
                             "metadata": {"clip_id": clip_id}})
            queued.append(clip_id)
            marker = Path(output_dir) / f"{clip_id}_seo_failed.json"
            with open(marker, "w", encoding="utf-8") as f:
                json.dump({"clip_id": clip_id, "reason": str(e),
                           "queued_at": __import__("datetime").datetime.now().isoformat(),
                           "transcript": transcript, "video_title": video_title,
                           "scorecard": scorecard, "trend_topics": trend_topics or [],
                           "live_stream_url": live_stream_url},
                          f, ensure_ascii=False, indent=2)
            if idx < len(clips):
                time.sleep(5)
            continue

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

    log.info("SEO complete: %d ok, %d queued-for-retry → %s",
             len(all_results), len(queued), output_dir)
    return str(combined_path)


# ── Retry queue: re-run SEO for clips that were queued on prior failure ───────

def retry_failed_seo(output_dir: str) -> Dict:
    """Re-attempt SEO for clips queued by a prior failure (``*_seo_failed.json``).

    Closes the escalate-not-degrade loop: a clip that failed all LLM attempts
    earlier is retried from its self-contained marker (which stores transcript +
    context). On success the ``*_metadata.json`` is written and the marker is
    removed (so the upload phase will now pick it up). On repeated failure the
    marker is left in place (still queued). Returns a summary dict.
    """
    out = Path(output_dir)
    markers = sorted(out.glob("*_seo_failed.json")) if out.exists() else []
    if not markers:
        log.info("No queued SEO failures to retry in %s", output_dir)
        return {"retried": 0, "recovered": 0, "still_failed": 0}

    recovered, still_failed = 0, 0
    for i, marker in enumerate(markers):
        try:
            ctx = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Skipping unreadable marker %s: %s", marker.name, e)
            continue
        clip_id = ctx.get("clip_id") or marker.stem.replace("_seo_failed", "")
        log.info("Retrying queued SEO [%d/%d]: %s", i + 1, len(markers), clip_id)
        res = generate_seo_for_exported_clip(
            clip_id=clip_id,
            transcript=ctx.get("transcript", "Cricket Live"),
            output_dir=output_dir,
            video_title=ctx.get("video_title", ""),
            scorecard=ctx.get("scorecard", ""),
            trend_topics=ctx.get("trend_topics", []),
            live_stream_url=ctx.get("live_stream_url", ""),
            inter_clip_pause=8.0 if i > 0 else 0.0,
        )
        if res.get("_seo_failed"):
            still_failed += 1
        else:
            recovered += 1
            try:
                marker.unlink()  # success -> dequeue
            except OSError:
                pass

    log.info("SEO retry complete: %d recovered, %d still queued", recovered, still_failed)
    return {"retried": len(markers), "recovered": recovered, "still_failed": still_failed}

if __name__ == "__main__":
    process_all_seo("highlights/video.yaml", "shorts/test")
