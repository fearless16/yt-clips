"""seo.py — Per-clip sequential SEO generation for Indian cricket live Shorts.

One AI call per clip with retry + backoff. No batching = no 429 storms.
"""
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

from utils.config import load_config
from utils.logger import get_logger
from utils.ai_client import AIClient
from trends import TEAM_MAPPINGS
from seo_learner import enhance_seo_prompt, generate_performance_report, learn_from_clip_performance

cfg = load_config()
log = get_logger("seo", cfg["logging"]["log_file"], cfg["logging"]["level"])
ai = AIClient()

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
    "Return ONLY valid JSON — no markdown, no explanation."
)

_PROMPT_TMPL = """
CONTEXT:
  Match: {video_title}
  Scorecard: {scorecard}
  Live Trending: {trend_topics}
  CTA: {live_cta}

CLIP CONTENT:
  Transcript: {transcript}
  Key moments: {local_kw}

TASK: Generate PERFECT SEO metadata for ONE viral cricket short.

═══════════════════════════════════════════════════════════════════════════════
TITLE FORMULA (pick ONE and make it SPECIFIC):
═══════════════════════════════════════════════════════════════════════════════
  1. SHOCK: "cricket live: <unexpected moment> | IPL 2026"
  2. STAR: "IPL 2026: <star player> <action> <result>"
  3. CLOSE: "cricket live: <close call> - did they?! | IPL 2026"
  4. NUMBERS: "IPL 2026: <score> runs in <overs> - game changer!"
  5. EMOTION: "Ye toh <emotion> hai bhai! <moment> | IPL 2026"

RULES:
  - MUST start with "cricket live:" or "IPL 2026" (SEO gold)
  - Inject 1-2 trending topics naturally into title
  - Max 100 chars, MAX 1 emoji
  - NEVER generic like "Cricket Amazing!"

═══════════════════════════════════════════════════════════════════════════════
DESCRIPTION FORMULA:
═══════════════════════════════════════════════════════════════════════════════
  First 100 chars: "{{hook}}" + what happened + "IPL 2026"
  Next 200 chars: Hindi/English mix - emotional reaction
  Last 100 chars: "{{cta}}"
  Then 3-5 hashtags (tournament + teams + #Shorts)

═══════════════════════════════════════════════════════════════════════════════
HASHTAGS (exactly 4, strategic order):
═══════════════════════════════════════════════════════════════════════════════
  1. #IPL or #T20WorldCup or #Cricket
  2. Winning team #RCB #CSK #MI etc
  3. Losing team or star player
  4. #Shorts (ALWAYS)

═══════════════════════════════════════════════════════════════════════════════
SEARCH TERMS (15-25, super targeted):
═══════════════════════════════════════════════════════════════════════════════
  - Include ALL trending topics as search terms
  - Player names + "catch" / "six" / "boundary" / "wicket"
  - NO generic "cricket viral"

Return ONLY JSON:
{{
  "clip_id": "{clip_id}",
  "title": "...",
  "description": "...",
  "hashtags": ["#...", "#...", "#...", "#..."],
  "search_terms": ["...", "..."]
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


def _inject_viral_elements(title: str, description: str, hashtags: List[str]) -> Dict:
    """Inject viral hooks and CTAs into SEO output."""
    import random

    # Pick random hook and CTA
    hook = random.choice(VIRAL_HOOKS)
    cta = random.choice(ENGAGING_CTAS)

    # Ensure description has hook and CTA
    if hook not in description and len(description) > 50:
        # Insert hook near beginning
        desc_parts = description.split("\n\n")
        if desc_parts:
            desc_parts[0] = f"{hook} {desc_parts[0][:100]}"
        description = "\n\n".join(desc_parts)

    # Ensure CTA at end
    if cta not in description:
        if len(description) > 100:
            description = f"{description}\n\n{cta}"
        else:
            description = f"{description} {cta}"

    # Ensure hashtags have proper structure
    if len(hashtags) < 3:
        hashtags = ["#IPL2026", "#Cricket", "#Shorts"] + hashtags

    return {
        "title": title,
        "description": description,
        "hashtags": hashtags[:5]
    }


def _generate_template_seo(
    clip_id: str,
    transcript: str,
    video_title: str,
    scorecard: str,
    trend_topics: List[str],
) -> Dict:
    """
    Template-based fallback when AI fails.
    Generates solid SEO without API calls.
    """
    import random
    import json

    local_kw = _extract_keywords(transcript, limit=8)

    # Extract team/moment from transcript
    teams_found = []
    for team_code, aliases in TEAM_MAPPINGS.items():
        for alias in aliases:
            if alias in transcript.lower()[:200]:
                teams_found.append(team_code)
                break
    teams_found = list(set(teams_found))[:2]

    # Build title from template — NO garbage transcript keywords
    team_str = " vs ".join(teams_found) if teams_found else "Cricket Live"
    score_str = scorecard.replace("Live: ", "").strip() if scorecard else ""

    title_variants = [
        f"{team_str} - Unbelievable Finish! 🔥 | IPL 2026",
        f"{team_str} - Match Ka Turning Point! | IPL 2026",
        f"Arey Yeh Kya Ho Gaya?! {team_str} | IPL 2026 🏏",
        f"{score_str} - {team_str} Full Drama! | IPL 2026" if score_str else f"{team_str} Full Drama! | IPL 2026",
        f"Last Over Ka Dhamaal! {team_str} | IPL 2026 🔥",
        f"{team_str} - Brutal Sixes & Wickets! | IPL 2026 🏏",
    ]
    title = random.choice(title_variants)[:100]

    # Build description
    hook = random.choice(VIRAL_HOOKS)
    cta = random.choice(ENGAGING_CTAS)
    description = (
        f"{hook}\n\n"
        f"Match: {scorecard[:100] if scorecard else video_title}\n"
        f"Moment: {', '.join(local_kw[:5])}\n\n"
        f"{cta}\n\n"
        f"#IPL2026 #{teams_found[0] if teams_found else 'Cricket'} #Shorts"
    )

    # Build hashtags
    hashtags = [
        "#IPL2026",
        f"#{teams_found[0]}" if teams_found else "#Cricket",
        f"#{teams_found[1]}" if len(teams_found) > 1 else "#T20",
        "#Shorts"
    ]

    # Search terms
    search_terms = (
        [t for t in trend_topics[:10]] +
        [f"{w} six" for w in local_kw[:3]] +
        [f"{w} wicket" for w in local_kw[:2]]
    )[:20]

    return {
        "clip_id": clip_id,
        "title": title,
        "description": description,
        "hashtags": hashtags,
        "search_terms": search_terms
    }


def _enforce_limits(item: Dict) -> Dict:
    title = (item.get("title") or "")[:100]
    description = (item.get("description") or "")[:5000]

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

    return {**item, "title": title, "description": description,
            "hashtags": hashtags, "search_terms": cleaned}


def _parse_json_response(text: str) -> Optional[Dict]:
    """Extract and parse the first JSON object from model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ── Per-clip SEO (one AI call, retries with backoff) ──────────────────────────

def generate_clip_seo(
    clip_id: str,
    transcript: str,
    video_title: str = "",
    scorecard: str = "",
    trend_topics: List[str] = None,
    live_stream_url: str = "",
) -> Dict:
    """
    Generate SEO metadata for a single clip.
    Retries up to 3 times with exponential backoff on 429/593.
    """
    trend_topics = trend_topics or []
    local_kw_list = _extract_keywords(transcript)
    local_kw = ", ".join(local_kw_list)
    
    # [NEW] Intercept exact players/events to ping YouTube Suggest API
    try:
        from trends import fetch_clip_specific_suggestions
        clip_suggestions = fetch_clip_specific_suggestions(local_kw_list)
        if clip_suggestions:
            trend_topics = list(trend_topics) + clip_suggestions
    except Exception as e:
        log.warning("Could not fetch clip-specific suggestions: %s", e)

    trend_str = ", ".join(trend_topics) or "IPL 2026, cricket live"
    live_cta = (
        f"Watch LIVE: {live_stream_url}" if live_stream_url
        else "Match chal raha hai LIVE — channel pe aao."
    )

    prompt = _PROMPT_TMPL.format(
        video_title=video_title or "Cricket Live Match",
        scorecard=scorecard or "Live match in progress",
        trend_topics=trend_str,
        live_cta=live_cta,
        transcript=transcript[:2000],   # keep prompt tight
        local_kw=local_kw,
        clip_id=clip_id,
    )
    
    # Enhance prompt with learned insights from performance data
    prompt = enhance_seo_prompt(prompt)

    backoff = [0, 8, 20, 45]
    for attempt, delay in enumerate(backoff):
        if delay:
            log.info("[%s] SEO retry %d — waiting %ds...", clip_id, attempt, delay)
            time.sleep(delay)
        try:
            response_text = ai.generate_text(prompt, system_instruction=_SYSTEM)
            data = _parse_json_response(response_text)
            if not data:
                raise ValueError("No JSON in response")

            result = _enforce_limits({
                "clip_id": clip_id,
                "title": data.get("title", f"Cricket Live Highlights | {clip_id}"),
                "description": data.get("description", ""),
                "hashtags": data.get("hashtags", ["#IPL2026", "#Cricket", "#Shorts"]),
                "search_terms": data.get("search_terms", []),
            })

            # Inject viral hooks and CTAs
            result = _inject_viral_elements(
                result["title"],
                result["description"],
                result["hashtags"]
            )
            result["clip_id"] = clip_id

            log.info("[%s] SEO done — title: %s", clip_id, result["title"][:60])
            return result

        except Exception as e:
            msg = str(e)
            if "429" in msg or "593" in msg:
                log.warning("[%s] Rate limited (attempt %d): %s", clip_id, attempt + 1, msg)
                continue
            log.error("[%s] SEO error (attempt %d): %s", clip_id, attempt + 1, msg)
            if attempt < len(backoff) - 1:
                continue

    # ── Fallback: Template-based SEO (no AI needed) ─────────────────────────
    log.warning("[%s] AI failed, using template fallback", clip_id)
    result = _generate_template_seo(
        clip_id=clip_id,
        transcript=transcript[:1000],
        video_title=video_title,
        scorecard=scorecard,
        trend_topics=trend_topics,
    )

    # Inject viral elements
    result = _inject_viral_elements(
        result["title"],
        result["description"],
        result["hashtags"]
    )

    result["clip_id"] = clip_id
    return _enforce_limits(result)


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
    """
    if inter_clip_pause > 0:
        log.debug("[%s] Waiting %.0fs before SEO call...", clip_id, inter_clip_pause)
        time.sleep(inter_clip_pause)

    result = generate_clip_seo(
        clip_id=clip_id,
        transcript=transcript,
        video_title=video_title,
        scorecard=scorecard,
        trend_topics=trend_topics or [],
        live_stream_url=live_stream_url,
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
    from trends import get_trending_context
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

    # Fetch trend context ONCE for the whole session
    log.info("Fetching trend context...")
    trend = get_trending_context(domain="cricket", region="IN", video_title=video_title)
    live_stream_url = live_stream_url or trend.get("live_stream_url", "")
    scorecard = trend.get("scorecard", "")
    trend_topics = trend.get("topics", [])

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
