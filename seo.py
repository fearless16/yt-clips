"""seo.py — Hybrid Batch SEO generation for Indian cricket live Shorts.

SEO Output Format:
You are an elite YouTube Shorts SEO and metadata generation engine for Indian cricket live-stream clips.

Your ONLY job is to generate:

* Titles
* Descriptions
* Search terms
* Hashtags

STRICTLY optimized for:

* YouTube Shorts discovery
* CTR
* Retention
* Search relevance
* Live-stream conversion

You MUST obey ALL metadata constraints EXACTLY.

━━━━━━━━━━━━━━━━━━
YOUTUBE API HARD LIMITS
━━━━━━━━━━━━━━━━━━

TITLE:

* MAXIMUM 100 characters
* IDEAL RANGE: 55–85 chars
* NEVER exceed 100
* Front-load the most important keywords

DESCRIPTION:

* MAXIMUM 5000 chars
* IDEAL RANGE: 400–1200 chars
* First 150 chars MUST contain:

  * main match keyword
  * player/moment keyword
  * “IPL 2026” or tournament keyword
* NEVER generate walls of text
* NEVER use markdown
* NEVER use AI-style wording

SEARCH TERMS:

* Total combined length MUST stay under 500 characters
* Each term:

  * lowercase
  * 2–5 words
  * highly searchable
* NO generic spam:

  * cricket
  * viral
  * shorts
  * trending
* NEVER duplicate hashtags inside search terms
* Target:

  * 18–30 search terms max

HASHTAGS:

* EXACTLY 3–5 hashtags
* NEVER exceed 15 hashtags
* Must be:

  * match-specific
  * tournament-specific
  * team-specific
* NO spam hashtags

━━━━━━━━━━━━━━━━━━
CONTENT RULES
━━━━━━━━━━━━━━━━━━

Your writing style:

* Natural Hinglish
* Emotional but believable
* Like a passionate cricket fan
* NOT like ChatGPT
* NOT corporate
* NOT clickbait garbage

DO:

* Mention:

  * teams
  * player names
  * match situation
  * wickets/runs/pressure moments
* Include:

  * “live match”
  * “match review”
  * “Hindi analysis”
  * “reaction”
  * “powerplay”
  * “live score”
    naturally when relevant

DO NOT:

* Use fake hype
* Use repetitive phrases
* Use emojis excessively
* Use markdown
* Use bullet spam
* Use hashtags in title unless necessary

━━━━━━━━━━━━━━━━━━
TITLE GENERATION RULES
━━━━━━━━━━━━━━━━━━

GOOD TITLE FORMAT: <TEAM1> vs <TEAM2> Live Match Today | <Moment> | IPL 2026

Examples:
DC vs KKR Live Match Today | KKR Chase Drama | IPL 2026
RCB vs CSK Live Match Today | Kohli Pressure Knock | IPL 2026
MI vs GT Live Match Today | Bumrah Deadly Spell | IPL 2026

TITLE REQUIREMENTS:

* Include BOTH teams
* Include “Live Match Today”
* Include emotional/context phrase
* Include tournament/year
* Avoid excessive punctuation
* Max 1 emoji allowed
* NEVER exceed 100 chars

━━━━━━━━━━━━━━━━━━
DESCRIPTION FORMAT
━━━━━━━━━━━━━━━━━━

STRICT TEMPLATE:

Line 1:

<TITLE>

Paragraph 1:
<2–3 factual match lines>

Paragraph 2:
<Natural Hindi/Hinglish analysis paragraph>

Paragraph 3:
<Short CTA for live stream/channel>

Then:
3–5 hashtags

Then:
Search Terms: <one search term per line>

━━━━━━━━━━━━━━━━━━
SEARCH TERM STRATEGY
━━━━━━━━━━━━━━━━━━

Search terms MUST target:

1. Main match
2. Tournament
3. Player moments
4. Hindi viewers
5. Live viewers
6. Reaction audience

GOOD:
dc vs kkr live
ipl 2026 live match
kkr batting collapse
dc vs kkr hindi commentary

BAD:
cricket
viral shorts
awesome match
sports

━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━

Return ONLY valid JSON.

NO markdown.
NO explanation.
NO commentary.

Schema:

{
"clips_seo": [
{
"clip_id": "clip1",
"title": "...",
"description": "...",
"hashtags": [
"#DCvsKKR",
"#IPL2026"
],
"search_terms": [
"dc vs kkr live",
"ipl 2026 live match"
]
}
]
}

━━━━━━━━━━━━━━━━━━
VALIDATION RULES
━━━━━━━━━━━━━━━━━━

Before returning output:

* Verify title <= 100 chars
* Verify description <= 5000 chars
* Verify hashtags count <= 5
* Verify total search_terms character count <= 500
* Remove duplicates
* Remove generic spam
* Ensure JSON is valid
* Ensure no field is empty

If constraints fail:
REGENERATE internally before responding.

NEVER explain failures.
ONLY return final valid JSON.


Rules:
  - search_terms and hashtags are SEPARATE. Never reuse hashtags as search_terms.
  - search_terms drive discoverability (teams, players, moments, intent).
  - hashtags drive category/trending (match, tournament, Shorts).
  - Description: emotional hook, core moment early, natural CTA.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

from utils.config import load_config
from utils.logger import get_logger
from utils.ai_client import AIClient

cfg = load_config()
log = get_logger("seo", cfg["logging"]["log_file"], cfg["logging"]["level"])
ai = AIClient()

STOP_WORDS = {
    "i", "me", "my", "you", "your", "we", "our", "they", "their", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
    "a", "an", "the", "and", "or", "but", "if", "as", "of", "to", "in", "on", "at", "for", "from",
    "with", "by", "about", "into", "over", "under", "again", "then", "here", "there", "when", "where",
    "why", "how", "all", "any", "more", "most", "some", "such", "no", "nor", "not", "only", "very",
}

GENERIC_TAGS = {
    "cricket", "shorts", "viral", "trending", "youtube", "video", "sports",
    "highlight", "highlights", "amazing", "awesome", "incredible", "wow"
}

# ── Proven high-CTR title formulas for cricket Shorts ──────────────────────────
# %PLAYER% = player name, %MOMENT% = what happened, %CONTEXT% = match context
# %TREND% = trending topic, %STAT% = a number/stat
TITLE_FORMULAS = [
    # Formula A: Curiosity-gap + drama
    "Nobody Expected %PLAYER% To Do THIS 😱 | %CONTEXT%",
    "%PLAYER% ne kya kar diya yaar 😤 | %MOMENT%",
    "Wait For It… %PLAYER%'s %MOMENT% Changed Everything 🔥",
    "This %MOMENT% by %PLAYER% broke the internet 💥 | %CONTEXT%",
    # Formula B: Stat-shock
    "%STAT% in %CONTEXT% — %PLAYER% is Built Different 🏏",
    "Only %PLAYER% Can Do This In %CONTEXT% 👀",
    # Formula C: Live stream hook
    "🔴 LIVE NOW: %TREND% | %PLAYER%'s Best Shots REACTION",
    "Watch Full Match LIVE 🔴 | %PLAYER% %MOMENT% Highlights",
]


def inject_trend_topics_into_tags(
    base_tags: List[str],
    trend_topics: List[str],
    player_name: str = ""
) -> List[str]:
    if not trend_topics:
        return base_tags
    result = base_tags.copy()
    seen_lower = {tag.lower() for tag in result}
    for topic in trend_topics[:3]:
        topic_lower = topic.lower()
        if topic_lower in seen_lower:
            continue
        if player_name:
            combo = f"{player_name} {topic_lower}"
            if combo not in seen_lower:
                result.append(combo)
                seen_lower.add(combo)
        if topic_lower not in seen_lower:
            result.append(topic_lower)
            seen_lower.add(topic_lower)
    return result


def ensure_trend_in_title(title: str, trend_topics: List[str]) -> str:
    """
    Inject trend into title ONLY if it fits naturally — never prepend blindly.
    Preserves curiosity gap structure.
    """
    if not trend_topics:
        return title[:100]
    title_lower = title.lower()
    for trend in trend_topics:
        trend_words = trend.lower().split()
        if len(trend_words) >= 2 and " ".join(trend_words[:2]) in title_lower:
            return title[:100]
        if trend.lower() in title_lower:
            return title[:100]
    # Append as context tag rather than prepend — keeps hook intact
    top = trend_topics[0]
    candidate = f"{title} | {top}"
    return candidate[:100] if len(candidate) <= 100 else title[:100]


def validate_hinglish_content(text: str) -> Tuple[bool, str]:
    hindi_words = {
        'ka', 'ki', 'ke', 'ko', 'mein', 'se', 'par', 'aur', 'hai', 'tha', 'thi',
        'dhamaakedaar', 'jabardast', 'shandaar', 'kya', 'yeh', 'toh', 'bhi',
        'maara', 'khela', 'dekho', 'bolo', 'arey', 'yaar', 'bhai', 'log', 'ne',
    }
    words = text.lower().split()
    count = sum(1 for w in words if w in hindi_words)
    if count >= 2:
        return True, "hinglish"
    elif count == 1:
        return True, "light_hinglish"
    return True, "english"


def validate_emoji_usage(text: str) -> bool:
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002700-\U000027BF\U00002600-\U000026FF]"
    )
    return len(emoji_pattern.findall(text)) <= 5


def _validate_search_terms(terms: List[str], min_words: int = 2) -> List[str]:
    """Validate search terms: must be 2+ words, not generic spam."""
    if not terms:
        return []
    valid = []
    for term in terms:
        tl = term.lower().strip()
        if not tl:
            continue
        if len(tl.split()) < min_words:
            continue
        if tl in GENERIC_TAGS:
            continue
        if set(tl.split()).issubset(GENERIC_TAGS):
            continue
        valid.append(term)
    return valid


def _extract_keywords(text: str, limit: int = 14) -> List[str]:
    words = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    freq: Dict[str, int] = {}
    for w in keywords:
        freq[w] = freq.get(w, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:limit]]


def batch_generate_seo(clips: List[Dict], domain: str = "cricket", region: str = "IN") -> List[Dict]:
    """
    Generate SEO for multiple clips in one AI call.

    Returns per clip:
      - title_variants: [A, B, C]  ← pick best performing via YouTube analytics
      - title: variant A (default upload title)
      - thumbnail_text: 3-4 word bold overlay copy
      - description: with live stream CTA
      - tags: long-tail specific
      - hashtags, trend_topics
    """
    from trends import get_trending_context

    # Load video metadata
    video_title = ""
    live_stream_url = ""
    meta_file = Path(cfg["paths"]["input"]) / "video_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
                video_title = meta.get("title", "")
                live_stream_url = meta.get("live_stream_url", "")  # ← add this to your metadata
        except Exception:
            pass

    # Trend context
    trend = get_trending_context(domain=domain, region=region, video_title=video_title)
    live_stream_url = live_stream_url or trend.get("live_stream_url", "")

    clips_context = []
    for c in clips:
        local_kw = _extract_keywords(c["text"])
        clips_context.append({
            "clip_id": c["clip_id"],
            "transcript": c["text"],
            "local_keywords": local_kw,
        })

    trend_topics_str = ", ".join(trend.get("topics", [])) or "IPL 2026, cricket live"
    scorecard = trend.get("scorecard", "")

    # ── System instruction ─────────────────────────────────────────────────────
    system_instr = (
        "You are a YouTube growth expert for small Indian cricket channels (140 subs). "
        "You write titles that cause thumb-stops and clicks — not recaps. "
        "You know IPL/International cricket deeply: players, rivalries, stats, pressure situations. "
        "Every title must create a CURIOSITY GAP or SHOCK. Viewers must feel they'll miss something if they don't click. "
        "Use Hinglish naturally (not forced). Think like a passionate fan who also knows SEO. "
        "Channel has a LIVE STREAM running — every Short must drive viewers there."
    )

    # ── Prompt ─────────────────────────────────────────────────────────────────
    live_cta = (
        f"🔴 Watch LIVE: {live_stream_url}" if live_stream_url
        else "🔴 Match chal raha hai LIVE — Channel pe aao! Link in bio."
    )

    prompt = f"""
Match: {video_title}
Scorecard: {scorecard}
Trending Topics: {trend_topics_str}
Live Stream CTA: {live_cta}

Generate YouTube Shorts SEO for these {len(clips)} cricket highlights:
{json.dumps(clips_context, indent=2)}

For EACH clip return:

1. **title_variants** (array of 3, each max 100 chars):
   - Variant A — Curiosity-gap / Drama formula. Example: "Kohli ne kya kar diya yaar 😤 | RCB vs CSK"
   - Variant B — Stat-shock / Insider formula. Example: "Only Kohli Can Hit This In a Chase 👀 | IPL 2026"
   - Variant C — Live stream hook. Example: "🔴 LIVE NOW: RCB chasing 200 | Kohli ki innings LIVE"
   Rules:
   - Must include specific player name + what happened
   - Curiosity gap: hint at drama without giving it away
   - Variant C MUST reference the live stream
   - Use Hinglish in at least one variant
   - NO generic openers like "Amazing shot" or "Incredible moment"

2. **thumbnail_text** (3-4 words, ALL CAPS, punchy — for text overlay on thumbnail):
   Examples: "KOHLI NE KAR DIYA", "IMPOSSIBLE CATCH 😱", "YEH KYA THA YAAR"

3. **description**:
   You MUST strictly follow this exact template structure. Replace the bracketed placeholders with relevant Hindi/Hinglish content. Use the provided Scorecard. If the Scorecard is empty, hallucinate realistic plausible stats based on the Match context.

   🏏 <Curiosity Hook in English/Hinglish> | <Player or Moment> | {video_title}

   🏏 <1-2 sentences summarizing the highlight and the player's performance in Hinglish/English. End with "Here are the highlights.">

   📋 Match Summary:
   Focus: <Key player stat or moment>
   Scorecard: {scorecard if scorecard else "TBD - Live Match"}
   Context: <Brief match context or turning point>
   Toss: <Opted to bat/bowl>

   💬 Your Call: <Engaging question for the viewers in Hinglish>

   📢 New creator yahan! Cricket insights ke liye SUBSCRIBE karo aur bell 🔔 dabao. Goal 200 subs tak pahunchne ka!

   <Insert the 3-5 hashtags here>

4. **search_terms** (15-20 highly searchable phrases, lowercase, comma-separated):
   Purpose: These go into YouTube tags for discoverability.
   CRITICAL: Heavily prioritize the MAIN MATCH (from '{video_title}') and TRENDS, rather than just the isolated clip content.
   Mix of:
   - Main Match/Tournament: e.g., "lsg vs rcb 2026", "ipl 2026 highlights", "today match highlights"
   - Match Context: e.g., "why play stopped lsg vs rcb", "cricket highlights today"
   - Specific Moment/Player from the clip: e.g., "mitchell marsh fastest century"
   Rules:
   - Each term MUST be 2+ words
   - NEVER use single generic words: "cricket", "shorts", "viral", "trending"
   - Optimized SEPARATELY from hashtags — do NOT reuse hashtags as search terms

5. **hashtags** (exactly 3-5, with # prefix):
   Prioritize: match name, teams, tournament, #Shorts
   Examples: ["#RCBvsCSK", "#IPL2026", "#CricketShorts", "#Kohli"]
   Rules:
   - Only 3-5. No more.
   - No generic spam hashtags like #viral #trending
   - Must be specific to this match/moment

Return a JSON object:
{{"clips_seo": [{{"clip_id": "...", "title_variants": ["A","B","C"], "thumbnail_text": "...", "description": "...", "search_terms": [...], "hashtags": [...]}}]}}
"""

    log.info("🚀 Sending Batch SEO request to AI (%d clips)...", len(clips))
    response_text = ai.generate_text(prompt, system_instruction=system_instr)

    results = []
    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            results = data.get("clips_seo", [])
    except Exception as e:
        log.error("Batch SEO parsing failed: %s", e)

    if not results:
        log.error("Batch SEO returned no results. Raising — SEO is required.")
        raise ValueError("SEO Generation Failed: No results from AI.")

    hashtags_from_trend = trend.get("tags", ["#Shorts", "#Cricket"])
    trend_topics_list = trend.get("topics", [])

    final_results = []
    for c in clips:
        seo = next((item for item in results if item["clip_id"] == c["clip_id"]), {})

        # Search terms: validated, trend-injected
        raw_terms = seo.get("search_terms", seo.get("tags", []))  # fallback to tags if AI used old key
        validated_terms = _validate_search_terms(raw_terms, min_words=2)

        # Extract player name for combo search terms
        variants = seo.get("title_variants", [])
        primary_title = variants[0] if variants else seo.get("title", f"Cricket Highlights {c['clip_id']}")
        names = re.findall(r'\b[A-Z][a-z]+\b', primary_title)
        player_name = names[0].lower() if names else ""

        combined_terms = inject_trend_topics_into_tags(validated_terms, trend_topics_list, player_name)

        # Hashtags: from AI (3-5) with fallback to trend hashtags
        ai_hashtags = seo.get("hashtags", [])
        if isinstance(ai_hashtags, list) and len(ai_hashtags) >= 3:
            # Ensure # prefix, cap at 5
            hashtags = [h if h.startswith("#") else f"#{h}" for h in ai_hashtags][:5]
        else:
            hashtags = hashtags_from_trend[:5]

        # Ensure trend in primary title (non-destructive)
        title_a = ensure_trend_in_title(primary_title, trend_topics_list)
        title_b = variants[1] if len(variants) > 1 else title_a
        title_c = variants[2] if len(variants) > 2 else title_a

        final_results.append({
            "clip_id": c["clip_id"],
            "title": title_a,
            "title_variants": [title_a, title_b, title_c],
            "thumbnail_text": seo.get("thumbnail_text", "WATCH THIS NOW"),
            "description": seo.get("description", "")[:5000],
            "search_terms": combined_terms[:30],
            "hashtags": hashtags,
            "trend_topics": trend_topics_list,
            "live_stream_url": live_stream_url,
        })

    return final_results


def generate_seo(clip_text: str, clip_id: str) -> Dict:
    """Legacy single-clip wrapper."""
    results = batch_generate_seo([{"clip_id": clip_id, "text": clip_text}])
    return results[0] if results else {}


def process_all_seo(highlights_path: str, output_dir: str):
    h_path = Path(highlights_path)
    if not h_path.exists():
        log.error("Highlights not found: %s", h_path)
        return

    with open(h_path, "r", encoding="utf-8") as f:
        import yaml
        highlights = yaml.safe_load(f) or {}

    log.info("Generating BATCH AI SEO for %d clips...", len(highlights))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    clips_to_process = [
        {"clip_id": clip_id, "text": info.get("text", "Cricket Live Highlights")}
        for clip_id, info in highlights.items()
    ]

    SEO_BATCH_SIZE = 3
    SEO_RETRY_DELAYS = [8, 20, 45]
    
    pending = []
    all_seo = []
    
    import time
    
    for i, clip in enumerate(clips_to_process):
        pending.append(clip)
        is_last = (i == len(clips_to_process) - 1)
        
        if len(pending) >= SEO_BATCH_SIZE or is_last:
            success = False
            for delay in SEO_RETRY_DELAYS:
                try:
                    seo_batch = batch_generate_seo(pending, domain="cricket", region="IN")
                    
                    for seo_data in seo_batch:
                        all_seo.append(seo_data)
                        clip_id = seo_data["clip_id"]
                        seo_file = Path(output_dir) / f"{clip_id}_metadata.json"
                        with open(seo_file, "w", encoding="utf-8") as f:
                            json.dump(seo_data, f, indent=2, ensure_ascii=False)
                            
                    pending.clear()
                    success = True
                    if not is_last:
                        time.sleep(8)
                    break
                except Exception as e:
                    if "429" in str(e):
                        log.warning("429 Rate Limit Hit. Retrying in %d seconds...", delay)
                        time.sleep(delay)
                        continue
                    else:
                        log.error("Batch SEO Failed: %s", e)
                        break
            
            if not success:
                log.error("Failed to generate SEO for batch after retries. Clearing pending.")
                pending.clear()

    log.info("✅ Batch SEO done → %s (Total clips: %d)", output_dir, len(all_seo))


if __name__ == "__main__":
    process_all_seo("highlights/video.yaml", "shorts/test")