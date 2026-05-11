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

# ── Prompt ─────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an elite YouTube Shorts SEO engine for Indian cricket live-stream clips. "
    "Return ONLY valid JSON — no markdown, no explanation, no preamble."
)

_PROMPT_TMPL = """
Match: {video_title}
Scorecard: {scorecard}
Trending topics: {trend_topics}
Live stream CTA: {live_cta}

Clip transcript:
{transcript}

Local keywords extracted from transcript: {local_kw}

Generate YouTube Shorts metadata for this ONE clip. Rules:

TITLE (≤100 chars):
  - Format: <TEAM1> vs <TEAM2> Live | <Moment> | IPL 2026
  - Include both teams, "Live", emotional moment, tournament + year
  - Max 1 emoji. NEVER exceed 100 chars.

DESCRIPTION (≤5000 chars, ideal 400-1200):
  - First 150 chars must have: match keyword + player/moment + "IPL 2026"
  - Natural Hinglish, NOT ChatGPT tone
  - Paragraphs: (1) 2-3 factual match lines, (2) Hinglish analysis, (3) short CTA
  - Then 3-5 hashtags. Then search terms one per line.
  - NO markdown, NO bullets, NO excessive emojis

HASHTAGS (exactly 3-5, match/tournament/team specific, NO spam):

SEARCH TERMS (18-30 terms, lowercase, 2-5 words each, total ≤500 chars):
  - Target: match, tournament, players, Hindi viewers, live viewers
  - NO generic: cricket, viral, shorts, trending
  - NO duplicates of hashtags

CRITICAL: Whisper auto-transcription has phonetic errors.
Silently correct player names from context (e.g. "Chakris Gale" → "Chris Gayle").

Return ONLY this JSON (no other text):
{{
  "clip_id": "{clip_id}",
  "title": "...",
  "description": "...",
  "hashtags": ["#A", "#B", "#C"],
  "search_terms": ["term one", "term two"]
}}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_keywords(text: str, limit: int = 14) -> List[str]:
    words = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    kw = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    freq: Dict[str, int] = {}
    for w in kw:
        freq[w] = freq.get(w, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:limit]]


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
    local_kw = ", ".join(_extract_keywords(transcript))
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

    log.error("[%s] SEO generation failed after all retries", clip_id)
    return {
        "clip_id": clip_id,
        "title": f"Cricket Live Highlights | {clip_id}",
        "description": "",
        "hashtags": ["#IPL2026", "#Cricket", "#Shorts"],
        "search_terms": [],
    }


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
