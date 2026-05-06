"""seo.py — Hybrid Batch SEO generation for Indian cricket live Shorts."""
import json
import re
from pathlib import Path
from typing import List, Dict

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


def _extract_keywords(text: str, limit: int = 14) -> List[str]:
    words = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    freq: Dict[str, int] = {}
    for w in keywords:
        freq[w] = freq.get(w, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:limit]]


def batch_generate_seo(clips: List[Dict], domain: str = "cricket", region: str = "IN") -> List[Dict]:
    """
    Generate SEO for multiple clips in a single AI call to optimize tokens and consistency.
    Optimized for Small YouTubers (140 subs) — focuses on long-tail search and high-retention hooks.
    """
    from trends import get_trending_context
    
    # 0. Load Global Video Metadata (Title)
    video_title = ""
    meta_file = Path(cfg["paths"]["input"]) / "video_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
                video_title = meta.get("title", "")
        except:
            pass

    # 1. Gather Global Trend Context
    trend = get_trending_context(domain=domain, region=region, video_title=video_title)
    
    clips_context = []
    for c in clips:
        local_kw = _extract_keywords(c['text'])
        clips_context.append({
            "clip_id": c['clip_id'],
            "transcript": c['text'],
            "local_keywords": local_kw
        })

    system_instr = (
        "You are a senior cricket analyst and YouTube SEO expert for small channels (140 subscribers). "
        "You have deep knowledge of every IPL/International team, player stats, rivalries, and match situations. "
        "Your job: write metadata that proves deep cricket understanding while creating curiosity gaps that force viewers to click. "
        "Every title must feel like an insider insight, not a generic recap. "
        "Every description must show you know exactly what happened, why it mattered, and what the player/crowd felt. "
        "Use natural, passionate language with strategic emojis."
    )
    
    prompt = f"""
    Overall Match: {video_title}
    Scorecard snapshot: {trend.get('scorecard', '')}
    Trending Topics: {', '.join(trend.get('topics', []))}
    
    Generate high-impact YouTube Shorts metadata for the following {len(clips)} highlights:
    {json.dumps(clips_context, indent=2)}
    
    Requirements for EACH clip:
    1. **Title**: Curiosity-gap, high-CTR, max 100 characters. Must include specific player name(s) and a hint of drama or skill. Example: "Kohli's Six Over Cover to Win It 💥 No One Believed!"
    2. **Description**: Write a rich, 3-5 line description (80-120 words) that:
       - Opens with a short hook line (sensory/emotional) and an emoji.
       - Explains the exact moment: what happened, who was bowling, the delivery, the shot, the context (chasing target, powerplay, etc.).
       - Adds a deeper insight: why this moment was special (stats, rivalry, pressure situation, player form).
       - End with a call-to-action that builds community (subscribe, comment on the next match, etc.) using a friendly tone.
       - Integrate the match scorecard context naturally (e.g., "Chasing 200, RCB were 45/3 in the powerplay when…").
       - Use 1-2 relevant hashtags at the end.
    3. **Tags**: 15-20 long-tail, ultra-specific tags. Format them as lowercase comma-separated keywords. Examples: "kohli six vs csk 2024, rcb run chase thriller, kohli cover drive, indian cricket shorts, dhoni reaction, ipl 2024 highlights". NEVER use generic tags like "cricket" or "shorts".
    
    Return a JSON object with a 'clips_seo' key containing a list of objects (clip_id, title, description, tags).
    """
    
    log.info("🚀 Sending Batch SEO request to AI...")
    response_text = ai.generate_text(prompt, system_instruction=system_instr)
    
    results = []
    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            results = data.get("clips_seo", [])
    except Exception as e:
        log.error(f"Batch SEO parsing failed: {e}")

    # 3. Finalize & Merge with trends
    hashtags = trend.get("tags", ["#Shorts", "#Cricket"])
    final_results = []
    for c in clips:
        seo = next((item for item in results if item["clip_id"] == c["clip_id"]), {})
        final_results.append({
            "clip_id": c["clip_id"],
            "title": seo.get("title", f"Cricket Highlights {c['clip_id']}")[:100],
            "description": seo.get("description", "")[:5000],
            "tags": seo.get("tags", [])[:30],
            "hashtags": hashtags,
            "trend_topics": trend.get("topics", []),
        })
    return final_results


def generate_seo(clip_text: str, clip_id: str) -> Dict:
    """Legacy wrapper for single-clip SEO generation. Uses batch logic internally."""
    clips = [{"clip_id": clip_id, "text": clip_text}]
    results = batch_generate_seo(clips)
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

    clips_to_process = []
    for clip_id, info in highlights.items():
        clips_to_process.append({
            "clip_id": clip_id,
            "text": info.get("text", "Cricket Live Highlights")
        })

    all_seo = batch_generate_seo(clips_to_process, domain="cricket", region="IN")

    for seo_data in all_seo:
        clip_id = seo_data["clip_id"]
        seo_file = Path(output_dir) / f"{clip_id}_metadata.json"
        with open(seo_file, "w", encoding="utf-8") as f:
            json.dump(seo_data, f, indent=2, ensure_ascii=False)

    log.info("✅ Batch AI SEO Metadata generated in %s", output_dir)


if __name__ == "__main__":
    process_all_seo("highlights/video.yaml", "shorts/test")