"""seo.py — Hybrid SEO generation for Indian cricket live Shorts."""
import json
import re
from pathlib import Path
from typing import List, Dict

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("seo", cfg["logging"]["log_file"], cfg["logging"]["level"])

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


def _hybrid_keywords(local_keywords: List[str], trend_topics: List[str], competitor_tokens: List[str]) -> List[str]:
    trend_tokens = []
    for t in trend_topics:
        trend_tokens.extend(re.findall(r"[A-Za-z0-9]{3,}", t.lower()))

    merged = list(dict.fromkeys(local_keywords + trend_tokens + competitor_tokens))
    return merged[:20]


def generate_seo(clip_text: str, clip_id: str, domain: str = "cricket", region: str = "IN") -> Dict:
    from trends import get_trending_context, humanize_title

    local_keywords = _extract_keywords(clip_text)
    trend = get_trending_context(domain=domain, region=region)

    hybrid_keywords = _hybrid_keywords(
        local_keywords,
        trend.get("topics", []),
        trend.get("competitor_tokens", []),
    )

    title = humanize_title(hybrid_keywords, trend_topics=trend.get("topics", []), vibe="excited_funny")

    hashtags = trend["tags"]
    tags = list(dict.fromkeys(hybrid_keywords[:20] + [h.replace("#", "") for h in hashtags]))[:30]

    description = (
        f"{title}\n\n"
        f"{trend['hook']}\n"
        f"Hybrid SEO: transcript + India Google trends + YouTube suggestions + competitor patterns.\n"
        f"Trending context: {', '.join(trend.get('topics', [])[:5])}\n\n"
        f"{' '.join(hashtags)}"
    )

    return {
        "clip_id": clip_id,
        "title": title[:100],
        "description": description[:5000],
        "tags": tags,
        "hashtags": hashtags,
        "trend_source": trend.get("source", "fallback"),
        "trend_topics": trend.get("topics", []),
        "competitor_tokens": trend.get("competitor_tokens", []),
        "keywords_local": local_keywords,
        "keywords_hybrid": hybrid_keywords,
    }


def process_all_seo(highlights_path: str, output_dir: str):
    h_path = Path(highlights_path)
    if not h_path.exists():
        log.error("Highlights not found: %s", h_path)
        return

    with open(h_path, "r", encoding="utf-8") as f:
        import yaml
        highlights = yaml.safe_load(f)

    log.info("Generating SEO for %d clips...", len(highlights))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for clip_id, info in highlights.items():
        text = info.get("text", "Cricket Live Highlights")
        seo_data = generate_seo(text, clip_id, domain="cricket", region="IN")

        seo_file = Path(output_dir) / f"{clip_id}_metadata.json"
        with open(seo_file, "w", encoding="utf-8") as f:
            json.dump(seo_data, f, indent=2, ensure_ascii=False)

    log.info("✅ SEO Metadata generated in %s", output_dir)


if __name__ == "__main__":
    process_all_seo("highlights/video.yaml", "shorts/test")
