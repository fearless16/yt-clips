"""seo.py — Hybrid Batch SEO generation for Indian cricket live Shorts."""
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

# Generic tags that should be filtered out - they waste tag slots
GENERIC_TAGS = {
    "cricket", "shorts", "viral", "trending", "youtube", "video", "sports",
    "highlight", "highlights", "amazing", "awesome", "incredible", "wow"
}


def inject_trend_topics_into_tags(
    base_tags: List[str], 
    trend_topics: List[str],
    player_name: str = ""
) -> List[str]:
    """
    Inject trending topics into tag list intelligently.
    
    Args:
        base_tags: Original tags from AI
        trend_topics: Trending topics to inject
        player_name: Optional player name for combination tags
        
    Returns:
        Combined list with no duplicates
    """
    if not trend_topics:
        return base_tags
    
    result = base_tags.copy()
    seen_lower = {tag.lower() for tag in result}
    
    for topic in trend_topics[:3]:  # Top 3 trends
        topic_lower = topic.lower()
        
        # Skip if already present
        if topic_lower in seen_lower:
            continue
        
        # Create combination tag with player name if available
        if player_name:
            combo_tag = f"{player_name} {topic_lower}"
            if combo_tag not in seen_lower:
                result.append(combo_tag)
                seen_lower.add(combo_tag)
        
        # Add plain topic as tag
        if topic_lower not in seen_lower:
            result.append(topic_lower)
            seen_lower.add(topic_lower)
    
    return result


def ensure_trend_in_title(title: str, trend_topics: List[str]) -> str:
    """
    Ensure title contains at least one trending topic.
    
    Args:
        title: Original title from AI
        trend_topics: List of trending topics
        
    Returns:
        Title with trend topic prepended if missing
    """
    if not trend_topics:
        return title[:100]
    
    # Check if any trend topic (or its key words) is already in title
    title_lower = title.lower()
    has_trend = False
    
    for trend in trend_topics:
        trend_lower = trend.lower()
        # Check full trend phrase
        if trend_lower in title_lower:
            has_trend = True
            break
        # Also check if main keywords from trend are present (e.g., "IPL 2024" in "IPL 2024 Playoffs")
        trend_words = trend_lower.split()
        if len(trend_words) >= 2:
            # Check if the first 2 significant words are in title
            main_keywords = ' '.join(trend_words[:2])
            if main_keywords in title_lower:
                has_trend = True
                break
    
    if not has_trend:
        # Prepend top trend topic
        top_trend = trend_topics[0]
        new_title = f"{top_trend}: {title}"
        return new_title[:100]
    
    return title[:100]


def validate_hinglish_content(text: str) -> Tuple[bool, str]:
    """
    Validate if content has proper Hindi/Hinglish mix.
    
    Args:
        text: Title or description text
        
    Returns:
        Tuple of (is_valid, language_mix_description)
    """
    hindi_words = {
        'ka', 'ki', 'ke', 'ko', 'mein', 'se', 'par', 'aur', 'hai', 'tha', 'thi',
        'dhamaakedaar', 'jabardast', 'shandaar', 'kya', 'yeh', 'toh', 'bhi',
        'maara', 'khela', 'dekho', 'bolo', 'arey', 'yaar', 'bhai', 'log'
    }
    
    words = text.lower().split()
    hindi_count = sum(1 for w in words if w in hindi_words)
    
    if hindi_count >= 2:
        return True, "hinglish"
    elif hindi_count == 1:
        return True, "light_hinglish"
    else:
        return True, "english"  # Still valid, just English


def validate_description_hooks(description: str) -> Tuple[bool, str]:
    """
    Validate if description has proper hook in first line.
    
    Args:
        description: Video description
        
    Returns:
        Tuple of (has_hook, hook_type)
    """
    if not description:
        return False, "none"
    
    first_line = description.split('\n')[0].lower()
    
    # Check for Hindi hooks
    hindi_hooks = ['kya', 'arey', 'dekh', 'bolo', 'yeh', 'kaise']
    if any(hook in first_line for hook in hindi_hooks):
        return True, "hindi"
    
    # Check for emotional hooks
    emotional_words = ['insane', 'unbelievable', 'crazy', 'amazing', 'wow', 'omg']
    if any(word in first_line for word in emotional_words):
        return True, "emotional"
    
    # Check for question hooks
    if '?' in first_line or '!' in first_line:
        return True, "question_or_exclamation"
    
    return True, "standard"  # Still acceptable


def validate_emoji_usage(text: str) -> bool:
    """
    Validate emoji usage (1-3 emojis optimal).
    
    Args:
        text: Title or description
        
    Returns:
        True if emoji count is acceptable (0-5), False if excessive (>5)
    """
    # Extended pattern to include dingbats and other symbol blocks
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F"  # Emoticons
        r"\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
        r"\U0001F680-\U0001F6FF"  # Transport and Map
        r"\U0001F1E0-\U0001F1FF"  # Flags
        r"\U00002700-\U000027BF"  # Dingbats (includes ✨)
        r"\U00002600-\U000026FF]" # Misc symbols
    )
    emojis = emoji_pattern.findall(text)
    
    # Allow 0-5 emojis, flag excessive usage (>5)
    return len(emojis) <= 5


def _validate_tags(tags: List[str], min_words: int = 2) -> List[str]:
    """
    Validate and filter tags to ensure specificity.
    
    Args:
        tags: List of tags to validate
        min_words: Minimum number of words required (prevents single generic words)
    
    Returns:
        Filtered list of specific, long-tail tags
    """
    if not tags:
        return []
    
    valid_tags = []
    for tag in tags:
        tag_lower = tag.lower().strip()
        
        # Skip empty tags
        if not tag_lower:
            continue
        
        # Count words in tag
        word_count = len(tag_lower.split())
        
        # Skip if too short (single word generics)
        if word_count < min_words:
            continue
        
        # Skip if tag is purely generic
        if tag_lower in GENERIC_TAGS:
            continue
        
        # Skip if tag contains only generic terms
        tag_words = set(tag_lower.split())
        if tag_words.issubset(GENERIC_TAGS):
            continue
        
        valid_tags.append(tag)
    
    return valid_tags


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

    # CRITICAL FIX: Enhanced system instruction for Hindi/Hinglish content
    system_instr = (
        "You are a senior cricket analyst and YouTube SEO expert for small Indian channels (140 subscribers). "
        "You specialize in Hinglish (Hindi+English mix) content for cricket audiences. "
        "You have deep knowledge of every IPL/International team, player stats, rivalries, and match situations. "
        "Your job: write metadata that proves deep cricket understanding while creating curiosity gaps that force viewers to click. "
        "Every title must feel like an insider insight, not a generic recap. Use Hinglish naturally where appropriate. "
        "Every description must show you know exactly what happened, why it mattered, and what the player/crowd felt. "
        "Use natural, passionate language with strategic emojis. Mix Hindi words like 'dhamaakedaar', 'jabardast', 'shandaar' where fitting."
    )
    
    # CRITICAL FIX: Inject trend topics directly into requirements to ensure AI uses them
    trend_topics_str = ', '.join(trend.get('topics', [])) if trend.get('topics') else 'None available'
    
    prompt = f"""
    Overall Match: {video_title}
    Scorecard snapshot: {trend.get('scorecard', '')}
    Trending Topics: {trend_topics_str}
    
    Generate high-impact YouTube Shorts metadata for the following {len(clips)} highlights:
    {json.dumps(clips_context, indent=2)}
    
    Requirements for EACH clip:
    1. **Title**: Curiosity-gap, high-CTR, max 100 characters. Must include specific player name(s) and a hint of drama or skill. 
       - INCORPORATE trending topics naturally if relevant (e.g., "IPL 2024", team names).
       - Use Hinglish where it adds authenticity (e.g., "Kohli ka Dhamaakedaar Six! 💥").
       - Example: "Kohli's Six Over Cover to Win It 💥 No One Believed!" or "Rohit ne maara winner six! 🔥"
    
    2. **Description**: Write a rich, 3-5 line description (80-120 words) that:
       - Opens with a short hook line (sensory/emotional) and an emoji. Use Hindi/Hinglish hooks like "Kya shot tha yaar!", "Believe nahi hoga!"
       - Explains the exact moment: what happened, who was bowling, the delivery, the shot, the context (chasing target, powerplay, etc.).
       - Adds a deeper insight: why this moment was special (stats, rivalry, pressure situation, player form).
       - End with a call-to-action that builds community (subscribe, comment on the next match, etc.) using a friendly Hinglish tone.
       - Integrate the match scorecard context naturally (e.g., "Chasing 200, RCB were 45/3 in the powerplay when…").
       - MUST include at least ONE trending topic from: {trend_topics_str}
       - Use 1-2 relevant hashtags at the end.
    
    3. **Tags**: 15-20 long-tail, ultra-specific tags. Format them as lowercase comma-separated keywords. 
       - Examples: "kohli six vs csk 2024, rcb run chase thriller, kohli cover drive, indian cricket shorts, dhoni reaction, ipl 2024 highlights"
       - NEVER use generic tags like "cricket", "shorts", "viral", "trending".
       - Include player names, team names, match context, and specific moments.
       - Mix Hindi terms: "kohli six hindi, cricket highlights hindi commentary"
    
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

    # 3. Finalize & Merge with trends - CRITICAL FIX: Validate tags and inject trends
    hashtags = trend.get("tags", ["#Shorts", "#Cricket"])
    trend_topics_list = trend.get("topics", [])
    
    final_results = []
    for c in clips:
        seo = next((item for item in results if item["clip_id"] == c["clip_id"]), {})
        
        # Get raw tags from AI and validate them
        raw_tags = seo.get("tags", [])
        validated_tags = _validate_tags(raw_tags, min_words=2)
        
        # Extract player name from title for combination tags
        import re as regex
        potential_names = regex.findall(r'\b[A-Z][a-z]+\b', seo.get("title", ""))
        player_name = potential_names[0].lower() if potential_names else ""
        
        # Inject trending topics into tags
        combined_tags = inject_trend_topics_into_tags(
            validated_tags, 
            trend_topics_list,
            player_name=player_name
        )
        
        # Ensure title has trending topic
        title = ensure_trend_in_title(seo.get("title", f"Cricket Highlights {c['clip_id']}"), trend_topics_list)
        
        final_results.append({
            "clip_id": c["clip_id"],
            "title": title,
            "description": seo.get("description", "")[:5000],
            "tags": combined_tags[:30],  # YouTube allows up to 500 chars total
            "hashtags": hashtags,
            "trend_topics": trend_topics_list,
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