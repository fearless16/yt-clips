"""
seo.py — Phase 6: Automated SEO Generation for YouTube Shorts.

Generates Titles, Descriptions, and Tags based on the transcript content.
"""
import json
import re
from pathlib import Path
from typing import List, Dict

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("seo", cfg["logging"]["log_file"], cfg["logging"]["level"])

# Common "filler" words to ignore when generating keywords
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

def generate_seo(clip_text: str, clip_id: str) -> Dict:
    """
    Generate high-CTR YouTube SEO metadata based on trends and transcript.
    """
    from trends import get_trending_context, humanize_title

    # Clean text and find keywords
    words = re.findall(r'\w+', clip_text.lower())
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 3]
    
    # Simple frequency analysis
    freq = {}
    for w in keywords:
        freq[w] = freq.get(w, 0) + 1
    sorted_keys = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [x[0] for x in sorted_keys[:10]]
    
    # Trend Analysis
    trend = get_trending_context()
    
    # Humanized Title
    title = humanize_title(top_keywords)
    
    # Tags & Hashtags
    hashtags = trend["tags"] + ["#Shorts", "#Cricket", "#Viral"]
    
    # Generate Description
    description = (
        f"{title}\n\n"
        f"{trend['hook']} Check out this epic Cricket moment! \n\n"
        f"Subscribe for daily highlights! 🏏🔥\n\n"
        f"{' '.join(hashtags)}"
    )
    
    return {
        "title": title[:100],
        "description": description,
        "tags": top_keywords + [t.replace('#', '') for t in hashtags],
        "hashtags": hashtags
    }

def process_all_seo(highlights_path: str, output_dir: str):
    """
    Generate SEO JSON files for all clips in a highlight file.
    """
    h_path = Path(highlights_path)
    if not h_path.exists():
        log.error(f"Highlights not found: {h_path}")
        return

    with open(h_path, 'r') as f:
        import yaml
        highlights = yaml.safe_load(f)

    log.info(f"Generating SEO for {len(highlights)} clips...")
    
    for clip_id, info in highlights.items():
        # Get the text for this clip segment (heuristically from the highlight if stored)
        # Note: In version 3 we will ideally pass the actual transcript slice here
        text = info.get("text", "Cricket Highlights") 
        
        seo_data = generate_seo(text, clip_id)
        
        seo_file = Path(output_dir) / f"{clip_id}_metadata.json"
        with open(seo_file, 'w') as f:
            json.dump(seo_data, f, indent=2)
            
    log.info(f"✅ SEO Metadata generated in {output_dir}")

if __name__ == "__main__":
    # Test run
    process_all_seo("highlights/video.yaml", "shorts/test")
