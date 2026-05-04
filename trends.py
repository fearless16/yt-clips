"""
trends.py — Hybrid trend intelligence for Indian cricket Shorts.

Sources (live when available):
  1) Google Trends RSS (geo=IN)
  2) YouTube suggest API (ds=yt)
  3) Competitor channel query signals (title tokens from search RSS)
"""
import random
import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("trends", cfg["logging"]["log_file"], cfg["logging"]["level"])

GOOGLE_TRENDS_RSS_IN = "https://trends.google.com/trending/rss?geo=IN"
YT_SUGGEST = "https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&q={q}"
YT_CHANNEL_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={id}"

# Map competitor names to their official YouTube Channel IDs (UC...)
# This avoids the 400 Bad Request errors from search-based RSS.
COMPETITOR_CHANNELS = {
    "sports tak": "UCVXCo0W9pk2dDkEBNLhTt7A",
    "iqbal sports": "UC91500_n_hM-wzH4y9g2MmA",
    "ab cricinfo": "UCDp2t-2y-Wl-9J61t6001Ig",
    "sports yaari": None,      # Fallback to Suggest API
}

EXCITED_HOOKS = [
    "Arey yeh kya tha?! 😱", "Bhailog this was insane! 🔥", "Clutch moment alert 🚨",
    "Has has ke pagal ho jaoge 😂", "Unreal finish, full goosebumps! 🏏",
    "Ye over toh history ban gaya 👀",
]

BASE_HASHTAGS = ["#Shorts", "#ViralShorts", "#TrendingNow", "#CricketReels"]
CRICKET_HASHTAGS = ["#Cricket", "#CricketHighlights", "#IPL", "#INDvs", "#SportsTak"]



def _session() -> requests.Session:
    """HTTP session tuned for public feed endpoints with retries and browser-like headers."""
    sess = requests.Session()
    sess.trust_env = False  # avoid broken proxy env causing 403 in some runtimes
    retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[429,500,502,503,504])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-IN,en;q=0.9",
    })
    return sess


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9\s]", " ", text)).strip()


def _extract_topics_from_rss(xml_text: str, max_topics: int = 12) -> List[str]:
    titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", xml_text)
    return [_clean(t) for t in titles[1:max_topics + 1] if _clean(t)]


def fetch_google_trends_in() -> List[str]:
    try:
        r = _session().get(GOOGLE_TRENDS_RSS_IN, timeout=6)
        r.raise_for_status()
        topics = _extract_topics_from_rss(r.text, max_topics=15)
        if topics:
            return topics
    except Exception as e:
        log.warning("Google Trends IN fetch failed: %s", e)
    return []


def fetch_youtube_suggestions(seed_query: str = "cricket live") -> List[str]:
    try:
        url = YT_SUGGEST.format(q=urllib.parse.quote(seed_query))
        r = _session().get(url, timeout=6)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            return [_clean(x) for x in data[1][:10] if _clean(x)]
    except Exception as e:
        log.warning("YouTube suggestion fetch failed: %s", e)
    return []


def _extract_tokens(texts: List[str], limit: int = 20) -> List[str]:
    words: Dict[str, int] = {}
    stop = {"live", "vs", "and", "the", "for", "with", "from", "today", "match", "cricket"}
    for t in texts:
        for w in re.findall(r"[A-Za-z0-9]{3,}", t.lower()):
            if w in stop:
                continue
            words[w] = words.get(w, 0) + 1
    return [k for k, _ in sorted(words.items(), key=lambda x: x[1], reverse=True)[:limit]]


def fetch_competitor_signals() -> List[str]:
    titles: List[str] = []
    for name, channel_id in COMPETITOR_CHANNELS.items():
        try:
            if channel_id:
                # Use official Channel RSS feed (Fast & Reliable)
                url = YT_CHANNEL_RSS.format(id=channel_id)
                r = _session().get(url, timeout=6)
                r.raise_for_status()
                root = ET.fromstring(r.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns)[:8]:
                    title_el = entry.find("atom:title", ns)
                    if title_el is not None and title_el.text:
                        titles.append(_clean(title_el.text))
            else:
                # No Channel ID? Fallback to Suggest API
                log.info("No ID for '%s', using Suggest API fallback.", name)
                titles.extend(fetch_youtube_suggestions(f"{name} cricket")[:5])

        except Exception as e:
            log.warning("Competitor RSS failed for '%s': %s. Falling back to suggest API.", name, e)
            titles.extend(fetch_youtube_suggestions(f"{name} cricket")[:5])
    return _extract_tokens(titles, limit=16)


def fetch_match_scorecard(query: str) -> str:
    """
    Fetch a summary of the match scorecard.
    In a production environment, this calls a Cricket API.
    Agentic Tip: Use search_web + read_url_content to get live scores for {query}.
    """
    log.info(f"🏏 Identifying match context for: {query}")
    
    # Comprehensive team list (International + IPL)
    teams_list = [
        "RCB", "CSK", "MI", "GT", "LSG", "SRH", "DC", "PBKS", "RR", "KKR",
        "INDIA", "AUSTRALIA", "ENGLAND", "PAKISTAN", "SOUTH AFRICA", "NEW ZEALAND", 
        "WEST INDIES", "AFGHANISTAN", "SRI LANKA", "BANGLADESH", "IND", "AUS", "ENG", "PAK", "SA", "NZ", "WI", "AFG", "SL", "BAN"
    ]
    
    found_teams = []
    q_upper = query.upper()
    for team in teams_list:
        if re.search(rf"\b{team}\b", q_upper):
            found_teams.append(team)
    
    # Remove duplicates (e.g., INDIA and IND)
    found_teams = list(dict.fromkeys(found_teams))
    
    if len(found_teams) >= 2:
        return f"Live Match Context: {found_teams[0]} vs {found_teams[1]}. (Refine with search for live score)."
    elif len(found_teams) == 1:
        return f"Match Context: {found_teams[0]} in action. (Refine with search)."
        
    return ""


def get_trending_context(domain: str = "cricket", region: str = "IN", video_title: str = "") -> Dict:
    google_topics = fetch_google_trends_in() if region.upper() == "IN" else []
    yt_suggestions = fetch_youtube_suggestions("cricket live hindi")
    competitor_tokens = fetch_competitor_signals()
    
    scorecard = ""
    if domain == "cricket" and video_title:
        scorecard = fetch_match_scorecard(video_title)

    hashtags = BASE_HASHTAGS + CRICKET_HASHTAGS if domain in {"cricket", "sports"} else BASE_HASHTAGS
    hashtags = list(dict.fromkeys(hashtags))[:8]

    topics = list(dict.fromkeys(google_topics[:6] + yt_suggestions[:6] + competitor_tokens[:6]))

    return {
        "hook": random.choice(EXCITED_HOOKS),
        "tags": hashtags,
        "topics": topics,
        "competitor_tokens": competitor_tokens,
        "scorecard": scorecard,
        "source": "google_trends_in+youtube_suggest+competitor_rss" if topics else "fallback",
    }


def humanize_title(keywords: List[str], trend_topics: List[str] | None = None, vibe: str = "excited_funny") -> str:
    hook = random.choice(EXCITED_HOOKS)
    main_topic = " ".join([k.strip().title() for k in keywords[:3] if k.strip()]) or "Cricket Live"
    trend = f" | {trend_topics[0][:22]}" if trend_topics else ""

    templates = [
        f"{hook} {main_topic} moment nobody expected{trend}",
        f"{main_topic} turned comedy real quick 😂{trend}",
        f"{main_topic} clutch + funny reactions 🔥{trend}",
    ]
    return random.choice(templates)[:100]
