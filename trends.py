"""
trends.py — Hybrid trend intelligence for Indian cricket Shorts.

Sources (live when available):
  1) Google Trends RSS (geo=IN)
  2) YouTube suggest API (ds=yt)
  3) Competitor channel query signals (title tokens from search RSS)
  4) Cricbuzz live scores (web scraping)
"""
import random
import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import hashlib

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("trends", cfg["logging"]["log_file"], cfg["logging"]["level"])

GOOGLE_TRENDS_RSS_IN = "https://trends.google.com/trending/rss?geo=IN"
YT_SUGGEST = "https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&q={q}"
YT_CHANNEL_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={id}"
CRICBUZZ_BASE = "https://www.cricbuzz.com"
CRICBUZZ_SEARCH = "https://www.cricbuzz.com/cricket-match/live-scores"

COMPETITOR_CHANNELS = {
    "sports tak": "UCVXCo0W9pk2dDkEBNLhTt7A",
    "iqbal sports": "UC91500_n_hM-wzH4y9g2MmA",
    "ab cricinfo": "UCDp2t-2y-Wl-9J61t6001Ig",
    "sports yaari": "UCjFw-0Vdfy2KW78NClGECXw"
}

# Comprehensive team mappings for URL extraction
TEAM_MAPPINGS = {
    # IPL Teams
    "RCB": ["rcb", "royal challengers bangalore", "bangalore", "royal challengers"],
    "CSK": ["csk", "chennai super kings", "chennai"],
    "MI": ["mi", "mumbai indians", "mumbai"],
    "GT": ["gt", "gujarat titans", "gujarat"],
    "LSG": ["lsg", "lucknow super giants", "lucknow"],
    "SRH": ["srh", "sunrisers hyderabad", "hyderabad"],
    "DC": ["dc", "delhi capitals", "delhi"],
    "PBKS": ["pbks", "punjab kings", "punjab"],
    "RR": ["rr", "rajasthan royals", "rajasthan"],
    "KKR": ["kkr", "kolkata knight riders", "kolkata"],
    # International Teams
    "INDIA": ["india", "ind", "team india"],
    "AUSTRALIA": ["australia", "aus", "aussies"],
    "ENGLAND": ["england", "eng"],
    "PAKISTAN": ["pakistan", "pak"],
    "SOUTH AFRICA": ["south africa", "sa"],
    "NEW ZEALAND": ["new zealand", "nz"],
    "WEST INDIES": ["west indies", "wi"],
    "AFGHANISTAN": ["afghanistan", "afg"],
    "SRI LANKA": ["sri lanka", "sl"],
    "BANGLADESH": ["bangladesh", "ban"]
}

MATCH_TYPE_KEYWORDS = {
    "ipl": ["ipl", "indian premier league", "t20 league"],
    "international": ["test", "odi", "t20i", "world cup", "champions trophy"],
    "t20": ["t20", "twenty20"],
    "domestic": ["ranji", "vijay hazare", "syed mushtaq ali"]
}

EXCITED_HOOKS = [
    "Arey yeh kya tha?! 😱", "Bhailog this was insane! 🔥", "Clutch moment alert 🚨",
    "Has has ke pagal ho jaoge 😂", "Unreal finish, full goosebumps! 🏏",
    "Ye over toh history ban gaya 👀",
]

BASE_HASHTAGS = ["#Shorts", "#ViralShorts", "#TrendingNow", "#CricketReels"]
CRICKET_HASHTAGS = ["#Cricket", "#CricketHighlights", "#IPL", "#INDvs", "#SportsTak"]

# Rotating hashtag pools for variety
HASHTAG_POOLS = {
    "ipl": [
        ["#IPL", "#IPL2024", "#TataIPL", "#Cricket", "#Shorts"],
        ["#IPLHighlights", "#CricketLovers", "#T20", "#ViratKohli", "#MSDhoni"],
        ["#RCB", "#CSK", "#MI", "#CricketFever", "#IPLMatches"],
        ["#Playoffs", "#Qualifier", "#Eliminator", "#Finals", "#CricketTime"]
    ],
    "international": [
        ["#INDvs", "#TeamIndia", "#Cricket", "#International", "#Shorts"],
        ["#TestCricket", "#ODI", "#T20I", "#WorldCup", "#BleedBlue"],
        ["#ViratKohli", "#RohitSharma", "#JaspritBumrah", "#CricketStars", "#IndVsAus"],
        ["#CricketMatch", "#LiveCricket", "#CricketFans", "#Sports", "#CricketLove"]
    ],
    "t20": [
        ["#T20", "#T20Cricket", "#Cricket", "#Shorts", "#CricketTime"],
        ["#BigHits", "#Sixes", "#CricketAction", "#T20Matches", "#CricketLovers"],
        ["#PowerHitting", "#FastBowling", "#CricketSkills", "#T20League", "#CricketFans"]
    ]
}



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


def extract_match_teams(video_title: str) -> Tuple[List[str], str]:
    """
    Extract team names and match type from video title.
    
    Args:
        video_title: YouTube video title
        
    Returns:
        Tuple of (list of teams, match_type)
    """
    title_lower = video_title.lower()
    found_teams = []
    
    # Search for team mappings
    for team_code, aliases in TEAM_MAPPINGS.items():
        for alias in aliases:
            if re.search(rf"\b{alias}\b", title_lower):
                if team_code not in found_teams:
                    found_teams.append(team_code)
                break
    
    # Determine match type - check international FIRST (more specific), then ipl, then t20
    match_type = "t20"  # default
    
    # Check international first (test, odi, t20i, world cup are more specific)
    if any(kw in title_lower for kw in MATCH_TYPE_KEYWORDS["international"]):
        match_type = "international"
    elif any(kw in title_lower for kw in MATCH_TYPE_KEYWORDS["ipl"]):
        match_type = "ipl"
    elif any(kw in title_lower for kw in MATCH_TYPE_KEYWORDS["domestic"]):
        match_type = "domestic"
    elif any(kw in title_lower for kw in MATCH_TYPE_KEYWORDS["t20"]):
        match_type = "t20"
    
    return found_teams[:2], match_type  # Return max 2 teams


def get_rotated_hashtags(match_type: str = "ipl", seed: Optional[int] = None) -> List[str]:
    """
    Get rotating hashtags based on match type to avoid repetition.
    
    Args:
        match_type: Type of match (ipl, international, t20)
        seed: Optional seed for deterministic rotation (useful for A/B testing)
        
    Returns:
        List of 5-8 hashtags
    """
    pools = HASHTAG_POOLS.get(match_type, HASHTAG_POOLS["t20"])
    
    # Use seed or time-based rotation
    if seed is None:
        # Rotate every hour
        hour_index = datetime.now().hour % len(pools)
    else:
        hour_index = seed % len(pools)
    
    base_tags = pools[hour_index].copy()
    
    # Add some variety by mixing with base hashtags
    extra_tags = random.sample(BASE_HASHTAGS, 2)
    
    # Combine and deduplicate while preserving order
    seen = set()
    result = []
    for tag in base_tags + extra_tags:
        if tag not in seen:
            seen.add(tag)
            result.append(tag)
    
    return result[:8]  # Max 8 hashtags


def parse_cricbuzz_scorecard(html: str) -> str:
    """
    Parse live scorecard from Cricbuzz HTML.
    
    Args:
        html: Raw HTML from Cricbuzz
        
    Returns:
        Formatted scorecard string
    """
    try:
        soup = BeautifulSoup(html, 'lxml')
        
        # Find score sections
        score_divs = soup.find_all('div', class_='team-score')
        scores = []
        
        for div in score_divs[:2]:  # Max 2 teams
            text = div.get_text(strip=True)
            if text:
                scores.append(text)
        
        # Find current batsman/bowler if available
        batsman = soup.find('span', class_='name')
        batsman_info = ""
        if batsman:
            parent = batsman.find_parent()
            if parent:
                batsman_info = parent.get_text(strip=True)[:50]
        
        if scores:
            scorecard = " | ".join(scores)
            if batsman_info:
                scorecard += f" | {batsman_info}"
            return scorecard
        
        # Fallback: look for any score-like patterns
        score_pattern = r'([A-Z]{2,4}|[A-Za-z]+)\s+(\d+/\d+)'
        matches = re.findall(score_pattern, html)
        if matches:
            return " | ".join([f"{m[0]} {m[1]}" for m in matches[:2]])
            
    except Exception as e:
        log.warning(f"Cricbuzz parsing failed: {e}")
    
    return ""


def fetch_cricbuzz_live_score(query: str, match_type: str = "ipl") -> Dict:
    """
    Fetch live/recent match score from Cricbuzz.
    
    Args:
        query: Match query (e.g., "RCB vs CSK")
        match_type: Type of match
        
    Returns:
        Dict with scorecard, match_url, and metadata
    """
    try:
        # Step 1: Search for the match
        search_url = f"{CRICBUZZ_SEARCH}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        session = _session()
        response = session.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Find match links - look for team names in titles
        match_link = None
        teams_in_query, _ = extract_match_teams(query)
        
        # Search for match cards
        match_cards = soup.find_all('a', href=re.compile(r'/cricket-match/'))
        
        for card in match_cards[:10]:  # Check first 10 matches
            card_text = card.get_text().lower()
            
            # Check if any team from query is in this match
            for team in teams_in_query:
                team_aliases = TEAM_MAPPINGS.get(team, [team.lower()])
                if any(alias in card_text for alias in team_aliases):
                    match_link = card.get('href')
                    break
            
            if match_link:
                break
        
        if not match_link:
            # Fallback: use first live match
            for card in match_cards[:5]:
                if 'live' in card.get_text().lower():
                    match_link = card.get('href')
                    break
        
        if not match_link:
            return {"error": "No matching live match found", "scorecard": ""}
        
        # Step 2: Fetch match details page
        match_url = f"{CRICBUZZ_BASE}{match_link}" if match_link.startswith('/') else match_link
        log.info(f"🏏 Fetching scorecard from: {match_url}")
        
        match_response = session.get(match_url, headers=headers, timeout=10)
        match_response.raise_for_status()
        
        scorecard = parse_cricbuzz_scorecard(match_response.text)
        
        return {
            "scorecard": scorecard,
            "match_url": match_url,
            "teams": teams_in_query,
            "match_type": match_type
        }
        
    except Exception as e:
        log.warning(f"Cricbuzz fetch failed: {e}")
        return {"error": str(e), "scorecard": ""}


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
    Fetch a summary of the match scorecard from Cricbuzz.
    
    Args:
        query: Video title or match query
        
    Returns:
        Formatted scorecard string
    """
    log.info(f"🏏 Identifying match context for: {query}")
    
    # Extract teams and match type from query
    teams, match_type = extract_match_teams(query)
    
    if len(teams) >= 2:
        base_context = f"{teams[0]} vs {teams[1]}"
    elif len(teams) == 1:
        base_context = f"{teams[0]} in action"
    else:
        return ""
    
    # Try to fetch live score from Cricbuzz
    try:
        score_data = fetch_cricbuzz_live_score(query, match_type)
        
        if "error" not in score_data and score_data.get("scorecard"):
            full_context = f"{base_context}: {score_data['scorecard']}"
            log.info(f"✅ Live scorecard fetched: {full_context[:80]}...")
            return full_context
        else:
            log.warning("Cricbuzz fetch returned no scorecard, using basic context")
    except Exception as e:
        log.warning(f"Cricbuzz integration failed: {e}")
    
    # Fallback to basic context
    return f"Match Context: {base_context}"


def get_trending_context(domain: str = "cricket", region: str = "IN", video_title: str = "") -> Dict:
    google_topics = fetch_google_trends_in() if region.upper() == "IN" else []
    yt_suggestions = fetch_youtube_suggestions("cricket live hindi")
    competitor_tokens = fetch_competitor_signals()
    
    scorecard = ""
    if domain == "cricket" and video_title:
        scorecard = fetch_match_scorecard(video_title)

    # Use rotated hashtags based on match type
    teams, match_type = extract_match_teams(video_title) if video_title else ([], "t20")
    hashtags = get_rotated_hashtags(match_type)

    topics = list(dict.fromkeys(google_topics[:6] + yt_suggestions[:6] + competitor_tokens[:6]))

    return {
        "hook": random.choice(EXCITED_HOOKS),
        "tags": hashtags,
        "topics": topics,
        "competitor_tokens": competitor_tokens,
        "scorecard": scorecard,
        "match_type": match_type,
        "teams": teams,
        "source": "google_trends_in+youtube_suggest+competitor_rss+cricbuzz" if scorecard else "google_trends_in+youtube_suggest+competitor_rss",
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
