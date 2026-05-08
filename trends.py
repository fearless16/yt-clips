"""
trends.py — Hybrid trend intelligence for Indian cricket Shorts.

Sources (live when available):
  1) Google Trends RSS (geo=IN)
  2) YouTube suggest API (ds=yt)
  3) Competitor channel query signals (title tokens from search RSS)
  4) Cricbuzz live scores (web scraping)
  5) [NEW] YouTube Data API / channel search for your own live stream URL
"""
import random
import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
from datetime import datetime
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
YT_SEARCH_RSS = "https://www.youtube.com/results?search_query={q}&sp=EgJAAQ%3D%3D"  # live filter
CRICBUZZ_BASE = "https://www.cricbuzz.com"
CRICBUZZ_SEARCH = "https://www.cricbuzz.com/cricket-match/live-scores"

COMPETITOR_CHANNELS = {
    "sports tak": "UCVXCo0W9pk2dDkEBNLhTt7A",
    "iqbal sports": "",
    "ab cricinfo": "",
    "sports yaari": "UCjFw-0Vdfy2KW78NClGECXw"
}

TEAM_MAPPINGS = {
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
    "INDIA": ["india", "ind", "team india"],
    "AUSTRALIA": ["australia", "aus", "aussies"],
    "ENGLAND": ["england", "eng"],
    "PAKISTAN": ["pakistan", "pak"],
    "SOUTH AFRICA": ["south africa", "sa"],
    "NEW ZEALAND": ["new zealand", "nz"],
    "WEST INDIES": ["west indies", "wi"],
    "AFGHANISTAN": ["afghanistan", "afg"],
    "SRI LANKA": ["sri lanka", "sl"],
    "BANGLADESH": ["bangladesh", "ban"],
}

MATCH_TYPE_KEYWORDS = {
    "ipl": ["ipl", "indian premier league", "t20 league"],
    "international": ["test", "odi", "t20i", "world cup", "champions trophy"],
    "t20": ["t20", "twenty20"],
    "domestic": ["ranji", "vijay hazare", "syed mushtaq ali"],
}

EXCITED_HOOKS = [
    "Arey yeh kya tha?! 😱", "Bhailog this was insane! 🔥", "Clutch moment alert 🚨",
    "Has has ke pagal ho jaoge 😂", "Unreal finish, full goosebumps! 🏏",
    "Ye over toh history ban gaya 👀",
]

BASE_HASHTAGS = ["#Shorts", "#ViralShorts", "#TrendingNow", "#CricketReels"]

HASHTAG_POOLS = {
    "ipl": [
        ["#IPL", "#IPL2026", "#TataIPL", "#Cricket", "#Shorts"],
        ["#IPLHighlights", "#CricketLovers", "#T20", "#ViratKohli", "#MSDhoni"],
        ["#RCB", "#CSK", "#MI", "#CricketFever", "#IPLMatches"],
        ["#Playoffs", "#Qualifier", "#Eliminator", "#Finals", "#CricketTime"],
    ],
    "international": [
        ["#INDvs", "#TeamIndia", "#Cricket", "#International", "#Shorts"],
        ["#TestCricket", "#ODI", "#T20I", "#WorldCup", "#BleedBlue"],
        ["#ViratKohli", "#RohitSharma", "#JaspritBumrah", "#CricketStars", "#IndVsAus"],
        ["#CricketMatch", "#LiveCricket", "#CricketFans", "#Sports", "#CricketLove"],
    ],
    "t20": [
        ["#T20", "#T20Cricket", "#Cricket", "#Shorts", "#CricketTime"],
        ["#BigHits", "#Sixes", "#CricketAction", "#T20Matches", "#CricketLovers"],
        ["#PowerHitting", "#FastBowling", "#CricketSkills", "#T20League", "#CricketFans"],
    ],
}


# ── HTTP session ───────────────────────────────────────────────────────────────

def _session() -> requests.Session:
    sess = requests.Session()
    sess.trust_env = False
    retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-IN,en;q=0.9",
    })
    return sess


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9\s]", " ", text)).strip()


# ── [NEW] Live stream URL detection ───────────────────────────────────────────

def fetch_own_live_stream_url(channel_id: str = "") -> str:
    """
    Detect your own active live stream URL.

    Priority:
      1. channel_id from config → scrape channel's /live page
      2. Fallback: config live_stream_url (static override)

    Add to config.yaml:
      channel:
        id: "UCxxxxxxxxxxxxxxxxx"       # your channel ID
        live_stream_url: ""             # static fallback if scraping fails
    """
    try:
        channel_cfg = cfg.get("channel", {})
        cid = channel_id or channel_cfg.get("id", "")
        static_url = channel_cfg.get("live_stream_url", "")

        if not cid:
            return static_url

        # YouTube channel live page reliably shows active stream
        live_page = f"https://www.youtube.com/channel/{cid}/live"
        r = _session().get(live_page, timeout=6)

        # Extract canonical watch URL from meta tags
        match = re.search(r'"canonicalBaseUrl":"(/watch\?v=[^"]+)"', r.text)
        if match:
            return f"https://www.youtube.com{match.group(1)}"

        # Fallback: og:url meta tag
        soup = BeautifulSoup(r.text, "lxml")
        og = soup.find("meta", property="og:url")
        if og and "watch" in (og.get("content") or ""):
            return og["content"]

        return static_url

    except Exception as e:
        log.warning("Live stream URL fetch failed: %s", e)
        return cfg.get("channel", {}).get("live_stream_url", "")


# ── Team / match extraction ────────────────────────────────────────────────────

def extract_match_teams(video_title: str) -> Tuple[List[str], str]:
    title_lower = video_title.lower()
    found_teams = []
    for team_code, aliases in TEAM_MAPPINGS.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", title_lower):
                if team_code not in found_teams:
                    found_teams.append(team_code)
                break

    match_type = "t20"
    if any(kw in title_lower for kw in MATCH_TYPE_KEYWORDS["international"]):
        match_type = "international"
    elif any(kw in title_lower for kw in MATCH_TYPE_KEYWORDS["ipl"]):
        match_type = "ipl"
    elif any(kw in title_lower for kw in MATCH_TYPE_KEYWORDS["domestic"]):
        match_type = "domestic"

    return found_teams[:2], match_type


def get_rotated_hashtags(match_type: str = "ipl", seed: Optional[int] = None) -> List[str]:
    pools = HASHTAG_POOLS.get(match_type, HASHTAG_POOLS["t20"])
    hour_index = (seed % len(pools)) if seed is not None else (datetime.now().hour % len(pools))
    base_tags = pools[hour_index].copy()
    extra_tags = random.sample(BASE_HASHTAGS, 2)
    seen, result = set(), []
    for tag in base_tags + extra_tags:
        if tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result[:8]


# ── Cricbuzz ──────────────────────────────────────────────────────────────────

def parse_cricbuzz_scorecard(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        score_divs = soup.select(
            ".cb-hm-scg-bat-txt, .cb-hm-scg-bwl-txt, .team-score, .ui-score, .cb-font-16"
        )
        scores = [d.get_text(strip=True) for d in score_divs[:2] if d.get_text(strip=True)]
        if scores:
            return " | ".join(scores)
        matches = re.findall(r"([A-Z]{2,4}|[A-Za-z]+)\s+(\d+/\d+)", html)
        if matches:
            return " | ".join(f"{m[0]} {m[1]}" for m in matches[:2])
    except Exception as e:
        log.warning("Cricbuzz parsing failed: %s", e)
    return ""


def fetch_cricbuzz_live_score(query: str, match_type: str = "ipl") -> Dict:
    try:
        session = _session()
        response = session.get(CRICBUZZ_SEARCH, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        teams_in_query, _ = extract_match_teams(query)
        match_link = None
        for card in soup.find_all("a", href=re.compile(r"/cricket-match/"))[:10]:
            card_text = card.get_text().lower()
            for team in teams_in_query:
                if any(alias in card_text for alias in TEAM_MAPPINGS.get(team, [team.lower()])):
                    match_link = card.get("href")
                    break
            if match_link:
                break

        if not match_link:
            for card in soup.find_all("a", href=re.compile(r"/cricket-match/"))[:5]:
                if "live" in card.get_text().lower():
                    match_link = card.get("href")
                    break

        if not match_link:
            return {"error": "No matching live match found", "scorecard": ""}

        match_url = f"{CRICBUZZ_BASE}{match_link}" if match_link.startswith("/") else match_link
        log.info("🏏 Fetching scorecard from: %s", match_url)
        mr = session.get(match_url, timeout=5)
        mr.raise_for_status()
        return {
            "scorecard": parse_cricbuzz_scorecard(mr.text),
            "match_url": match_url,
            "teams": teams_in_query,
            "match_type": match_type,
        }
    except Exception as e:
        log.warning("Cricbuzz fetch failed: %s", e)
        return {"error": str(e), "scorecard": ""}


# ── Trend sources ──────────────────────────────────────────────────────────────

def _extract_topics_from_rss(xml_text: str, max_topics: int = 12) -> List[str]:
    titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", xml_text)
    return [_clean(t) for t in titles[1 : max_topics + 1] if _clean(t)]


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
            if w not in stop:
                words[w] = words.get(w, 0) + 1
    return [k for k, _ in sorted(words.items(), key=lambda x: x[1], reverse=True)[:limit]]


def fetch_competitor_signals() -> List[str]:
    titles: List[str] = []
    for name, channel_id in COMPETITOR_CHANNELS.items():
        try:
            if channel_id:
                r = _session().get(YT_CHANNEL_RSS.format(id=channel_id), timeout=6)
                r.raise_for_status()
                root = ET.fromstring(r.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns)[:8]:
                    el = entry.find("atom:title", ns)
                    if el is not None and el.text:
                        titles.append(_clean(el.text))
            else:
                titles.extend(fetch_youtube_suggestions(f"{name} cricket")[:5])
        except Exception as e:
            log.warning("Competitor RSS failed for '%s': %s", name, e)
            titles.extend(fetch_youtube_suggestions(f"{name} cricket")[:5])
    return _extract_tokens(titles, limit=16)


def fetch_smart_search_summary(query: str) -> str:
    try:
        url = f"https://www.bing.com/search?q={urllib.parse.quote(query + ' cricket match scorecard')}"
        r = _session().get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        snippets = [
            s.get_text(strip=True)
            for s in soup.select(".b_algo p, .b_ans, .b_algo .b_vlist2col")
        ]
        return " ".join(snippets[:2]) if snippets else ""
    except Exception as e:
        log.warning("Smart search fallback failed: %s", e)
    return ""


def fetch_match_scorecard(query: str) -> str:
    teams, match_type = extract_match_teams(query)
    base_context = (
        f"{teams[0]} vs {teams[1]}" if len(teams) >= 2
        else (f"{teams[0]} in action" if teams else "")
    )
    if not base_context:
        return ""

    try:
        score_data = fetch_cricbuzz_live_score(query, match_type)
        if "error" not in score_data and score_data.get("scorecard"):
            full = f"{base_context}: {score_data['scorecard']}"
            log.info("✅ Live scorecard: %s", full[:80])
            return full
    except Exception as e:
        log.warning("Cricbuzz integration failed: %s", e)

    summary = fetch_smart_search_summary(query)
    if summary:
        return f"{base_context}: {summary[:200]}"
    return f"Match Context: {base_context}"


# ── Main entry ─────────────────────────────────────────────────────────────────

def get_trending_context(domain: str = "cricket", region: str = "IN", video_title: str = "") -> Dict:
    google_topics = fetch_google_trends_in() if region.upper() == "IN" else []
    yt_suggestions = fetch_youtube_suggestions("cricket live hindi")
    competitor_tokens = fetch_competitor_signals()

    scorecard = ""
    if domain == "cricket" and video_title:
        scorecard = fetch_match_scorecard(video_title)

    teams, match_type = extract_match_teams(video_title) if video_title else ([], "t20")
    hashtags = get_rotated_hashtags(match_type)

    # [NEW] Detect own live stream URL
    live_stream_url = fetch_own_live_stream_url()

    topics = list(dict.fromkeys(google_topics[:6] + yt_suggestions[:6] + competitor_tokens[:6]))

    return {
        "hook": random.choice(EXCITED_HOOKS),
        "tags": hashtags,
        "topics": topics,
        "competitor_tokens": competitor_tokens,
        "scorecard": scorecard,
        "match_type": match_type,
        "teams": teams,
        "live_stream_url": live_stream_url,          # ← passed to seo.py
        "source": (
            "google_trends_in+youtube_suggest+competitor_rss+cricbuzz"
            if scorecard else
            "google_trends_in+youtube_suggest+competitor_rss"
        ),
    }


def humanize_title(
    keywords: List[str],
    trend_topics: Optional[List[str]] = None,
    vibe: str = "excited_funny",
) -> str:
    hook = random.choice(EXCITED_HOOKS)
    main_topic = " ".join(k.strip().title() for k in keywords[:3] if k.strip()) or "Cricket Live"
    trend = f" | {trend_topics[0][:22]}" if trend_topics else ""
    templates = [
        f"{hook} {main_topic} moment nobody expected{trend}",
        f"{main_topic} turned comedy real quick 😂{trend}",
        f"{main_topic} clutch + funny reactions 🔥{trend}",
    ]
    return random.choice(templates)[:100]
