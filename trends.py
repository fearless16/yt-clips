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
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

from utils.config import load_config
from utils.logger import get_logger
from utils.resilience import CircuitBreaker, retry_with_backoff

cfg = load_config()
log = get_logger("trends", cfg["logging"]["log_file"], cfg["logging"]["level"])

# Circuit breakers for external APIs
_cricbuzz_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
_google_trends_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=120.0)
_yt_suggest_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

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
    """
    Extract structured match context from Cricbuzz match page HTML.
    Returns a detailed text summary including:
    - Venue, match status
    - Team scores with overs
    - Individual batter stats (runs, balls, 4s, 6s, SR)
    - Individual bowler stats (overs, runs, wickets, economy)
    - Current run rate, required rate, match phase
    """
    try:
        soup = BeautifulSoup(html, "lxml")
        parts = []

        # 1. Match status / header context
        status_el = soup.select_one(
            ".cb-min-stts, .cb-match-status, .cb-text-gray, "
            ".cb-col-100 .cb-text-complete, .cb-text-live, .cb-text-preview"
        )
        if status_el:
            status_text = status_el.get_text(strip=True)
            if status_text:
                parts.append(f"Status: {status_text}")

        # 2. Venue extraction
        venue_el = soup.select_one(
            ".cb-match-venue, .cb-text-gray, "
            ".cb-col-100 .cb-text-gray, [class*=venue]"
        )
        if not venue_el:
            venue_patterns = re.findall(
                r"at\s+([A-Za-z\s]+(?:Stadium|Ground|Oval|Park|Sports\s+Complex|Cricket\s+Ground))",
                html
            )
            if venue_patterns:
                parts.append(f"Venue: {venue_patterns[0].strip()}")
        else:
            vt = venue_el.get_text(strip=True)
            if vt and len(vt) > 5:
                parts.append(f"Venue: {vt}")

        # 3. Team scores (primary scorecard area)
        score_divs = soup.select(
            ".cb-hm-scg-bat-txt, .cb-hm-scg-bwl-txt, .cb-sc-hm-runs, "
            ".cb-ovr-num, .cb-font-16, .cb-col-100, .team-score, .ui-score "
        )
        for div in score_divs[:6]:
            text = div.get_text(strip=True)
            if text and len(text) > 3:
                parts.append(text)

        # 4. Full batting scorecard — individual player stats
        batting_table = soup.select_one("table.cb-table, .cb-series-summary table, .cb-score-bat")
        if batting_table:
            rows = batting_table.find_all("tr")
            batter_stats = []
            for row in rows[1:6]:  # Top 5 batters
                cells = row.find_all("td")
                if len(cells) >= 7:
                    name = cells[0].get_text(strip=True) if cells[0] else ""
                    runs = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    balls = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                    fours = cells[5].get_text(strip=True) if len(cells) > 5 else ""
                    sixes = cells[6].get_text(strip=True) if len(cells) > 6 else ""
                    sr = cells[7].get_text(strip=True) if len(cells) > 7 else ""
                    if name and runs:
                        detail = f"{name}: {runs}({balls}b {fours}x4 {sixes}x6 SR:{sr})"
                        batter_stats.append(detail)
            if batter_stats:
                parts.append("Batters: " + " | ".join(batter_stats))
        else:
            # Fallback: use compact batting display
            batting_section = soup.select_one(".cb-min-bat-runs, .cb-min-itm-rw")
            if batting_section:
                rows = batting_section.find_all("div", class_="cb-col cb-col-100")
                for row in rows[:4]:
                    cells = row.find_all(["div", "span"], class_=re.compile(r"cb-col-\d+"))
                    btext = " ".join(c.get_text(strip=True) for c in cells if c.get_text(strip=True))
                    if btext and len(btext) > 5:
                        batter_stats.append(btext)
                if batter_stats:
                    parts.append("Batting: " + " | ".join(batter_stats))

        # 5. Full bowling scorecard — individual bowler stats
        bowling_table = soup.select_one(".cb-score-bwl table, .cb-series-summary table + table")
        if bowling_table:
            rows = bowling_table.find_all("tr")
            bowler_stats = []
            for row in rows[1:5]:  # Top 4 bowlers
                cells = row.find_all("td")
                if len(cells) >= 7:
                    name = cells[0].get_text(strip=True) if cells[0] else ""
                    overs = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                    runs = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                    wickets = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    econ = cells[5].get_text(strip=True) if len(cells) > 5 else ""
                    if name and runs:
                        detail = f"{name}: {wickets}-{runs}({overs}ov econ:{econ})"
                        bowler_stats.append(detail)
            if bowler_stats:
                parts.append("Bowlers: " + " | ".join(bowler_stats))
        else:
            # Fallback: compact bowling display
            bowling_section = soup.select_one(".cb-min-bwl-figures, .cb-min-itm-rw")
            if bowling_section:
                rows = bowling_section.find_all("div", class_="cb-col cb-col-100")
                for row in rows[:4]:
                    cells = row.find_all(["div", "span"], class_=re.compile(r"cb-col-\d+"))
                    btext = " ".join(c.get_text(strip=True) for c in cells if c.get_text(strip=True))
                    if btext and len(btext) > 5:
                        bowler_stats.append(btext)
                if bowler_stats:
                    parts.append("Bowling: " + " | ".join(bowler_stats))

        # 6. Recent overs / commentary
        recent = soup.select_one(
            ".cb-min-rcnt, .cb-col-100.cb-min-rcnt, "
            ".cb-tms-bd-list, .cb-list-item"
        )
        if recent:
            rtext = recent.get_text(strip=True)[:300]
            if rtext:
                parts.append(f"Recent: {rtext}")

        # 7. Match phase — target, overs remaining, required rate
        innings_info = soup.select_one(".cb-text-innings, .cb-sc-otr, .cb-ovr-num")
        if innings_info:
            itext = innings_info.get_text(strip=True)
            if itext:
                parts.append(f"Innings: {itext}")

        target_pattern = re.findall(r"target[:\s]*(\d+)", html, re.IGNORECASE)
        if target_pattern:
            parts.append(f"Target: {target_pattern[0]}")

        # CRR + RRR always useful
        crr_pattern = re.findall(r"CRR[:\s]*(\d+\.\d+)", html)
        if crr_pattern:
            parts.append(f"CRR: {crr_pattern[0]}")
        rrr_pattern = re.findall(r"Req\.? RR[:\s]*(\d+\.\d+)", html, re.IGNORECASE)
        if rrr_pattern:
            parts.append(f"Req RR: {rrr_pattern[0]}")

        # 8. Fallback: basic regex for scores + overs (if structured parse yielded nothing)
        if not parts:
            team_score_pattern = re.findall(
                r"([A-Z]{2,5}|[A-Za-z\s]{3,20})\s+(\d{1,3}[/-]\d{1,2})\s+\((\d{1,2}\.\d)\)",
                html
            )
            for ts in team_score_pattern[:2]:
                parts.append(f"{ts[0].strip()} {ts[1]} ({ts[2]} ov)")
            if crr_pattern:
                parts.append(f"CRR: {crr_pattern[0]}")
            if rrr_pattern:
                parts.append(f"Req RR: {rrr_pattern[0]}")

        combined = " | ".join(dict.fromkeys(parts))
        return combined[:1500] if combined else ""

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
        for card in soup.find_all("a", href=re.compile(r"/live-cricket-scores/"))[:10]:
            card_text = card.get_text().lower()
            for team in teams_in_query:
                if any(alias in card_text for alias in TEAM_MAPPINGS.get(team, [team.lower()])):
                    match_link = card.get("href")
                    break
            if match_link:
                break

        if not match_link:
            for card in soup.find_all("a", href=re.compile(r"/live-cricket-scores/"))[:5]:
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
    if _google_trends_breaker.state == "OPEN":
        log.warning("Circuit breaker OPEN for Google Trends, skipping...")
        return []
    try:
        result = _google_trends_breaker.call(_fetch_google_trends_internal)
        return result if result else []
    except Exception as e:
        log.warning("Google Trends IN fetch failed: %s", e)
        return []


def _fetch_google_trends_internal() -> List[str]:
    r = _session().get(GOOGLE_TRENDS_RSS_IN, timeout=6)
    r.raise_for_status()
    topics = _extract_topics_from_rss(r.text, max_topics=15)
    return topics if topics else []


def fetch_youtube_suggestions(seed_query: str = "cricket live") -> List[str]:
    if _yt_suggest_breaker.state == "OPEN":
        log.warning("Circuit breaker OPEN for YouTube Suggest, skipping...")
        return []
    try:
        result = _yt_suggest_breaker.call(_fetch_yt_suggest_internal, seed_query)
        return result if result else []
    except Exception as e:
        log.warning("YouTube suggestion fetch failed: %s", e)
        return []


def _fetch_yt_suggest_internal(seed_query: str) -> List[str]:
    url = YT_SUGGEST.format(q=urllib.parse.quote(seed_query))
    r = _session().get(url, timeout=6)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
        return [_clean(x) for x in data[1][:10] if _clean(x)]
    return []


CRICKET_KEYWORDS = {
    "kohli", "dhoni", "rohit", "gill", "raina", "sky", "pandya", "bumrah", "shami",
    "siraj", "jaddu", "jadeja", "ashwin", "chahal", "russell", "narine", "moeen",
    "maxwell", "faf", "rashid", "warner", "rahul", "pooran", "stoinis", "iyer",
    "shankar", "sundar", "kishan", "suryakumar", "sharma", "gaikwad", "dube",
    "rahane", "miller", "klassen", "chahar", "umesh", "deepak", "boult", "gayle",
    "abd", "virat", "sachin", "yuvi", "yuvraj", "starc", "cummins", "patel",
    "axar", "thakur", "krunal", "mayank", "pandey", "samson", "jaiswal", "tripathi",
    "rishabh", "pant", "dinesh", "karthik", "ishant", "rayudu", "harbhajan",
    "singh", "pathan", "irfan", "yusuf", "malik", "jordan", "watson", "buttler",
    "morgan", "stokes", "curran", "livingstone", "root", "six", "four", "wicket",
    "catch", "runout", "stump", "bowled", "lbw", "century", "half", "yorker",
    "bouncer", "fulltoss", "drive", "pull", "hook", "sweep", "reverse", "slog",
    "ipl", "t20", "cricket", "super", "final", "trophy", "cup", "match", "run",
    "score", "target", "chase", "win", "drama", "tension", "intense",
    "crazy", "unbelievable", "incredible", "amazing", "fantastic", "stunning",
    "shot", "bowling", "batting", "fielding", "captain", "umpire",
    "review", "decision", "controversy", "argument", "dismissal", "partnership",
}

def fetch_clip_specific_suggestions(local_keywords: List[str]) -> List[str]:
    """
    Ping YouTube Suggest API using specific entities found in the clip's transcript.
    Example: if keywords have 'kohli' and 'umpire', query 'kohli umpire' to find real-time 
    search spikes like 'kohli angry on umpire'.
    """
    if not local_keywords:
        return []
    
    # Only use cricket-relevant keywords, filter Whisper noise
    noise_words = {"oh", "ah", "ha", "he", "she", "it", "do", "go", "so", "yeah", "hey",
                   "come", "get", "got", "let", "put", "say", "see", "use", "way", "like",
                   "know", "take", "tell", "make", "think", "give", "will", "would", "could",
                   "should", "can", "may", "might", "shall", "now", "then", "just", "also",
                   "dumbing", "think", "him", "will", "video", "like", "chicken"}
    cricket_keywords = [k for k in local_keywords if k in CRICKET_KEYWORDS and k not in noise_words]
    if not cricket_keywords:
        log.info("No cricket-specific keywords found in clip — skipping YouTube Suggest ping")
        return []
        
    # Take top 2-3 significant keywords (e.g. players, events) to form a focused query
    focused_query = " ".join(cricket_keywords[:3])
        
    log.info("🔍 Pinging YouTube Suggest API for clip-specific keywords: '%s'", focused_query)
    
    try:
        url = YT_SUGGEST.format(q=urllib.parse.quote(focused_query))
        r = _session().get(url, timeout=4)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            suggestions = [_clean(x) for x in data[1][:5] if _clean(x)]
            log.info("🎯 Found specific search intents: %s", suggestions)
            return suggestions
    except Exception as e:
        log.warning("Clip-specific YouTube suggestion fetch failed for '%s': %s", focused_query, e)
        
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
    # Try alternative providers via Bing to bypass bot protection blocks on sites like Crex
    search_strategies = [
        f"site:crex.live {query} live score cricket",
        f"site:sportskeeda.com {query} live cricket score",
        f"{query} cricket match scorecard"
    ]
    
    session = _session()
    
    for search_query in search_strategies:
        try:
            url = f"https://www.bing.com/search?q={urllib.parse.quote(search_query)}"
            r = session.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=5)
            soup = BeautifulSoup(r.text, "html.parser")
            snippets = [
                s.get_text(strip=True)
                for s in soup.select(".b_algo p, .b_ans, .b_algo .b_vlist2col")
            ]
            if snippets:
                combined = " ".join(snippets[:2])
                # Only return if it looks like it actually found something relevant
                if len(combined) > 20:
                    log.info("✅ Alternative scorecard found via %s", search_query.split()[0])
                    return combined
        except Exception as e:
            log.debug("Smart search strategy '%s' failed: %s", search_query, e)
            continue
            
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
