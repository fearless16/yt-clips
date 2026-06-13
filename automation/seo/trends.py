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

TEAM_MAPPINGS = {
    "csk": "CSK", "chennai": "CSK", "chennai super kings": "CSK",
    "mi": "MI", "mumbai": "MI", "mumbai indians": "MI",
    "rcb": "RCB", "bangalore": "RCB", "royal challengers": "RCB",
    "kkr": "KKR", "kolkata": "KKR", "kolkata knight riders": "KKR",
    "srh": "SRH", "hyderabad": "SRH", "sunrisers": "SRH",
    "dc": "DC", "delhi": "DC", "delhi capitals": "DC",
    "pbks": "PBKS", "pbk": "PBKS", "punjab": "PBKS",
    "rr": "RR", "rajasthan": "RR", "rajasthan royals": "RR",
    "lsg": "LSG", "lucknow": "LSG", "lucknow super giants": "LSG",
    "gt": "GT", "gujarat": "GT", "gujarat titans": "GT",
}


def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=4, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def fetch_own_live_stream_url(channel_id: str = "") -> str:
    """Fetch current live stream URL from YouTube channel."""
    if not channel_id:
        channel_id = cfg.get("youtube", {}).get("channel_id", "")
    if not channel_id:
        return ""
    try:
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from pathlib import Path
        token_path = cfg.get("youtube", {}).get("token_path", "yt_channel_token.json")
        if not Path(token_path).exists():
            return ""
        creds = Credentials.from_authorized_user_file(token_path)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if not creds:
            return ""
        youtube = build("youtube", "v3", credentials=creds)
        req = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            eventType="live",
            type="video",
        )
        res = req.execute()
        items = res.get("items", [])
        if items:
            video_id = items[0]["id"]["videoId"]
            return f"https://www.youtube.com/watch?v={video_id}"
    except Exception as e:
        log.warning("Live stream URL fetch failed: %s", e)
    return ""


def extract_match_teams(video_title: str) -> Tuple[List[str], str]:
    """Extract team names and match type from video title."""
    title_lower = video_title.lower()
    found = []
    for abbr, name in sorted(TEAM_MAPPINGS.items(), key=lambda x: -len(x[0])):
        if abbr in title_lower:
            found.append(name)
    match_type = "ipl"
    if "t20" in title_lower:
        match_type = "t20"
    elif "test" in title_lower:
        match_type = "test"
    elif "odi" in title_lower:
        match_type = "odi"
    return found, match_type


def get_rotated_hashtags(match_type: str = "ipl", seed: Optional[int] = None, domain: str = "cricket") -> List[str]:
    """Return a diverse set of hashtags for variety across clips based on domain."""
    if seed is not None:
        random.seed(seed)

    if domain == "football":
        fb_stars = ["#Mbappe", "#Ronaldo", "#Messi", "#Griezmann", "#Neymar"]
        generic = ["#FIFAWorldCup", "#WorldCup2026", "#Football", "#SoccerHighlights"]
        tags = ["#FIFA2026", "#Shorts"]
        star_tag = random.choice(fb_stars)
        generic_tag = random.choice(generic)
        tags.extend([star_tag, generic_tag])
        return tags
    elif domain == "general":
        generic = ["#Trending", "#Viral", "#Foryou", "#ShortsVideo"]
        tags = ["#Shorts"]
        generic_tag = random.choice(generic)
        tags.append(generic_tag)
        return tags
    else:  # cricket
        ipl_teams = ["#RCB", "#CSK", "#MI", "#KKR", "#SRH", "#DC", "#PBKS", "#RR", "#LSG", "#GT"]
        generic = ["#CricketShorts", "#IPLHighlights", "#T20Highlights"]
        tags = ["#IPL2026", "#Shorts"]
        team_tag = random.choice(ipl_teams)
        generic_tag = random.choice(generic)
        tags.extend([team_tag, generic_tag])
        return tags


def parse_cricbuzz_scorecard(html: str) -> str:
    """Parse Cricbuzz scorecard HTML into summary text."""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1")
    match_title = title_tag.get_text(strip=True) if title_tag else "Match"

    team_summaries = []
    for innings_div in soup.find_all("div", class_="cb-col cb-col-100 cb-ltst-wgt-hdr"):
        team_name_tag = innings_div.find("span", class_="cb-mat-total")
        innings_name = innings_div.find("span", class_="cb-text-gray")
        team_name = team_name_tag.get_text(strip=True) if team_name_tag else "Team"
        innings = innings_name.get_text(strip=True) if innings_name else ""

        top_scorers = []
        bowler_stats = []

        for score_row in innings_div.find_all("div", class_="cb-col cb-col-100 cb-scrd-itms"):
            cells = score_row.find_all("div", class_=re.compile(r"cb-col.*"))
            if len(cells) >= 3:
                player = cells[0].get_text(strip=True) if len(cells) > 0 else ""
                detail = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                runs_or_wickets = cells[2].get_text(strip=True) if len(cells) > 2 else ""

                if "run" in detail.lower() or "bowl" in detail.lower():
                    bowler_stats.append(f"{player} {detail} {runs_or_wickets}")
                elif "b" in detail or "c" in detail:
                    top_scorers.append(f"{player} {detail} {runs_or_wickets}")

        parts = [f"{team_name} {innings}"]
        if top_scorers:
            parts.append("  Top: " + ", ".join(top_scorers[:3]))
        if bowler_stats:
            parts.append("  Bowling: " + ", ".join(bowler_stats[:2]))
        team_summaries.append("\n".join(parts))

    return f"Match: {match_title}\n" + "\n\n".join(team_summaries)


def fetch_cricbuzz_live_score(query: str, match_type: str = "ipl") -> Dict:
    """Fetch live scorecard from Cricbuzz for a cricket match."""
    if not _cricbuzz_breaker.allow_request():
        return {"scorecard": "", "url": ""}
    try:
        search_url = "https://www.cricbuzz.com/cricket-match/live-scores"
        resp = _session().get(search_url, timeout=10)
        if resp.status_code == 200:
            parsed = parse_cricbuzz_scorecard(resp.text)
            _cricbuzz_breaker.record_success()
            return {"scorecard": parsed, "url": search_url}
        _cricbuzz_breaker.record_failure()
    except Exception as e:
        log.warning("Cricbuzz error: %s", e)
        _cricbuzz_breaker.record_failure()
    return {"scorecard": "", "url": ""}


def _extract_topics_from_rss(xml_text: str, max_topics: int = 12) -> List[str]:
    """Parse RSS XML into topic strings."""
    topics = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//entry", ns) or root.findall(".//item"):
            title_el = entry.find("title") or entry.find("title", ns)
            if title_el is not None and title_el.text:
                t = _clean(title_el.text)
                if t:
                    topics.append(t)
    except ET.ParseError:
        pass
    return topics[:max_topics]


def fetch_google_trends_in() -> List[str]:
    """Fetch Indian Google Trends RSS feed."""
    try:
        resp = _session().get(
            "https://trends.google.com/trending/rss?geo=IN", timeout=10
        )
        if resp.status_code == 200:
            return _extract_topics_from_rss(resp.text)
    except Exception as e:
        log.warning("Google Trends RSS error: %s", e)
    return []


def _fetch_google_trends_internal() -> List[str]:
    return fetch_google_trends_in()


def fetch_youtube_suggestions(seed_query: str = "cricket live") -> List[str]:
    """Fetch YouTube autocomplete suggestions for a seed query."""
    results = []
    base_queries = [
        f"{seed_query}", f"{seed_query} ipl", f"{seed_query} cricket",
        f"cricket {seed_query}", f"ipl {seed_query}",
    ]
    for query in base_queries:
        try:
            params = urllib.parse.urlencode({"client": "firefox", "ds": "yt", "q": query})
            resp = _session().get(f"https://suggestqueries.google.com/complete/search?{params}", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                suggestions = data[1] if len(data) > 1 else []
                for s in suggestions:
                    term = s[0] if isinstance(s, list) else s
                    if term not in results:
                        results.append(term)
        except Exception:
            continue
    return results[:30]


def _fetch_yt_suggest_internal(seed_query: str) -> List[str]:
    return fetch_youtube_suggestions(seed_query)


def fetch_clip_specific_suggestions(local_keywords: List[str]) -> List[str]:
    """Fetch YouTube suggestions based on clip-specific keywords."""
    all_suggestions = []
    player_queries = [kw for kw in local_keywords[:5]]
    for kw in player_queries:
        try:
            params = urllib.parse.urlencode({"client": "firefox", "ds": "yt", "q": f"{kw} cricket"})
            resp = _session().get(f"https://suggestqueries.google.com/complete/search?{params}", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                suggestions = data[1] if len(data) > 1 else []
                for s in suggestions:
                    term = s[0] if isinstance(s, list) else s
                    if term not in all_suggestions:
                        all_suggestions.append(term)
        except Exception:
            continue
    return all_suggestions[:20]


def fetch_enhanced_clip_suggestions(
    local_keywords: List[str],
    teams: List[str] = None,
    match_type: str = "ipl",
) -> List[str]:
    """Multi-question YouTube autocomplete focused on clip keywords."""
    questions = []
    teams = teams or []

    for kw in local_keywords[:5]:
        questions.append(f"{kw} {match_type}")
        questions.append(f"{kw} cricket")
    for t in teams:
        questions.append(f"{t} {match_type}")
        questions.append(f"{t} vs")
    questions.extend([
        f"{' vs '.join(teams[:2])}" if len(teams) >= 2 else "",
        f"{match_type} 2026 highlights",
        f"{match_type} live",
    ])
    questions = [q for q in questions if q.strip() and len(q) > 3]

    all_suggestions = []
    seen = set()
    for q in questions[:12]:
        try:
            params = urllib.parse.urlencode({"client": "firefox", "ds": "yt", "q": q})
            resp = _session().get(f"https://suggestqueries.google.com/complete/search?{params}", timeout=3)
            if resp.status_code != 200:
                continue
            data = resp.json()
            suggestions = data[1] if len(data) > 1 else []
            for s in suggestions[:8]:
                term = s[0] if isinstance(s, list) else s
                term_lower = term.lower()
                if term_lower not in seen:
                    seen.add(term_lower)
                    all_suggestions.append(term)
        except Exception:
            continue
    return all_suggestions[:30]


def _extract_tokens(texts: List[str], limit: int = 20) -> List[str]:
    """Extract common tokens from a list of texts."""
    tokens = []
    for t in texts:
        for word in t.split():
            cleaned = word.strip("#.,!?;:")
            if cleaned and len(cleaned) > 2:
                tokens.append(cleaned.lower())
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda x: -x[1])[:limit]]


def fetch_competitor_signals() -> List[str]:
    """Fetch competitor channel signals via search RSS."""
    try:
        resp = _session().get(
            "https://news.google.com/rss/search?q=cricket+live+score+today+IPL&hl=en-IN&gl=IN", timeout=10
        )
        if resp.status_code == 200:
            topics = _extract_topics_from_rss(resp.text, max_topics=20)
            return topics
    except Exception as e:
        log.warning("Competitor RSS error: %s", e)
    return []


def fetch_smart_search_summary(query: str) -> str:
    """Fetch a summary of trending topics for a search query."""
    try:
        param = urllib.parse.urlencode({"q": f"{query} cricket ipl 2026", "hl": "en-IN", "gl": "IN"})
        resp = _session().get(f"https://news.google.com/rss/search?{param}", timeout=10)
        if resp.status_code == 200:
            topics = _extract_topics_from_rss(resp.text, max_topics=8)
            if topics:
                return "\n".join(f"- {t}" for t in topics)
    except Exception as e:
        log.warning("Smart search error: %s", e)
    return ""


def fetch_match_scorecard(query: str) -> str:
    """Fetch a match scorecard summary from Cricbuzz."""
    if not _cricbuzz_breaker.allow_request():
        return ""
    try:
        search_url = f"https://www.cricbuzz.com/api/search/{urllib.parse.quote(query)}"
        resp = _session().get(search_url, timeout=10)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        matches = data.get("matches", [])
        if not matches:
            return ""
        match_id = matches[0].get("match_id", "")
        if not match_id:
            return ""

        scorecard_url = f"https://www.cricbuzz.com/api/cricket-match/scorecard/{match_id}"
        score_resp = _session().get(scorecard_url, timeout=10)
        if score_resp.status_code == 200:
            parsed = parse_cricbuzz_scorecard(score_resp.text)
            _cricbuzz_breaker.record_success()
            return parsed
    except Exception as e:
        log.warning("Scorecard fetch error: %s", e)
        _cricbuzz_breaker.record_failure()
    return ""


def detect_video_domain(video_title: str, transcript: str = "") -> Tuple[str, str, List[str]]:
    """Detect domain, primary topic/query, and keywords from title and transcript.

    Returns:
        Tuple[str, str, List[str]]: (domain, query, keywords)
    """
    text = f"{video_title} {transcript}".lower()
    
    football_kw = {
        "fifa", "world cup", "football", "mbappe", "ronaldo", "messi", 
        "griezmann", "soccer", "estadio azteca", "morocco", "france", 
        "argentina", "portugal", "jiménez", "south africa", "england roast", 
        "cricfy", "dai dai"
    }
    cricket_kw = {
        "cricket", "ipl", "t20", "odi", "test match", "wankhede", "chinnaswamy", 
        "rcb", "mi", "csk", "kohli", "rohit sharma", "bumrah", "dhoni", 
        "cricbuzz", "wicket", "run rate", "rinku singh", "padikkal", "patidar"
    }
    
    fb_count = sum(1 for kw in football_kw if kw in text)
    cr_count = sum(1 for kw in cricket_kw if kw in text)
    
    if fb_count > cr_count and fb_count > 0:
        domain = "football"
    elif cr_count > fb_count and cr_count > 0:
        domain = "cricket"
    else:
        if fb_count > 0:
            domain = "football"
        elif cr_count > 0:
            domain = "cricket"
        else:
            domain = "general"
            
    words = re.findall(r"\b[A-Za-z0-9]+\b", video_title)
    stopwords = {
        "vs", "live", "match", "today", "commentary", "highlights", "watchalong", 
        "watch", "along", "the", "and", "for", "with", "from", "shorts", "video", 
        "show", "epic", "dhamaka", "mein", "gayi", "phat", "ki", "ka", "ko", "ne", 
        "aur", "se", "bhi", "ke", "hai", "aaj", "ab", "is", "in", "it", "to", "effect"
    }
    
    keywords = []
    for w in words:
        wl = w.lower()
        if wl not in stopwords and len(w) > 2 and w not in keywords:
            keywords.append(w)
            
    if len(keywords) < 3 and transcript:
        t_words = re.findall(r"\b[A-Z][a-z]+\b", transcript)
        for tw in t_words:
            if tw.lower() not in stopwords and tw not in keywords:
                keywords.append(tw)
                
    if not keywords:
        keywords = ["Sports" if domain != "general" else "Trending"]
        
    query = " ".join(keywords[:3])
    return domain, query, keywords


def get_trending_context(domain: str = "cricket", region: str = "IN", video_title: str = "") -> Dict:
    """Aggregate all trend sources into a single context dict."""
    topics = []
    
    detected_domain = domain
    query_topic = f"{domain} live"
    teams = []
    match_type = "ipl"
    
    if video_title:
        detected_domain, query_topic, keywords = detect_video_domain(video_title)
        teams, match_type = extract_match_teams(video_title)
    
    trends = fetch_google_trends_in()
    topics.extend(trends)
    
    if video_title and query_topic:
        try:
            q_encoded = urllib.parse.quote_plus(f"{query_topic} live today")
            resp = _session().get(
                f"https://news.google.com/rss/search?q={q_encoded}&hl={en-IN if 'region' not in locals() else region}&gl={IN if 'region' not in locals() else region}", timeout=10
            )
            if resp.status_code == 200:
                topics.extend(_extract_topics_from_rss(resp.text, max_topics=20))
        except Exception as e:
            log.warning("Competitor RSS error: %s", e)
    else:
        competitor = fetch_competitor_signals()
        topics.extend(competitor)
        
    yt_suggestions = fetch_youtube_suggestions(query_topic)
    topics.extend(yt_suggestions)
    
    scorecard_data = {}
    if detected_domain == "cricket":
        scorecard_data = fetch_cricbuzz_live_score(video_title, match_type)
        
    live_stream_url = fetch_own_live_stream_url()
    
    seen = set()
    unique_topics = []
    for t in topics:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            unique_topics.append(t)
    top_topics = unique_topics[:20]
    
    return {
        "topics": top_topics,
        "scorecard": scorecard_data.get("scorecard", "") if scorecard_data else "",
        "live_stream_url": live_stream_url,
        "teams": teams,
        "domain": detected_domain,
    }


def humanize_title(raw_title: str) -> str:
    """Convert raw video title to SEO-friendly human-readable form."""
    title = raw_title.strip()
    title = re.sub(r"[#_/]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title[:120]
