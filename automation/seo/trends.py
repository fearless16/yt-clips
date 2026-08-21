"""
trends.py — Hybrid trend intelligence for Indian cricket Shorts.

Sources (live when available):
  1) Google Trends RSS (geo=IN)
  2) YouTube suggest API (ds=yt)
  3) Competitor channel query signals (title tokens from search RSS)
  4) Cricbuzz live scores (web scraping)
  5) [NEW] YouTube Data API / channel search for your own live stream URL
"""
import json
import random
import re
import threading
import urllib.parse
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

from utils.config import load_config
from utils.logger import get_logger
from utils.resilience import CircuitBreaker
from automation.seo.cricket_context import (
    correct_cricket_spelling,
    discover_grounded_player_aliases,
    discover_grounded_player_names,
    find_canonical_entities,
    is_cricket_content,
)
from automation.seo.context_engine import build_grounded_search_queries

cfg = load_config()
log = get_logger("trends", cfg["logging"]["log_file"], cfg["logging"]["level"])

# Circuit breakers for external APIs
_cricbuzz_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
_http_local = threading.local()

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
    existing = getattr(_http_local, "session", None)
    if existing is not None:
        return existing
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=4, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131 Safari/537.36"
        )
    })
    _http_local.session = s
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
    corrected = correct_cricket_spelling(video_title)
    title_lower = corrected.lower()
    found = list(find_canonical_entities(corrected)["teams"])
    for abbr, name in sorted(TEAM_MAPPINGS.items(), key=lambda x: -len(x[0])):
        if abbr in title_lower and name not in found:
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


def parse_cricbuzz_search_results(html: str) -> str:
    """Return the first real Cricbuzz scorecard URL from DuckDuckGo HTML."""
    soup = BeautifulSoup(html or "", "html.parser")
    for anchor in soup.select("a.result__a"):
        href = str(anchor.get("href") or "")
        if href.startswith("//"):
            href = "https:" + href
        parsed = urllib.parse.urlparse(href)
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
        target = urllib.parse.unquote(target)
        target_url = urllib.parse.urlparse(target)
        host = (target_url.hostname or "").casefold()
        if (
            (host == "cricbuzz.com" or host.endswith(".cricbuzz.com"))
            and "live-cricket-scorecard" in target_url.path
        ):
            return target
    return ""


def parse_cricbuzz_match_page(html: str) -> Dict[str, List[str]]:
    """Parse current Next.js Cricbuzz pages into facts and player entities."""
    raw = html or ""
    normalized = raw.replace(r'\"', '"')
    soup = BeautifulSoup(raw, "html.parser")
    facts = []
    title_tag = soup.find("h1")
    if title_tag:
        title = _clean(title_tag.get_text(" ", strip=True))
        title = re.sub(r"\s+-\s+Scorecard.*$", "", title, flags=re.I)
        if title:
            facts.append(title)

    status_match = re.search(r'"status"\s*:\s*"([^"\\]+)', normalized)
    if status_match:
        status = _clean(status_match.group(1))
        if status and status not in facts:
            facts.append(status)

    players = []
    seen = set()
    for key in ("batName", "bowlName"):
        for name in re.findall(
            rf'"{key}"\s*:\s*"([^"\\]+)', normalized
        ):
            clean_name = _clean(name).removesuffix(" (c)").removesuffix(" (wk)")
            marker = clean_name.casefold()
            if clean_name and marker not in seen:
                seen.add(marker)
                players.append(clean_name)
    return {"facts": facts, "player_names": players}


@lru_cache(maxsize=128)
def _find_cricbuzz_scorecard_url(query: str) -> str:
    search = f"site:cricbuzz.com/live-cricket-scorecard {query}"
    params = urllib.parse.urlencode({"q": search})
    response = _session().get(
        f"https://html.duckduckgo.com/html/?{params}", timeout=10
    )
    if response.status_code != 200:
        return ""
    return parse_cricbuzz_search_results(response.text)


def _fetch_cricbuzz_match(query: str) -> Dict:
    url = _find_cricbuzz_scorecard_url(query)
    if not url:
        return {"facts": [], "player_names": [], "source_url": ""}
    response = _session().get(url, timeout=10)
    if response.status_code != 200:
        return {"facts": [], "player_names": [], "source_url": ""}
    parsed = parse_cricbuzz_match_page(response.text)
    return {**parsed, "source_url": url}


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


@lru_cache(maxsize=128)
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


def parse_youtube_search_titles(html: str, limit: int = 10) -> List[str]:
    """Extract video titles from YouTube's server-rendered search payload."""
    pattern = re.compile(
        r'"videoRenderer"\s*:\s*\{.*?"title"\s*:\s*\{\s*"runs"\s*:\s*'
        r'\[\s*\{\s*"text"\s*:\s*"((?:\\.|[^"\\])*)"',
        re.DOTALL,
    )
    titles = []
    seen = set()
    for encoded in pattern.findall(html or ""):
        try:
            title = json.loads(f'"{encoded}"')
        except (ValueError, json.JSONDecodeError):
            title = encoded.replace(r'\"', '"')
        title = _clean(title)
        if title and title.casefold() not in seen:
            seen.add(title.casefold())
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


@lru_cache(maxsize=128)
def fetch_youtube_search_signals(query: str, limit: int = 10) -> List[str]:
    """Fetch current YouTube result titles for query-specific search intent."""
    params = urllib.parse.urlencode({"search_query": query, "hl": "en", "gl": "IN"})
    response = _session().get(
        f"https://www.youtube.com/results?{params}",
        timeout=8,
    )
    if response.status_code != 200:
        return []
    return parse_youtube_search_titles(response.text, limit=limit)


def fetch_verified_match_context(query: str) -> Dict:
    """Fetch query-specific match facts and retain their source URL.

    This deliberately does not use the generic live-scores page: a random live
    match must never become evidence for an unrelated archived clip.
    """
    if not query.strip() or not _cricbuzz_breaker.allow_request():
        return {"facts": [], "player_names": [], "source_url": ""}
    try:
        match = _fetch_cricbuzz_match(query)
        if not match["facts"]:
            _cricbuzz_breaker.record_failure()
            return match
        _cricbuzz_breaker.record_success()
        return match
    except Exception as exc:
        log.warning("Verified match context error: %s", exc)
        _cricbuzz_breaker.record_failure()
        return {"facts": [], "player_names": [], "source_url": ""}


def _research_query(video_title: str, video_description: str, transcript: str) -> str:
    """Build a compact match lookup from all local source evidence."""
    clip_text = correct_cricket_spelling(transcript or "")
    clip_entities = find_canonical_entities(clip_text)
    combined = correct_cricket_spelling(" ".join((
        video_title or "", video_description or "", transcript or ""
    )))
    combined_entities = find_canonical_entities(combined)
    teams = list(clip_entities["teams"])
    if len(teams) < 2:
        teams.extend(team for team in combined_entities["teams"] if team not in teams)
    entities = {
        "teams": teams,
        "players": clip_entities["players"] or combined_entities["players"],
    }
    entity_text = " ".join(entities["teams"] + entities["players"][:2])
    format_terms = [
        term for term in ("Test", "ODI", "T20", "IPL", "World Cup", "series")
        if re.search(r"\b" + re.escape(term) + r"\b", combined, re.I)
    ]
    title = re.sub(r"\s+", " ", correct_cricket_spelling(video_title)).strip()
    parts = (
        [entity_text, " ".join(format_terms), "cricket"]
        if entity_text
        else [title, " ".join(format_terms)]
    )
    return re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()[:240]


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


def get_trending_context(
    domain: str = "cricket",
    region: str = "IN",
    video_title: str = "",
    video_description: str = "",
    transcript: str = "",
    include_live_stream_url: bool = True,
) -> Dict:
    """Build current, query-specific research context with provenance."""
    combined = " ".join((video_title, video_description, transcript))
    detected_domain, detected_query, _ = detect_video_domain(video_title, combined)
    if not video_title:
        detected_domain = domain
    elif domain == "cricket" and is_cricket_content(combined, combined):
        detected_domain = "cricket"
    teams, _match_type = extract_match_teams(combined)
    query_topic = _research_query(video_title, video_description, transcript) or detected_query

    try:
        suggestions = fetch_youtube_suggestions(query_topic)
    except Exception as exc:
        log.warning("YouTube suggest unavailable: %s", exc)
        suggestions = []
    try:
        recent_youtube_titles = fetch_youtube_search_signals(query_topic)
    except Exception as exc:
        log.warning("YouTube search unavailable: %s", exc)
        recent_youtube_titles = []

    match_facts = []
    player_names = []
    sources = []
    query_teams = find_canonical_entities(query_topic)["teams"]
    if detected_domain == "cricket" and len(set(query_teams)) == 2:
        try:
            match_context = fetch_verified_match_context(query_topic)
        except Exception as exc:
            log.warning("Match research unavailable: %s", exc)
            match_context = {"facts": [], "player_names": [], "source_url": ""}
        match_facts = list(match_context.get("facts") or [])
        player_names = list(match_context.get("player_names") or [])
        if match_context.get("source_url"):
            sources.append({
                "kind": "match",
                "url": match_context["source_url"],
                "query": query_topic,
            })
    if recent_youtube_titles:
        sources.append({
            "kind": "youtube_search",
            "url": "https://www.youtube.com/results?" + urllib.parse.urlencode({
                "search_query": query_topic,
            }),
            "query": query_topic,
        })

    # Global trends are allowed only when they overlap the source evidence.
    anchors = set(re.findall(r"[a-z0-9]+", query_topic.casefold()))
    try:
        global_topics = fetch_google_trends_in()
    except Exception:
        global_topics = []
    relevant_topics = [
        topic for topic in global_topics
        if set(re.findall(r"[a-z0-9]+", topic.casefold())) & anchors
    ]
    topics = list(dict.fromkeys([
        *suggestions,
        *recent_youtube_titles,
        *relevant_topics,
    ]))[:20]
    discovered_players = discover_grounded_player_names(
        combined, [*recent_youtube_titles, *suggestions]
    )
    player_names = list(dict.fromkeys([*player_names, *discovered_players]))
    player_aliases = discover_grounded_player_aliases(
        combined, [*recent_youtube_titles, *suggestions], player_names
    )
    search_queries = build_grounded_search_queries(
        video_title,
        video_description,
        transcript,
        suggestions,
        player_names,
        player_aliases,
    )

    return {
        "topics": topics,
        "scorecard": "\n".join(match_facts),
        "match_facts": match_facts,
        "player_names": player_names,
        "player_aliases": player_aliases,
        "search_queries": search_queries,
        "sources": sources,
        "research_query": query_topic,
        "live_stream_url": fetch_own_live_stream_url() if include_live_stream_url else "",
        "teams": teams,
        "domain": detected_domain,
        "region": region,
    }
