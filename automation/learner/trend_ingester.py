"""trend_ingester.py — multi-source trend ingestion for the cricket learning engine.

Pulls cricket trends from free, no-quota sources:
  1. YouTube autocomplete (search suggestions) — no API quota needed
  2. ESPN Cricinfo RSS feed (top stories) — no API quota needed
  3. Hardcoded fixtures + recent results — no network needed

Each source returns ``TrendInput`` objects that can be passed to
``TrendEngine.ingest()``. Run via:

    .venv/bin/python -m automation.learner.trend_ingester

Or programmatically:
    from automation.learner.trend_ingester import TrendIngester
    ingester = TrendIngester(state_store)
    trends = ingester.ingest_all()
"""

import json
import logging
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

from automation.learner.trend_engine import TrendInput, HALF_LIVES
from automation.learner.entity_learner import CRICKET_ENTITIES


INTERNATIONAL_TEAMS = {
    "india": ["india", "indian", "team india", "men in blue"],
    "australia": ["australia", "aussie", "aussies", "aus"],
    "england": ["england", "english"],
    "south_africa": ["south africa", "proteas", "sa "],
    "new_zealand": ["new zealand", "blackcaps", "kiwis", "nz"],
    "pakistan": ["pakistan", "pak", "shaheens", "green shirts"],
    "sri_lanka": ["sri lanka", "sri", "lanka", "lions"],
    "bangladesh": ["bangladesh", "tigers", "bangla"],
    "west_indies": ["west indies", "windies", "caribbean"],
    "afghanistan": ["afghanistan"],
}


log = logging.getLogger(__name__)


def _word_match(keyword: str, text: str) -> bool:
    kw = keyword.rstrip()
    if not kw:
        return False
    return bool(re.search(r'\b' + re.escape(kw) + r'\b', text))


YOUTUBE_AUTOCOMPLETE_URL = (
    "https://suggestqueries.google.com/complete/search"
    "?client=youtube&ds=yt&q={query}"
)
ESPN_CRICKINFO_RSS_URL = "https://www.espncricinfo.com/rss/content/story/feeds/6.xml"
GOOGLE_TRENDS_EXPLORE_URL = "https://trends.google.com/trends/explore"
GOOGLE_TRENDS_WIDGET_BASE = "https://trends.google.com/trends/api/widgetdata"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


CRICKET_SEED_QUERIES = [
    "cricket highlights",
    "IPL 2026",
    "T20 World Cup 2026",
    "India cricket",
    "Bumrah bowling",
    "Virat Kohli",
    "Rohit Sharma",
    "Mumbai Indians",
    "Chennai Super Kings",
    "cricket viral video",
    "cricket funny moment",
    "cricket controversy",
    "cricket today match",
    "cricket live",
]


GOOGLE_TRENDS_KEYWORDS = [
    "cricket",
    "IPL",
    "T20 World Cup",
    "Bumrah",
    "Virat Kohli",
]


UPCOMING_FIXTURES = [
    {
        "trend_id": "fixture-ipl-2026-final",
        "query": "IPL 2026 Final",
        "category": "match",
        "entities": {"teams": ["rcb", "gt"], "players": [], "series": ["ipl"]},
        "trend_score": 0.85,
        "velocity": 0.7,
    },
    {
        "trend_id": "fixture-t20-wc-2026",
        "query": "T20 World Cup 2026",
        "category": "tournament",
        "entities": {"teams": [], "players": [], "series": ["t20i"]},
        "trend_score": 0.75,
        "velocity": 0.4,
    },
    {
        "trend_id": "fixture-ind-vs-aus-test",
        "query": "India vs Australia Test Series",
        "category": "series",
        "entities": {"teams": [], "players": [], "series": ["test"]},
        "trend_score": 0.65,
        "velocity": 0.3,
    },
]


RECENT_RESULTS = [
    {
        "trend_id": "result-ipl-2026-final-gt-wins",
        "query": "GT beat RCB IPL 2026 Final",
        "category": "match_result",
        "entities": {"teams": ["gt", "rcb"], "players": [], "series": ["ipl"]},
        "trend_score": 0.80,
        "velocity": 0.6,
    },
    {
        "trend_id": "result-t20-wc-india-wins",
        "query": "India wins T20 World Cup 2026",
        "category": "match_result",
        "entities": {"teams": [], "players": [], "series": ["t20i"]},
        "trend_score": 0.70,
        "velocity": 0.5,
    },
]


class TrendIngester:
    """Multi-source trend ingester. Aggregates from YouTube, ESPN, fixtures, results."""

    def __init__(self, state_store, http_timeout: float = 8.0):
        """Initialize TrendIngester.

        Args:
            state_store: PersistentStateStore instance
            http_timeout: HTTP request timeout in seconds
        """
        self._state = state_store
        self._timeout = http_timeout

    def ingest_all(self) -> list[TrendInput]:
        """Ingest trends from all sources. Returns list of TrendInput created.

        Sources (in order):
          1. UPCOMING_FIXTURES (always available)
          2. RECENT_RESULTS (always available)
          3. YouTube autocomplete (network)
          4. ESPN Cricinfo RSS (network)
          5. Google Trends (network, two-step explore + widget)
        """
        trends: list[TrendInput] = []

        trends.extend(self._build_fixture_trends())
        log.info("Built %d fixture trends", len(trends))

        result_count = len(trends)
        trends.extend(self._build_recent_result_trends())
        log.info("Built %d recent-result trends", len(trends) - result_count)

        try:
            yt_count = len(trends)
            trends.extend(self._fetch_youtube_autocomplete_trends())
            log.info("Built %d YouTube trends", len(trends) - yt_count)
        except Exception as e:
            log.warning("YouTube autocomplete ingestion failed: %s", e)

        try:
            rss_count = len(trends)
            trends.extend(self._fetch_espn_cricinfo_trends())
            log.info("Built %d ESPN Cricinfo trends", len(trends) - rss_count)
        except Exception as e:
            log.warning("ESPN Cricinfo RSS ingestion failed: %s", e)

        try:
            gt_count = len(trends)
            trends.extend(self._fetch_google_trends())
            log.info("Built %d Google Trends trends", len(trends) - gt_count)
        except Exception as e:
            log.warning("Google Trends ingestion failed: %s", e)

        return trends

    def _build_fixture_trends(self) -> list[TrendInput]:
        """Convert UPCOMING_FIXTURES to TrendInput objects."""
        trends = []
        now = datetime.now(timezone.utc)
        for fx in UPCOMING_FIXTURES:
            half_life = HALF_LIVES.get(fx["category"], 72.0)
            expires_at = (now + timedelta(hours=half_life * 2)).isoformat()
            trends.append(TrendInput(
                trend_id=fx["trend_id"],
                source="fixtures",
                query=fx["query"],
                trend_score=fx["trend_score"],
                velocity=fx["velocity"],
                category=fx["category"],
                entities=fx["entities"],
                half_life_hours=half_life,
                expires_at=expires_at,
            ))
        return trends

    def _build_recent_result_trends(self) -> list[TrendInput]:
        """Convert RECENT_RESULTS to TrendInput objects."""
        trends = []
        now = datetime.now(timezone.utc)
        for r in RECENT_RESULTS:
            half_life = HALF_LIVES.get(r["category"], 36.0)
            expires_at = (now + timedelta(hours=half_life * 2)).isoformat()
            trends.append(TrendInput(
                trend_id=r["trend_id"],
                source="results",
                query=r["query"],
                trend_score=r["trend_score"],
                velocity=r["velocity"],
                category=r["category"],
                entities=r["entities"],
                half_life_hours=half_life,
                expires_at=expires_at,
            ))
        return trends

    def _http_get(self, url: str) -> str:
        """Make HTTP GET request with user agent. Returns response text.

        Uses certifi's CA bundle if available, otherwise falls back to an
        unverified SSL context. macOS system Python often lacks proper CA
        certificates.
        """
        import ssl
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=self._timeout, context=ctx) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def _fetch_youtube_autocomplete_trends(self) -> list[TrendInput]:
        """Fetch YouTube search autocomplete suggestions for cricket queries.

        YouTube's autocomplete endpoint returns 10-15 related search terms per query.
        We deduplicate and extract entity mentions to build trends.
        """
        all_suggestions: list[str] = []
        for query in CRICKET_SEED_QUERIES:
            try:
                url = YOUTUBE_AUTOCOMPLETE_URL.format(
                    query=urllib.parse.quote(query)
                )
                raw = self._http_get(url)
                suggestions = self._parse_youtube_autocomplete(raw)
                all_suggestions.extend(suggestions)
                time.sleep(0.1)
            except (urllib.error.URLError, TimeoutError, Exception) as e:
                log.debug("YouTube autocomplete for '%s' failed: %s", query, e)
                continue

        unique = list(dict.fromkeys(all_suggestions))
        log.info("Got %d unique YouTube suggestions from %d queries",
                 len(unique), len(CRICKET_SEED_QUERIES))

        trends = []
        now = datetime.now(timezone.utc)
        for i, suggestion in enumerate(unique[:30]):
            entities = self._extract_entities_from_text(suggestion)
            if not entities["players"] and not entities["teams"]:
                continue

            category = self._categorize_query(suggestion)
            half_life = HALF_LIVES.get(category, 72.0)
            score = self._score_suggestion(suggestion)
            velocity = self._velocity_from_position(i, len(unique))

            trend_id = f"yt-suggest-{hash(suggestion) & 0xFFFFFFFF:08x}"
            expires_at = (now + timedelta(hours=half_life * 2)).isoformat()

            trends.append(TrendInput(
                trend_id=trend_id,
                source="youtube_autocomplete",
                query=suggestion,
                trend_score=score,
                velocity=velocity,
                category=category,
                entities=entities,
                half_life_hours=half_life,
                expires_at=expires_at,
            ))
        return trends

    def _parse_youtube_autocomplete(self, raw: str) -> list[str]:
        """Parse YouTube autocomplete response (JSONP or JSON).

        Returns a list of suggestion strings.
        """
        text = raw.strip()
        if text.startswith("window.google.ac.h("):
            text = text[text.index("(") + 1:text.rindex(")")]
        elif text.startswith("(") and text.endswith(")"):
            text = text[1:-1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []

        if isinstance(data, list) and len(data) >= 2:
            suggestions_raw = data[1]
            if isinstance(suggestions_raw, list):
                result = []
                for item in suggestions_raw:
                    if isinstance(item, list) and len(item) >= 1:
                        result.append(str(item[0]))
                    elif isinstance(item, str):
                        result.append(item)
                return result
        return []

    def _fetch_espn_cricinfo_trends(self) -> list[TrendInput]:
        """Fetch top stories from ESPN Cricinfo RSS feed."""
        raw = self._http_get(ESPN_CRICKINFO_RSS_URL)
        root = ET.fromstring(raw)

        trends = []
        now = datetime.now(timezone.utc)
        for i, item in enumerate(root.findall(".//item")[:15]):
            title_el = item.find("title")
            if title_el is None or not title_el.text:
                continue
            title = title_el.text.strip()
            entities = self._extract_entities_from_text(title)
            if not entities["players"] and not entities["teams"]:
                continue

            category = self._categorize_query(title)
            half_life = HALF_LIVES.get(category, 72.0)
            score = max(0.4, 0.9 - i * 0.05)
            velocity = max(0.0, 0.5 - i * 0.05)

            trend_id = f"espn-{hash(title) & 0xFFFFFFFF:08x}"
            expires_at = (now + timedelta(hours=half_life * 2)).isoformat()

            trends.append(TrendInput(
                trend_id=trend_id,
                source="espn_cricinfo",
                query=title,
                trend_score=score,
                velocity=velocity,
                category=category,
                entities=entities,
                half_life_hours=half_life,
                expires_at=expires_at,
            ))
        return trends

    def _fetch_google_trends(self) -> list[TrendInput]:
        """Fetch trends from Google Trends (YouTube property, 7-day window).

        Uses the 2-step pattern:
          1. GET the explore page to extract the widget token
          2. GET the widget data endpoint with the token to get related queries

        The user-provided reference URL is:
            https://trends.google.com/trends/explore?date=now%207-d&gprop=youtube&q=cricket

        Returns 0 trends on 429 (rate limited), 404, or any network error.
        Graceful degradation: never raises.
        """
        trends: list[TrendInput] = []
        for keyword in GOOGLE_TRENDS_KEYWORDS:
            try:
                trends.extend(self._fetch_google_trends_for_keyword(keyword))
                time.sleep(1.0)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    log.warning("Google Trends rate-limited (429) — skipping remaining keywords")
                    return trends
                log.debug("Google Trends HTTP %s for %r: %s", e.code, keyword, e)
                continue
            except (urllib.error.URLError, TimeoutError, Exception) as e:
                log.debug("Google Trends for %r failed: %s", keyword, e)
                continue
        return trends

    def _fetch_google_trends_for_keyword(self, keyword: str) -> list[TrendInput]:
        """Fetch related-queries trends for a single keyword via Google Trends.

        Two-step flow:
          1. GET explore page → extract widget tokens
          2. GET widget data endpoint with token → related queries (top + rising)
        """
        import ssl
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        params = urllib.parse.urlencode({
            "q": keyword,
            "date": "now 7-d",
            "gprop": "youtube",
        })
        explore_url = f"{GOOGLE_TRENDS_EXPLORE_URL}?{params}"
        req = urllib.request.Request(explore_url, headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        })
        with urllib.request.urlopen(req, timeout=self._timeout, context=ctx) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        token = self._extract_google_trends_token(html)
        if not token:
            log.debug("No Google Trends token found in explore page for %r", keyword)
            return []

        request_body = {
            "restriction": {"type": "COUNTRY", "countryId": "IN"},
            "keywordType": "ENTITY",
            "metric": ["TOP", "RISING"],
            "trendDepth": 7,
            "timeRange": "now 7-d",
            "gprop": "youtube",
            "comparisonItem": [{"keyword": keyword, "geo": {"country": "IN"}}],
            "category": 0,
        }
        widget_url = (
            f"{GOOGLE_TRENDS_WIDGET_BASE}/relatedsearches"
            f"?req={urllib.parse.quote(json.dumps(request_body))}"
            f"&token={urllib.parse.quote(token)}"
        )
        req2 = urllib.request.Request(widget_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req2, timeout=self._timeout, context=ctx) as resp2:
            raw = resp2.read().decode("utf-8", errors="replace")

        data = self._parse_google_trends_widget(raw)
        if not data:
            return []

        return self._build_google_trend_inputs(keyword, data)

    def _extract_google_trends_token(self, html: str) -> str | None:
        """Extract widget token from Google Trends explore HTML.

        Token is embedded in script data as "token":"..." or token":\\"...".
        """
        patterns = [
            r'"token"\s*:\s*"([^"]+)"',
            r"\\\"token\\\"\s*:\s*\\\"([^\\]+)\\\"",
        ]
        for pat in patterns:
            m = re.search(pat, html)
            if m:
                return m.group(1)
        return None

    def _parse_google_trends_widget(self, raw: str) -> dict | None:
        """Parse Google Trends widget data response.

        Strips the JSONP prefix `)]}',\\n` and parses the remaining JSON.
        """
        text = raw.strip()
        if text.startswith(")]}',"):
            text = text[5:].lstrip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _build_google_trend_inputs(
        self, seed_keyword: str, widget_data: dict
    ) -> list[TrendInput]:
        """Convert Google Trends widget data into TrendInput objects.

        widget_data structure (from Google Trends API):
        {
          "default": {
            "rankedList": [
              {"rankedKeyword": [{"query": "...", "value": int, "formattedValue": "..."}, ...]},
              {"rankedKeyword": [{"query": "...", "value": int, "link": "..."}, ...]}
            ]
          }
        }
        The first rankedList is TOP queries, the second is RISING queries.
        """
        trends: list[TrendInput] = []
        default = widget_data.get("default", {})
        ranked_lists = default.get("rankedList", [])
        if len(ranked_lists) < 1:
            return trends

        now = datetime.now(timezone.utc)
        for list_idx, ranked_list in enumerate(ranked_lists[:2]):
            is_rising = (list_idx == 1)
            for rank, kw in enumerate(ranked_list.get("rankedKeyword", [])[:10]):
                query = kw.get("query", "")
                if not query or query == seed_keyword:
                    continue
                value = kw.get("value", 0)
                formatted = kw.get("formattedValue", str(value))

                entities = self._extract_entities_from_text(query)
                if not entities["players"] and not entities["teams"]:
                    continue

                category = self._categorize_query(query)
                half_life = HALF_LIVES.get(category, 72.0)

                if is_rising:
                    if isinstance(value, (int, float)) and value > 0:
                        raw_score = min(1.0, 0.5 + math.log10(max(1, value)) * 0.15)
                    else:
                        raw_score = 0.6
                    velocity = min(0.8, max(0.0, raw_score * 0.6))
                else:
                    raw_score = min(1.0, 0.45 + rank * -0.02) if rank > 0 else 0.7
                    velocity = 0.2

                trend_id = f"gtrends-{hash(query) & 0xFFFFFFFF:08x}"
                expires_at = (now + timedelta(hours=half_life * 2)).isoformat()

                trends.append(TrendInput(
                    trend_id=trend_id,
                    source="google_trends",
                    query=query,
                    trend_score=raw_score,
                    velocity=velocity,
                    category=category,
                    entities=entities,
                    half_life_hours=half_life,
                    expires_at=expires_at,
                ))
        return trends

    def _extract_entities_from_text(self, text: str) -> dict:
        """Extract cricket entities (players, teams, series) from text.

        Returns dict with keys: players, teams, series
        """
        text_lower = text.lower()
        found = {"players": [], "teams": [], "series": []}
        for entity_type, entities in CRICKET_ENTITIES.items():
            for name, keywords in entities.items():
                if any(_word_match(kw, text_lower) for kw in keywords):
                    found[entity_type].append(name)

        for name, keywords in INTERNATIONAL_TEAMS.items():
            if any(_word_match(kw, text_lower) for kw in keywords):
                found["teams"].append(name)

        return found

    def _categorize_query(self, text: str) -> str:
        """Categorize a query for half-life assignment."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["live", "today", "now", "streaming"]):
            return "live_moment"
        if any(w in text_lower for w in ["won", "beats", "defeats", "wins", "result", "final"]):
            return "match_result"
        if any(w in text_lower for w in ["controversy", "scandal", "drama", "beef", "fight"]):
            return "controversy"
        if any(w in text_lower for w in ["world cup", "ipl", "t20", "bbl", "psl"]):
            return "tournament"
        if any(w in text_lower for w in ["select", "squad", "team announcement"]):
            return "selection"
        if any(w in text_lower for w in ["test", "odi"]):
            return "match"
        return "match"

    def _score_suggestion(self, suggestion: str) -> float:
        """Score a YouTube suggestion based on entity richness and recency hints.

        Returns a score 0.0-1.0.
        """
        text_lower = suggestion.lower()
        score = 0.5

        entity_count = sum(
            1 for word in text_lower.split()
            if any(kw == word or word.startswith(kw)
                   for ent in CRICKET_ENTITIES.values()
                   for kws in ent.values() for kw in kws)
        )
        score += min(0.3, entity_count * 0.1)

        if any(w in text_lower for w in ["2026", "2025", "live", "today"]):
            score += 0.15
        if "highlights" in text_lower or "viral" in text_lower:
            score += 0.1

        return min(1.0, max(0.1, score))

    def _velocity_from_position(self, position: int, total: int) -> float:
        """Higher velocity for earlier (more popular) suggestions."""
        if total == 0:
            return 0.0
        return max(0.0, 0.5 * (1.0 - position / total))

    def ingest_into_engine(self, engine) -> int:
        """Run ingest_all() and feed results into the given TrendEngine.

        Args:
            engine: TrendEngine instance

        Returns:
            Number of trends ingested
        """
        trends = self.ingest_all()
        for trend in trends:
            engine.ingest(trend)
        log.info("Ingested %d trends into TrendEngine", len(trends))
        return len(trends)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ingest trends into self_learner.db."""
    import argparse
    from automation.learner.state_store import PersistentStateStore
    from automation.learner.trend_engine import TrendEngine

    parser = argparse.ArgumentParser(
        description="Ingest cricket trends from multiple sources"
    )
    parser.add_argument(
        "--db", default="self_learner.db",
        help="Path to self_learner.db (default: ./self_learner.db)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be ingested without writing",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    state = PersistentStateStore(args.db)
    ingester = TrendIngester(state)
    trends = ingester.ingest_all()

    print(f"\n{'=' * 60}")
    print(f"TREND INGESTION — {len(trends)} trends collected")
    print(f"{'=' * 60}")

    by_source: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for t in trends:
        by_source[t.source] = by_source.get(t.source, 0) + 1
        by_category[t.category] = by_category.get(t.category, 0) + 1

    print(f"\nBy source:")
    for src, n in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {src:25s} {n:3d}")

    print(f"\nBy category:")
    for cat, n in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s} {n:3d}")

    print(f"\nTop 10 trends (by score):")
    sorted_trends = sorted(trends, key=lambda t: -t.trend_score)
    for t in sorted_trends[:10]:
        ents = t.entities
        ents_str = ", ".join(
            ents.get("players", [])[:2] + ents.get("teams", [])[:2]
        ) or "(no entities)"
        print(f"  [{t.trend_score:.2f}] {t.query} ({t.category}) — {ents_str}")

    if args.dry_run:
        print(f"\nDRY RUN — no writes performed")
        state.close()
        return 0

    engine = TrendEngine(state)
    ingested = ingester.ingest_into_engine(engine)
    print(f"\nIngested {ingested} trends into TrendEngine")
    state.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
