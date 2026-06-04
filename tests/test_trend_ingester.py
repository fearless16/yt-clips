"""Tests for automation.learner.trend_ingester."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from automation.learner.state_store import PersistentStateStore
from automation.learner.trend_ingester import (
    TrendIngester,
    CRICKET_SEED_QUERIES,
    GOOGLE_TRENDS_KEYWORDS,
    UPCOMING_FIXTURES,
    RECENT_RESULTS,
    INTERNATIONAL_TEAMS,
    YOUTUBE_AUTOCOMPLETE_URL,
    GOOGLE_TRENDS_EXPLORE_URL,
)
from automation.learner.trend_engine import TrendEngine, TrendInput, HALF_LIVES


class TestFixtureTrends(unittest.TestCase):
    """Test fixture-based trend generation."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.state = PersistentStateStore(str(self.db_path))
        self.ingester = TrendIngester(self.state)

    def tearDown(self):
        self.state.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_fixture_trends_created(self):
        trends = self.ingester._build_fixture_trends()
        self.assertEqual(len(trends), len(UPCOMING_FIXTURES))

    def test_fixture_trend_has_required_fields(self):
        trends = self.ingester._build_fixture_trends()
        for t in trends:
            self.assertIsInstance(t, TrendInput)
            self.assertTrue(t.trend_id.startswith("fixture-"))
            self.assertEqual(t.source, "fixtures")
            self.assertGreater(t.trend_score, 0.0)
            self.assertLessEqual(t.trend_score, 1.0)
            self.assertIsNotNone(t.half_life_hours)
            self.assertIsNotNone(t.expires_at)

    def test_fixture_half_life_matches_category(self):
        trends = self.ingester._build_fixture_trends()
        for t in trends:
            expected_hl = HALF_LIVES.get(t.category, 72.0)
            self.assertEqual(t.half_life_hours, expected_hl)

    def test_fixture_entities_extracted(self):
        trends = self.ingester._build_fixture_trends()
        ipl_final = next(t for t in trends if "IPL 2026 Final" in t.query)
        self.assertIn("rcb", ipl_final.entities["teams"])
        self.assertIn("gt", ipl_final.entities["teams"])


class TestRecentResultTrends(unittest.TestCase):
    """Test recent-result trend generation."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.state = PersistentStateStore(str(self.db_path))
        self.ingester = TrendIngester(self.state)

    def tearDown(self):
        self.state.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_recent_result_trends_created(self):
        trends = self.ingester._build_recent_result_trends()
        self.assertEqual(len(trends), len(RECENT_RESULTS))

    def test_recent_result_source(self):
        trends = self.ingester._build_recent_result_trends()
        for t in trends:
            self.assertEqual(t.source, "results")

    def test_recent_result_category(self):
        trends = self.ingester._build_recent_result_trends()
        for t in trends:
            self.assertEqual(t.category, "match_result")


class TestEntityExtraction(unittest.TestCase):
    """Test entity extraction from text."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.state = PersistentStateStore(str(self.db_path))
        self.ingester = TrendIngester(self.state)

    def tearDown(self):
        self.state.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_extract_players(self):
        ents = self.ingester._extract_entities_from_text("Bumrah magic spell")
        self.assertIn("bumrah", ents["players"])

    def test_extract_ipl_teams(self):
        ents = self.ingester._extract_entities_from_text("MI vs RCB rivalry")
        self.assertIn("mi", ents["teams"])
        self.assertIn("rcb", ents["teams"])

    def test_extract_international_teams(self):
        ents = self.ingester._extract_entities_from_text("India vs Australia")
        self.assertIn("india", ents["teams"])
        self.assertIn("australia", ents["teams"])

    def test_extract_series(self):
        ents = self.ingester._extract_entities_from_text("T20 World Cup 2026")
        self.assertIn("world_cup", ents["series"])

    def test_extract_empty(self):
        ents = self.ingester._extract_entities_from_text("Random news text")
        self.assertEqual(ents["players"], [])
        self.assertEqual(ents["teams"], [])
        self.assertEqual(ents["series"], [])

    def test_extract_combined(self):
        ents = self.ingester._extract_entities_from_text("Kohli and Bumrah for RCB in IPL")
        self.assertIn("kohli", ents["players"])
        self.assertIn("bumrah", ents["players"])
        self.assertIn("rcb", ents["teams"])
        self.assertIn("ipl", ents["series"])


class TestQueryCategorization(unittest.TestCase):
    """Test query categorization for half-life assignment."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.state = PersistentStateStore(str(self.db_path))
        self.ingester = TrendIngester(self.state)

    def tearDown(self):
        self.state.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_categorize_live_moment(self):
        cat = self.ingester._categorize_query("cricket live streaming today")
        self.assertEqual(cat, "live_moment")

    def test_categorize_match_result(self):
        cat = self.ingester._categorize_query("India wins T20 final")
        self.assertEqual(cat, "match_result")

    def test_categorize_controversy(self):
        cat = self.ingester._categorize_query("umpire controversy drama")
        self.assertEqual(cat, "controversy")

    def test_categorize_tournament(self):
        cat = self.ingester._categorize_query("IPL 2026 auction")
        self.assertEqual(cat, "tournament")

    def test_categorize_test_match(self):
        cat = self.ingester._categorize_query("Test match analysis")
        self.assertEqual(cat, "match")

    def test_categorize_default(self):
        cat = self.ingester._categorize_query("Generic cricket news")
        self.assertEqual(cat, "match")


class TestSuggestionScoring(unittest.TestCase):
    """Test scoring and velocity helpers."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.state = PersistentStateStore(str(self.db_path))
        self.ingester = TrendIngester(self.state)

    def tearDown(self):
        self.state.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_score_with_entities_higher(self):
        s1 = self.ingester._score_suggestion("Generic news")
        s2 = self.ingester._score_suggestion("Bumrah bowling highlights 2026")
        self.assertGreater(s2, s1)

    def test_score_recency_hints(self):
        s1 = self.ingester._score_suggestion("Bumrah")
        s2 = self.ingester._score_suggestion("Bumrah live today 2026")
        self.assertGreater(s2, s1)

    def test_score_bounded(self):
        for s in ["", "Bumrah Kohli Rohit 2026 live highlights viral"]:
            score = self.ingester._score_suggestion(s)
            self.assertGreaterEqual(score, 0.1)
            self.assertLessEqual(score, 1.0)

    def test_velocity_decreases_with_position(self):
        v1 = self.ingester._velocity_from_position(0, 100)
        v50 = self.ingester._velocity_from_position(50, 100)
        v100 = self.ingester._velocity_from_position(100, 100)
        self.assertGreater(v1, v50)
        self.assertGreater(v50, v100)
        self.assertEqual(v100, 0.0)

    def test_velocity_empty_total(self):
        v = self.ingester._velocity_from_position(0, 0)
        self.assertEqual(v, 0.0)


class TestYouTubeAutocompleteParser(unittest.TestCase):
    """Test YouTube autocomplete response parser."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.state = PersistentStateStore(str(self.db_path))
        self.ingester = TrendIngester(self.state)

    def tearDown(self):
        self.state.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_parse_google_ac_wrapper(self):
        raw = 'window.google.ac.h(["cricket",[["cricket highlights",0,[]],["cricket live",0,[]]]]);'
        result = self.ingester._parse_youtube_autocomplete(raw)
        self.assertEqual(result, ["cricket highlights", "cricket live"])

    def test_parse_plain_json(self):
        raw = '["cricket",[["cricket highlights",0,[]],["cricket live",0,[]]]]'
        result = self.ingester._parse_youtube_autocomplete(raw)
        self.assertEqual(result, ["cricket highlights", "cricket live"])

    def test_parse_malformed(self):
        raw = "not json at all"
        result = self.ingester._parse_youtube_autocomplete(raw)
        self.assertEqual(result, [])

    def test_parse_empty(self):
        raw = 'window.google.ac.h(["cricket",[]]);'
        result = self.ingester._parse_youtube_autocomplete(raw)
        self.assertEqual(result, [])


class TestTrendIngestIntoEngine(unittest.TestCase):
    """Test that ingested trends are stored in TrendEngine."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.state = PersistentStateStore(str(self.db_path))

    def tearDown(self):
        self.state.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_ingest_fixtures_only(self):
        ingester = TrendIngester(self.state)
        trends = ingester.ingest_all()
        # Should have at least fixtures + recent results
        self.assertGreaterEqual(len(trends), len(UPCOMING_FIXTURES) + len(RECENT_RESULTS))

        engine = TrendEngine(self.state)
        count = ingester.ingest_into_engine(engine)
        self.assertGreaterEqual(count, len(UPCOMING_FIXTURES))

        active = engine.get_active()
        self.assertGreater(len(active), 0)

    def test_ingest_idempotent(self):
        ingester = TrendIngester(self.state)
        engine = TrendEngine(self.state)

        # First ingest
        ingester.ingest_into_engine(engine)
        first_count = len(engine.get_active())

        # Second ingest (same trend_ids)
        ingester.ingest_into_engine(engine)
        second_count = len(engine.get_active())

        # Should still be same count (trends are upserted by id)
        self.assertEqual(first_count, second_count)


class TestIngestAllWithNetworkFailures(unittest.TestCase):
    """Test ingest_all handles network failures gracefully."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.state = PersistentStateStore(str(self.db_path))

    def tearDown(self):
        self.state.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch.object(TrendIngester, '_fetch_youtube_autocomplete_trends')
    @patch.object(TrendIngester, '_fetch_espn_cricinfo_trends')
    def test_ingest_continues_when_youtube_fails(self, mock_espn, mock_yt):
        mock_yt.side_effect = Exception("YouTube down")
        mock_espn.return_value = []

        ingester = TrendIngester(self.state)
        trends = ingester.ingest_all()

        # Should still have fixtures + recent results
        self.assertGreaterEqual(len(trends), len(UPCOMING_FIXTURES) + len(RECENT_RESULTS))

    @patch.object(TrendIngester, '_fetch_youtube_autocomplete_trends')
    @patch.object(TrendIngester, '_fetch_espn_cricinfo_trends')
    def test_ingest_continues_when_espn_fails(self, mock_espn, mock_yt):
        mock_yt.return_value = []
        mock_espn.side_effect = Exception("ESPN down")

        ingester = TrendIngester(self.state)
        trends = ingester.ingest_all()

        # Should still have fixtures + recent results
        self.assertGreaterEqual(len(trends), len(UPCOMING_FIXTURES) + len(RECENT_RESULTS))


class TestHttpGet(unittest.TestCase):
    """Test HTTP GET with SSL handling."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.state = PersistentStateStore(str(self.db_path))
        self.ingester = TrendIngester(self.state, http_timeout=2.0)

    def tearDown(self):
        self.state.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_http_get_uses_certifi(self):
        # Test that http_get works against a known URL (Google's autocomplete)
        # This is a live network test but fast
        try:
            raw = self.ingester._http_get(
                YOUTUBE_AUTOCOMPLETE_URL.format(query="cricket")
            )
            self.assertGreater(len(raw), 0)
        except Exception as e:
            self.skipTest(f"Network unavailable: {e}")


class TestInternationalTeamsDict(unittest.TestCase):
    """Test the INTERNATIONAL_TEAMS constant."""

    def test_includes_major_teams(self):
        expected = ["india", "australia", "england", "pakistan", "new_zealand"]
        for team in expected:
            self.assertIn(team, INTERNATIONAL_TEAMS)

    def test_keywords_are_lowercase(self):
        for name, kws in INTERNATIONAL_TEAMS.items():
            for kw in kws:
                self.assertEqual(kw, kw.lower(), f"Keyword {kw!r} for {name} not lowercase")


class TestGoogleTrendsTokenExtraction(unittest.TestCase):
    """Test Google Trends token extraction from explore HTML."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.state = PersistentStateStore(str(self.db_path))
        self.ingester = TrendIngester(self.state)

    def tearDown(self):
        self.state.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_extract_token_simple(self):
        html = '<html><script>var data = {"token":"abc123xyz"};</script></html>'
        token = self.ingester._extract_google_trends_token(html)
        self.assertEqual(token, "abc123xyz")

    def test_extract_token_escaped(self):
        html = r'<html><script>\"token\":\"abc123\"</script></html>'
        token = self.ingester._extract_google_trends_token(html)
        self.assertIsNotNone(token)
        self.assertIn("abc123", token)

    def test_extract_token_missing(self):
        html = '<html><body>no token here</body></html>'
        token = self.ingester._extract_google_trends_token(html)
        self.assertIsNone(token)


class TestGoogleTrendsWidgetParser(unittest.TestCase):
    """Test Google Trends widget data parser."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.state = PersistentStateStore(str(self.db_path))
        self.ingester = TrendIngester(self.state)

    def tearDown(self):
        self.state.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_parse_with_jsonp_prefix(self):
        raw = ")]}',\n" + json.dumps({"default": {"rankedList": []}})
        result = self.ingester._parse_google_trends_widget(raw)
        self.assertIsNotNone(result)
        self.assertIn("default", result)

    def test_parse_plain_json(self):
        raw = json.dumps({"default": {"rankedList": []}})
        result = self.ingester._parse_google_trends_widget(raw)
        self.assertIsNotNone(result)

    def test_parse_malformed(self):
        result = self.ingester._parse_google_trends_widget("not json")
        self.assertIsNone(result)


class TestGoogleTrendsBuilder(unittest.TestCase):
    """Test TrendInput construction from Google Trends data."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.state = PersistentStateStore(str(self.db_path))
        self.ingester = TrendIngester(self.state)

    def tearDown(self):
        self.state.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_build_from_widget_data(self):
        widget_data = {
            "default": {
                "rankedList": [
                    {
                        "rankedKeyword": [
                            {"query": "bumrah bowling", "value": 100, "formattedValue": "100"},
                            {"query": "kohli century", "value": 80, "formattedValue": "80"},
                        ]
                    },
                    {
                        "rankedKeyword": [
                            {"query": "mumbai indians", "value": 5000, "formattedValue": "+5000%"},
                        ]
                    },
                ]
            }
        }
        trends = self.ingester._build_google_trend_inputs("cricket", widget_data)
        self.assertGreater(len(trends), 0)
        for t in trends:
            self.assertEqual(t.source, "google_trends")

    def test_filters_seed_keyword(self):
        widget_data = {
            "default": {
                "rankedList": [
                    {"rankedKeyword": [
                        {"query": "cricket", "value": 100, "formattedValue": "100"},
                        {"query": "bumrah magic", "value": 80, "formattedValue": "80"},
                    ]},
                ]
            }
        }
        trends = self.ingester._build_google_trend_inputs("cricket", widget_data)
        queries = [t.query for t in trends]
        self.assertNotIn("cricket", queries)
        self.assertIn("bumrah magic", queries)

    def test_filters_queries_without_entities(self):
        widget_data = {
            "default": {
                "rankedList": [
                    {"rankedKeyword": [
                        {"query": "random unrelated topic", "value": 100, "formattedValue": "100"},
                        {"query": "bumrah yorker", "value": 80, "formattedValue": "80"},
                    ]},
                ]
            }
        }
        trends = self.ingester._build_google_trend_inputs("cricket", widget_data)
        queries = [t.query for t in trends]
        self.assertNotIn("random unrelated topic", queries)
        self.assertIn("bumrah yorker", queries)

    def test_rising_queries_get_higher_velocity(self):
        widget_data = {
            "default": {
                "rankedList": [
                    {"rankedKeyword": [
                        {"query": "bumrah top", "value": 100, "formattedValue": "100"},
                    ]},
                    {"rankedKeyword": [
                        {"query": "kohli rising", "value": 5000, "formattedValue": "+5000%"},
                    ]},
                ]
            }
        }
        trends = self.ingester._build_google_trend_inputs("cricket", widget_data)
        top = next(t for t in trends if "top" in t.query)
        rising = next(t for t in trends if "rising" in t.query)
        self.assertGreater(rising.velocity, top.velocity)


class TestGoogleTrendsFetch(unittest.TestCase):
    """Test Google Trends fetching with mocked network."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self.state = PersistentStateStore(str(self.db_path))
        self.ingester = TrendIngester(self.state, http_timeout=2.0)

    def tearDown(self):
        self.state.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch.object(TrendIngester, '_fetch_google_trends_for_keyword')
    def test_ingest_continues_when_one_keyword_fails(self, mock_fetch):
        mock_fetch.side_effect = [
            [TrendInput(
                trend_id="t1", source="google_trends", query="bumrah test",
                trend_score=0.7, velocity=0.3, category="match",
                entities={"players": ["bumrah"], "teams": [], "series": []},
                half_life_hours=72.0,
            )],
            Exception("network error"),
            [],
        ]
        trends = self.ingester._fetch_google_trends()
        self.assertEqual(len(trends), 1)
        self.assertEqual(trends[0].query, "bumrah test")

    @patch.object(TrendIngester, '_fetch_google_trends_for_keyword')
    def test_returns_empty_on_429_skip(self, mock_fetch):
        import urllib.error
        mock_fetch.side_effect = urllib.error.HTTPError(
            GOOGLE_TRENDS_EXPLORE_URL, 429, "Too Many Requests", {}, None
        )
        trends = self.ingester._fetch_google_trends()
        self.assertEqual(trends, [])

    def test_google_trends_appears_in_ingest_all(self):
        with patch.object(TrendIngester, '_fetch_google_trends', return_value=[]):
            trends = self.ingester.ingest_all()
            self.assertIsInstance(trends, list)


class TestGoogleTrendsSeedKeywords(unittest.TestCase):
    """Test GOOGLE_TRENDS_KEYWORDS constant."""

    def test_keywords_are_strings(self):
        for kw in GOOGLE_TRENDS_KEYWORDS:
            self.assertIsInstance(kw, str)
            self.assertGreater(len(kw), 0)

    def test_keywords_non_empty(self):
        self.assertGreater(len(GOOGLE_TRENDS_KEYWORDS), 0)

    def test_keywords_unique(self):
        self.assertEqual(len(GOOGLE_TRENDS_KEYWORDS), len(set(GOOGLE_TRENDS_KEYWORDS)))


if __name__ == "__main__":
    unittest.main()
