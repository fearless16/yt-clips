"""
test_cricbuzz_integration.py — Tests for Cricbuzz scraping, hashtag rotation,
and trend tag/title injection (from trends.py + seo.py).
"""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestCricbuzzScraper:
    """Tests for Cricbuzz match data scraping"""

    def test_match_url_extraction(self):
        """Should extract match URL from video title"""
        from trends import extract_match_teams
        
        # Test IPL match
        title = "RCB vs CSK IPL 2024 Live Match"
        teams, match_type = extract_match_teams(title)
        assert "RCB" in teams or "CSK" in teams
        assert match_type == "ipl"
        
        # Test International match
        title2 = "India vs Australia Test Match 2024"
        teams2, match_type2 = extract_match_teams(title2)
        assert "INDIA" in teams2 or "IND" in teams2
        assert "AUSTRALIA" in teams2 or "AUS" in teams2
        assert match_type2 == "international"

    def test_scorecard_parsing(self):
        """Should parse live scorecard from Cricbuzz HTML"""
        from trends import parse_cricbuzz_scorecard
        
        # Mock HTML response
        mock_html = """
        <div class="match-score">
            <div class="team-score">RCB 185/4 (18.2 Ov)</div>
            <div class="team-score">CSK 178/6 (20 Ov)</div>
        </div>
        <div class="current-batsman">
            <span class="name">Virat Kohli</span> - 67* (34)
        </div>
        """
        
        scorecard = parse_cricbuzz_scorecard(mock_html)
        assert "RCB" in scorecard or "185" in scorecard
        assert "CSK" in scorecard or "178" in scorecard

    def test_fetch_live_score_integration(self):
        """Integration test for fetching live score from Cricbuzz"""
        from trends import fetch_cricbuzz_live_score
        
        # This will be implemented to actually scrape Cricbuzz
        # For now, test that it returns a dict with expected keys
        result = fetch_cricbuzz_live_score("RCB vs CSK", "ipl")
        
        assert isinstance(result, dict)
        assert "scorecard" in result or "error" in result  # Either valid data or error message


class TestHashtagRotation:
    """Tests for dynamic hashtag generation and rotation"""

    def test_hashtag_rotation_by_match_type(self):
        """Should rotate hashtags based on match type — uses seeds for determinism."""
        from trends import get_rotated_hashtags, HASHTAG_POOLS

        # IPL: at least one pool across all seeds must contain an IPL-related tag
        ipl_all_tags = set()
        for seed in range(len(HASHTAG_POOLS["ipl"])):
            ipl_all_tags.update(get_rotated_hashtags("ipl", seed=seed))
        assert any("IPL" in t or "Tata" in t or "RCB" in t or "CSK" in t or "MI" in t
                   for t in ipl_all_tags), f"No IPL tag found across all pools: {ipl_all_tags}"

        # International: at least one pool must have an international cricket tag
        intl_all_tags = set()
        for seed in range(len(HASHTAG_POOLS["international"])):
            intl_all_tags.update(get_rotated_hashtags("international", seed=seed))
        assert any(t in intl_all_tags for t in
                   ("#INDvs", "#Cricket", "#TestCricket", "#ODI", "#BleedBlue", "#TeamIndia"))

        # T20: at least one pool must have a T20-related tag
        t20_all_tags = set()
        for seed in range(len(HASHTAG_POOLS["t20"])):
            t20_all_tags.update(get_rotated_hashtags("t20", seed=seed))
        assert any(t in t20_all_tags for t in
                   ("#T20", "#Cricket", "#T20Matches", "#BigHits", "#CricketAction", "#T20Cricket"))

    def test_hashtag_uniqueness(self):
        """Should have variation across different seeds."""
        from trends import get_rotated_hashtags
        tags_set1 = set(get_rotated_hashtags("ipl", seed=0))
        tags_set2 = set(get_rotated_hashtags("ipl", seed=1))
        tags_set3 = set(get_rotated_hashtags("ipl", seed=2))
        all_tags = tags_set1 | tags_set2 | tags_set3
        assert len(all_tags) > len(tags_set1)  # More unique tags across rotations

    def test_hashtag_limits(self):
        """Should return between 3 and 15 hashtags."""
        from trends import get_rotated_hashtags
        tags = get_rotated_hashtags("ipl", seed=0)
        assert 3 <= len(tags) <= 15


class TestTrendTagInjection:
    """Tests for injecting trending topics into tags"""

    def test_trend_injection_into_tags(self):
        """Should inject trending topics into tag list"""
        from seo import inject_trend_topics_into_tags
        
        base_tags = ["kohli six", "rcb highlights"]
        trend_topics = ["IPL 2024", "Mumbai Indians", "Playoffs"]
        
        result = inject_trend_topics_into_tags(base_tags, trend_topics)
        
        # Should have original tags plus injected ones
        assert "kohli six" in result
        assert "rcb highlights" in result
        
        # Should have at least one trend-related tag
        has_trend_tag = any(
            any(trend.lower() in tag.lower() for trend in trend_topics)
            for tag in result
        )
        assert has_trend_tag

    def test_trend_injection_with_player_name(self):
        """Should combine player names with trend topics"""
        from seo import inject_trend_topics_into_tags
        
        base_tags = ["virat kohli", "cover drive"]
        trend_topics = ["IPL 2024", "RCB vs CSK"]
        player_name = "virat kohli"
        
        result = inject_trend_topics_into_tags(
            base_tags, 
            trend_topics, 
            player_name=player_name
        )
        
        # Should have combination tags like "virat kohli ipl 2024"
        has_combo_tag = any(
            "virat" in tag.lower() and "ipl" in tag.lower()
            for tag in result
        )
        assert has_combo_tag

    def test_no_duplicate_injection(self):
        """Should not inject duplicate tags"""
        from seo import inject_trend_topics_into_tags
        
        base_tags = ["ipl 2024", "kohli six"]
        trend_topics = ["IPL 2024"]  # Already in base tags
        
        result = inject_trend_topics_into_tags(base_tags, trend_topics)
        
        # Count occurrences of "ipl 2024"
        ipl_count = sum(1 for tag in result if tag.lower() == "ipl 2024")
        assert ipl_count == 1  # No duplicates


class TestTitleTrendInjection:
    """Tests for injecting trends into titles"""

    def test_title_injection_when_missing_trend(self):
        """Should prepend trend topic to title if missing"""
        from seo import ensure_trend_in_title
        
        title = "Kohli's Amazing Six!"
        trend_topics = ["IPL 2024 Playoffs", "RCB Qualifier"]
        
        result = ensure_trend_in_title(title, trend_topics)
        
        # Should have trend topic prepended
        assert "IPL 2024" in result or "RCB" in result
        assert "Kohli" in result  # Original content preserved
        assert len(result) <= 100  # YouTube limit

    def test_title_unchanged_when_has_trend(self):
        """Should not modify title if it already has trend topic"""
        from seo import ensure_trend_in_title
        
        title = "IPL 2024: Kohli's Amazing Six!"
        trend_topics = ["IPL 2024 Playoffs"]
        
        result = ensure_trend_in_title(title, trend_topics)
        
        # Title should remain unchanged
        assert result == title

    def test_title_truncation(self):
        """Should truncate title to 100 chars after injection"""
        from seo import ensure_trend_in_title
        
        long_title = "This is a very long title that exceeds YouTube's 100 character limit when combined with trend topics" * 2
        trend_topics = ["IPL 2024"]
        
        result = ensure_trend_in_title(long_title, trend_topics)
        assert len(result) <= 100


class TestFetchOwnLiveStreamUrl:
    """Tests for trends.fetch_own_live_stream_url — the new live stream detection."""

    def test_returns_static_fallback_when_no_channel_id(self, monkeypatch):
        """No channel ID → falls back to static config URL."""
        import trends
        monkeypatch.setattr(trends, "cfg", {
            "channel": {"id": "", "live_stream_url": "https://youtube.com/watch?v=STATIC"},
            "logging": trends.cfg.get("logging", {"log_file": "test.log", "level": "INFO"}),
        })
        result = trends.fetch_own_live_stream_url()
        assert result == "https://youtube.com/watch?v=STATIC"

    def test_returns_string(self, monkeypatch):
        """Should always return a string (never raise)."""
        import trends
        # Simulate network failure
        monkeypatch.setattr(trends._session(), "get",
                            lambda *a, **kw: (_ for _ in ()).throw(Exception("network error")),
                            raising=False)
        result = trends.fetch_own_live_stream_url("")
        assert isinstance(result, str)

    def test_extracts_watch_url_from_canonical(self, monkeypatch):
        """When channel page returns canonicalBaseUrl, extract watch URL."""
        import trends
        from unittest.mock import MagicMock

        fake_response = MagicMock()
        fake_response.text = '"canonicalBaseUrl":"/watch?v=ABC123"'
        fake_session = MagicMock()
        fake_session.get.return_value = fake_response

        monkeypatch.setattr(trends, "_session", lambda: fake_session)
        monkeypatch.setattr(trends, "cfg", {
            "channel": {"id": "UCtest123", "live_stream_url": ""},
            "logging": trends.cfg.get("logging", {"log_file": "test.log", "level": "INFO"}),
        })
        result = trends.fetch_own_live_stream_url("UCtest123")
        assert result == "https://www.youtube.com/watch?v=ABC123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
