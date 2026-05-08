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
        """Should rotate hashtags based on match type"""
        from trends import get_rotated_hashtags
        
        # IPL match hashtags
        ipl_tags = get_rotated_hashtags("ipl")
        assert any("IPL" in t for t in ipl_tags)
        assert len(ipl_tags) >= 5
        
        # International match hashtags
        intl_tags = get_rotated_hashtags("international")
        assert any(t in intl_tags for t in ("#INDvs", "#Cricket", "#TestCricket", "#ODI", "#BleedBlue"))
        
        # T20 match hashtags
        t20_tags = get_rotated_hashtags("t20")
        assert any(t in t20_tags for t in ("#T20", "#Cricket", "#T20Matches", "#BigHits", "#CricketAction"))

    def test_hashtag_uniqueness(self):
        """Should not repeat same hashtags across multiple videos"""
        from trends import get_rotated_hashtags
        
        # Simulate generating hashtags for 3 consecutive videos
        tags_set1 = set(get_rotated_hashtags("ipl", seed=1))
        tags_set2 = set(get_rotated_hashtags("ipl", seed=2))
        tags_set3 = set(get_rotated_hashtags("ipl", seed=3))
        
        # At least some variation between sets
        all_tags = tags_set1 | tags_set2 | tags_set3
        assert len(all_tags) > len(tags_set1)  # More unique tags overall

    def test_hashtag_limits(self):
        """Should respect YouTube hashtag limits (max 15, but 3-5 optimal)"""
        from trends import get_rotated_hashtags
        
        tags = get_rotated_hashtags("ipl")
        # YouTube shows only first 3 hashtags above title, but allows up to 15
        assert len(tags) <= 15
        assert len(tags) >= 3  # Minimum for discoverability


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
