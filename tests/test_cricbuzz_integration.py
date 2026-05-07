"""
Test suite for Cricbuzz integration and enhanced SEO features - TDD approach
Tests for:
1. Cricbuzz score scraping
2. Hashtag validation and rotation
3. Tag injection from trends
4. Upload A/B testing framework
"""
import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, '/workspace')


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
        assert match_type == "international"

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
        assert "#IPL" in ipl_tags or "#IPL2024" in ipl_tags
        assert len(ipl_tags) >= 5
        
        # International match hashtags
        intl_tags = get_rotated_hashtags("international")
        assert "#INDvs" in intl_tags or "#Cricket" in intl_tags
        
        # T20 match hashtags
        t20_tags = get_rotated_hashtags("t20")
        assert "#T20" in t20_tags or "#Cricket" in t20_tags

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


class TestUploadABTesting:
    """Tests for A/B testing framework in uploads"""

    def test_generate_thumbnail_variants(self):
        """Should generate multiple thumbnail variants"""
        from thumbnail import generate_thumbnail_variants
        
        # Mock video path and metadata
        mock_video_path = "/tmp/test_video.mp4"
        mock_metadata = {
            "title": "Kohli's Six",
            "clip_id": "test_001"
        }
        
        # Create dummy video file for testing
        Path(mock_video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(mock_video_path).touch()
        
        # Test variant generation (will be implemented)
        try:
            variants = generate_thumbnail_variants(mock_video_path, mock_metadata, count=3)
            assert len(variants) == 3
            for v in variants:
                assert Path(v).exists()
                assert "_v" in str(v)  # Variant naming
        except NotImplementedError:
            # Expected until implementation
            pytest.skip("generate_thumbnail_variants not yet implemented")

    def test_upload_schedule_optimization(self):
        """Should schedule uploads for peak IST hours"""
        from upload import calculate_optimal_upload_time
        
        # Test for different current times
        # Peak hours: 7 PM - 10 PM IST (13:30 - 16:30 UTC)
        
        # If current time is 10 AM IST, should schedule for 7 PM IST
        optimal = calculate_optimal_upload_time(hour=10, minute=0, timezone="IST")
        assert optimal.hour >= 19  # 7 PM or later
        
        # If current time is 8 PM IST, should schedule for next day 7 PM or same day if before 10 PM
        optimal2 = calculate_optimal_upload_time(hour=20, minute=0, timezone="IST")
        # Should be next day 7 PM
        assert optimal2.hour == 19


class TestHindiHinglishSEO:
    """Tests for Hindi/Hinglish SEO generation"""

    def test_hinglish_title_generation(self):
        """Should generate titles with natural Hinglish mix"""
        from seo import validate_hinglish_content
        
        # Test title with Hinglish
        title = "Kohli ka Dhamaakedaar Six! 💥"
        is_valid, language_mix = validate_hinglish_content(title)
        
        assert is_valid
        assert "hindi" in language_mix.lower() or "hinglish" in language_mix.lower()

    def test_hindi_description_hooks(self):
        """Should include Hindi hooks in descriptions"""
        from seo import validate_description_hooks
        
        description = "Kya shot tha yaar! 😱 Virat Kohli hits a massive six..."
        has_hook, hook_type = validate_description_hooks(description)
        
        assert has_hook
        assert hook_type in ["hindi", "hinglish", "emotional"]

    def test_emoji_usage_validation(self):
        """Should validate proper emoji usage (1-3 per title/description)"""
        from seo import validate_emoji_usage
        
        # Good emoji usage
        good_title = "Kohli's Six! 💥🔥"
        assert validate_emoji_usage(good_title) is True
        
        # Too many emojis
        bad_title = "Kohli's Six! 💥🔥😱🎉🏆✨"
        assert validate_emoji_usage(bad_title) is False
        
        # No emojis (acceptable but not optimal)
        no_emoji_title = "Kohli's Amazing Six"
        assert validate_emoji_usage(no_emoji_title) is True  # Not an error, just suboptimal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
