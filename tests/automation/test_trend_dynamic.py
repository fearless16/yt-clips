"""TDD tests for dynamic domain/topic detection and trend routing in SEO.
"""
import pytest
from unittest.mock import patch

from automation.seo.trends import detect_video_domain, get_trending_context, get_rotated_hashtags
from automation.seo.seo import generate_clip_seo


def test_detect_video_domain_cricket():
    """Should correctly identify cricket domain and extract entities/keywords."""
    domain, query, keywords = detect_video_domain(
        "🔴LIVE: Bangalore vs Mumbai, IPL 2026 Commentary | Live Match Today | RCB vs MI Live",
        "Virat Kohli is batting and Jasprit Bumrah is bowling. What a clash at Wankhede!"
    )
    assert domain == "cricket"
    assert "RCB" in keywords or "MI" in keywords or "Kohli" in keywords
    assert "IPL" in query or "RCB" in query or "MI" in query or "cricket" in query


def test_detect_video_domain_football():
    """Should correctly identify football domain for FIFA videos."""
    domain, query, keywords = detect_video_domain(
        "🇫🇷 France ki Fanbase PHAT GAYI! Mbappé Effect | WC 2026 Watchalong 🔥 #Shorts",
        "Mbappe scored 2 goals in 97 seconds in the 2022 World Cup final against Argentina. Griezmann assisted."
    )
    assert domain == "football"
    assert "Mbappe" in keywords or "France" in keywords or "Griezmann" in keywords
    assert "Mbappe" in query or "France" in query or "FIFA" in query or "World Cup" in query


def test_detect_video_domain_general():
    """Should fallback to general domain for non-sports videos."""
    domain, query, keywords = detect_video_domain(
        "Tech Bro Life in Bangalore | SDE Salary & Rents",
        "Talking about software engineering jobs, salaries, and apartment rents in Bangalore tech hubs."
    )
    assert domain == "general"
    assert "Salary" in keywords or "Rent" in keywords or "Tech" in keywords
    assert "Tech" in query or "Salary" in query or "Bangalore" in query


def test_get_rotated_hashtags_by_domain():
    """Should return domain-appropriate hashtags."""
    cricket_tags = get_rotated_hashtags(domain="cricket")
    assert "#IPL2026" in cricket_tags
    assert "#Shorts" in cricket_tags

    football_tags = get_rotated_hashtags(domain="football")
    assert "#FIFA2026" in football_tags
    assert "#Shorts" in football_tags

    general_tags = get_rotated_hashtags(domain="general")
    assert "#Shorts" in general_tags


@patch("automation.seo.trends.fetch_youtube_suggestions")
@patch("automation.seo.trends.fetch_youtube_search_signals")
@patch("automation.seo.trends.fetch_google_trends_in")
def test_get_trending_context_uses_dynamic_football_queries(
    mock_google, mock_yt_search, mock_yt_suggest
):
    """Should not call Cricbuzz when domain is football, and should query with dynamic topic."""
    mock_yt_suggest.return_value = ["Mbappe skills", "France World Cup"]
    mock_yt_search.return_value = ["Mbappe World Cup fan reaction"]
    mock_google.return_value = ["World Cup 2026"]
    context = get_trending_context(
        video_title="France ki Fanbase PHAT GAYI! Mbappé Effect"
    )

    assert context["scorecard"] == ""

    # YouTube suggestions should have been called with football queries, not "cricket live"
    called_queries = [call.args[0] for call in mock_yt_suggest.call_args_list if call.args]
    assert len(called_queries) > 0
    # None of the queries should contain cricket-related keywords like "cricket live"
    for q in called_queries:
        assert "cricket" not in q.lower() or "france" in q.lower() or "mbappe" in q.lower()


@patch("automation.seo.seo._get_ai")
def test_generator_rejects_football_before_ai(mock_get_ai):
    """The channel generator is cricket-only, including direct API calls."""
    from automation.seo.seo import SEOGenerationError

    with pytest.raises(SEOGenerationError, match="non-cricket"):
        generate_clip_seo(
            clip_id="clip_002",
            transcript="Mbappe is the best player",
            video_title="Mbappé vs Messi World Cup Final",
            trend_topics=["Mbappe", "World Cup"],
        )

    mock_get_ai.assert_not_called()
