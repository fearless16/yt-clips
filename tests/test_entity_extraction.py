"""Tests for cricket entity extraction from transcript."""

import pytest

from automation.seo.trends import extract_cricket_entities_from_transcript, TEAM_MAPPINGS


class TestEntityExtraction:
    def test_extracts_player_names(self):
        transcript = "Virat Kohli smashed a six! Kohli is the best."
        result = extract_cricket_entities_from_transcript(transcript)
        assert "Virat Kohli" in result["players"]
        assert len(result["players"]) == 1

    def test_extracts_team_names(self):
        transcript = "RCB vs MI match was amazing!"
        result = extract_cricket_entities_from_transcript(transcript)
        assert "RCB" in result["teams"]
        assert "MI" in result["teams"]

    def test_extracts_match_context(self):
        transcript = "What a six! Target 189 and the wickets keep falling."
        result = extract_cricket_entities_from_transcript(transcript)
        assert "six" in result["match_context"]
        assert "wicket" in result["match_context"]
        assert "target" in result["match_context"]

    def test_extracts_scores(self):
        transcript = "The score is 198/4 after 18 overs"
        result = extract_cricket_entities_from_transcript(transcript)
        assert "198/4" in result["match_context"]

    def test_empty_transcript(self):
        result = extract_cricket_entities_from_transcript("")
        assert result["players"] == []
        assert result["teams"] == []

    def test_multiple_players_deduplicated(self):
        transcript = "Kohli scored 67. Virat Kohli is amazing."
        result = extract_cricket_entities_from_transcript(transcript)
        assert result["players"].count("Virat Kohli") == 1

    def test_case_insensitive(self):
        transcript = "rcb vs csk at chepauk"
        result = extract_cricket_entities_from_transcript(transcript)
        assert "RCB" in result["teams"]
        assert "CSK" in result["teams"]
