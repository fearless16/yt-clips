import pytest
from unittest.mock import patch, MagicMock
from automation.seo.seo import generate_clip_seo
from automation.seo.cricket_context import correct_cricket_spelling

def test_cricket_spelling_correction():
    text = "coaly and bumra played well at wankhede against csk"
    corrected = correct_cricket_spelling(text)
    assert "Kohli" in corrected
    assert "Bumrah" in corrected
    assert "Wankhede Stadium" in corrected
    assert "Chennai Super Kings" in corrected

def test_shorts_specific_seo_generation():
    with patch("utils.ai_client.AIClient.generate_fastest_first", return_value='{"title": "Epic Kohli Six! 😱🔥 #Shorts", "description": "Viral description", "search_terms": ["kohli six", "cricket shorts"], "hashtags": ["#Shorts", "#Kohli"]}') as mock_generate:
        res = generate_clip_seo(
            clip_id="clip123",
            transcript="coaly hit a six",
            video_title="IND vs PAK",
            is_shorts=True
        )
        assert res["title"] == "Epic Kohli Six! 😱🔥 #Shorts"
        assert res["ai_generated"] is True
        
        _, kwargs = mock_generate.call_args
        system_instruction = kwargs.get("system_instruction", "")
        assert "shorts" in system_instruction.lower()
        assert "viral" in system_instruction.lower()

def test_standard_seo_generation():
    with patch("utils.ai_client.AIClient.generate_fastest_first", return_value='{"title": "LIVE IND vs PAK | India vs Pakistan T20 | Aaj Ka Match", "description": "LIVE: India vs Pakistan\\n🇮🇳 India: JioHotstar\\n🇵🇰 Pakistan: Yupp TV\\n\\nCHAPTERS\\n00:00 Live Start", "search_terms": ["ind vs pak live", "cricket match today"], "hashtags": ["#INDvsPAK", "#Cricket"]}') as mock_generate:
        res = generate_clip_seo(
            clip_id="clip123",
            transcript="kohli hit a six",
            video_title="IND vs PAK",
            is_shorts=False
        )
        assert "LIVE IND vs PAK" in res["title"]
        
        _, kwargs = mock_generate.call_args
        system_instruction = kwargs.get("system_instruction", "")
        # The standard system instructions says "You are an elite YouTube Shorts SEO expert..." but is for long-form structure
        assert "shorts" in system_instruction.lower()
