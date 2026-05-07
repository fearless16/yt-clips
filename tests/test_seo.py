"""
Test suite for SEO generation - TDD approach
Tests for Hindi/Hinglish support, trend integration, and tag specificity
"""
import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, '/workspace')

from seo import _extract_keywords, batch_generate_seo


class TestKeywordExtraction:
    """Tests for keyword extraction from transcripts"""
    
    def test_hindi_words_extraction(self):
        """Should extract Hindi/Hinglish words correctly"""
        text = "Kohli ne amazing six maara stadium mein"
        keywords = _extract_keywords(text)
        assert "kohli" in keywords
        assert "amazing" in keywords
        assert "six" in keywords
        assert "maara" in keywords  # Hindi word preserved
        assert "stadium" in keywords
    
    def test_stop_words_filtered(self):
        """Common stop words should be filtered out"""
        text = "the player is hitting the ball very well"
        keywords = _extract_keywords(text)
        assert "the" not in keywords
        assert "is" not in keywords
        assert "very" not in keywords
        assert "player" in keywords or "hitting" in keywords or "ball" in keywords
    
    def test_short_words_filtered(self):
        """Words with length <= 2 should be filtered"""
        text = "I am a pro at this game"
        keywords = _extract_keywords(text)
        assert "i" not in keywords
        assert "am" not in keywords
        assert "a" not in keywords
        assert "pro" in keywords
        assert "this" not in keywords  # Stop word
        assert "game" in keywords


class TestSEOBatchGeneration:
    """Tests for batch SEO generation structure"""
    
    def test_batch_structure(self):
        """Batch SEO should return list with required fields per clip"""
        clips = [
            {"clip_id": "test1", "text": "Kohli hits massive six over cover"},
            {"clip_id": "test2", "text": "Dhoni stunning stumping in last over"}
        ]
        
        # Mock AI response to avoid API calls in tests
        import seo
        original_generate = seo.ai.generate_text
        
        def mock_generate(prompt, system_instruction=None):
            return json.dumps({
                "clips_seo": [
                    {
                        "clip_id": "test1",
                        "title": "Kohli's MASSIVE Six! 💥",
                        "description": "Virat Kohli hits a huge six over cover drive. #Cricket",
                        "tags": ["kohli six", "rcb highlights", "ipl 2024"]
                    },
                    {
                        "clip_id": "test2",
                        "title": "Dhoni Magic Stumping! 🔥",
                        "description": "MS Dhoni does a quick stumping. #CSK",
                        "tags": ["dhoni stumping", "csk moments", "msd magic"]
                    }
                ]
            })
        
        seo.ai.generate_text = mock_generate
        
        try:
            results = batch_generate_seo(clips)
            
            assert len(results) == 2
            for result in results:
                assert "clip_id" in result
                assert "title" in result
                assert "description" in result
                assert "tags" in result
                assert "hashtags" in result
                assert "trend_topics" in result
                
                # Title length constraint
                assert len(result["title"]) <= 100
            
            # Verify specific clip IDs preserved
            clip_ids = [r["clip_id"] for r in results]
            assert "test1" in clip_ids
            assert "test2" in clip_ids
        finally:
            seo.ai.generate_text = original_generate
    
    def test_trend_topics_integration(self):
        """Trending topics should be included in results"""
        clips = [{"clip_id": "t1", "text": "Rohit Sharma pull shot"}]
        
        import seo
        import trends
        
        # Mock both AI and trends
        original_generate = seo.ai.generate_text
        original_trend = trends.get_trending_context
        
        def mock_generate(prompt, system_instruction=None):
            return json.dumps({
                "clips_seo": [{
                    "clip_id": "t1",
                    "title": "Rohit's Pull Shot Masterclass",
                    "description": "Amazing shot",
                    "tags": ["rohit sharma", "pull shot"]
                }]
            })
        
        def mock_trend(domain, region, video_title):
            return {
                "topics": ["IPL 2024", "Mumbai Indians"],
                "tags": ["#Shorts", "#Cricket"],
                "scorecard": "MI vs CSK"
            }
        
        seo.ai.generate_text = mock_generate
        trends.get_trending_context = mock_trend
        
        try:
            results = batch_generate_seo(clips)
            assert len(results) == 1
            result = results[0]
            
            # Trend topics should be present
            assert "trend_topics" in result
            assert len(result["trend_topics"]) > 0
        finally:
            seo.ai.generate_text = original_generate
            trends.get_trending_context = original_trend


class TestTagSpecificity:
    """Tests for tag quality and specificity validation"""
    
    def test_generic_tags_should_be_filtered(self):
        """Generic tags like 'cricket' or 'shorts' should be rejected"""
        # This will be implemented in the fix
        from seo import _validate_tags
        
        generic_tags = ["cricket", "shorts", "viral", "trending"]
        valid_tags = _validate_tags(generic_tags)
        
        # All generic tags should be filtered
        assert len(valid_tags) < len(generic_tags)
    
    def test_specific_tags_should_pass(self):
        """Long-tail specific tags should be preserved"""
        from seo import _validate_tags
        
        specific_tags = [
            "kohli six vs csk 2024",
            "rcb run chase thriller", 
            "dhoni reaction ipl"
        ]
        valid_tags = _validate_tags(specific_tags)
        
        # All specific tags should pass
        assert len(valid_tags) == len(specific_tags)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
