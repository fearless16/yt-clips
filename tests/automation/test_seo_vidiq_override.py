import pytest
from unittest.mock import patch
from automation.seo import seo

def test_generate_clip_seo_preserves_llm_title_over_vidiq_override(monkeypatch):
    """
    Ensures that when VidIQ provides SEO context and titles, the AI-generated
    title is preserved, preventing VidIQ from enforcing English algorithmic titles
    over desi-style/Hinglish engaging titles.
    """
    def mock_attempt_seo(*args, **kwargs):
        return {
            "title": "Virat Kohli's Massive Six! 🤯",
            "description": "Watch this shot." * 100, 
            "search_terms": ["kohli", "cricket"],
            "primary_search_terms": ["kohli six"],
            "tags": ["cricket", "kohli"],
            "thumbnail_instruction": "A frame of Kohli hitting the ball."
        }
        
    monkeypatch.setattr(seo, "_attempt_seo_generation", mock_attempt_seo)
    monkeypatch.setattr(seo, "_validate_seo_quality", lambda x: True)
    
    def mock_vidiq_context(*args, **kwargs):
        keywords = ["virat kohli cricket"] + [f"keyword {i}" for i in range(15)]
        return "Mocked Context", {"used": True}, {
            "titles": ["Virat Kohli Best Cricket Highlights 2026"],
            "keywords": keywords
        }
        
    monkeypatch.setattr(seo, "_get_vidiq_context", mock_vidiq_context)
    
    original_cfg_get = seo.cfg.get
    def mock_cfg_get(key, default=None):
        if key == "seo":
            val = original_cfg_get(key, default)
            val["vidiq"] = val.get("vidiq", {})
            val["vidiq"]["required"] = True
            val["vidiq"]["min_keywords_required"] = 12
            return val
        return original_cfg_get(key, default)
        
    monkeypatch.setattr(seo.cfg, "get", mock_cfg_get)
    monkeypatch.setattr(seo, "_get_learner_context", lambda: "")
    
    result = seo.generate_clip_seo(
        clip_id="test_clip_1",
        transcript="Kohli hits it for six! Unbelievable shot.",
        video_title="India vs Pakistan Live Stream",
        video_description="Live coverage",
        scorecard="IND 100/1",
    )
    
    assert result is not None
    assert result["title"] == "Virat Kohli's Massive Six! 🤯"
    assert "virat kohli cricket" in result["search_terms"], "VidIQ keywords should be included"

