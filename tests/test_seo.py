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



# ─── feat/seo-quality: no generic fallback, shorts fix, escalation ────────────

import json
from datetime import datetime
import automation.seo.seo as seo_mod
from automation.seo.seo import (
    SEOGenerationError, _enforce_limits, _attempt_seo_generation,
    generate_seo_for_exported_clip, _default_hashtags,
)
from automation.seo.cricket_context import find_canonical_entities


def test_enforce_limits_has_no_generic_padding():
    """The forbidden generic 'safe_defaults' padding must be gone."""
    # Only one specific tag, no fallback terms -> must NOT be padded to 10 generics.
    out = _enforce_limits({
        "title": "T", "description": "D",
        "hashtags": ["#Shorts"],
        "search_terms": ["kohli cover drive"],
    }, fallback_terms=None)
    generic_phrases = {
        "cricket highlights", "cricket live match", "ipl match video",
        "t20 cricket live", "best cricket moments", "cricket shorts live",
        "indian cricket team", "cricket action", "match highlights",
        "cricket viral shorts",
    }
    assert not (set(out["search_terms"]) & generic_phrases), \
        "generic safe_defaults padding must not be injected"
    assert out["search_terms"] == ["kohli cover drive"]


def test_default_hashtags_use_current_season_not_hardcoded_2026():
    tags = _default_hashtags()
    year = datetime.now().year
    assert f"#IPL{year}" in tags
    assert "#Cricket" in tags and "#Shorts" in tags


def test_shorts_description_is_preserved_not_overwritten():
    """Shorts must keep the LLM's short description, not the long template."""
    ai_json = json.dumps({
        "title": "KOHLI SIX! 🔥 #Shorts",
        "description": "Kohli ne maara chakka! Subscribe for more 🔥 #Shorts",
        "search_terms": ["virat kohli six", "ipl live hindi"],
        "hashtags": ["#Shorts", "#Kohli"],
    })
    with patch("utils.ai_client.AIClient.generate_fastest_first", return_value=ai_json):
        res = generate_clip_seo(clip_id="c1", transcript="kohli hit six",
                                video_title="RCB vs CSK", is_shorts=True)
    # No long-form template markers leaked into the Shorts description.
    assert "CHAPTERS" not in res["description"]
    assert "Disclaimer:" not in res["description"]
    assert "Kohli ne maara chakka" in res["description"]
    assert res["ai_generated"] is True
    assert res["is_shorts"] is True


def test_long_form_gets_structured_description():
    ai_json = json.dumps({
        "title": "LIVE RCB vs CSK | Royal Challengers vs Chennai",
        "description": "short ai text",
        "search_terms": ["rcb vs csk live", "ipl live hindi"],
        "hashtags": ["#RCBvCSK"],
    })
    with patch("utils.ai_client.AIClient.generate_fastest_first", return_value=ai_json):
        res = generate_clip_seo(clip_id="c2", transcript="kohli hit six",
                                video_title="RCB vs CSK", is_shorts=False)
    # Long-form path builds the structured layout.
    assert "CHAPTERS" in res["description"]
    assert res["ai_generated"] is True


def test_escalation_then_queue_on_total_failure_no_generic():
    """When the racer returns nothing AND escalation fails, raise (queue),
    never emit generic SEO."""
    with patch("utils.ai_client.AIClient.generate_fastest_first", return_value=""), \
         patch("utils.ai_client.AIClient.generate_text", side_effect=RuntimeError("all down")):
        with pytest.raises(SEOGenerationError):
            generate_clip_seo(clip_id="c3", transcript="kohli six", video_title="RCB vs CSK")


def test_escalation_recovers_via_failover_chain():
    """If the racer fails but the stricter-prompt failover retry succeeds,
    we get AI content (escalation, not degradation)."""
    good = json.dumps({"title": "KOHLI 100! #Shorts", "description": "ton up! #Shorts",
                       "search_terms": ["kohli century", "ipl live"], "hashtags": ["#Shorts"]})
    with patch("utils.ai_client.AIClient.generate_fastest_first", return_value=""), \
         patch("utils.ai_client.AIClient.generate_text", return_value=good):
        res = generate_clip_seo(clip_id="c4", transcript="kohli century", video_title="RCB vs CSK")
    assert res["ai_generated"] is True
    assert "KOHLI 100" in res["title"]


def test_queue_marker_written_and_no_metadata(tmp_path):
    """generate_seo_for_exported_clip must NOT write *_metadata.json on failure
    (so upload skips it); it writes a *_seo_failed.json queue marker instead."""
    with patch("utils.ai_client.AIClient.generate_fastest_first", return_value=""), \
         patch("utils.ai_client.AIClient.generate_text", side_effect=RuntimeError("down")):
        out = generate_seo_for_exported_clip(
            clip_id="clipX", transcript="kohli six", output_dir=str(tmp_path),
            video_title="RCB vs CSK",
        )
    assert out.get("_seo_failed") is True
    assert not (tmp_path / "clipX_metadata.json").exists()
    assert (tmp_path / "clipX_seo_failed.json").exists()


def test_find_canonical_entities_grounds_names():
    ents = find_canonical_entities("Bumrah bowled a yorker to Babar Azam at Wankhede")
    assert "Jasprit Bumrah" in ents["players"]
    assert "Babar Azam" in ents["players"]


def test_head_false_positive_removed():
    """'head' must no longer be force-corrected to 'Travis Head' (English word)."""
    assert "Travis Head" not in correct_cricket_spelling("over the head for six")



# ─── feat: SEO retry queue (closes the escalate-not-degrade loop) ─────────────

from automation.seo.seo import retry_failed_seo


def test_retry_failed_seo_recovers_and_dequeues(tmp_path):
    """A queued clip (*_seo_failed.json) is retried; on success the metadata is
    written and the marker removed."""
    # First, force a failure so a self-contained marker is written.
    with patch("utils.ai_client.AIClient.generate_fastest_first", return_value=""), \
         patch("utils.ai_client.AIClient.generate_text", side_effect=RuntimeError("down")):
        generate_seo_for_exported_clip(
            clip_id="clipR", transcript="kohli six at wankhede",
            output_dir=str(tmp_path), video_title="RCB vs CSK",
        )
    marker = tmp_path / "clipR_seo_failed.json"
    assert marker.exists()
    assert not (tmp_path / "clipR_metadata.json").exists()
    # Marker is self-contained (carries transcript/context for retry).
    ctx = json.loads(marker.read_text())
    assert ctx["transcript"] == "kohli six at wankhede"
    assert ctx["video_title"] == "RCB vs CSK"

    # Now the LLM recovers -> retry must succeed, write metadata, drop marker.
    good = json.dumps({"title": "KOHLI SIX! #Shorts", "description": "chakka! #Shorts",
                       "search_terms": ["kohli six", "ipl live hindi"], "hashtags": ["#Shorts"]})
    with patch("utils.ai_client.AIClient.generate_fastest_first", return_value=good):
        summary = retry_failed_seo(str(tmp_path))

    assert summary["recovered"] == 1
    assert summary["still_failed"] == 0
    assert (tmp_path / "clipR_metadata.json").exists()
    assert not marker.exists()  # dequeued


def test_retry_failed_seo_keeps_marker_on_repeated_failure(tmp_path):
    with patch("utils.ai_client.AIClient.generate_fastest_first", return_value=""), \
         patch("utils.ai_client.AIClient.generate_text", side_effect=RuntimeError("down")):
        generate_seo_for_exported_clip(
            clip_id="clipS", transcript="dhoni finish", output_dir=str(tmp_path),
            video_title="CSK vs MI",
        )
        # Still down on retry -> marker stays, no metadata.
        summary = retry_failed_seo(str(tmp_path))
    assert summary["still_failed"] == 1
    assert (tmp_path / "clipS_seo_failed.json").exists()
    assert not (tmp_path / "clipS_metadata.json").exists()


def test_retry_failed_seo_noop_when_empty(tmp_path):
    assert retry_failed_seo(str(tmp_path)) == {"retried": 0, "recovered": 0, "still_failed": 0}


# ─── Root seo.py stub must be clean re-export (no generic fallback drift) ─────

def test_root_seo_stub_is_clean_re_export():
    """Root seo.py must NOT contain generic fallback logic (Invariant 3+4).
    It must be a pure re-export from automation.seo.seo."""
    import seo as root_seo
    # Should NOT have the forbidden functions from the old implementation
    forbidden = ["_generate_template_seo", "_translate_hindi_to_english", "_salvaged"]
    for name in forbidden:
        assert not hasattr(root_seo, name), \
            "root seo.py must not contain %s — generic fallback drift" % name
    # Should HAVE the canon functions via re-export
    assert hasattr(root_seo, "SEOGenerationError")
    assert hasattr(root_seo, "process_all_seo")
    assert hasattr(root_seo, "generate_seo_for_exported_clip")
    # The canon _enforce_limits must NOT have safe_defaults padding
    assert "safe_defaults" not in root_seo.__dict__, \
        "root seo.py must not define safe_defaults — would be generic tag padding"
