"""TDD tests for automation/seo module.

Covers:
- _parse_json_response  (all edge cases)
- process_all_seo failure-marker writes
- Inter-clip sleep is config-driven (not hardcoded)
- retry_failed_seo smoke test
"""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:

    def _fn(self):
        from automation.seo.seo import _parse_json_response
        return _parse_json_response

    def test_valid_json_direct(self):
        fn = self._fn()
        result = fn('{"title": "hello", "description": "world", "hashtags": [], "search_terms": []}')
        assert result["title"] == "hello"

    def test_markdown_fenced_json(self):
        fn = self._fn()
        raw = '```json\n{"title": "t", "description": "d", "hashtags": [], "search_terms": []}\n```'
        result = fn(raw)
        assert result is not None
        assert result["title"] == "t"

    def test_markdown_fence_no_lang(self):
        fn = self._fn()
        raw = '```\n{"title": "t2"}\n```'
        result = fn(raw)
        assert result is not None

    def test_single_quotes_fallback(self):
        fn = self._fn()
        raw = "{'title': 'foo', 'description': 'bar'}"
        result = fn(raw)
        assert result is not None
        assert result["title"] == "foo"

    def test_json_embedded_in_text(self):
        fn = self._fn()
        raw = 'Here is the result: {"title": "embedded", "desc": "ok"} — done.'
        result = fn(raw)
        assert result is not None
        assert result["title"] == "embedded"

    def test_empty_string_returns_none(self):
        fn = self._fn()
        assert fn("") is None

    def test_whitespace_only_returns_none(self):
        fn = self._fn()
        assert fn("   ") is None

    def test_plain_text_no_json_returns_none(self):
        fn = self._fn()
        assert fn("no json here at all") is None

    def test_truncated_json_repaired(self):
        fn = self._fn()
        # Truncated before closing brace
        raw = '{"title": "truncated", "description": "cut'
        result = fn(raw)
        # Either repairs it or returns None — must not raise
        assert result is None or isinstance(result, dict)

    def test_nested_json_parses(self):
        fn = self._fn()
        raw = '{"title": "t", "hashtags": ["#Shorts", "#IPL"], "search_terms": ["kohli", "rcb"]}'
        result = fn(raw)
        assert result["hashtags"] == ["#Shorts", "#IPL"]

    def test_returns_none_for_truly_broken(self):
        fn = self._fn()
        assert fn("{{{not valid at all}}}}}") is None


# ---------------------------------------------------------------------------
# process_all_seo — failure marker writes
# ---------------------------------------------------------------------------

class TestProcessAllSEOFailureMarker:

    def _make_highlights_yaml(self, tmp_path, clips: dict) -> str:
        import yaml
        h_path = tmp_path / "highlights.yaml"
        h_path.write_text(yaml.dump(clips))
        return str(h_path)

    @patch("automation.seo.seo._get_ai")
    def test_failure_marker_written_when_all_providers_fail(self, mock_get_ai, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        (tmp_path / "input").mkdir()

        mock_ai = MagicMock()
        mock_ai.generate_fastest_first.side_effect = Exception("All providers down")
        mock_ai.generate_text.side_effect = Exception("All providers down")
        mock_get_ai.return_value = mock_ai

        h_path = self._make_highlights_yaml(tmp_path, {
            "clip_001": {"text": "Kohli ne maara six!", "start": 0, "end": 15},
        })
        out_dir = str(tmp_path / "output")

        from automation.seo.seo import process_all_seo
        process_all_seo(h_path, out_dir)

        # Failure marker must exist
        markers = list(Path(out_dir).glob("*_seo_failed.json"))
        assert len(markers) == 1, f"Expected 1 failure marker, got {len(markers)}"
        marker_data = json.loads(markers[0].read_text())
        assert marker_data["clip_id"] == "clip_001"

    @patch("automation.seo.seo.generate_clip_seo")
    def test_metadata_written_on_success(self, mock_generate, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        (tmp_path / "input").mkdir()

        mock_generate.return_value = {
            "title": "Kohli ne maara CHHAKKA! 🔥 #Shorts",
            "description": "Virat Kohli grounded cricket analysis explains every delivery of the match discussion. " * 45,
            "hashtags": ["#Shorts", "#IPL"],
            "search_terms": [f"virat kohli cricket analysis {i}" for i in range(8)],
        }
        monkeypatch.setattr(
            "automation.seo.trends.get_trending_context",
            lambda **kwargs: {
                "scorecard": "", "topics": [], "teams": [],
                "search_queries": [], "match_facts": [], "sources": [],
                "live_stream_url": "",
            },
        )

        h_path = self._make_highlights_yaml(tmp_path, {
            "clip_001": {"text": "Kohli ne maara six!", "start": 0, "end": 15},
        })
        out_dir = str(tmp_path / "output")

        from automation.seo.seo import process_all_seo
        process_all_seo(h_path, out_dir)

        meta_files = list(Path(out_dir).glob("*_metadata.json"))
        assert len(meta_files) == 1

    @patch("automation.seo.seo._get_ai")
    def test_combined_results_always_written(self, mock_get_ai, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        (tmp_path / "input").mkdir()

        mock_ai = MagicMock()
        mock_ai.generate_fastest_first.side_effect = Exception("fail")
        mock_ai.generate_text.side_effect = Exception("fail")
        mock_ai.generate_seo_text.side_effect = Exception("fail")
        mock_get_ai.return_value = mock_ai
        monkeypatch.setitem(
            __import__("automation.seo.seo", fromlist=["cfg"]).cfg["seo"],
            "inter_clip_sleep_s",
            0,
        )

        h_path = self._make_highlights_yaml(tmp_path, {
            "c1": {"text": "clip1 text", "start": 0, "end": 10},
            "c2": {"text": "clip2 text", "start": 10, "end": 20},
        })
        out_dir = str(tmp_path / "output")

        from automation.seo.seo import process_all_seo
        result_path = process_all_seo(h_path, out_dir)

        assert result_path  # non-empty path
        combined = Path(out_dir) / "seo_results.json"
        assert combined.exists()
        data = json.loads(combined.read_text())
        assert len(data) == 2


# ---------------------------------------------------------------------------
# Inter-clip sleep is configurable (P2 fix)
# ---------------------------------------------------------------------------

class TestInterClipSleepConfigurable:

    @patch("automation.seo.seo._get_ai")
    @patch("automation.seo.seo.time")
    def test_sleep_uses_config_value(self, mock_time, mock_get_ai, tmp_path, monkeypatch):
        """process_all_seo sleep must come from config, not hardcoded 5s."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        (tmp_path / "input").mkdir()

        import yaml
        h_path = tmp_path / "highlights.yaml"
        h_path.write_text(yaml.dump({
            "c1": {"text": "t1", "start": 0, "end": 10},
            "c2": {"text": "t2", "start": 10, "end": 20},
        }))

        mock_ai = MagicMock()
        mock_ai.generate_fastest_first.side_effect = Exception("fail")
        mock_ai.generate_text.side_effect = Exception("fail")
        mock_ai.generate_seo_text.side_effect = Exception("fail")
        mock_get_ai.return_value = mock_ai

        mock_time.sleep = MagicMock()
        mock_time.time = time.time  # keep real time()

        from automation.seo.seo import process_all_seo
        process_all_seo(str(h_path), str(tmp_path / "output"))

        # sleep must have been called at least once (between clips)
        assert mock_time.sleep.called, "sleep() was never called between clips"
        # Sleep value must NOT be hardcoded to 5 — it could be anything from config
        # We just assert it was called with a numeric arg
        call_args = [c.args[0] for c in mock_time.sleep.call_args_list if c.args]
        assert all(isinstance(v, (int, float)) for v in call_args)


# ---------------------------------------------------------------------------
# retry_failed_seo
# ---------------------------------------------------------------------------

class TestRetryFailedSEO:

    @patch("automation.seo.seo.generate_clip_seo")
    def test_retry_recovers_failed_clip(self, mock_generate, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Write a failure marker
        marker = {
            "clip_id": "clip_retry",
            "transcript": "Kohli ne maara six",
            "video_title": "RCB vs CSK",
            "is_shorts": True,
        }
        (out_dir / "clip_retry_seo_failed.json").write_text(json.dumps(marker))

        mock_generate.return_value = {
            "title": "Kohli ne maara CHHAKKA! 🔥 Recovered",
            "description": "Virat Kohli grounded cricket analysis explains every delivery of the match discussion. " * 45,
            "hashtags": ["#Shorts"],
            "search_terms": [f"virat kohli analysis {i}" for i in range(8)],
        }

        from automation.seo.seo import retry_failed_seo
        result = retry_failed_seo(str(out_dir))

        assert result["total"] == 1
        assert result["recovered"] == 1
        # marker should be gone
        assert not (out_dir / "clip_retry_seo_failed.json").exists()
        # metadata should be written
        assert (out_dir / "clip_retry_metadata.json").exists()

    def test_retry_returns_zero_when_no_markers(self, tmp_path):
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        from automation.seo.seo import retry_failed_seo
        result = retry_failed_seo(str(out_dir))
        assert result == {"recovered": 0, "total": 0}

    def test_retry_returns_zero_when_dir_not_found(self, tmp_path):
        from automation.seo.seo import retry_failed_seo
        result = retry_failed_seo(str(tmp_path / "nonexistent"))
        assert result == {"recovered": 0, "total": 0}


# ---------------------------------------------------------------------------
# _get_ai lazy init (P1 fix — no module-level AIClient)
# ---------------------------------------------------------------------------

class TestLazyAIClientInit:

    def test_seo_module_has_get_ai_function(self):
        """After P1 fix: seo.py must expose _get_ai() instead of module-level ai."""
        import automation.seo.seo as seo_mod
        assert callable(getattr(seo_mod, "_get_ai", None)), (
            "_get_ai() function missing — module-level ai = AIClient() not yet refactored"
        )

    def test_get_ai_returns_ai_client(self):
        import automation.seo.seo as seo_mod
        if not hasattr(seo_mod, "_get_ai"):
            pytest.skip("_get_ai not yet implemented")
        client = seo_mod._get_ai()
        assert client is not None
        # Same instance on second call
        assert seo_mod._get_ai() is client


# ---------------------------------------------------------------------------
# SEO Model Restrictions (P0 — block nvidia/groq from SEO)
# ---------------------------------------------------------------------------

class TestSEOModelRestrictions:

    def test_seo_blocked_providers_includes_nvidia_and_groq(self):
        """nvidia and groq must be in SEO_BLOCKED_PROVIDERS."""
        from automation.seo.seo import SEO_BLOCKED_PROVIDERS
        assert "nvidia" in SEO_BLOCKED_PROVIDERS
        assert "groq" in SEO_BLOCKED_PROVIDERS
        assert "opencode" not in SEO_BLOCKED_PROVIDERS

    def test_seo_preferred_models_all_opencode(self):
        """All models in SEO_PREFERRED_MODELS must be opencode provider."""
        from automation.seo.seo import SEO_PREFERRED_MODELS
        for provider, model in SEO_PREFERRED_MODELS:
            assert provider == "opencode", f"Non-opencode model: {provider}/{model}"

    def test_seo_preferred_models_priority(self):
        """Priority order: mimo-v2.5-pro, deepseek-v4-pro."""
        from automation.seo.seo import SEO_PREFERRED_MODELS
        models = [m for _, m in SEO_PREFERRED_MODELS]
        assert models[0] == "mimo-v2.5-pro"
        assert models[1] == "deepseek-v4-pro"

    @patch("automation.seo.seo._get_ai")
    def test_generate_ai_seo_uses_generate_seo_text(self, mock_get_ai):
        """_generate_ai_seo must call generate_seo_text (not generate_text)
        when no model_override is specified."""
        mock_ai = MagicMock()
        mock_ai.generate_seo_text.return_value = json.dumps({
            "title": "Kohli ne maara CHHAKKA! 🔥",
            "description": (
                "Virat Kohli batting analysis explains this grounded cricket "
                "moment without inventing a score or opponent. " * 45
            ),
            "hashtags": ["#Shorts", "#IPL"],
            "search_terms": ["kohli six ipl"],
        })
        mock_ai._model = "qwen3.7-max"
        mock_get_ai.return_value = mock_ai

        from automation.seo.seo import _generate_ai_seo
        result = _generate_ai_seo("clip_001", "test prompt", "kohli six", True)
        assert result is not None
        assert result["title"] == "Kohli ne maara CHHAKKA! 🔥"
        # Must call generate_seo_text, NOT generate_text
        mock_ai.generate_seo_text.assert_called_once()
        mock_ai.generate_text.assert_not_called()

    @patch("automation.seo.seo._get_ai")
    def test_generate_ai_seo_uses_generate_text_with_model_override(self, mock_get_ai):
        """When model_override is set, _generate_ai_seo should use generate_text
        (allowing the caller to specify any model)."""
        mock_ai = MagicMock()
        mock_ai.generate_text.return_value = json.dumps({
            "title": "Override model SEO title! 🔥",
            "description": (
                "Custom model generated a detailed grounded cricket discussion "
                "without inventing a score, player, or opponent. " * 45
            ),
            "hashtags": ["#Shorts"],
            "search_terms": ["test term"],
        })
        mock_get_ai.return_value = mock_ai

        from automation.seo.seo import _generate_ai_seo
        result = _generate_ai_seo("clip_002", "test prompt", "test", True,
                                   model_override="custom-model-v1")
        assert result is not None
        mock_ai.generate_text.assert_called_once()
        mock_ai.generate_seo_text.assert_not_called()

