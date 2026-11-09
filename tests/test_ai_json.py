"""Native structured output must never treat truncated reasoning as a decision."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from utils.ai_client import AIClient, JSONResponseError


@pytest.fixture
def route(monkeypatch):
    ai = AIClient()
    ai._provider, ai._model = "deepseek", "deepseek-v4-pro"
    ai.deepseek_api_key = "test-key-not-real"
    monkeypatch.setattr(ai, "_check_and_consume_token", lambda *_: True)
    sdk = MagicMock()
    client = sdk.return_value.__enter__.return_value
    monkeypatch.setattr("utils.ai_client.OpenAI", sdk)
    return ai, sdk, client.chat.completions.create


def reply(content, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish,
        message=SimpleNamespace(content=content, reasoning_content='{"selected": []}'))])


def test_json_requests_native_format_on_exact_configured_route(route):
    ai, sdk, create = route
    create.return_value = reply('{"selected": []}')
    original = ai._provider, ai._model, ai._reasoning_effort
    assert json.loads(ai.generate_json("Return JSON", "Review only source evidence")) == {"selected": []}
    kwargs = create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["model"] == "deepseek-v4-pro"
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_tokens"] == 16384
    assert sdk.call_args.kwargs["max_retries"] == 0  # Arbiter owns its retry budget.
    assert (ai._provider, ai._model, ai._reasoning_effort) == original


@pytest.mark.parametrize("content,finish", [
    ('{"selected": []}', "length"),  # Even parseable content is not a finished decision.
    ('{"selected": [', "stop"),
    (None, "stop"),
    ("   ", "stop"),
    ('{"selected": []}', "content_filter"),
])
def test_incomplete_blank_or_invalid_answer_is_held_without_reasoning_fallback(route, content, finish):
    ai, sdk, create = route
    create.return_value = reply(content, finish)
    with pytest.raises(JSONResponseError) as exc:
        ai.generate_json("Return JSON")
    assert exc.value.finish_reason == finish
    assert exc.value.response_text == (content or "")
    assert create.call_count == sdk.call_count == 1


def test_valid_json_array_is_not_a_decision_object(route):
    ai, _, create = route
    create.return_value = reply("[]")
    with pytest.raises(ValueError, match="must be an object"):
        ai.generate_json("Return JSON")


def test_missing_key_does_not_use_another_provider(route):
    ai, sdk, _ = route
    ai.deepseek_api_key = None
    ai.opencode_api_key = "another-provider-key"
    with pytest.raises(ValueError, match="API key required"):
        ai.generate_json("Return JSON")
    sdk.assert_not_called()


def test_failed_json_request_does_not_silently_fall_back(route):
    ai, sdk, create = route
    create.side_effect = RuntimeError("provider down")
    with pytest.raises(RuntimeError, match="provider down"):
        ai.generate_json("Return JSON")
    assert sdk.call_count == 1


def test_error_evidence_is_bounded(route):
    ai, _, create = route
    create.return_value = reply("x" * 20000, "length")
    with pytest.raises(JSONResponseError) as exc:
        ai.generate_json("Return JSON")
    assert len(exc.value.response_text) == 16000
    assert exc.value.response_chars == 20000


def test_editorial_high_thinking_is_explicit_and_has_its_own_budget(route):
    ai, sdk, create = route
    create.return_value = reply('{"selected": []}')
    ai.generate_json("Return JSON", reasoning_effort="high", max_tokens=32768, timeout=300)
    request = create.call_args.kwargs
    assert request["reasoning_effort"] == "high"
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert request["max_tokens"] == 32768
    assert sdk.call_args.kwargs["timeout"] == 300
    assert request["model"] == "deepseek-v4-pro"
