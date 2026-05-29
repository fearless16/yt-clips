import pytest
import time
from unittest.mock import patch, MagicMock
from utils.ai_client import AIClient

def reset_shared_state():
    with AIClient._shared_lock:
        AIClient._provider_failures.clear()
        AIClient._provider_cooldown_until.clear()
        AIClient._provider_token_buckets.clear()

def test_deepseek_failover():
    reset_shared_state()
    ai = AIClient()
    ai.groq_api_key = "mock_groq"
    ai.deepseek_api_key = "mock_deepseek"
    ai.openrouter_api_key = None
    ai.nvidia_api_key = None
    
    with patch.object(ai, "generate_groq", side_effect=RuntimeError("Groq Down")), \
         patch.object(ai, "generate_deepseek", return_value="DeepSeek Success") as mock_ds:
        
        res = ai.generate_text("test prompt")
        assert res == "DeepSeek Success"
        mock_ds.assert_called_once_with("test prompt", None)

def test_provider_circuit_breaker():
    reset_shared_state()
    ai = AIClient()
    ai.groq_api_key = "mock_groq"
    ai.deepseek_api_key = None
    ai.openrouter_api_key = None
    ai.nvidia_api_key = None
    
    with patch.object(ai, "generate_groq", side_effect=RuntimeError("Groq Down")), \
         patch.object(ai, "generate_ollama", side_effect=RuntimeError("Ollama Down")):
        # Fail 5 times to trigger circuit breaker
        for _ in range(5):
            try:
                ai.generate_text("test prompt")
            except RuntimeError:
                pass
                
        # 6th attempt should skip Groq due to cooldown and raise RuntimeError (or try Ollama and fail)
        with pytest.raises(RuntimeError) as exc_info:
            ai.generate_text("test prompt")
        assert "All LLM providers failed" in str(exc_info.value)

def test_rate_limit_token_bucket():
    reset_shared_state()
    # Empty token bucket for groq
    with AIClient._shared_lock:
        AIClient._provider_token_buckets["groq"] = {
            "capacity": 30.0,
            "tokens": 0.0,
            "last_update": time.time(),
            "refill_rate": 0.5
        }
    ai = AIClient()
    ai.groq_api_key = "mock_groq"
    ai.deepseek_api_key = None
    ai.openrouter_api_key = None
    ai.nvidia_api_key = None
    
    # Should skip groq due to rate limit, then fall back to it
    with patch.object(ai, "generate_groq", return_value="Groq Fallback") as mock_groq, \
         patch.object(ai, "generate_ollama", side_effect=RuntimeError("Ollama Down")):
        res = ai.generate_text("test prompt")
        assert res == "Groq Fallback"
        mock_groq.assert_called_once()



# ─── 429 / Retry-After + racer health integration (feat/llm-orchestration) ────

class _FakeResp:
    def __init__(self, status_code, retry_after=None):
        self.status_code = status_code
        self.headers = {}
        if retry_after is not None:
            self.headers["retry-after"] = str(retry_after)


class FakeRateLimitError(Exception):
    """Mimics openai.RateLimitError (status_code + response.headers)."""
    def __init__(self, retry_after=None):
        super().__init__("rate limited")
        self.status_code = 429
        self.response = _FakeResp(429, retry_after)


def test_classify_rate_limit_and_retry_after():
    e = FakeRateLimitError(retry_after=12)
    assert AIClient._is_rate_limited(e) is True
    assert AIClient._is_retryable(e) is True
    assert AIClient._retry_after_seconds(e) == 12.0


def test_429_sets_cooldown_from_retry_after():
    reset_shared_state()
    AIClient._note_provider_error("groq", FakeRateLimitError(retry_after=120))
    # Provider should now be in cooldown (skips further calls).
    assert AIClient._in_cooldown("groq") is True
    cooled = AIClient._provider_cooldown_until["groq"]
    assert cooled > time.time() + 100  # ~120s honored


def test_generic_error_does_not_cooldown():
    reset_shared_state()
    AIClient._note_provider_error("groq", RuntimeError("boom"))
    assert AIClient._in_cooldown("groq") is False  # only rate-limits cool down


def test_fastest_first_records_success_and_skips_cooldown():
    reset_shared_state()
    ai = AIClient()
    ai.groq_api_key = "k"
    ai.deepseek_api_key = None
    ai.openrouter_api_key = None
    ai.nvidia_api_key = None

    # Patch the per-thread make_call path by patching generate_groq on the class
    with patch.object(AIClient, "generate_groq", return_value="RACED OK"):
        out = ai.generate_fastest_first("p", "s")
    assert out == "RACED OK"
    assert ai.get_used_provider() == "groq"
    # success resets failure count / cooldown
    assert AIClient._in_cooldown("groq") is False


def test_fastest_first_returns_empty_when_all_fail_no_generic_fallback():
    reset_shared_state()
    ai = AIClient()
    ai.groq_api_key = "k"
    ai.deepseek_api_key = None
    ai.openrouter_api_key = None
    ai.nvidia_api_key = None

    with patch.object(AIClient, "generate_groq", side_effect=FakeRateLimitError(retry_after=60)):
        out = ai.generate_fastest_first("p", "s")
    # Escalation contract: empty result, NOT a generic fallback string.
    assert out == ""
    # The 429 should have cooled groq down.
    assert AIClient._in_cooldown("groq") is True


def test_fastest_first_keeps_slow_but_valid_response():
    """A slow valid response within the tier deadline must NOT be dropped."""
    reset_shared_state()
    ai = AIClient()
    ai.groq_api_key = "k"
    ai.deepseek_api_key = None
    ai.openrouter_api_key = None
    ai.nvidia_api_key = None

    def slow_ok(prompt, system_instruction=None):
        time.sleep(0.4)
        return "SLOW VALID"

    with patch("utils.config.load_config", return_value={"ai": {"race_tier_timeout_seconds": 5.0}, "logging": {}}):
        with patch.object(AIClient, "generate_groq", side_effect=slow_ok):
            out = ai.generate_fastest_first("p", "s")
    assert out == "SLOW VALID"
