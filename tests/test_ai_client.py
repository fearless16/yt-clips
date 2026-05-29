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



# ─── Bug 2: failover model selection is no longer deterministic ──────────────

def test_call_provider_random_model_on_failover_not_fixed_index():
    """_call_provider must NOT always land on a fixed models[0]/[1] when the
    current model doesn't belong to the target provider. Over many calls it
    should exercise more than one of the provider's models (Bug 2)."""
    reset_shared_state()
    ai = AIClient()
    ai.groq_api_key = "k"
    ai._model = "some-non-groq-model"  # forces selection within groq's list

    seen = set()

    def capture(prompt, system_instruction=None):
        seen.add(ai._model)
        return "ok"

    with patch.object(ai, "generate_groq", side_effect=capture):
        for _ in range(40):
            ai._call_provider("groq", "p", None)

    # All chosen models must be valid groq models...
    assert seen.issubset(set(AIClient.PROVIDER_MODELS["groq"]))
    # ...and selection must not be pinned to a single fixed model.
    assert len(seen) > 1


def test_call_provider_honors_prefer_model():
    reset_shared_state()
    ai = AIClient()
    ai.groq_api_key = "k"
    ai._model = "x"
    target = AIClient.PROVIDER_MODELS["groq"][2]
    used = {}

    def capture(prompt, system_instruction=None):
        used["model"] = ai._model
        return "ok"

    with patch.object(ai, "generate_groq", side_effect=capture):
        ai._call_provider("groq", "p", None, prefer_model=target)
    assert used["model"] == target


def test_call_provider_restores_state_on_exception():
    """A failing provider must not leave the singleton mis-pointed (try/finally)."""
    reset_shared_state()
    ai = AIClient()
    ai.groq_api_key = "k"
    ai._provider, ai._model = "groq", "meta-llama/llama-4-scout-17b-16e-instruct"
    with patch.object(ai, "generate_nvidia", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            ai._call_provider("nvidia", "p", None)
    assert ai._provider == "groq"
    assert ai._model == "meta-llama/llama-4-scout-17b-16e-instruct"


# ─── Bug 3: token bucket is per (provider, model), not per provider ──────────

def test_token_bucket_is_per_model_not_per_provider():
    """Racing several models of the same provider must draw from independent
    buckets. Exhausting one model's bucket must NOT block its siblings."""
    reset_shared_state()
    m0, m1 = AIClient.PROVIDER_MODELS["groq"][0], AIClient.PROVIDER_MODELS["groq"][1]
    # Drain ONLY m0's bucket.
    with AIClient._shared_lock:
        AIClient._provider_token_buckets[f"groq:{m0}"] = {
            "capacity": 30.0, "tokens": 0.0, "last_update": time.time(), "refill_rate": 0.5,
        }
    assert AIClient._check_and_consume_token("groq", m0) is False  # drained
    assert AIClient._check_and_consume_token("groq", m1) is True   # sibling unaffected


def test_token_bucket_cooldown_is_per_provider():
    """A provider cooldown (circuit breaker / 429) blocks ALL of its models."""
    reset_shared_state()
    with AIClient._shared_lock:
        AIClient._provider_cooldown_until["groq"] = time.time() + 300
    for m in AIClient.PROVIDER_MODELS["groq"]:
        assert AIClient._check_and_consume_token("groq", m) is False


def test_token_bucket_capacity_is_config_driven():
    reset_shared_state()
    fake_cfg = {"ai": {"rate_limit": {"capacity": 3, "refill_per_sec": 0.0}}, "logging": {}}
    with patch("utils.ai_client._cfg", fake_cfg):
        prov, model = "groq", "m"
        ok = sum(1 for _ in range(10) if AIClient._check_and_consume_token(prov, model))
    # Exactly `capacity` calls succeed before the (non-refilling) bucket empties.
    assert ok == 3


# ─── Bugs 1 & 5: racer plan is diverse + honors prefer_provider/prefer_model ──

def _plan_first_picks(ai, n=60, **kw):
    return [plan[0][0] for plan in (ai._available_plan(**kw) for _ in range(n)) if plan]


def test_available_plan_shuffles_for_diversity():
    """Across calls, tier-1 must not always lead with the same model (Bug 5)."""
    reset_shared_state()
    ai = AIClient()
    ai.groq_api_key = "k"  # only groq -> tier 1 has 3 groq models
    ai.deepseek_api_key = ai.openrouter_api_key = ai.nvidia_api_key = None
    leaders = {ai._available_plan()[0][0][1] for _ in range(60)}
    assert len(leaders) > 1  # more than one model leads over time


def test_available_plan_prefer_model_wins_front():
    reset_shared_state()
    ai = AIClient()
    ai.groq_api_key = "k"
    ai.deepseek_api_key = ai.openrouter_api_key = ai.nvidia_api_key = None
    target = AIClient.PROVIDER_MODELS["groq"][2]
    for _ in range(30):
        first_provider, first_model = ai._available_plan(prefer_model=target)[0][0]
        assert first_model == target  # exact model always boosted to the front


def test_available_plan_prefer_provider_boosts_provider():
    reset_shared_state()
    ai = AIClient()
    ai.groq_api_key = ai.deepseek_api_key = "k"
    ai.openrouter_api_key = ai.nvidia_api_key = None
    # Tier 2 contains both deepseek and (no) others available; ensure deepseek
    # leads its tier when preferred.
    for _ in range(20):
        plan = ai._available_plan(prefer_provider="deepseek")
        # find the tier that contains deepseek
        for tier in plan:
            provs = [p for p, _ in tier]
            if "deepseek" in provs:
                assert tier[0][0] == "deepseek"
                break
