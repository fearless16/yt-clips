import pytest
import time
from unittest.mock import patch, MagicMock
from utils.ai_client import AIClient, classify_error, ErrorCategory


def reset_shared_state():
    with AIClient._shared_lock:
        AIClient._provider_failures.clear()
        AIClient._provider_cooldown_until.clear()
        AIClient._provider_token_buckets.clear()


def test_provider_circuit_breaker():
    reset_shared_state()
    ai = AIClient()
    ai.opencode_api_key = "mock_oc"
    ai.nvidia_api_key = None
    ai.groq_api_key = None

    with patch.object(ai, "generate_opencode", side_effect=RuntimeError("OC Down")), \
         patch.object(ai, "generate_ollama", side_effect=RuntimeError("Ollama Down")):
        for _ in range(5):
            try:
                ai.generate_text("test prompt")
            except RuntimeError:
                pass

        with pytest.raises(RuntimeError) as exc_info:
            ai.generate_text("test prompt")
        assert "All LLM providers failed" in str(exc_info.value)


def test_rate_limit_token_bucket():
    reset_shared_state()
    with AIClient._shared_lock:
        AIClient._provider_token_buckets["opencode"] = {
            "capacity": 30.0,
            "tokens": 0.0,
            "last_update": time.time(),
            "refill_rate": 0.5
        }
    ai = AIClient()
    ai.opencode_api_key = "mock_oc"
    ai.nvidia_api_key = None
    ai.groq_api_key = None

    with patch.object(ai, "generate_opencode", return_value="OC Fallback") as mock_oc, \
         patch.object(ai, "generate_ollama", side_effect=RuntimeError("Ollama Down")):
        res = ai.generate_text("test prompt")
        assert res == "OC Fallback"
        mock_oc.assert_called_once()


# --- Error classification tests ---

def test_classify_error_auth_failure():
    exc = Exception("Invalid API key")
    exc.status_code = 401
    assert classify_error(exc) == ErrorCategory.AUTH_FAILURE

def test_classify_error_rate_limit():
    exc = Exception("rate limited")
    exc.status_code = 429
    assert classify_error(exc) == ErrorCategory.RATE_LIMIT

def test_classify_error_quota():
    exc = Exception("quota exceeded for this month")
    assert classify_error(exc) == ErrorCategory.QUOTA_EXHAUSTED

def test_classify_error_model_not_found():
    exc = Exception("model not found")
    exc.status_code = 404
    assert classify_error(exc) == ErrorCategory.MODEL_NOT_FOUND

def test_classify_error_timeout():
    exc = TimeoutError("connection timed out")
    assert classify_error(exc) == ErrorCategory.TIMEOUT

def test_classify_error_server_error():
    exc = Exception("internal server error")
    exc.status_code = 500
    assert classify_error(exc) == ErrorCategory.SERVER_ERROR

def test_auth_failure_stops_retry():
    reset_shared_state()
    ai = AIClient()
    ai.opencode_api_key = "mock_oc"
    ai.nvidia_api_key = None
    ai.groq_api_key = None

    auth_exc = Exception("Invalid API key")
    auth_exc.status_code = 401

    with patch.object(ai, "generate_opencode", side_effect=auth_exc):
        with pytest.raises(RuntimeError, match="Auth failures detected"):
            ai.generate_text("test prompt")

def test_quota_exhaustion_skips_provider():
    reset_shared_state()
    ai = AIClient()
    ai.opencode_api_key = "mock_oc"
    ai.nvidia_api_key = "mock_nv"
    ai.groq_api_key = None
    ai._model = "qwen3.6-plus"

    quota_exc = Exception("quota exceeded")
    with patch.object(ai, "generate_opencode", side_effect=quota_exc), \
         patch.object(ai, "generate_nvidia", return_value="NV OK") as mock_nv:
        res = ai.generate_text("test prompt")
        assert res == "NV OK"
        assert AIClient._in_cooldown("opencode")


# --- 429 / Retry-After + racer health integration ---

class _FakeResp:
    def __init__(self, status_code, retry_after=None):
        self.status_code = status_code
        self.headers = {}
        if retry_after is not None:
            self.headers["retry-after"] = str(retry_after)

class FakeRateLimitError(Exception):
    def __init__(self, retry_after=None):
        super().__init__("rate limited")
        self.status_code = 429
        self.response = _FakeResp(429, retry_after)

def test_classify_rate_limit_and_retry_after():
    e = FakeRateLimitError(retry_after=12)
    assert classify_error(e) == ErrorCategory.RATE_LIMIT
    assert AIClient._is_rate_limited(e) is True

def test_429_sets_cooldown_from_retry_after():
    reset_shared_state()
    AIClient._note_provider_error("opencode", FakeRateLimitError(retry_after=120))
    assert AIClient._in_cooldown("opencode") is True
    cooled = AIClient._provider_cooldown_until["opencode"]
    assert cooled > time.time() + 100

def test_generic_error_does_not_cooldown():
    reset_shared_state()
    AIClient._note_provider_error("opencode", RuntimeError("boom"))
    assert AIClient._in_cooldown("opencode") is False


def test_fastest_first_records_success_and_skips_cooldown():
    reset_shared_state()
    ai = AIClient()
    ai.opencode_api_key = "k"
    ai.nvidia_api_key = None

    with patch.object(AIClient, "generate_opencode", return_value="RACED OK"):
        out = ai.generate_fastest_first("p", "s")
    assert out == "RACED OK"
    assert ai.get_used_provider() == "opencode"
    assert AIClient._in_cooldown("opencode") is False


def test_fastest_first_returns_empty_when_all_fail():
    reset_shared_state()
    ai = AIClient()
    ai.opencode_api_key = "k"
    ai.nvidia_api_key = None
    ai.groq_api_key = None

    with patch.object(AIClient, "generate_opencode", side_effect=FakeRateLimitError(retry_after=60)):
        out = ai.generate_fastest_first("p", "s")
    assert out == ""
    assert AIClient._in_cooldown("opencode") is True


def test_fastest_first_keeps_slow_but_valid_response():
    reset_shared_state()
    ai = AIClient()
    ai.opencode_api_key = "k"
    ai.nvidia_api_key = None
    ai.groq_api_key = None

    def slow_ok(prompt, system_instruction=None):
        time.sleep(0.4)
        return "SLOW VALID"

    with patch("utils.config.load_config", return_value={"ai": {"race_tier_timeout_seconds": 5.0}, "logging": {}}):
        with patch.object(AIClient, "generate_opencode", side_effect=slow_ok):
            out = ai.generate_fastest_first("p", "s")
    assert out == "SLOW VALID"


def test_call_provider_random_model_on_failover_not_fixed_index():
    reset_shared_state()
    ai = AIClient()
    ai.opencode_api_key = "k"
    ai._model = "some-non-opencode-model"

    seen = set()
    def capture(prompt, system_instruction=None):
        seen.add(ai._model)
        return "ok"

    with patch.object(ai, "generate_opencode", side_effect=capture):
        for _ in range(40):
            ai._call_provider("opencode", "p", None)

    assert seen.issubset(set(AIClient.PROVIDER_MODELS["opencode"]))
    assert len(seen) > 1


def test_call_provider_honors_prefer_model():
    reset_shared_state()
    ai = AIClient()
    ai.opencode_api_key = "k"
    ai._model = "x"
    target = AIClient.PROVIDER_MODELS["opencode"][2]
    used = {}

    def capture(prompt, system_instruction=None):
        used["model"] = ai._model
        return "ok"

    with patch.object(ai, "generate_opencode", side_effect=capture):
        ai._call_provider("opencode", "p", None, prefer_model=target)
    assert used["model"] == target


def test_call_provider_restores_state_on_exception():
    reset_shared_state()
    ai = AIClient()
    ai.opencode_api_key = "k"
    ai._provider, ai._model = "opencode", "qwen3.6-plus"
    with patch.object(ai, "generate_nvidia", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            ai._call_provider("nvidia", "p", None)
    assert ai._provider == "opencode"
    assert ai._model == "qwen3.6-plus"


def test_token_bucket_is_per_model_not_per_provider():
    reset_shared_state()
    models = AIClient.PROVIDER_MODELS["opencode"]
    m0, m1 = models[0], models[1]
    with AIClient._shared_lock:
        AIClient._provider_token_buckets[f"opencode:{m0}"] = {
            "capacity": 30.0, "tokens": 0.0, "last_update": time.time(), "refill_rate": 0.5,
        }
    assert AIClient._check_and_consume_token("opencode", m0) is False
    assert AIClient._check_and_consume_token("opencode", m1) is True


def test_token_bucket_cooldown_is_per_provider():
    reset_shared_state()
    with AIClient._shared_lock:
        AIClient._provider_cooldown_until["opencode"] = time.time() + 300
    for m in AIClient.PROVIDER_MODELS["opencode"]:
        assert AIClient._check_and_consume_token("opencode", m) is False


def test_token_bucket_capacity_is_config_driven():
    reset_shared_state()
    fake_cfg = {"ai": {"rate_limit": {"capacity": 3, "refill_per_sec": 0.0}}, "logging": {}}
    with patch("utils.ai_client._cfg", fake_cfg):
        # Use a provider NOT in PROVIDER_RATE_LIMITS to test config-driven path
        prov, model = "test_prov", "m"
        ok = sum(1 for _ in range(10) if AIClient._check_and_consume_token(prov, model))
    assert ok == 3


# --- Racer plan tests ---

def test_all_models_shuffles_for_diversity():
    reset_shared_state()
    ai = AIClient()
    ai.opencode_api_key = "k"
    ai.nvidia_api_key = "k"
    first_models = set()
    for _ in range(120):
        models = ai._all_models()
        if models:
            first_models.add(models[0])
    # With 15 models across 2 providers, should see at least 2 different first picks
    assert len(first_models) > 1, f"Only saw {first_models}"


def test_all_models_prefer_model_wins_front():
    reset_shared_state()
    ai = AIClient()
    ai.opencode_api_key = "k"
    ai.nvidia_api_key = None
    target = AIClient.PROVIDER_MODELS["opencode"][2]
    for _ in range(30):
        models = ai._all_models(prefer_model=target)
        assert models[0][1] == target


def test_all_models_prefer_provider_boosts_provider():
    reset_shared_state()
    ai = AIClient()
    ai.opencode_api_key = "k"
    ai.nvidia_api_key = "k"
    for _ in range(20):
        models = ai._all_models(prefer_provider="nvidia")
        assert models[0][0] == "nvidia"


def test_no_deepseek_provider():
    assert "deepseek" not in AIClient.PROVIDER_MODELS


def test_failover_chain_excludes_deepseek():
    reset_shared_state()
    ai = AIClient()
    chain = ai._get_failover_chain("opencode")
    assert "deepseek" not in chain


def test_opencode_is_primary_in_chain():
    reset_shared_state()
    ai = AIClient()
    chain = ai._get_failover_chain("opencode")
    assert chain[0] == "opencode"
    chain2 = ai._get_failover_chain("nvidia")
    assert "opencode" in chain2


def test_opencode_provider_models():
    assert "opencode" in AIClient.PROVIDER_MODELS
    assert "mimo-v2.5-pro" in AIClient.PROVIDER_MODELS["opencode"]
    assert "mimo-v2.5" in AIClient.PROVIDER_MODELS["opencode"]
    assert "minimax-m2.5" in AIClient.PROVIDER_MODELS["opencode"]
    assert "minimax-m3" in AIClient.PROVIDER_MODELS["opencode"]
    assert "qwen3.7-max" in AIClient.PROVIDER_MODELS["opencode"]
    assert "qwen3.7-plus" in AIClient.PROVIDER_MODELS["opencode"]


def test_model_timeouts_include_qwen37():
    assert "qwen3.7-max" in AIClient.MODEL_TIMEOUTS
    assert AIClient.MODEL_TIMEOUTS["qwen3.7-max"] == 180.0
    assert len(AIClient.MODEL_TIMEOUTS) == 1


def test_providers_only_opencode_nvidia_groq():
    assert set(AIClient.PROVIDER_MODELS.keys()) == {"opencode", "nvidia", "groq"}
    assert "openrouter" not in AIClient.PROVIDER_MODELS


def test_nvidia_models():
    assert "nvidia/llama-3.3-nemotron-super-49b-v1" in AIClient.PROVIDER_MODELS["nvidia"]
    assert "meta/llama-3.3-70b-instruct" in AIClient.PROVIDER_MODELS["nvidia"]


def test_xiaomi_mimo_in_opencode():
    assert "mimo-v2.5-pro" in AIClient.PROVIDER_MODELS["opencode"]
    assert "mimo-v2.5" in AIClient.PROVIDER_MODELS["opencode"]
    assert "qwen3.7-plus" in AIClient.PROVIDER_MODELS["opencode"]


def test_total_model_count():
    total = sum(len(v) for v in AIClient.PROVIDER_MODELS.values())
    assert total == 19, f"Expected 19 models, got {total}"


def test_get_available_providers_only_enabled():
    reset_shared_state()
    ai = AIClient()
    ai.opencode_api_key = "k"
    ai.nvidia_api_key = None
    ai.groq_api_key = None
    avail = ai.get_available_providers()
    assert "opencode" in avail
    assert "nvidia" not in avail
    assert "groq" not in avail


# --- SEO-specific generation (OpenCode Go only) ---

class TestGenerateSeoText:

    def setup_method(self):
        reset_shared_state()

    def test_seo_preferred_models_all_opencode(self):
        """SEO_PREFERRED_MODELS must only contain opencode provider."""
        for provider, model in AIClient.SEO_PREFERRED_MODELS:
            assert provider == "opencode", f"Non-opencode model in SEO list: {provider}/{model}"

    def test_seo_preferred_models_priority_order(self):
        """Priority: mimo-v2.5-pro → deepseek-v4-pro."""
        models = [m for _, m in AIClient.SEO_PREFERRED_MODELS]
        assert models == ["mimo-v2.5-pro", "deepseek-v4-pro"]

    def test_seo_text_uses_opencode_only(self):
        """generate_seo_text must only call opencode, never nvidia/groq/ollama."""
        ai = AIClient()
        ai.opencode_api_key = "mock"
        ai.nvidia_api_key = "mock"
        ai.groq_api_key = "mock"

        with patch.object(ai, "generate_opencode", return_value="SEO OK") as mock_oc, \
             patch.object(ai, "generate_nvidia") as mock_nv, \
             patch.object(ai, "generate_groq") as mock_gr, \
             patch.object(ai, "generate_ollama") as mock_ol:
            result = ai.generate_seo_text("test seo prompt")
            assert result == "SEO OK"
            assert mock_oc.called
            mock_nv.assert_not_called()
            mock_gr.assert_not_called()
            mock_ol.assert_not_called()

    def test_seo_text_raises_without_opencode_key(self):
        """generate_seo_text must raise if OpenCode key is missing."""
        ai = AIClient()
        ai.opencode_api_key = None
        with pytest.raises(RuntimeError, match="OpenCode Go API key missing"):
            ai.generate_seo_text("test")

    def test_seo_text_tries_all_models_on_failure(self):
        """Should try mimo-v2.5-pro, deepseek-v4-pro before raising."""
        ai = AIClient()
        ai.opencode_api_key = "mock"

        models_tried = []

        def capture_model(prompt, system_instruction=None):
            models_tried.append(ai._model)
            raise RuntimeError("model down")

        with patch.object(ai, "generate_opencode", side_effect=capture_model):
            with pytest.raises(RuntimeError, match="all OpenCode Go models exhausted"):
                ai.generate_seo_text("test")

        # Should have tried both models at least once (primary + retry rounds)
        assert "mimo-v2.5-pro" in models_tried
        assert "deepseek-v4-pro" in models_tried

    def test_seo_text_falls_through_to_second_model(self):
        """If mimo-v2.5-pro fails, should try deepseek-v4-pro."""
        ai = AIClient()
        ai.opencode_api_key = "mock"

        call_count = [0]

        def selective_fail(prompt, system_instruction=None):
            call_count[0] += 1
            if ai._model == "mimo-v2.5-pro":
                raise RuntimeError("mimo down")
            return "DEEPSEEK OK"

        with patch.object(ai, "generate_opencode", side_effect=selective_fail):
            result = ai.generate_seo_text("test")
            assert result == "DEEPSEEK OK"

    def test_seo_text_never_calls_ollama_fallback(self):
        """Even when all 3 models fail, must NOT fall back to ollama."""
        ai = AIClient()
        ai.opencode_api_key = "mock"

        with patch.object(ai, "generate_opencode", side_effect=RuntimeError("down")), \
             patch.object(ai, "generate_ollama") as mock_ol:
            with pytest.raises(RuntimeError):
                ai.generate_seo_text("test")
            mock_ol.assert_not_called()
