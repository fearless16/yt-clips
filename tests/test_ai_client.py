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
