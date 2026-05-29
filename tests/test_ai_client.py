import pytest
import time
from unittest.mock import patch, MagicMock
from utils.ai_client import AIClient

def test_deepseek_failover():
    ai = AIClient()
    ai.groq_api_key = "mock_groq"
    ai.deepseek_api_key = "mock_deepseek"
    
    with patch.object(ai, "generate_groq", side_effect=RuntimeError("Groq Down")), \
         patch.object(ai, "generate_deepseek", return_value="DeepSeek Success") as mock_ds:
        
        res = ai.generate_text("test prompt")
        assert res == "DeepSeek Success"
        mock_ds.assert_called_once_with("test prompt", None)

def test_provider_circuit_breaker():
    ai = AIClient()
    ai.groq_api_key = "mock_groq"
    
    with AIClient._shared_lock:
        AIClient._provider_failures["groq"] = 0
        AIClient._provider_cooldown_until["groq"] = 0.0
        
    with patch.object(ai, "generate_groq", side_effect=RuntimeError("Groq Down")):
        # Fail 5 times to trigger circuit breaker
        for _ in range(5):
            try:
                ai.generate_text("test prompt")
            except RuntimeError:
                pass
                
        # 6th attempt should skip Groq due to cooldown and raise RuntimeError
        with patch.object(ai, "generate_ollama", side_effect=RuntimeError("Ollama Down")):
            with pytest.raises(RuntimeError) as exc_info:
                ai.generate_text("test prompt")
            assert "All LLM providers failed" in str(exc_info.value)

def test_rate_limit_token_bucket():
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
    
    # Should skip groq due to rate limit, then fall back to it
    with patch.object(ai, "generate_groq", return_value="Groq Fallback") as mock_groq:
        res = ai.generate_text("test prompt")
        assert res == "Groq Fallback"
        mock_groq.assert_called_once()
