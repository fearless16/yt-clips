import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, FIRST_COMPLETED, wait
from pathlib import Path
from typing import Dict, List, Optional, Callable

import requests
from dotenv import load_dotenv
from openai import OpenAI

from utils.config import load_config
from utils.logger import get_logger

load_dotenv()

_cfg = load_config()
log = get_logger("ai_client", _cfg.get("logging", {}).get("log_file", "logs/pipeline.log"), _cfg.get("logging", {}).get("level", "INFO"))


# --- Error classification ---

class ErrorCategory:
    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    MODEL_NOT_FOUND = "model_not_found"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    AUTH_FAILURE = "auth_failure"
    UNKNOWN = "unknown"


def _extract_status(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if isinstance(val, int):
            return val
    return None


def _extract_retry_after(exc: Exception) -> Optional[float]:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        raw = None
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def classify_error(exc: Exception) -> str:
    status = _extract_status(exc)
    msg = str(exc).lower()
    name = type(exc).__name__.lower()

    if status in (401, 403) or ("invalid" in msg and "key" in msg) or "auth" in name:
        return ErrorCategory.AUTH_FAILURE
    if status == 429 or "ratelimit" in name or "429" in msg:
        return ErrorCategory.RATE_LIMIT
    if any(w in msg for w in ("quota", "billing", "payment", "insufficient", "exceeded")):
        return ErrorCategory.QUOTA_EXHAUSTED
    if status == 404 or ("model" in msg and ("not found" in msg or "not available" in msg)):
        return ErrorCategory.MODEL_NOT_FOUND
    if status == 504 or "timeout" in name or "timeout" in msg:
        return ErrorCategory.TIMEOUT
    if status is not None and 500 <= status < 600:
        return ErrorCategory.SERVER_ERROR
    if any(s in name for s in ("internalserver", "serviceunavailable", "connectionerror")):
        return ErrorCategory.SERVER_ERROR
    return ErrorCategory.UNKNOWN


class AIClient:
    _shared_lock = threading.Lock()
    _provider_failures = {}
    _provider_cooldown_until = {}
    _provider_token_buckets = {}

    PROVIDER_MODELS = {
        "opencode": ["mimo-v2.5-pro", "mimo-v2.5",
                     "kimi-k2.5", "kimi-k2.6",
                     "glm-5", "glm-5.1",
                     "deepseek-v4-pro", "deepseek-v4-flash",
                     "minimax-m2.5", "minimax-m2.7", "minimax-m3",
                     "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus"],
        "nvidia": ["nvidia/llama-3.3-nemotron-super-49b-v1", "meta/llama-3.3-70b-instruct",
                   "xiaomi/mimo-v2.5-pro", "xiaomi/mimo-v2.5"],
        "groq": ["llama-3.3-70b-versatile", "gemma2-9b-it"],
    }

    # Per-provider rate limit overrides. Groq has strict TPM limits.
    PROVIDER_RATE_LIMITS = {
        "opencode": {"capacity": 30.0, "refill_per_sec": 0.5},
        "nvidia": {"capacity": 30.0, "refill_per_sec": 0.5},
        "groq": {"capacity": 10.0, "refill_per_sec": 0.15},  # ~6K TPM limit
    }

    # Per-model timeouts (seconds) for non-blocking race.
    # qwen3.7-max is slow (~20-120s); others return in 1-10s.
    MODEL_TIMEOUTS = {
        "qwen3.7-max": 180.0,
    }

    def __init__(self):
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        self.nvidia_base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.opencode_api_key = os.getenv("OPENCODE_ZEN_API_KEY")
        self.opencode_base_url = os.getenv("OPENCODE_ZEN_BASE_URL", "https://opencode.ai/zen/go/v1")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        self.ollama_url = "http://localhost:11434/api/generate"
        ai_cfg = _cfg.get("ai", {})
        self._provider = ai_cfg.get("provider", "opencode")
        self._model = ai_cfg.get("model", "mimo-v2.5-pro")
        self._last_provider = None
        self._last_model = None

    @staticmethod
    def _bucket_key(provider: str, model: Optional[str] = None) -> str:
        return f"{provider}:{model}" if model else provider

    @classmethod
    def _rate_limit_cfg(cls, provider: str = None) -> tuple:
        """Get rate limit config for a provider. Uses per-provider overrides if available."""
        if provider and provider in cls.PROVIDER_RATE_LIMITS:
            rl = cls.PROVIDER_RATE_LIMITS[provider]
            return float(rl.get("capacity", 30.0)), float(rl.get("refill_per_sec", 0.5))
        rl = _cfg.get("ai", {}).get("rate_limit", {})
        capacity = float(rl.get("capacity", 30.0))
        refill = float(rl.get("refill_per_sec", 0.5))
        return capacity, refill

    @classmethod
    def _check_and_consume_token(cls, provider: str, model: Optional[str] = None) -> bool:
        key = cls._bucket_key(provider, model)
        capacity, refill_rate = cls._rate_limit_cfg(provider)
        with cls._shared_lock:
            now = time.time()
            cooldown = cls._provider_cooldown_until.get(provider, 0.0)
            if cooldown > now:
                log.debug("Token gate: provider %s in cooldown for %.1fs", provider, cooldown - now)
                return False
            bucket = cls._provider_token_buckets.get(key)
            if bucket is None:
                bucket = {"capacity": capacity, "tokens": capacity, "last_update": now, "refill_rate": refill_rate}
                cls._provider_token_buckets[key] = bucket
            elapsed = now - bucket["last_update"]
            bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + elapsed * bucket["refill_rate"])
            bucket["last_update"] = now
            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True
            log.debug("Token gate: bucket %s exhausted (%.2f tokens)", key, bucket["tokens"])
            return False

    @classmethod
    def _record_success(cls, provider: str):
        with cls._shared_lock:
            cls._provider_failures[provider] = 0
            cls._provider_cooldown_until[provider] = 0.0

    @classmethod
    def _record_failure(cls, provider: str):
        with cls._shared_lock:
            cls._provider_failures[provider] = cls._provider_failures.get(provider, 0) + 1
            if cls._provider_failures[provider] >= 5:
                cls._provider_cooldown_until[provider] = time.time() + 300
                log.warning("Provider %s hit circuit breaker! Cooldown for 5 minutes.", provider)

    @classmethod
    def _is_rate_limited(cls, exc: Exception) -> bool:
        return classify_error(exc) == ErrorCategory.RATE_LIMIT

    @staticmethod
    def _retry_after_seconds(exc: Exception):
        return _extract_retry_after(exc)

    @classmethod
    def _note_provider_error(cls, provider: str, exc: Exception, default_cooldown: float = 30.0):
        cat = classify_error(exc)
        if cat == ErrorCategory.AUTH_FAILURE:
            with cls._shared_lock:
                cls._provider_cooldown_until[provider] = time.time() + 3600
            log.error("Provider %s auth failure - marking unavailable for 1 hour", provider)
        elif cat == ErrorCategory.QUOTA_EXHAUSTED:
            with cls._shared_lock:
                cls._provider_cooldown_until[provider] = time.time() + 3600
            log.warning("Provider %s quota exhausted - switching to next provider", provider)
        elif cat == ErrorCategory.RATE_LIMIT:
            wait_s = _extract_retry_after(exc) or default_cooldown
            with cls._shared_lock:
                prev = cls._provider_cooldown_until.get(provider, 0.0)
                cls._provider_cooldown_until[provider] = max(prev, time.time() + wait_s)
            log.warning("Provider %s rate-limited (429) - cooling down %.0fs", provider, wait_s)
        elif cat == ErrorCategory.MODEL_NOT_FOUND:
            with cls._shared_lock:
                cls._provider_cooldown_until[provider] = time.time() + 60
            log.warning("Provider %s model not found - switching to next compatible model", provider)
        elif cat in (ErrorCategory.TIMEOUT, ErrorCategory.SERVER_ERROR):
            with cls._shared_lock:
                cls._provider_cooldown_until[provider] = time.time() + 15
            log.warning("Provider %s transient error (%s) - short cooldown", provider, cat)
        cls._record_failure(provider)

    @classmethod
    def _in_cooldown(cls, provider: str) -> bool:
        with cls._shared_lock:
            return cls._provider_cooldown_until.get(provider, 0.0) > time.time()

    def _get_failover_chain(self, start_provider: str) -> List[str]:
        chain = ["opencode", "nvidia", "groq"]
        if start_provider in chain:
            chain.remove(start_provider)
            chain.insert(0, start_provider)
        return chain

    def _select_model_for_provider(self, provider: str, prefer_model: Optional[str] = None) -> str:
        models = self.PROVIDER_MODELS.get(provider, [])
        if not models:
            return self._model
        if prefer_model and prefer_model in models:
            return prefer_model
        if self._model in models:
            return self._model
        import random as _random
        return _random.choice(models)

    def _call_provider(self, provider: str, prompt: str, system_instruction: Optional[str] = None,
                       prefer_model: Optional[str] = None) -> str:
        old_provider = self._provider
        old_model = self._model
        chosen_model = self._select_model_for_provider(provider, prefer_model=prefer_model)
        self._provider = provider
        self._model = chosen_model
        log.debug("Routing call to %s/%s", provider, chosen_model)
        try:
            if provider == "opencode":
                return self.generate_opencode(prompt, system_instruction)
            elif provider == "nvidia":
                return self.generate_nvidia(prompt, system_instruction)
            elif provider == "groq":
                return self.generate_groq(prompt, system_instruction)
            else:
                raise ValueError(f"Unknown provider {provider}")
        finally:
            self._provider = old_provider
            self._model = old_model

    def _log_cost(self, provider: str, model: str, input_chars: int, output_chars: int):
        in_tokens = input_chars // 4
        out_tokens = output_chars // 4
        rates = {
            "opencode": {"in": 0.0, "out": 0.0},
            "nvidia": {"in": 0.07, "out": 0.07},
            "ollama": {"in": 0.0, "out": 0.0}
        }
        rate = rates.get(provider, {"in": 0.50, "out": 0.50})
        cost = (in_tokens * rate["in"] + out_tokens * rate["out"]) / 1_000_000
        log.info("LLM cost: %s/%s input=%d output=%d est=$%.6f",
                 provider, model, in_tokens, out_tokens, cost,
                 extra={"stage": "llm_cost", "metadata": {"provider": provider, "model": model, "cost": cost}})

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None,
                      prefer_model: Optional[str] = None) -> str:
        """Text generation with error-classified failover.

        Error handling policy (hardcoded):
        - 429: respect Retry-After, retry same provider
        - quota/billing: mark unavailable, next provider
        - model not found: next compatible model/provider
        - timeout/server: retry once then fallback
        - auth: fail fast, no retry
        """
        ai_cfg = _cfg.get("ai", {})
        base_delay = float(ai_cfg.get("retry_base_delay_seconds", 2.0))
        max_delay = float(ai_cfg.get("retry_max_delay_seconds", 30.0))

        chain = self._get_failover_chain(self._provider)

        available_chain = []
        for p in chain:
            if p == "opencode" and self.opencode_api_key:
                available_chain.append(p)
            elif p == "nvidia" and self.nvidia_api_key:
                available_chain.append(p)
            elif p == "groq" and self.groq_api_key:
                available_chain.append(p)

        errors = []
        retryable_seen = False
        fatal_seen = False
        for provider in available_chain:
            if not self._check_and_consume_token(provider):
                log.info("Skipping provider %s (rate-limited or circuit-breaker cooldown)", provider)
                continue

            try:
                res = self._call_provider(provider, prompt, system_instruction, prefer_model=prefer_model)
                if res:
                    self._record_success(provider)
                    self._log_cost(provider, self.get_used_model(), len(prompt) + len(system_instruction or ""), len(res))
                    log.info("LLM ok via %s/%s (primary round)", provider, self.get_used_model())
                    return res
            except Exception as e:
                cat = classify_error(e)
                retryable_seen = retryable_seen or cat in (ErrorCategory.RATE_LIMIT, ErrorCategory.TIMEOUT, ErrorCategory.SERVER_ERROR)
                if cat == ErrorCategory.AUTH_FAILURE:
                    fatal_seen = True
                    log.error("Auth failure on %s — will not retry this provider", provider)
                self._note_provider_error(provider, e)
                log.warning("Provider %s failed (%s): %s", provider, cat, e)
                errors.append(f"{provider} [{cat}]: {e}")

        # If only auth failures occurred, fail fast — no backoff, no retry round
        if fatal_seen and not retryable_seen:
            raise RuntimeError(f"Auth failures detected — stopping. Errors: {'; '.join(errors)}")

        # Backoff once before retry round, only if failures were transient
        if retryable_seen:
            delay = min(max_delay, base_delay)
            log.info("Transient LLM errors - backing off %.1fs before retry round", delay)
            time.sleep(delay)

        # Retry round: ignore token bucket but still respect provider cooldowns
        for provider in available_chain:
            try:
                if self._in_cooldown(provider):
                    log.info("Retry round: skipping %s (still in cooldown)", provider)
                    continue
                res = self._call_provider(provider, prompt, system_instruction, prefer_model=prefer_model)
                if res:
                    self._record_success(provider)
                    self._log_cost(provider, self.get_used_model(), len(prompt) + len(system_instruction or ""), len(res))
                    log.info("LLM ok via %s/%s (retry round)", provider, self.get_used_model())
                    return res
            except Exception as e:
                cat = classify_error(e)
                if cat == ErrorCategory.AUTH_FAILURE:
                    log.error("Auth failure on %s retry — skipping", provider)
                    continue
                self._note_provider_error(provider, e)
                log.warning("Provider %s failed on retry (%s): %s", provider, cat, e)
                errors.append(f"{provider} (retry) [{cat}]: {e}")

        # Local fallback
        try:
            log.info("All primary LLM providers failed, attempting Ollama...")
            res = self.generate_ollama(prompt, system_instruction)
            if res:
                return res
        except Exception as e:
            errors.append(f"ollama: {e}")

        raise RuntimeError(f"All LLM providers failed. Errors: {'; '.join(errors)}")

    def _all_models(self, prefer_provider: Optional[str] = None,
                    prefer_model: Optional[str] = None) -> List[tuple]:
        import random as _random
        models = []
        for p, ms in self.PROVIDER_MODELS.items():
            if p == "opencode" and not self.opencode_api_key:
                continue
            if p == "nvidia" and not self.nvidia_api_key:
                continue
            if p == "groq" and not self.groq_api_key:
                continue
            for m in ms:
                models.append((p, m))
        _random.shuffle(models)
        if prefer_provider:
            preferred = [x for x in models if x[0] == prefer_provider]
            others = [x for x in models if x[0] != prefer_provider]
            models = preferred + others
        if prefer_model:
            exact = [x for x in models if x[1] == prefer_model]
            rest = [x for x in models if x[1] != prefer_model]
            models = exact + rest
        return models

    def generate_fastest_first(self, prompt: str, system_instruction: Optional[str] = None,
                               prefer_provider: Optional[str] = None,
                               prefer_model: Optional[str] = None) -> str:
        all_models = self._all_models(prefer_provider=prefer_provider, prefer_model=prefer_model)
        runnable = [(p, m) for p, m in all_models if self._check_and_consume_token(p, m)]
        if not runnable:
            log.info("Racer: no API-keyed providers available - falling back to Ollama")
            return self.generate_ollama(prompt, system_instruction)

        def make_call(p, m):
            thread_client = AIClient()
            thread_client._provider = p
            thread_client._model = m
            fn = getattr(thread_client, f"generate_{p}")
            return fn(prompt, system_instruction)

        log.info("Racer: racing %d models simultaneously: %s",
                 len(runnable), ", ".join(f"{p}/{m}" for p, m in runnable))

        default_timeout = float(_cfg.get("ai", {}).get("race_tier_timeout_seconds", 45.0))
        deadline = time.monotonic() + max(
            self.MODEL_TIMEOUTS.get(m, default_timeout) for _, m in runnable
        )

        with ThreadPoolExecutor(max_workers=len(runnable)) as exc:
            fut_map = {exc.submit(make_call, p, m): (p, m) for p, m in runnable}
            pending = set(fut_map)
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for fut in done:
                    p, m = fut_map[fut]
                    try:
                        text = fut.result()
                    except Exception as e:
                        self._note_provider_error(p, e)
                        log.warning("Racer %s/%s failed: %s", p, m, e)
                        continue
                    if text and text.strip() and "API key missing" not in text:
                        self._record_success(p)
                        self._last_provider = p
                        self._last_model = m
                        log.info("Racer winner: %s/%s", p, m)
                        for f in pending:
                            f.cancel()
                        return text.strip()
            for f in pending:
                f.cancel()

        log.warning("Racer: all available models failed or timed out")
        return ""

    # --- OPENCODE GO ---

    def generate_opencode(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generate via OpenCode Go (mimo-v2.5-pro primary, $0 with Go subscription)."""
        if not self.opencode_api_key:
            raise ValueError("OpenCode Zen API key missing")
        client = OpenAI(api_key=self.opencode_api_key, base_url=self.opencode_base_url)
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        self._last_provider = "opencode"
        self._last_model = self._model
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.7, max_tokens=8192,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        tokens = getattr(response.usage, "total_tokens", 0) if getattr(response, "usage", None) else 0
        log.info("LLM: opencode/%s (%dms, %d tokens)", self._model, duration_ms, tokens,
                 extra={"stage": "llm_generate", "duration_ms": duration_ms,
                        "metadata": {"provider": "opencode", "model": self._model, "tokens": tokens}})
        content = response.choices[0].message.content
        # Some reasoning models put output in reasoning_content only
        if not content and hasattr(response.choices[0].message, "reasoning_content"):
            reasoning = response.choices[0].message.reasoning_content
            if reasoning:
                content = reasoning
        if content is None:
            raise ValueError(f"OpenCode returned empty (finish_reason={response.choices[0].finish_reason})")
        return content.strip()

    # --- NVIDIA ---

    def generate_nvidia(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        if not self.nvidia_api_key:
            raise ValueError("NVIDIA API key missing")
        client = OpenAI(api_key=self.nvidia_api_key, base_url=self.nvidia_base_url)
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        self._last_provider = "nvidia"
        self._last_model = self._model
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.7, max_tokens=8192,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        tokens = getattr(response.usage, "total_tokens", 0) if getattr(response, "usage", None) else 0
        log.info("LLM: nvidia/%s (%dms, %d tokens)", self._model, duration_ms, tokens,
                 extra={"stage": "llm_generate", "duration_ms": duration_ms,
                        "metadata": {"provider": "nvidia", "model": self._model, "tokens": tokens}})
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"NVIDIA returned empty (finish_reason={response.choices[0].finish_reason})")
        return content.strip()

    # --- GROQ ---

    def generate_groq(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generate via Groq (fast inference, strict TPM limits).

        Groq has ~6K TPM on free tier. Rate limit is handled by the
        per-provider token bucket (capacity=10, refill=0.15/s).
        """
        if not self.groq_api_key:
            raise ValueError("Groq API key missing")
        client = OpenAI(api_key=self.groq_api_key, base_url=self.groq_base_url)
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        self._last_provider = "groq"
        self._last_model = self._model
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.7, max_tokens=4096,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        tokens = getattr(response.usage, "total_tokens", 0) if getattr(response, "usage", None) else 0
        log.info("LLM: groq/%s (%dms, %d tokens)", self._model, duration_ms, tokens,
                 extra={"stage": "llm_generate", "duration_ms": duration_ms,
                        "metadata": {"provider": "groq", "model": self._model, "tokens": tokens}})
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"Groq returned empty (finish_reason={response.choices[0].finish_reason})")
        return content.strip()

    def get_used_provider(self) -> str:
        return self._last_provider or self._provider

    def get_used_model(self) -> str:
        return self._last_model or self._model

    def get_available_providers(self) -> Dict:
        available = {}
        if self.opencode_api_key:
            available["opencode"] = self.PROVIDER_MODELS["opencode"]
        if self.nvidia_api_key:
            available["nvidia"] = self.PROVIDER_MODELS["nvidia"]
        if self.groq_api_key:
            available["groq"] = self.PROVIDER_MODELS["groq"]
        return available

    def generate_image(self, prompt: str, output_path: str) -> bool:
        return False

    # --- OLLAMA ---

    def generate_ollama(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        full_prompt = prompt
        if system_instruction:
            full_prompt = f"System: {system_instruction}\n\nUser: {prompt}"
        payload = {"model": "llama3.2", "prompt": full_prompt, "stream": False}
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.ConnectionError:
            raise ConnectionError("Ollama connection refused")
        except requests.Timeout:
            raise TimeoutError("Ollama request timed out (>120s)")
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

    # --- PARALLEL TEST ---

    def compare_models(self, prompt: str, system_instruction: Optional[str] = None) -> Dict:
        providers = {
            "nvidia": lambda: self.generate_nvidia(prompt, system_instruction),
            "ollama": lambda: self.generate_ollama(prompt, system_instruction),
        }
        results = {}
        with ThreadPoolExecutor(max_workers=len(providers)) as executor:
            futures = {executor.submit(fn): name for name, fn in providers.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    output = future.result()
                    results[name] = {"success": True, "response": output}
                except Exception as e:
                    results[name] = {"success": False, "error": str(e)}
        return results


if __name__ == "__main__":
    ai = AIClient()
    prompt = "Explain black holes in simple words"
    results = ai.compare_models(prompt)
    for provider, data in results.items():
        print("\n" + "=" * 80)
        print(f"PROVIDER: {provider.upper()}")
        print("=" * 80)
        if data["success"]:
            print(data["response"])
        else:
            print("ERROR:", data["error"])
