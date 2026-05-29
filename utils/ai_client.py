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


class AIClient:
    # Shared state for health and rate limiting across all AIClient instances
    _shared_lock = threading.Lock()
    _provider_failures = {}        # provider -> count
    _provider_cooldown_until = {}  # provider -> timestamp
    _provider_token_buckets = {}   # provider -> dict

    PROVIDER_MODELS = {
        "groq": ["meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile", "qwen/qwen3-32b"],
        "deepseek": ["deepseek-chat"],
        "openrouter": ["deepseek/deepseek-v4-flash", "google/gemini-2.0-flash-001", "qwen/qwen3.5-122b-a10b", "anthropic/claude-sonnet-4"],
        "nvidia": ["meta/llama-3.3-70b-instruct", "nvidia/llama-3.3-nemotron-super-49b-v1"],
    }

    # Speed-ordered tiers — fastest first (tested latencies).
    # Tier 1: <1s (Groq models)
    # Tier 2: 1-3s (NVIDIA, OpenRouter fast models)
    # Tier 3: >10s (Claude - slower but high quality)
    FASTEST_TIERS = [
        [("groq", "meta-llama/llama-4-scout-17b-16e-instruct"), ("groq", "llama-3.3-70b-versatile"), ("groq", "qwen/qwen3-32b")],
        [("deepseek", "deepseek-chat"), ("nvidia", "nvidia/llama-3.3-nemotron-super-49b-v1"), ("openrouter", "deepseek/deepseek-v4-flash"), ("openrouter", "google/gemini-2.0-flash-001")],
        [("nvidia", "meta/llama-3.3-70b-instruct"), ("openrouter", "qwen/qwen3.5-122b-a10b")],
        [("openrouter", "anthropic/claude-sonnet-4")],  # Slow but smart
    ]

    def __init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        )

        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        self.nvidia_base_url = os.getenv(
            "NVIDIA_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        )

        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_base_url = os.getenv(
            "GROQ_BASE_URL",
            "https://api.groq.com/openai/v1",
        )

        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_base_url = os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/v1",
        )

        self.ollama_url = "http://localhost:11434/api/generate"
        ai_cfg = _cfg.get("ai", {})
        self._provider = ai_cfg.get("provider", "groq")
        self._model = ai_cfg.get("model", "qwen/qwen3-32b")
        self._last_provider = None
        self._last_model = None

    @classmethod
    def _check_and_consume_token(cls, provider: str) -> bool:
        with cls._shared_lock:
            now = time.time()
            if provider not in cls._provider_token_buckets:
                cls._provider_token_buckets[provider] = {
                    "capacity": 30.0,
                    "tokens": 30.0,
                    "last_update": now,
                    "refill_rate": 0.5
                }
            
            bucket = cls._provider_token_buckets[provider]
            elapsed = now - bucket["last_update"]
            bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + elapsed * bucket["refill_rate"])
            bucket["last_update"] = now
            
            cooldown = cls._provider_cooldown_until.get(provider, 0.0)
            if cooldown > now:
                return False
                
            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True
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
                log.warning(f"Provider {provider} hit circuit breaker! Cooldown for 5 minutes.")

    # ── Error classification + rate-limit aware cooldown ────────────────────────

    @staticmethod
    def _status_code(exc: Exception) -> Optional[int]:
        """Best-effort HTTP status extraction from openai/requests exceptions."""
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

    @classmethod
    def _is_rate_limited(cls, exc: Exception) -> bool:
        if cls._status_code(exc) == 429:
            return True
        name = type(exc).__name__.lower()
        return "ratelimit" in name or "429" in str(exc)

    @classmethod
    def _is_retryable(cls, exc: Exception) -> bool:
        """True for transient errors worth backing off on (429 + 5xx + network)."""
        status = cls._status_code(exc)
        if status is not None and (status == 429 or 500 <= status < 600):
            return True
        name = type(exc).__name__.lower()
        return any(s in name for s in (
            "ratelimit", "timeout", "apiconnection", "internalserver",
            "serviceunavailable", "connectionerror",
        ))

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> Optional[float]:
        """Read a Retry-After header (seconds) from the exception, if present."""
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

    @classmethod
    def _note_provider_error(cls, provider: str, exc: Exception, default_cooldown: float = 30.0):
        """Record a failure and, for rate-limit errors, set a provider cooldown.

        Honors a server-provided Retry-After when available; otherwise applies a
        conservative default so the racer/failover skips the provider until it
        is likely healthy again. Always feeds the circuit breaker.
        """
        if cls._is_rate_limited(exc):
            wait_s = cls._retry_after_seconds(exc) or default_cooldown
            with cls._shared_lock:
                prev = cls._provider_cooldown_until.get(provider, 0.0)
                cls._provider_cooldown_until[provider] = max(prev, time.time() + wait_s)
            log.warning("Provider %s rate-limited (429) — cooling down %.0fs", provider, wait_s)
        cls._record_failure(provider)

    @classmethod
    def _in_cooldown(cls, provider: str) -> bool:
        with cls._shared_lock:
            return cls._provider_cooldown_until.get(provider, 0.0) > time.time()

    def _get_failover_chain(self, start_provider: str) -> List[str]:
        chain = ["groq", "deepseek", "openrouter", "nvidia"]
        if start_provider in chain:
            chain.remove(start_provider)
            chain.insert(0, start_provider)
        return chain

    def _call_provider(self, provider: str, prompt: str, system_instruction: Optional[str] = None) -> str:
        old_provider = self._provider
        old_model = self._model
        
        # Temporarily route to correct model/provider
        self._provider = provider
        if provider == "groq":
            self._model = self.PROVIDER_MODELS["groq"][1] if self._model not in self.PROVIDER_MODELS["groq"] else self._model
            res = self.generate_groq(prompt, system_instruction)
        elif provider == "deepseek":
            self._model = "deepseek-chat"
            res = self.generate_deepseek(prompt, system_instruction)
        elif provider == "openrouter":
            self._model = self.PROVIDER_MODELS["openrouter"][0] if self._model not in self.PROVIDER_MODELS["openrouter"] else self._model
            res = self.generate_openrouter(prompt, system_instruction)
        elif provider == "nvidia":
            self._model = self.PROVIDER_MODELS["nvidia"][0] if self._model not in self.PROVIDER_MODELS["nvidia"] else self._model
            res = self.generate_nvidia(prompt, system_instruction)
        else:
            raise ValueError(f"Unknown provider {provider}")
            
        self._provider = old_provider
        self._model = old_model
        return res

    def _log_cost(self, provider: str, model: str, input_chars: int, output_chars: int):
        in_tokens = input_chars // 4
        out_tokens = output_chars // 4
        rates = {
            "groq": {"in": 0.59, "out": 0.79},
            "deepseek": {"in": 0.14, "out": 0.28},
            "openrouter": {"in": 0.20, "out": 0.40},
            "nvidia": {"in": 0.07, "out": 0.07},
            "ollama": {"in": 0.0, "out": 0.0}
        }
        rate = rates.get(provider, {"in": 0.50, "out": 0.50})
        cost = (in_tokens * rate["in"] + out_tokens * rate["out"]) / 1_000_000
        log.info(f"LLM request cost: {provider}/{model} -> input_tokens={in_tokens}, output_tokens={out_tokens}, est_cost=${cost:.6f}",
                 extra={"stage": "llm_cost", "metadata": {"provider": provider, "model": model, "cost": cost}})

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Text generation with rate-limit awareness, circuit breaker, 429-aware
        cooldown/backoff, failover, and token/cost tracking."""
        ai_cfg = _cfg.get("ai", {})
        base_delay = float(ai_cfg.get("retry_base_delay_seconds", 2.0))
        max_delay = float(ai_cfg.get("retry_max_delay_seconds", 30.0))

        chain = self._get_failover_chain(self._provider)

        available_chain = []
        for p in chain:
            if p == "groq" and self.groq_api_key:
                available_chain.append(p)
            elif p == "deepseek" and self.deepseek_api_key:
                available_chain.append(p)
            elif p == "openrouter" and self.openrouter_api_key:
                available_chain.append(p)
            elif p == "nvidia" and self.nvidia_api_key:
                available_chain.append(p)

        errors = []
        retryable_seen = False
        for provider in available_chain:
            if not self._check_and_consume_token(provider):
                log.info(f"Skipping provider {provider} due to rate limiting or circuit breaker cooldown.")
                continue

            try:
                res = self._call_provider(provider, prompt, system_instruction)
                if res:
                    self._record_success(provider)
                    self._log_cost(provider, self.get_used_model(), len(prompt) + len(system_instruction or ""), len(res))
                    return res
            except Exception as e:
                retryable_seen = retryable_seen or self._is_retryable(e)
                self._note_provider_error(provider, e)
                log.warning(f"Provider {provider} failed: {e}")
                errors.append(f"{provider}: {e}")

        # Backoff once before the retry round, but only if failures were transient
        # (429/5xx/network). Generic errors fail over immediately (no sleep).
        if retryable_seen:
            delay = min(max_delay, base_delay)
            log.info("Transient LLM errors — backing off %.1fs before retry round", delay)
            time.sleep(delay)

        # Retry round: ignore token bucket but still respect provider cooldowns.
        for attempt, provider in enumerate(available_chain):
            try:
                if self._in_cooldown(provider):
                    continue
                res = self._call_provider(provider, prompt, system_instruction)
                if res:
                    self._record_success(provider)
                    self._log_cost(provider, self.get_used_model(), len(prompt) + len(system_instruction or ""), len(res))
                    return res
            except Exception as e:
                self._note_provider_error(provider, e)
                log.warning(f"Provider {provider} failed on retry: {e}")
                errors.append(f"{provider} (retry): {e}")

        # Local fallback
        try:
            log.info("All primary LLM providers failed or unavailable, attempting Ollama...")
            res = self.generate_ollama(prompt, system_instruction)
            if res:
                return res
        except Exception as e:
            errors.append(f"ollama: {e}")

        raise RuntimeError(f"All LLM providers failed. Errors: {'; '.join(errors)}")

    def _available_plan(self) -> List[List[tuple]]:
        """Return FASTEST_TIERS filtered to only available (have API key) providers."""
        has_key = {
            "groq": bool(self.groq_api_key),
            "deepseek": bool(self.deepseek_api_key),
            "openrouter": bool(self.openrouter_api_key),
            "nvidia": bool(self.nvidia_api_key),
        }
        plan = []
        for tier in self.FASTEST_TIERS:
            available = [(p, m) for p, m in tier if has_key.get(p)]
            if available:
                plan.append(available)
        return plan

    def generate_fastest_first(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Race available models in parallel speed-tiers; return first valid response.

        Health-aware: each candidate is gated by the shared token bucket + circuit
        breaker (providers in cooldown are skipped), and success/failure (incl.
        429 Retry-After cooldown) is recorded so the racer cooperates with the
        rest of the pipeline. Within a tier it keeps collecting completed futures
        until one returns valid text or a tier deadline elapses — so a slow but
        valid response is no longer dropped by a short result() timeout.

        Returns "" only when every available provider/model failed or was in
        cooldown — callers should ESCALATE (next strategy / queue), never emit a
        generic fallback.
        """
        plan = self._available_plan()
        if not plan:
            return self.generate_ollama(prompt, system_instruction)

        ai_cfg = _cfg.get("ai", {})
        tier_timeout = float(ai_cfg.get("race_tier_timeout_seconds", 45.0))

        def make_call(p, m):
            thread_client = AIClient()
            thread_client._provider = p
            thread_client._model = m
            fn = getattr(thread_client, f"generate_{p}")
            return fn(prompt, system_instruction)

        for tier in plan:
            # Gate each candidate through the shared health layer (token bucket +
            # circuit-breaker cooldown). Skip what we shouldn't hit right now.
            runnable = []
            for provider, model in tier:
                if not self._check_and_consume_token(provider):
                    log.info("Racer skipping %s/%s (rate-limited or in cooldown)", provider, model)
                    continue
                runnable.append((provider, model))
            if not runnable:
                continue

            with ThreadPoolExecutor(max_workers=len(runnable)) as exc:
                fut_map = {exc.submit(make_call, p, m): (p, m) for p, m in runnable}
                pending = set(fut_map)
                deadline = time.monotonic() + tier_timeout
                winner = None
                while pending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
                    if not done:
                        break  # tier deadline hit
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
                            winner = text.strip()
                            break
                    if winner:
                        break
                for f in pending:
                    f.cancel()
                if winner:
                    return winner

        return ""

    # ---------------- OPENROUTER ----------------

    def generate_openrouter(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key missing")

        client = OpenAI(
            api_key=self.openrouter_api_key,
            base_url=self.openrouter_base_url,
        )

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        self._last_provider = "openrouter"
        self._last_model = self._model

        import time
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=8192,
            extra_headers={"HTTP-Referer": "https://github.com/prajwalbairagi/yt-clips"},
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        tokens = getattr(response.usage, "total_tokens", 0) if getattr(response, "usage", None) else 0
        log.info(f"LLM request: openrouter/{self._model} ({duration_ms}ms, {tokens} tokens)",
                 extra={"stage": "llm_generate", "duration_ms": duration_ms, 
                        "metadata": {"provider": "openrouter", "model": self._model, "tokens": tokens}})

        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"OpenRouter returned empty content (finish_reason={response.choices[0].finish_reason})")
        return content.strip()

    # ---------------- NVIDIA ----------------

    def generate_nvidia(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        if not self.nvidia_api_key:
            raise ValueError("NVIDIA API key missing")

        client = OpenAI(
            api_key=self.nvidia_api_key,
            base_url=self.nvidia_base_url,
        )

        messages = []

        if system_instruction:
            messages.append(
                {
                    "role": "system",
                    "content": system_instruction,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        self._last_provider = "nvidia"
        self._last_model = self._model

        import time
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=8192,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        tokens = getattr(response.usage, "total_tokens", 0) if getattr(response, "usage", None) else 0
        log.info(f"LLM request: nvidia/{self._model} ({duration_ms}ms, {tokens} tokens)",
                 extra={"stage": "llm_generate", "duration_ms": duration_ms, 
                        "metadata": {"provider": "nvidia", "model": self._model, "tokens": tokens}})

        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"NVIDIA returned empty content (finish_reason={response.choices[0].finish_reason})")
        return content.strip()

    # ---------------- GROQ ----------------

    def generate_groq(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        if not self.groq_api_key:
            raise ValueError("Groq API key missing")

        # Trim total prompt to stay within API limits
        total = len(system_instruction or "") + len(prompt)
        if total > 28000:
            excess = total - 28000
            prompt = prompt[:max(len(prompt) - excess - 500, 8000)]
            if system_instruction:
                system_instruction = system_instruction[:2000]

        client = OpenAI(
            api_key=self.groq_api_key,
            base_url=self.groq_base_url,
        )

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        self._last_provider = "groq"
        self._last_model = self._model

        import time
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=8192,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        tokens = getattr(response.usage, "total_tokens", 0) if getattr(response, "usage", None) else 0
        log.info(f"LLM request: groq/{self._model} ({duration_ms}ms, {tokens} tokens)",
                 extra={"stage": "llm_generate", "duration_ms": duration_ms, 
                        "metadata": {"provider": "groq", "model": self._model, "tokens": tokens}})

        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"Groq returned empty content (finish_reason={response.choices[0].finish_reason})")
        return content.strip()

    # ---------------- DEEPSEEK ----------------

    def generate_deepseek(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        if not self.deepseek_api_key:
            raise ValueError("DeepSeek API key missing")

        client = OpenAI(
            api_key=self.deepseek_api_key,
            base_url=self.deepseek_base_url,
        )

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        self._last_provider = "deepseek"
        self._last_model = "deepseek-chat"

        import time
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=self._last_model,
            messages=messages,
            temperature=0.7,
            max_tokens=8192,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        tokens = getattr(response.usage, "total_tokens", 0) if getattr(response, "usage", None) else 0
        log.info(f"LLM request: deepseek/{self._last_model} ({duration_ms}ms, {tokens} tokens)",
                 extra={"stage": "llm_generate", "duration_ms": duration_ms, 
                        "metadata": {"provider": "deepseek", "model": self._last_model, "tokens": tokens}})

        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"DeepSeek returned empty content (finish_reason={response.choices[0].finish_reason})")
        return content.strip()

    def get_used_provider(self) -> str:
        return self._last_provider or self._provider

    def get_used_model(self) -> str:
        return self._last_model or self._model

    def get_available_providers(self) -> Dict:
        """Return dict of available providers and their candidate models."""
        available = {}
        if self.groq_api_key:
            available["groq"] = self.PROVIDER_MODELS["groq"]
        if self.deepseek_api_key:
            available["deepseek"] = self.PROVIDER_MODELS["deepseek"]
        if self.openrouter_api_key:
            available["openrouter"] = self.PROVIDER_MODELS["openrouter"]
        if self.nvidia_api_key:
            available["nvidia"] = self.PROVIDER_MODELS["nvidia"]
        return available

    def generate_image(self, prompt: str, output_path: str) -> bool:
        """Stub: returns False to trigger ffmpeg frame extraction fallback."""
        return False

    # ---------------- OLLAMA ----------------

    def generate_ollama(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        full_prompt = prompt

        if system_instruction:
            full_prompt = (
                f"System: {system_instruction}\n\n"
                f"User: {prompt}"
            )

        payload = {
            "model": "llama3.2",
            "prompt": full_prompt,
            "stream": False,
        }

        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.ConnectionError:
            raise ConnectionError("Ollama connection refused — is Ollama running?")
        except requests.Timeout:
            raise TimeoutError("Ollama request timed out (>120s)")
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

    # ---------------- PARALLEL TEST ----------------

    def compare_models(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> Dict:

        providers = {
            "nvidia": lambda: self.generate_nvidia(
                prompt,
                system_instruction,
            ),
            "ollama": lambda: self.generate_ollama(
                prompt,
                system_instruction,
            ),
        }



        results = {}

        with ThreadPoolExecutor(max_workers=len(providers)) as executor:
            futures = {
                executor.submit(fn): name
                for name, fn in providers.items()
            }

            for future in as_completed(futures):
                name = futures[future]

                try:
                    output = future.result()

                    results[name] = {
                        "success": True,
                        "response": output,
                    }

                except Exception as e:
                    results[name] = {
                        "success": False,
                        "error": str(e),
                    }

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
