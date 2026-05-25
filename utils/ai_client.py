import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, FIRST_COMPLETED, wait
from pathlib import Path
from typing import Dict, List, Optional, Callable

import requests
from dotenv import load_dotenv
from openai import OpenAI

from utils.config import load_config

load_dotenv()

_cfg = load_config()


class AIClient:
    PROVIDER_MODELS = {
        "groq": ["meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile", "qwen/qwen3-32b"],
        "openrouter": ["deepseek/deepseek-v4-flash", "google/gemini-2.0-flash-001", "qwen/qwen3.5-122b-a10b", "anthropic/claude-sonnet-4"],
        "nvidia": ["meta/llama-3.3-70b-instruct", "nvidia/llama-3.3-nemotron-super-49b-v1"],
    }

    # Speed-ordered tiers — fastest first (tested latencies).
    # Tier 1: <1s (Groq models)
    # Tier 2: 1-3s (NVIDIA, OpenRouter fast models)
    # Tier 3: >10s (Claude - slower but high quality)
    FASTEST_TIERS = [
        [("groq", "meta-llama/llama-4-scout-17b-16e-instruct"), ("groq", "llama-3.3-70b-versatile"), ("groq", "qwen/qwen3-32b")],
        [("nvidia", "nvidia/llama-3.3-nemotron-super-49b-v1"), ("openrouter", "deepseek/deepseek-v4-flash"), ("openrouter", "google/gemini-2.0-flash-001")],
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

        self.ollama_url = "http://localhost:11434/api/generate"
        ai_cfg = _cfg.get("ai", {})
        self._provider = ai_cfg.get("provider", "groq")
        self._model = ai_cfg.get("model", "qwen/qwen3-32b")
        self._last_provider = None
        self._last_model = None

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generic text generation wrapper, prioritizes configured provider."""
        provider = self._provider
        if provider == "openrouter" and self.openrouter_api_key:
            return self.generate_openrouter(prompt, system_instruction)
        elif provider == "nvidia" and self.nvidia_api_key:
            return self.generate_nvidia(prompt, system_instruction)
        elif provider == "groq" and self.groq_api_key:
            return self.generate_groq(prompt, system_instruction)
        elif self.openrouter_api_key:
            return self.generate_openrouter(prompt, system_instruction)
        elif self.groq_api_key:
            return self.generate_groq(prompt, system_instruction)
        elif self.nvidia_api_key:
            return self.generate_nvidia(prompt, system_instruction)
        return self.generate_ollama(prompt, system_instruction)

    def _available_plan(self) -> List[List[tuple]]:
        """Return FASTEST_TIERS filtered to only available (have API key) providers."""
        has_key = {
            "groq": bool(self.groq_api_key),
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
        """Race multiple models in parallel tiers, return first successful response.

        Fires all models in tier 1 concurrently. If none succeed, moves to tier 2,
        and so on. Never blocks waiting for a slow model — first valid response wins.
        Cancels remaining in-flight calls once a result is received.
        """
        plan = self._available_plan()
        if not plan:
            return self.generate_ollama(prompt, system_instruction)

        provider_map = {
            "groq": self.generate_groq,
            "openrouter": self.generate_openrouter,
            "nvidia": self.generate_nvidia,
        }

        for tier in plan:
            calls: List[tuple] = []
            for provider, model in tier:
                fn = provider_map.get(provider)
                if fn is None:
                    continue
                old_p, old_m = self._provider, self._model
                def make_call(f=fn, p=provider, m=model, _op=old_p, _om=old_m):
                    self._provider = p
                    self._model = m
                    try:
                        return f(prompt, system_instruction)
                    finally:
                        self._provider = _op
                        self._model = _om
                calls.append((provider, model, make_call))

            if not calls:
                continue

            with ThreadPoolExecutor(max_workers=len(calls)) as exc:
                fut_map = {exc.submit(c[2]): (c[0], c[1]) for c in calls}
                try:
                    done, not_done = wait(
                        fut_map, return_when=FIRST_COMPLETED, timeout=45,
                    )
                    for fut in done:
                        p, m = fut_map[fut]
                        try:
                            text = fut.result(timeout=5)
                            if text and text.strip() and "API key missing" not in text:
                                self._last_provider = p
                                self._last_model = m
                                for f in not_done:
                                    f.cancel()
                                return text.strip()
                        except Exception:
                            continue
                    for f in not_done:
                        f.cancel()
                except Exception:
                    for f in fut_map:
                        f.cancel()

        return ""

    # ---------------- OPENROUTER ----------------

    def generate_openrouter(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        if not self.openrouter_api_key:
            return "OpenRouter API key missing"

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

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=8192,
            extra_headers={"HTTP-Referer": "https://github.com/prajwalbairagi/yt-clips"},
        )

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
            return "NVIDIA API key missing"

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

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=8192,
        )

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
            return "Groq API key missing"

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

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=8192,
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"Groq returned empty content (finish_reason={response.choices[0].finish_reason})")
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
            return "Ollama connection refused — is Ollama running?"
        except requests.Timeout:
            return "Ollama request timed out (>120s)"
        except requests.HTTPError as e:
            return f"Ollama HTTP error: {e}"
        except Exception as e:
            return f"Ollama error: {e}"

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

        if self.groq_api_key:
            providers["groq"] = lambda: self.generate_groq(
                prompt,
                system_instruction,
            )
        if self.deepseek_api_key:
            providers["deepseek"] = lambda: self.generate_deepseek(
                prompt,
                system_instruction,
            )
        if self.xai_api_key:
            providers["grok"] = lambda: self.generate_grok(
                prompt,
                system_instruction,
            )

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
