import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()


class AIClient:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("AI_API_KEY")

        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        self.nvidia_base_url = os.getenv(
            "NVIDIA_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        )

        self.xai_api_key = os.getenv("XAI_API_KEY")
        self.xai_base_url = os.getenv(
            "XAI_BASE_URL",
            "https://api.x.ai/v1",
        )

        self.ollama_url = "http://localhost:11434/api/generate"

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generic text generation wrapper, prioritizes Gemini then falls back."""
        if self.google_api_key:
            return self.generate_gemini(prompt, system_instruction)
        elif self.nvidia_api_key:
            return self.generate_nvidia(prompt, system_instruction)
        elif self.xai_api_key:
            return self.generate_grok(prompt, system_instruction)
        return self.generate_ollama(prompt, system_instruction)

    def generate_image(self, prompt: str, output_path: str) -> bool:
        """Generic image generation wrapper."""
        # Optional: Implement actual image generation via Gemini or DALL-E.
        # Returning False triggers the fallback to FFmpeg frame extraction.
        return False

    # ---------------- GEMINI ----------------

    def generate_gemini(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        if not self.google_api_key:
            return "Gemini API key missing"

        client = genai.Client(api_key=self.google_api_key)

        config = None
        if system_instruction:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction
            )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config,
        )

        return response.text.strip()

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

        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=messages,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    # ---------------- GROK ----------------

    def generate_grok(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        if not self.xai_api_key:
            return "Grok API key missing"

        client = OpenAI(
            api_key=self.xai_api_key,
            base_url=self.xai_base_url,
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

        response = client.chat.completions.create(
            model="grok-2-latest",
            messages=messages,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

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

        response = requests.post(
            self.ollama_url,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("response", "").strip()

    # ---------------- PARALLEL TEST ----------------

    def compare_models(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> Dict:

        providers = {
            "gemini": lambda: self.generate_gemini(
                prompt,
                system_instruction,
            ),
            "nvidia": lambda: self.generate_nvidia(
                prompt,
                system_instruction,
            ),
            "ollama": lambda: self.generate_ollama(
                prompt,
                system_instruction,
            ),
        }

        # Optional Grok
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