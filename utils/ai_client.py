import os
from typing import Optional, Dict
from openai import OpenAI
from utils.config import load_config
from utils.logger import get_logger
from google import genai
from google.genai import types
import time
import random
from dotenv import load_dotenv

# Load .env file at module level
load_dotenv()

cfg = load_config()
log = get_logger("ai_client", cfg["logging"]["log_file"], cfg["logging"]["level"])

class AIClient:
    def __init__(self):
        self.config = cfg.get("ai", {})
        self.provider = self.config.get("provider", "gemini")  # gemini | openai | ollama
        self.api_key = self.config.get("api_key")
        
        # Priority: Config > AI_API_KEY (.env) > Provider-specific Env Var
        if not self.api_key:
            self.api_key = os.environ.get("AI_API_KEY")
            
        if not self.api_key and self.provider != "ollama":
            if self.provider == "gemini":
                self.api_key = os.environ.get("GOOGLE_API_KEY")
            else:
                self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key and self.provider != "ollama":
            log.warning(f"No API key found for AI provider: {self.provider}")

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generates text using the configured AI provider."""
        if not self.api_key and self.provider != "ollama":
            return "Fallback: SEO metadata generation failed due to missing API key."

        try:
            if self.provider == "gemini":
                # Using the latest Gemini 3.1 Pro logic
                return self._generate_gemini(prompt, system_instruction)
            elif self.provider == "openai":
                return self._generate_openai(prompt, system_instruction)
            elif self.provider == "ollama":
                return self._generate_ollama(prompt, system_instruction)
            else:
                log.error(f"Unsupported AI provider: {self.provider}")
                return ""
        except Exception as e:
            log.error(f"AI Generation failed: {e}")
            return ""

    def generate_image(self, prompt: str, output_path: str) -> bool:
        """
        Bypasses AI image generation to avoid 429 Quota errors.
        Since YouTube Shorts do not support custom API thumbnails, this safely skips it.
        """
        log.info(f"⏭️ Skipping AI image generation for {output_path} (Quota protection)")
        return False

    def _generate_ollama(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        import requests
        model_name = self.config.get("model", "deepseek-r1:7b")
        url = "http://localhost:11434/api/generate"
        
        full_prompt = prompt
        if system_instruction:
            full_prompt = f"System: {system_instruction}\n\nUser: {prompt}"
            
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def _generate_gemini(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        from google import genai
        from google.genai import types
        import time
        import random
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = genai.Client(api_key=self.api_key)
                model_name = self.config.get("model", "gemini-1.5-flash")
                
                config = None
                if system_instruction:
                    config = types.GenerateContentConfig(system_instruction=system_instruction)
                    
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                return response.text.strip()
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10 + random.uniform(1, 5)
                    log.warning(f"⏳ Gemini Text Quota hit (429). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        return ""

    def _generate_openai(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        client = OpenAI(api_key=self.api_key)
        model_name = self.config.get("model", "gpt-4o-mini")
        
        messages =[]
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content.strip()