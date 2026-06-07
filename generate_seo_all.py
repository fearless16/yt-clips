"""
generate_seo_all.py — Fetch video metadata + generate SEO from multiple AI models.
Saves results to seo_output.txt for copy-paste into YouTube Studio.

Usage:
    .venv/bin/python generate_seo_all.py
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from utils.ai_client import AIClient
from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("generate_seo", cfg["logging"]["log_file"], cfg["logging"]["level"])

VIDEOS = [
    ("yhbGk20AFas", "https://youtube.com/shorts/yhbGk20AFas"),
    ("hgneVa0n_HM", "https://youtube.com/shorts/hgneVa0n_HM"),
    ("uRBajQdeILQ", "https://youtube.com/shorts/uRBajQdeILQ"),
    ("NPSPlNkkA6s", "https://youtube.com/shorts/NPSPlNkkA6s"),
    ("6RiKMn5EYlA", "https://youtube.com/shorts/6RiKMn5EYlA"),
]

# Models to race (provider/model tuples)
MODELS = [
    ("opencode", "minimax-m3"),
    ("opencode", "qwen3.7-plus"),
    ("opencode", "kimi-k2.5"),
    ("opencode", "deepseek-v4-pro"),
]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]


def fetch_video_info(video_id: str, url: str) -> dict:
    """Fetch video title and description using yt-dlp."""
    try:
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--cookies", "cookies.txt",
            "--dump-json",
            "--no-download",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"  ⚠ yt-dlp error for {video_id}: {result.stderr[:200]}")
            return {"id": video_id, "url": url, "title": "", "description": "", "tags": []}
        data = json.loads(result.stdout)
        return {
            "id": video_id,
            "url": url,
            "title": data.get("title", ""),
            "description": data.get("description", ""),
            "tags": data.get("tags", []),
        }
    except Exception as e:
        print(f"  ⚠ Failed to fetch {video_id}: {e}")
        return {"id": video_id, "url": url, "title": "", "description": "", "tags": []}


def get_transcript(video_id: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        data = transcript.to_raw_data()
        return " ".join(segment["text"] for segment in data)
    except Exception as e:
        log.warning("Transcript not available for %s: %s", video_id, e)
        return ""


def generate_seo_via_groq(api_key: str, model: str, title: str, description: str, transcript: str) -> dict:
    """Generate SEO using Groq API directly."""
    import requests

    prompt = f"""You are a YouTube SEO expert for cricket content. Generate viral-optimized metadata for YouTube Shorts.

CONTEXT:
Video Title: {title or "(no title)"}
Description: {description[:500] if description else "(no description)"}
Transcript: {transcript[:2000] if transcript else "(no transcript)"}

Return ONLY valid JSON, no markdown:
{{
  "title": "<max 80 chars, engaging English title with emojis, describe THIS clip>",
  "description": "<SEO-rich English description, match/player context, emojis, call to action>",
  "tags": ["<max 15 comma-separated tags>"],
  "hashtags": ["<max 5 hashtags>"]
}}"""

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a YouTube SEO expert. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json.loads(json_match.group())
        print(f"    ⚠ Groq {model} returned non-JSON: {raw[:100]}")
        return {}
    except Exception as e:
        print(f"    ⚠ Groq {model} error: {e}")
        return {}


def generate_seo_opencode(ai: AIClient, provider: str, model: str, title: str, transcript: str) -> dict:
    """Generate SEO using OpenCode Go model."""
    system = (
        "You are a YouTube SEO expert for cricket content. "
        "Generate viral-optimized metadata for YouTube Shorts. "
        "Only use player names, teams, and events from the transcript. "
        "NEVER invent or hallucinate. "
        "Return ONLY valid JSON — no markdown, no explanation."
    )

    prompt = f"""CONTEXT:
  Video Title: {title}
  Transcript: {transcript or "(no transcript available, infer from title only)"}

TASK: Generate YouTube SEO metadata for this Shorts video.

You MUST return valid JSON (no markdown, no other text):
{{{{
  "title": "<max 80 chars, engaging English title with emojis>",
  "description": "<full SEO-rich English description following the format below>",
  "tags": ["<max 15 comma-separated search tags>"],
  "hashtags": ["<max 5 hashtags>"]
}}}}

TITLE (max 80 chars, English with emojis):
- Start with the MOST EXCITING moment of this clip
- Example: "Kohli SMASHES 120m Six! RCB vs MI IPL 2026 🔥"
- Must describe THIS specific video content
- End with 1-2 relevant emojis

DESCRIPTION FORMAT (full English, SEO-rich, with line breaks):
First line: Brief summary of the key moment
Match/event context and key players
What makes this moment exciting
Call to action asking viewers to subscribe
Relevant emojis throughout
Line breaks between sections for readability

TAGS (max 15):
Player names, team names, tournament, keywords from the video

HASHTAGS (max 5):
Include #Shorts and event/cricket related hashtags

Generate the absolute best SEO to maximize reach."""

    try:
        text = ai.generate_text(prompt, system_instruction=system, prefer_model=model)
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group())
        print(f"    ⚠ {provider}/{model} non-JSON response: {text[:150]}")
        return {}
    except Exception as e:
        print(f"    ⚠ {provider}/{model} error: {e}")
        return {}


def format_output(video: dict, results: list) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"Video: {video['id']}")
    lines.append(f"URL: {video['url']}")
    lines.append(f"Current Title: {video['title'][:80]}")
    lines.append(f"Current Description: {len(video['description'])} chars")
    lines.append(f"Transcript: {'yes' if video.get('transcript') else 'no'}")
    lines.append("")

    for r in results:
        model_label = r["model"]
        seo = r.get("seo", {})
        lines.append(f"--- {model_label} ---")
        lines.append(f"Title: {seo.get('title', 'N/A')}")
        lines.append(f"Description:")
        desc = seo.get("description", "")
        if desc:
            for line in desc.split("\n"):
                lines.append(f"  {line}")
        else:
            lines.append("  (N/A)")
        tags = seo.get("tags", []) or []
        hashtags = seo.get("hashtags", []) or []
        all_tags = list(dict.fromkeys(tags + hashtags))
        lines.append(f"Tags ({len(all_tags)}):")
        for t in all_tags:
            lines.append(f"  #{t.replace(' ', '')}")
        lines.append("")

    lines.append("")
    return "\n".join(lines)


def main():
    api_key = os.getenv("OPENCODE_ZEN_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("OPENCODE_ZEN_API_KEY not set")
        sys.exit(1)

    ai = AIClient()

    print(f"Fetching metadata for {len(VIDEOS)} videos...")
    videos = []
    for vid_id, url in VIDEOS:
        print(f"\n  [{vid_id}] Fetching...")
        info = fetch_video_info(vid_id, url)
        if info["title"]:
            print(f"    Title: {info['title'][:80]}")
            print(f"    Description: {len(info['description'])} chars")
        info["transcript"] = get_transcript(vid_id)
        if info["transcript"]:
            print(f"    Transcript: {len(info['transcript'])} chars")
        videos.append(info)

    print(f"\n{'=' * 70}")
    print(f"Generating SEO for each video...")
    print(f"{'=' * 70}")

    all_output = []
    for video in videos:
        print(f"\n--- {video['id']}: {video['title'][:60]} ---")
        title = video["title"]
        transcript = video["transcript"]
        description = video["description"]
        results = []
        results_lock = threading.Lock()

        def run_opencode(p, m):
            seo = generate_seo_opencode(ai, p, m, title, transcript)
            if seo:
                with results_lock:
                    results.append({"model": f"{p}/{m}", "seo": seo})

        def run_groq(m):
            seo = generate_seo_via_groq(groq_key, m, title, description, transcript)
            if seo:
                with results_lock:
                    results.append({"model": f"groq/{m}", "seo": seo})

        tasks = []
        for provider, model in MODELS:
            tasks.append((run_opencode, (provider, model)))
        if groq_key:
            for model in GROQ_MODELS:
                tasks.append((run_groq, (model,)))

        print(f"  Racing {len(tasks)} models in parallel...")
        with ThreadPoolExecutor(max_workers=len(tasks)) as exc:
            futs = {exc.submit(fn, *args): f"{fn.__name__}({args})" for fn, args in tasks}
            for fut in as_completed(futs):
                label = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"    ⚠ {label} failed: {e}")

        if not results:
            print(f"  ⚠ No SEO generated for {video['id']}")
            continue

        video["results"] = results
        all_output.append(format_output(video, results))
        time.sleep(1)

    output_path = Path("seo_output.txt")
    output_path.write_text("\n".join(all_output))
    print(f"\n{'=' * 70}")
    print(f"SEO saved to {output_path}")
    print(f"Open it and copy-paste each section into YouTube Studio.")
    print(f"File size: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
