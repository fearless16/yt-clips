#!/usr/bin/env python3
"""Benchmark DeepSeek V4, MiniMax M2.5 Pro, Xiaomi Mimo 2.5 Pro, and Alibaba Qwen."""

import subprocess
import json
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

MODELS = {
    "DeepSeek V4 Pro":     "deepseek/deepseek-v4-pro",
    "DeepSeek V4 Flash":   "deepseek/deepseek-v4-flash",
    "MiniMax M2.5 Pro":    "openrouter/minimax/minimax-m2.5:free",
    "Xiaomi Mimo 2.5 Pro": "xiaomi/mimo-v2.5-pro",
    "Alibaba Qwen Max":    "alibaba/qwen-max",
    "Alibaba Qwen Plus":   "alibaba/qwen-plus",
}

PROMPTS = [
    # Coding
    ("coding_easy",
     "Write a Python function to find the longest palindromic substring in O(n^2). Return only code."),
    ("coding_medium",
     "Write a Python async web scraper that fetches 10 pages concurrently, parses all <a> hrefs, "
     "and saves unique URLs to a file. Include error handling and rate limiting."),
    ("coding_hard",
     "Write a Python implementation of a thread-safe LRU cache with TTL support. "
     "Must support get, put, and delete operations. Include type hints and a unit test."),

    # Reasoning
    ("logic",
     "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. "
     "How much does the ball cost? Explain step by step."),
    ("math",
     "If 3x + 7 = 22, and 2y - 5 = 3x, what is x + y? Show your work."),

    # General knowledge / writing
    ("explain",
     "Explain quantum entanglement in 3 sentences as if to a 12-year-old."),
    ("creative",
     "Write a 4-line poem about debugging in the style of Edgar Allan Poe."),

    # Instruction following
    ("structured",
     "Return a JSON object with keys: name, version, features (array of 3 strings). "
     "Use the name 'BenchmarkTest' and version '1.0.0'. No other text."),
]

def run_model(model_id: str, prompt: str, timeout: int = 60) -> dict:
    """Run a single model with a prompt and return timing/response."""
    start = time.time()
    try:
        result = subprocess.run(
            ["opencode", "run", "--model", model_id, prompt],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "OPENCODE_DISABLE_PROJECT_CONFIG": "1", "OPENCODE_PURE": "1"}
        )
        elapsed = time.time() - start
        output = result.stdout.strip()
        # opencode output typically has a header line then the response
        lines = output.split("\n")
        # Filter out header/banner lines
        response_lines = [l for l in lines if l.strip() and not l.startswith(">")]
        response = "\n".join(response_lines).strip() if response_lines else output
        return {
            "success": result.returncode == 0,
            "response": response[:500],
            "latency": round(elapsed, 2),
            "error": result.stderr[:200] if result.stderr else None,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "response": "", "latency": timeout, "error": "Timeout", "returncode": -1}
    except Exception as e:
        return {"success": False, "response": "", "latency": 0, "error": str(e), "returncode": -1}


def score_response(prompt_name: str, response: str) -> int:
    """Simple heuristic scoring of response quality (0-100)."""
    if not response:
        return 0
    score = 50  # base
    # Length bonus (up to 20)
    score += min(len(response) / 10, 20)
    # Code-specific scoring
    if "coding" in prompt_name:
        if any(kw in response for kw in ["def ", "class ", "import "]):
            score += 15
        if "```" in response:
            score += 5
        if "return" in response:
            score += 5
    # JSON scoring
    if prompt_name == "structured":
        try:
            json.loads(response)
            score += 30
        except (json.JSONDecodeError, ValueError):
            pass
    # Penalty for errors/refusals
    if any(kw in response.lower() for kw in ["i cannot", "i can't", "i am not able", "i'm sorry"]):
        score -= 30
    return max(0, min(100, int(score)))


def main():
    print("=" * 80)
    print("  MODEL BENCHMARK: DeepSeek V4 / MiniMax M2.5 Pro / Xiaomi Mimo 2.5 Pro / Alibaba Qwen")
    print("=" * 80)
    print(f"\n{'Model':<22} {'Prompt':<16} {'Latency':>8} {'Score':>6} {'Response':>8}")
    print("-" * 80)

    results = {}

    for model_name, model_id in MODELS.items():
        print(f"\n>>> Testing: {model_name} ({model_id})")
        results[model_name] = {"total_latency": 0, "total_score": 0, "count": 0, "details": []}

        for pname, prompt in PROMPTS:
            r = run_model(model_id, prompt)
            score = score_response(pname, r["response"]) if r["success"] else 0
            results[model_name]["details"].append({**r, "prompt": pname, "score": score})
            results[model_name]["total_latency"] += r["latency"]
            results[model_name]["total_score"] += score
            results[model_name]["count"] += 1

            resp_preview = r["response"][:60].replace("\n", " ") if r["response"] else "FAILED"
            status = "OK" if r["success"] else "ERR"
            print(f"  {pname:<16} {r['latency']:>6.1f}s  {score:>4}/100  [{status}] {resp_preview}")

        avg_lat = results[model_name]["total_latency"] / results[model_name]["count"]
        avg_score = results[model_name]["total_score"] / results[model_name]["count"]
        print(f"  {'':->62}")
        print(f"  {'AVERAGE':<16} {avg_lat:>6.1f}s  {avg_score:>4.0f}/100")

    print("\n" + "=" * 80)
    print("  FINAL RANKING (by avg score)")
    print("=" * 80)
    sorted_models = sorted(results.items(), key=lambda x: x[1]["total_score"] / x[1]["count"], reverse=True)
    print(f"\n{'Rank':<6} {'Model':<22} {'Avg Score':>10} {'Avg Latency':>12}")
    print("-" * 50)
    for i, (name, data) in enumerate(sorted_models, 1):
        avg_s = data["total_score"] / data["count"]
        avg_l = data["total_latency"] / data["count"]
        print(f"  {i:<4}  {name:<22} {avg_s:>6.0f}/100    {avg_l:>6.1f}s")

    print("\n" + "=" * 80)
    print("  DETAILED RESULTS TABLE")
    print("=" * 80)
    header = f"{'Prompt':<16}"
    for name, _ in sorted_models:
        header += f"  {name[:20]:<22}"
    print(f"\n{header}")
    print("-" * len(header))
    for pname, _ in PROMPTS:
        row = f"{pname:<16}"
        for name, _ in sorted_models:
            dets = [d for d in results[name]["details"] if d["prompt"] == pname]
            if dets and dets[0]["success"]:
                d = dets[0]
                row += f"  {d['latency']:>4.1f}s/{d['score']:>3}/100{'':>4}"
            else:
                row += f"  {'FAILED':>15}"
        print(row)

    # Summary JSON
    summary = {}
    for name, data in results.items():
        summary[name] = {
            "avg_score": round(data["total_score"] / data["count"], 1),
            "avg_latency": round(data["total_latency"] / data["count"], 2),
            "total_score": data["total_score"],
            "total_latency": round(data["total_latency"], 2),
        }
    print(f"\n\nJSON Summary:\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
