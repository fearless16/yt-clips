#!/usr/bin/env python3
"""Benchmark models via direct API calls. Fast model comparison tool."""
import json
import time
import urllib.request
import urllib.error
import os
import sys

# API configs
PROVIDERS = {
    "Groq Scout": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key": os.environ.get("GROQ_API_KEY", ""),
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "MiniMax M2.5 Pro": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key": os.environ.get("OPENROUTER_API_KEY", ""),
        "model": "minimax/minimax-m2.5:free",
    },
    "Xiaomi Mimo 2.5 Pro": {
        "url": "https://opengateway.gitlawb.com/v1/xiaomi-mimo/chat/completions",
        "key": os.environ.get("XIAOMI_API_KEY", ""),
        "model": "mimo-v2.5-pro",
    },
    "Alibaba Qwen Max": {
        "url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "model": "qwen-max",
    },
    "Alibaba Qwen Plus": {
        "url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "model": "qwen-plus",
    },
}

PROMPTS = [
    ("coding_easy", "Write a Python function to find the longest palindromic substring in O(n^2). Return only code."),
    ("coding_medium", "Write a Python async web scraper that fetches 10 pages concurrently, parses all <a> hrefs, and saves unique URLs to a file. Include error handling and rate limiting."),
    ("logic", "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost? Explain step by step."),
    ("math", "If 3x + 7 = 22, and 2y - 5 = 3x, what is x + y? Show your work."),
    ("explain", "Explain quantum entanglement in 3 sentences as if to a 12-year-old."),
    ("creative", "Write a 4-line poem about debugging in the style of Edgar Allan Poe."),
    ("structured", "Return a JSON object with keys: name, version, features (array of 3 strings). Use the name 'BenchmarkTest' and version '1.0.0'. No other text."),
]

def call_model(name, cfg, prompt, timeout=30):
    start = time.time()
    body = json.dumps({
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        cfg["url"],
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['key']}",
        },
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        data = json.loads(resp.read())
        elapsed = time.time() - start
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"success": True, "response": content[:500], "latency": round(elapsed, 2)}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()[:200]
        return {"success": False, "response": "", "latency": round(time.time() - start, 2), "error": f"HTTP {e.code}: {err_body}"}
    except Exception as e:
        return {"success": False, "response": "", "latency": round(time.time() - start, 2), "error": str(e)}

def score(pname, response):
    if not response:
        return 0
    s = 50
    s += min(len(response) / 10, 20)
    if "coding" in pname:
        if any(kw in response for kw in ["def ", "class ", "import "]):
            s += 15
        if "```" in response:
            s += 10
        if "return" in response:
            s += 5
    if pname == "structured":
        try:
            json.loads(response)
            s += 30
        except: pass
    if any(kw in response.lower() for kw in ["i cannot", "i can't", "i'm sorry"]):
        s -= 30
    return max(0, min(100, s))

def main():
    print("=" * 100)
    print("  MODEL BENCHMARK: DeepSeek V4 / MiniMax M2.5 Pro / Xiaomi Mimo 2.5 Pro / Alibaba Qwen")
    print("  Testing", len(list(PROVIDERS.keys())), "models x", len(PROMPTS), "prompts")
    print("=" * 100)

    all_results = {}

    for mname, mcfg in PROVIDERS.items():
        if not mcfg["key"]:
            print(f"\n  SKIPPING {mname} (no API key)")
            continue
        print(f"\n  >>> {mname} ({mcfg['model']})")
        all_results[mname] = {"total_latency": 0, "total_score": 0, "n": 0, "details": []}
        for pname, prompt in PROMPTS:
            r = call_model(mname, mcfg, prompt)
            s = score(pname, r["response"]) if r["success"] else 0
            all_results[mname]["details"].append({**r, "prompt": pname, "score": s})
            all_results[mname]["total_latency"] += r["latency"]
            all_results[mname]["total_score"] += s
            all_results[mname]["n"] += 1
            status = "OK" if r["success"] else "ERR"
            preview = r["response"][:70].replace("\n", " ") if r["response"] else r.get("error", "FAILED")[:70]
            print(f"    {pname:<18} {r['latency']:>5.1f}s  {s:>3}/100  [{status}] {preview}")

        avg_l = all_results[mname]["total_latency"] / all_results[mname]["n"]
        avg_s = all_results[mname]["total_score"] / all_results[mname]["n"]
        print(f"    {'─'*65}")
        print(f"    {'AVERAGE':<18} {avg_l:>5.1f}s  {avg_s:>3.0f}/100")

    print("\n" + "=" * 100)
    print("  RANKING")
    print("=" * 100)
    ranked = sorted(all_results.items(), key=lambda x: x[1]["total_score"]/x[1]["n"], reverse=True)
    print(f"\n  {'Rank':<5} {'Model':<24} {'Avg Score':>10} {'Avg Latency':>12}")
    print("  " + "-" * 55)
    for i, (name, d) in enumerate(ranked, 1):
        avg_s = d["total_score"] / d["n"]
        avg_l = d["total_latency"] / d["n"]
        bar = "█" * int(avg_s / 5)
        print(f"  {i:<5} {name:<24} {avg_s:>6.0f}/100    {avg_l:>5.1f}s   {bar}")

    print("\n" + "=" * 100)
    print("  PER-PROMPT COMPARISON")
    print("=" * 100)
    headers = ["Prompt"] + [n[:18] for n, _ in ranked]
    print(f"\n  {'Prompt':<18}", end="")
    for h in headers[1:]:
        print(f"  {h:<20}", end="")
    print()
    print("  " + "-" * (18 + 22 * len(ranked)))
    for pname, _ in PROMPTS:
        print(f"  {pname:<18}", end="")
        for mname, _ in ranked:
            dets = [d for d in all_results[mname]["details"] if d["prompt"] == pname]
            if dets and dets[0]["success"]:
                d = dets[0]
                print(f"  {d['latency']:>4.1f}s/{d['score']:>3}", end="")
            else:
                print(f"  {'FAILED':>9}", end="")
        print()

    # JSON output
    summary = {}
    for name, d in all_results.items():
        summary[name] = {
            "avg_score": round(d["total_score"] / d["n"], 1),
            "avg_latency": round(d["total_latency"] / d["n"], 2),
        }
    print(f"\n\n  JSON: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main()
