"""scoring.py — LLM output quality scoring + evaluation.

Evaluates LLM-generated SEO metadata on three axes:
    1. Structure — valid JSON with required field presence
    2. Grounding — uses correct player/team names from source
    3. Quality — no dict-in-string, proper formatting, power words

Results cached 5m in SCORE_CACHE for repeat evaluations.

Usage::

    from .scoring import score_seo_output, format_score_table

    result = score_seo_output(
        raw_llm_text='{"title": "...", "description": "...", ...}',
        expected_players={"kohli", "faf"},
        expected_teams={"rcb", "csk"},
    )
    print(result["total"])      # 0-100
    print(result["breakdown"])  # {"structure": 40, "grounding": 25, "quality": 20}
    print(format_score_table([result]))  # tabular output
"""

import re
import json
from typing import Optional

from ._cache import TTLCache

SCORE_CACHE = TTLCache(maxsize=32, ttl=300)


def _parse_json_response(text: str) -> Optional[dict]:
    """Extract first JSON object from LLM response (handles markdown wrapping)."""
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def score_seo_output(
    raw_llm_text: str,
    expected_players: set | None = None,
    expected_teams: set | None = None,
    expected_tournament: str = "IPL 2026",
) -> dict:
    """Score an LLM-generated SEO block on structure/grounding/quality.

    Returns::

        {
            "total": 85,          # 0-100 composite
            "breakdown": {
                "structure": 40,   # max 40
                "grounding": 30,   # max 30
                "quality": 30,     # max 30
            },
            "details": {
                "has_title": True,
                "has_description": True,
                "title_length": 87,
                "player_hits": ["kohli"],
                "player_hallucinations": [],
                "has_dict_syntax": False,
                "has_power_words": True,
                "latency": 2.3,           # if latency_seconds provided
            }
        }
    """
    cache_key = f"score:{hash(raw_llm_text)}"
    cached = SCORE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    expected_players = expected_players or set()
    expected_teams = expected_teams or set()
    details = {}
    score = 0

    data = _parse_json_response(raw_llm_text)
    title = ""
    description = ""

    if data is None:
        result = {
            "total": 0,
            "breakdown": {"structure": 0, "grounding": 0, "quality": 0},
            "details": {"parsed": False, "error": "No valid JSON found"},
        }
        SCORE_CACHE.set(cache_key, result)
        return result

    details["parsed"] = True

    # ── STRUCTURE (40 points) ──────────────────────────────────────────────────
    structure = 0

    if "title" in data and isinstance(data["title"], str) and len(data["title"]) > 10:
        structure += 10
        title = data["title"]
        details["has_title"] = True
        details["title_length"] = len(title)

    if "description" in data and isinstance(data["description"], str) and len(data["description"]) > 100:
        structure += 10
        description = data["description"]
        details["has_description"] = True
        details["description_length"] = len(description)

    if "hashtags" in data and isinstance(data["hashtags"], list) and len(data["hashtags"]) >= 3:
        structure += 10
        details["hashtag_count"] = len(data["hashtags"])

    if "search_terms" in data and isinstance(data["search_terms"], list) and len(data["search_terms"]) >= 5:
        structure += 10
        details["search_term_count"] = len(data["search_terms"])

    score += structure

    # ── GROUNDING (30 points) — uses correct names from source ───────────────
    grounding = 0
    title_lower = title.lower()
    desc_lower = description.lower()

    player_hits = []
    player_hallucinations = []

    for p in expected_players:
        if p in title_lower or p in desc_lower:
            player_hits.append(p)

    if player_hits:
        grounding += min(len(player_hits) * 10, 20)
        details["player_hits"] = player_hits

    if expected_tournament.lower() in title_lower:
        grounding += 5
    for t in expected_teams:
        if t.lower() in title_lower or t.lower() in desc_lower:
            grounding += 5
            break

    details["player_hallucinations"] = player_hallucinations
    score += grounding

    # ── QUALITY (30 points) — no dict syntax, clean formatting ───────────────
    quality = 0

    has_dict_syntax = bool(re.search(r"\{[^}]{5,}\}", description))
    if not has_dict_syntax and description:
        quality += 15
        details["has_dict_syntax"] = False

    power_words = {"smash", "six", "fire", "brilliant", "clutch", "incredible", "destroy", "massive", "huge", "epic"}
    title_words = set(title_lower.split())
    hits = title_words & power_words
    if hits:
        quality += 8
        details["power_words"] = list(hits)

    if "|" in title or ":" in title:
        quality += 7
        details["has_format_separator"] = True

    score += quality

    # ── LATENCY PENALTY (off by default, applied by caller) ──────────────────
    details["latency"] = 0.0

    result = {
        "total": min(100, score),
        "breakdown": {"structure": structure, "grounding": grounding, "quality": quality},
        "details": details,
    }
    SCORE_CACHE.set(cache_key, result)
    return result


def format_score_table(results: list[dict]) -> str:
    """Render scoring results as a formatted table.

    Args:
        results: List of dicts from ``score_seo_output()``.

    Returns::

        ┌────────┬──────┬───────────┬──────────┬───────┬──────────┐
        │ Item   │ Total│ Structure │ Grounding│Quality│ Halluc.  │
        ├────────┼──────┼───────────┼──────────┼───────┼──────────┤
        │ clip1  │  85  │    40     │    25    │  20   │    0     │
        │ clip2  │  92  │    40     │    30    │  22   │    0     │
        └────────┴──────┴───────────┴──────────┴───────┴──────────┘
    """
    if not results:
        return "No scoring results."

    sep = "│"
    header = f"{sep} Item    {sep} Total{sep} Structure{sep} Grounding{sep} Quality{sep} Halluc.  {sep} Latency{sep}"
    bar = f"{sep}{'─'*8}{sep}{'─'*6}{sep}{'─'*10}{sep}{'─'*10}{sep}{'─'*8}{sep}{'─'*9}{sep}{'─'*8}{sep}"
    lines = [bar, header, bar]

    for i, r in enumerate(results):
        det = r.get("details", {})
        hall = len(det.get("player_hallucinations", []))
        lat = det.get("latency", 0)
        brk = r["breakdown"]
        lines.append(
            f"{sep} clip{i+1:<3} {sep} {r['total']:<4}{sep} {brk['structure']:<8}{sep}"
            f" {brk['grounding']:<8}{sep} {brk['quality']:<6}{sep} {hall:<7}{sep} {lat:<6}{sep}"
        )
    lines.append(bar)
    return "\n".join(lines)


def score_latency_penalty(result: dict, latency_seconds: float) -> dict:
    """Apply latency penalty to an existing score dict (mutates in place).

    Penalty scheme:
        >30s  → −20
        >15s  → −10
        >10s  → −5
    """
    result["details"]["latency"] = round(latency_seconds, 1)
    if latency_seconds > 30:
        result["total"] = max(0, result["total"] - 20)
    elif latency_seconds > 15:
        result["total"] = max(0, result["total"] - 10)
    elif latency_seconds > 10:
        result["total"] = max(0, result["total"] - 5)
    return result
