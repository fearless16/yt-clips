"""Final Arbiter — combines agent scores and optionally runs LLM refinement.

Two-tier approach:
1. Weighted score from all agents (fast, always runs)
2. LLM arbiter pass for top candidates (optional, refines rankings)

Weights are auto-built from agent class attributes — single source of truth.
Explicit call-time overrides remain available for controlled experiments.
"""

import json
import math
import re
from typing import Any

from utils.config import load_config
from utils.logger import get_logger
from utils.ai_client import AIClient

from automation.clip_selection.agents import ALL_AGENTS

cfg = load_config()
log = get_logger("arbiter")


def _build_default_weights() -> dict[str, float]:
    total = sum(a.weight for a in ALL_AGENTS if a.name != "brutal_rejection")
    if total <= 0:
        return {}
    return {
        a.name: round(a.weight / total, 4)
        for a in ALL_AGENTS
        if a.name != "brutal_rejection" and a.weight > 0
    }


_DEFAULT_WEIGHTS: dict[str, float] = _build_default_weights()
AGENT_WEIGHTS = dict(_DEFAULT_WEIGHTS)

# AI client for LLM arbiter pass
_ai: AIClient | None = None


def _get_ai() -> AIClient:
    global _ai
    if _ai is None:
        _ai = AIClient()
    return _ai


def compute_weighted_score(
    agent_scores: dict[str, dict],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Combine all agent scores using configured weights.

    Args:
        agent_scores: {agent_name: {"score": 0-100, "reasoning": str, ...}}
        weights: Optional weight overrides (defaults to AGENT_WEIGHTS)

    Returns:
        dict with final_score, breakdown, rejection_reasons, should_reject
    """
    if weights is None:
        weights = AGENT_WEIGHTS

    total = 0.0
    breakdown = {}
    rejection_reasons = []

    for agent_name, result in agent_scores.items():
        weight = weights.get(agent_name, 0.0)
        if agent_name == "brutal_rejection":
            rejection_score = result.get("score", 0)
            if result.get("should_reject", False):
                rejection_reasons.append(result.get("reasoning", ""))
            continue

        score = result.get("score", 0)
        weighted = score * weight
        total += weighted
        breakdown[agent_name] = {
            "raw": score,
            "weight": weight,
            "weighted": round(weighted, 2),
        }

    # Rejection penalty
    if rejection_reasons:
        total *= 0.5
        breakdown["rejection_penalty"] = {
            "penalty": 0.5,
            "reasons": rejection_reasons,
        }

    final_score = max(0.0, min(100.0, total))

    return {
        "final_score": round(final_score, 2),
        "breakdown": breakdown,
        "rejection_reasons": rejection_reasons,
        "should_reject": len(rejection_reasons) >= 2,
    }


def _render_match_facts(match_context: Any) -> str:
    """Render verified match context as a compact evidence line for the LLM."""
    if not match_context:
        return ""
    try:
        rendered = json.dumps(match_context, ensure_ascii=False)
    except (TypeError, ValueError):
        return ""
    rendered = re.sub(r"\s+", " ", rendered).strip()
    return rendered[:600]


# Safety valve: normal complete thoughts are <=20s of speech (~350 chars).
# The cap only bites on pathological outliers so one runaway segment cannot
# blow the prompt budget; the marker keeps the truncation honest to the LLM.
MAX_SCRIPT_CHARS = 2000


def _render_candidate_script(text: Any) -> str:
    script = str(text or "").strip()
    if len(script) > MAX_SCRIPT_CHARS:
        log.warning("Candidate script capped at %d chars (was %d)",
                    MAX_SCRIPT_CHARS, len(script))
        return script[:MAX_SCRIPT_CHARS].rstrip() + " …[script truncated]"
    return script


def _coerce_candidate_index(candidate_id: Any, total: int) -> int:
    """Map an LLM-provided candidate_id (int, float, numeric string) to a
    valid list index. Returns -1 when unusable or out of range."""
    try:
        idx = int(candidate_id) - 1
    except (TypeError, ValueError):
        return -1
    return idx if 0 <= idx < total else -1


def llm_arbiter_refine(
    candidates_with_scores: list[dict],
    context: dict,
    max_selected: int = 10,
) -> list[dict]:
    """Optional LLM pass to refine rankings.

    Takes the top-k by weighted score and asks LLM to make final selection
    considering the full picture.

    Returns re-ranked candidates list.
    """
    if len(candidates_with_scores) <= 1:
        return candidates_with_scores

    top_n = min(len(candidates_with_scores), max_selected + 5)
    candidates = candidates_with_scores[:top_n]

    transcript_segments = context.get("transcript_segments", [])

    # Build candidate detail for LLM — full clip script, never silently
    # truncated: the arbiter cannot judge self-containedness from a fragment.
    lines = []
    for i, c in enumerate(candidates, 1):
        text = _render_candidate_script(c.get("text"))
        agent_breakdown = c.get("agent_scores", {})
        content_type = str(c.get("content_type") or "").strip()
        type_tag = f" type={content_type}" if content_type else ""
        source_tag = " source_match=yes" if c.get("source_match") else ""
        scores_str = " | ".join(
            f"{k}:{v.get('score', 0):.0f}" for k, v in sorted(agent_breakdown.items())
        )
        lines.append(
            f"{i}. [{c['start']:.1f}s-{c['end']:.1f}s] "
            f"weighted={c.get('final_score', 0):.1f}{type_tag}{source_tag} "
            f"agents=[{scores_str}] "
            f"script={text}"
        )

    candidates_str = "\n".join(lines)

    # Build transcript context
    candidate_min = min(c["start"] for c in candidates)
    candidate_max = max(c["end"] for c in candidates)
    transcript_snippets = [
        f"[{fmt_ts(s['start'])}] {s.get('text', '')}"
        for s in transcript_segments
        if s["end"] > candidate_min and s["start"] < candidate_max
    ][:30]
    transcript_text = "\n".join(transcript_snippets)

    match_facts = _render_match_facts(context.get("match_context"))
    match_line = f"\nVerified match facts (evidence boundary): {match_facts}\n" if match_facts else ""

    trend_topics = [
        str(t).strip() for t in (context.get("trend_topics") or [])
        if str(t).strip()
    ][:10]
    trend_lines = ""
    if trend_topics:
        rendered = "\n".join(f"- {t}" for t in trend_topics)
        trend_lines = (
            "\nCurrent YouTube search demand (IN), newest signals:\n"
            f"{rendered}\n"
        )

    system_prompt = (
        "You are the Final Clip Selection Arbiter. "
        "Your job: select the best clips for YouTube Shorts from scored candidates.\n\n"
        "Ranking priorities:\n"
        "1. Hook strength (first 3 seconds must grab)\n"
        "2. Emotional peak (crowd/commentator excitement)\n"
        "3. Cricket relevance (key players, big moments)\n"
        "4. Self-contained (makes sense without context)\n"
        "5. Viral potential (rare/controversial/shocking)\n\n"
        "Content angles:\n"
        "- Each candidate line carries type=moment|comedy|debate|news.\n"
        "- Moments with visible stakes outrank debate/chatter when scores tie.\n"
        "- Comedy and news angles are valid when genuinely strong.\n\n"
        "Source-grounding:\n"
        "- Prefer clips that directly match the source event/title\n"
        "- Keep an off-topic tangent only when it is exceptionally strong, self-contained, and cricket-grounded\n"
        "- Never invent a player, match, or expansion of an ambiguous nickname\n\n"
        "Search demand:\n"
        "- Trend lines show what viewers search right now; prefer candidates "
        "whose content matches that demand when quality is otherwise close\n"
        "- Demand lines are context only: never mention them, and never add a "
        "player or event to a clip just because demand lists it\n\n"
        "ID contract:\n"
        '- Every candidate line starts with its integer number. "candidate_id" '
        "MUST be that integer (1-based), copied exactly. Never invent other id formats.\n\n"
        "Rules:\n"
        "- Reject clips that are boring, repetitive, or incomplete\n"
        "- Prefer shorter clips (15-30s) for Shorts retention\n"
        "- Prefer clips with audio peaks (crowd eruption, commentator scream)\n"
        "- Publishable bar: these candidates already cleared a hard quality\n"
        "  floor — pick the strongest available rather than returning empty.\n"
        "  Go empty ONLY when EVERY candidate is unusable chatter (no cricket\n"
        "  content, broken sentences, pure filler)\n"
        "- Return valid JSON only\n"
        "- Max 10 clips"
    )

    source_title = str(context.get("source_title", "") or "").strip()
    source_line = f"Source video title: {source_title}\n" if source_title else ""

    user_prompt = (
        f"{source_line}{match_line}{trend_lines}\n"
        f"Here are {len(candidates)} scored candidates:\n\n"
        f"{candidates_str}\n\n"
        f"Transcript context:\n{transcript_text}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "selected": [\n'
        '    {"candidate_id": 1, "score": 0-100, "reason": "..."},\n'
        '    {"candidate_id": 3, "score": 0-100, "reason": "..."}\n'
        "  ],\n"
        '  "rejected": [\n'
        '    {"candidate_id": 2, "reason": "..."}\n'
        "  ],\n"
        '  "notes": {"overall_quality": "high|medium|low"}\n'
        "}"
    )

    result: dict | None = None
    last_error: str | None = None
    for attempt in (1, 2):
        try:
            log.info("LLM arbiter: refining %d candidates (attempt %d)...",
                     len(candidates), attempt)
            response = _get_ai().generate_text(user_prompt, system_instruction=system_prompt)

            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                last_error = "no JSON found in response"
                log.warning("LLM arbiter: %s (attempt %d)", last_error, attempt)
                continue

            try:
                result = json.loads(match.group(0))
                break
            except json.JSONDecodeError as exc:
                last_error = f"malformed JSON: {exc}"
                log.warning("LLM arbiter: %s (attempt %d)", last_error, attempt)

        except Exception as e:
            last_error = str(e)
            log.warning("LLM arbiter: generation failed (attempt %d): %s", attempt, e)

    if result is None:
        log.warning("LLM arbiter failed after retry (%s) — using weighted scores",
                    last_error or "unknown")
        return candidates_with_scores[:max_selected]

    selected = result.get("selected", [])
    log.info("LLM arbiter: selected %d of %d candidates",
             len(selected), len(candidates))

    # Apply LLM selection. LLM output is adversarial input: ids may arrive as
    # strings/floats/out of range, entries may not be objects, duplicates
    # happen. Skip anything unusable instead of crashing the stage.
    refined: list[dict] = []
    seen_indices: set[int] = set()
    for sel in selected:
        if not isinstance(sel, dict):
            log.warning("LLM arbiter: skipping non-object selection entry: %r", sel)
            continue
        idx = _coerce_candidate_index(sel.get("candidate_id"), len(candidates))
        if idx < 0:
            log.warning("LLM arbiter: invalid candidate_id %r (expected 1-%d)",
                        sel.get("candidate_id"), len(candidates))
            continue
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        c = dict(candidates[idx])
        c["ai_score"] = sel.get("score", c.get("final_score", 0))
        c["ai_reason"] = sel.get("reason", "")
        refined.append(c)

    refined.sort(key=lambda x: x.get("ai_score", x.get("final_score", 0)), reverse=True)
    return refined[:max_selected]


def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
