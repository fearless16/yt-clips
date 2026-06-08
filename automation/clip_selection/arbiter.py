"""Final Arbiter — combines agent scores and optionally runs LLM refinement.

Two-tier approach:
1. Weighted score from all 7 agents (fast, always runs)
2. LLM arbiter pass for top candidates (optional, refines rankings)

Weights can be overridden at call time (e.g. from weight_learner.py).
"""

import json
import re
from typing import Any

from utils.config import load_config
from utils.logger import get_logger
from utils.ai_client import AIClient

cfg = load_config()
log = get_logger("arbiter")


_DEFAULT_WEIGHTS: dict[str, float] = {
    "hook_expert": 0.35,
    "emotion_expert": 0.20,
    "viral_potential": 0.15,
    "cricket_context": 0.10,
    "viewer_psychology": 0.10,
    "retention_expert": 0.05,
    "technical_quality": 0.05,
}

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

    # Build candidate detail for LLM
    lines = []
    for i, c in enumerate(candidates, 1):
        text = c.get("text", "")[:150]
        agent_breakdown = c.get("agent_scores", {})
        scores_str = " | ".join(
            f"{k}:{v.get('score', 0):.0f}" for k, v in sorted(agent_breakdown.items())
        )
        lines.append(
            f"{i}. [{c['start']:.1f}s-{c['end']:.1f}s] "
            f"weighted={c.get('final_score', 0):.1f} "
            f"agents=[{scores_str}] "
            f"text={text}"
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

    system_prompt = (
        "You are the Final Clip Selection Arbiter. "
        "Your job: select the best clips for YouTube Shorts from scored candidates.\n\n"
        "Ranking priorities:\n"
        "1. Hook strength (first 3 seconds must grab)\n"
        "2. Emotional peak (crowd/commentator excitement)\n"
        "3. Cricket relevance (key players, big moments)\n"
        "4. Self-contained (makes sense without context)\n"
        "5. Viral potential (rare/controversial/shocking)\n\n"
        "Rules:\n"
        "- Reject clips that are boring, repetitive, or incomplete\n"
        "- Prefer shorter clips (15-30s) for Shorts retention\n"
        "- Prefer clips with audio peaks (crowd eruption, commentator scream)\n"
        "- Return valid JSON only\n"
        "- Max 10 clips"
    )

    user_prompt = (
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

    try:
        log.info("LLM arbiter: refining %d candidates...", len(candidates))
        response = _get_ai().generate_text(user_prompt, system_instruction=system_prompt)

        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            log.warning("LLM arbiter: no JSON found in response")
            return candidates_with_scores[:max_selected]

        result = json.loads(match.group(0))
        selected = result.get("selected", [])
        log.info("LLM arbiter: selected %d of %d candidates",
                 len(selected), len(candidates))

        # Apply LLM selection
        refined = []
        for sel in selected:
            idx = sel.get("candidate_id", 0) - 1
            if 0 <= idx < len(candidates):
                c = dict(candidates[idx])
                c["ai_score"] = sel.get("score", c.get("final_score", 0))
                c["ai_reason"] = sel.get("reason", "")
                refined.append(c)

        # Fill remaining slots if LLM didn't select enough
        if len(refined) < max_selected:
            selected_set = {s.get("candidate_id", 0) for s in selected}
            for i, c in enumerate(candidates):
                if (i + 1) not in selected_set and len(refined) < max_selected:
                    refined.append(c)

        refined.sort(key=lambda x: x.get("ai_score", x.get("final_score", 0)), reverse=True)
        return refined[:max_selected]

    except Exception as e:
        log.warning("LLM arbiter failed: %s — using weighted scores", e)
        return candidates_with_scores[:max_selected]


def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
