"""ClipSelector — orchestrates all 7 agents + Final Arbiter.

Replaces the old `_parallel_score_candidates` + `_refine_highlights_with_ai`
in ``highlight.py``.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from utils.config import load_config
from utils.logger import get_logger

from automation.clip_selection.agents import ALL_AGENTS, BrutalRejectionAgent
from automation.clip_selection.arbiter import compute_weighted_score, llm_arbiter_refine

cfg = load_config()
log = get_logger("clip_selector")


class ClipSelector:
    """Runs all 7 agents on candidates, then Final Arbiter ranks them."""

    def __init__(self, use_llm_arbiter: bool = True, weights: dict[str, float] | None = None):
        self.agents = ALL_AGENTS
        self.use_llm_arbiter = use_llm_arbiter
        self._weights = weights

    def score_candidates(
        self,
        candidates: list[dict],
        context: dict,
        max_workers: int = 8,
    ) -> list[dict]:
        """Score all candidates through all 7 agents in parallel.

        Args:
            candidates: List of candidate dicts with start/end/text
            context: Dict with rms_map, avg_rms, max_rms, transcript_segments, etc.
            max_workers: Thread pool size

        Returns:
            Candidates sorted by final weighted score, with agent_scores attached.
        """
        # Build set of all candidate texts for rejection duplicate detection
        all_texts = set()
        for c in candidates:
            t = c.get("text", "").strip()
            if t:
                all_texts.add(t)
        context["all_candidate_texts"] = all_texts

        scored_candidates = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for candidate in candidates:
                fut = executor.submit(self._score_single, candidate, context)
                futures[fut] = candidate

            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    result = future.result()
                    scored_candidates.append(result)
                except Exception as e:
                    log.warning("Agent scoring failed for %.1f-%.1f: %s",
                                candidate["start"], candidate["end"], e)
                    candidate["final_score"] = 0
                    candidate["agent_scores"] = {}
                    candidate["score_breakdown"] = {}
                    candidate["rejection_reasons"] = []
                    candidate["should_reject"] = True
                    scored_candidates.append(candidate)

        scored_candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return scored_candidates

    def select(
        self,
        scored_candidates: list[dict],
        context: dict,
        max_selected: int = 10,
        min_quality: float = 20.0,
    ) -> list[dict]:
        """Select top clips from scored candidates.

        Filters by quality threshold, rejects bad clips, runs optional LLM arbiter.

        Args:
            scored_candidates: Output of score_candidates()
            context: Full context dict (needed for LLM arbiter)
            max_selected: Max clips to select
            min_quality: Minimum weighted score to keep a clip (0-100)

        Returns:
            Final ranked clip list.
        """
        # Filter rejects
        keep = [c for c in scored_candidates if not c.get("should_reject", False)]
        rejected_count = sum(1 for c in scored_candidates if c.get("should_reject", False))
        if rejected_count:
            log.info("Rejected %d/%d candidates by agent consensus",
                     rejected_count, len(scored_candidates))

        # Quality threshold
        quality_pass = [c for c in keep if c.get("final_score", 0) >= min_quality]
        if not quality_pass:
            log.warning("No candidates passed quality threshold %.1f — using top %d",
                        min_quality, min(max_selected, len(keep)))
            quality_pass = keep[:max_selected] if keep else scored_candidates[:max_selected]

        log.info("Quality pass: %d/%d candidates above %.1f",
                 len(quality_pass), len(keep), min_quality)

        # LLM arbiter refinement
        if self.use_llm_arbiter and len(quality_pass) >= 2:
            final = llm_arbiter_refine(quality_pass, context, max_selected)
        else:
            final = quality_pass[:max_selected]

        return final

    def _score_single(self, candidate: dict, context: dict) -> dict:
        """Run all 7 agents on one candidate."""
        agent_scores = {}
        for agent in self.agents:
            try:
                result = agent.score(candidate, context)
                agent_scores[agent.name] = result
            except Exception as e:
                log.warning("Agent %s failed: %s", agent.name, e)
                agent_scores[agent.name] = {"score": 0, "reasoning": f"error: {e}"}

        # Combine into final score
        combined = compute_weighted_score(agent_scores, weights=self._weights)
        rejection = agent_scores.get("brutal_rejection", {})

        candidate["agent_scores"] = agent_scores
        candidate["final_score"] = combined["final_score"]
        candidate["score_breakdown"] = combined["breakdown"]
        candidate["rejection_reasons"] = combined["rejection_reasons"]
        candidate["should_reject"] = (
            combined["should_reject"]
            or rejection.get("should_reject", False)
        )

        return candidate


def select_best_clips(
    candidates: list[dict],
    transcript_segments: list[dict],
    rms_map: dict,
    avg_rms: float,
    max_rms: float,
    match_context: dict | None = None,
    max_selected: int = 10,
    use_llm_arbiter: bool = True,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Convenience function: score + select in one call.

    Args:
        candidates: Raw candidate list with start/end/text
        transcript_segments: Full transcript segment list
        rms_map: Audio RMS energy map (second -> rms)
        avg_rms: Average RMS across full video
        max_rms: Maximum RMS across full video
        match_context: Optional match info (players, teams, highlights)
        max_selected: Max clips to return
        use_llm_arbiter: Whether to run LLM refinement
        weights: Optional learned agent weights (from weight_learner)

    Returns:
        Final ranked clips with agent scores attached.
    """
    selector = ClipSelector(use_llm_arbiter=use_llm_arbiter, weights=weights)

    context = {
        "rms_map": rms_map,
        "avg_rms": avg_rms,
        "max_rms": max_rms,
        "transcript_segments": transcript_segments,
        "match_context": match_context or {},
    }

    scored = selector.score_candidates(candidates, context)
    final = selector.select(scored, context, max_selected=max_selected)

    return final
