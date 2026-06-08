"""HookAuditor — analyzes first 3 seconds of a clip for hook quality.

Phase 3 tool: examines the opening moment of each shortlisted clip
and scores its hook strength based on audio, video, and transcript cues.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("hook_auditor")


HOOK_TYPES = {
    "wicket": {"keywords": {"out", "bowled", "caught", "lbw", "stumped", "wicket", "gone"}},
    "six": {"keywords": {"six", "chhakka", "sixer", "maximum", "over the rope"}},
    "four": {"keywords": {"four", "chauka", "boundary"}},
    "crowd_eruption": {"keywords": {"crowd", "roar", "stadium", "fans", "audience"}},
    "commentator_scream": {"keywords": {"OH", "WOW", "WHAT A", "INCREDIBLE", "NO WAY"}},
    "reaction_face": {},
    "controversy": {"keywords": {"fight", "argument", "angry", "review", "drs", "controversy"}},
    "milestone": {"keywords": {"century", "fifty", "record", "hattrick", "maiden"}},
    "drama": {"keywords": {"review", "drs", "decision", "umpire", "controversy"}},
    "instant_payoff": {"keywords": {"out", "gone", "taken", "got him", "win", "victory"}},
}


def analyze_clip_hook(
    clip_path: str | Path,
    transcript_text: str,
    start_sec: float,
    clip_duration: float,
    rms_map: dict | None = None,
    avg_rms: float | None = None,
) -> dict[str, Any]:
    """Analyze the first 3 seconds of a clip for hook quality.

    Args:
        clip_path: Path to the exported MP4 clip
        transcript_text: Clip transcript text
        start_sec: Start time in the original video
        clip_duration: Duration of clip
        rms_map: Optional audio RMS energy map (second -> RMS)
        avg_rms: Optional average RMS for the full video

    Returns:
        Dict with hook_score, hook_type, swipe_risk, and reasoning.
    """
    score = 0
    reasons = []
    hook_types_found = set()
    swipe_risks = []

    text_lower = transcript_text.lower()

    # ── 1. Text-based hook detection ────────────────────────────────────────
    first_20_words = " ".join(transcript_text.split()[:20]).lower()

    for hook_type, config in HOOK_TYPES.items():
        if "keywords" in config:
            for kw in config["keywords"]:
                if kw in first_20_words or kw in text_lower[:60]:
                    score += 15
                    hook_types_found.add(hook_type)
                    reasons.append(f"hook_type={hook_type}")
                    break

    # ── 2. Reaction word opener ─────────────────────────────────────────────
    reaction_openers = {"oh", "wow", "what", "no", "yes", "arre", "kya", "dekho",
                        "oho", "accha", "haan", "bhai", "yaar", "whoa", "wait"}
    first_word = transcript_text.split()[0].lower().strip(".!?,") if transcript_text.split() else ""
    if first_word in reaction_openers:
        score += 25
        reasons.append(f"reaction_opener={first_word}")
        hook_types_found.add("reaction_face")

    # ── 3. Punctuation excitement ──────────────────────────────────────────
    if "!" in transcript_text[:40]:
        score += 10
        reasons.append("exclamation_early")
    if "?" in transcript_text[:40]:
        score += 8
        reasons.append("question_hook")

    # ── 4. Audio energy check ──────────────────────────────────────────────
    if rms_map and avg_rms and avg_rms > 0:
        hook_seconds = min(int(start_sec) + 3, int(start_sec + clip_duration))
        hook_rms_values = [
            rms_map.get(t, 0.0)
            for t in range(int(start_sec), hook_seconds + 1)
        ]
        if hook_rms_values:
            hook_avg = sum(hook_rms_values) / len(hook_rms_values)
            if hook_avg > avg_rms * 1.5:
                score += 20
                reasons.append(f"audio_spike={hook_avg/avg_rms:.1f}x")
                hook_types_found.add("crowd_eruption")
            elif hook_avg > avg_rms * 1.2:
                score += 10
                reasons.append(f"audio_boost={hook_avg/avg_rms:.1f}x")

    # ── 5. Swipe risk assessment ──────────────────────────────────────────
    if clip_duration < 8:
        swipe_risks.append("too_short_to_hook")

    if not reasons:
        swipe_risks.append("no_immediate_hook")
        score = 5

    if not transcript_text.strip():
        swipe_risks.append("silent_opening")
        score = 0

    if score < 30:
        swipe_risks.append("weak_opening")

    # ── 6. Hook type classification ─────────────────────────────────────────
    primary_hook = list(hook_types_found)[0] if hook_types_found else "generic_start"

    return {
        "hook_score": min(100, score),
        "hook_type": primary_hook,
        "hook_types_found": list(hook_types_found),
        "swipe_risk": "high" if not hook_types_found and score < 20 else (
            "medium" if score < 40 or len(swipe_risks) > 1 else "low"
        ),
        "swipe_risks": swipe_risks,
        "reason": "; ".join(reasons) if reasons else "no_hook_detected",
        "first_20_words": " ".join(transcript_text.split()[:20]),
    }


def analyze_multiple_clips(
    clip_data_list: list[dict],
    rms_map: dict | None = None,
    avg_rms: float | None = None,
) -> list[dict]:
    """Run hook analysis on multiple clips.

    Args:
        clip_data_list: List of dicts with clip_path, transcript_text, start_sec, clip_duration
        rms_map: Optional audio RMS map
        avg_rms: Optional average RMS

    Returns:
        List of hook analysis results.
    """
    results = []
    for data in clip_data_list:
        result = analyze_clip_hook(
            clip_path=data["clip_path"],
            transcript_text=data["transcript"],
            start_sec=data["start_sec"],
            clip_duration=data["clip_duration"],
            rms_map=rms_map,
            avg_rms=avg_rms,
        )
        result["clip_id"] = data.get("clip_id", "unknown")
        results.append(result)
    return results


class HookAuditor:
    """Thin class wrapper around ``analyze_clip_hook`` for orchestrator integration.

    Mirrors the ``ClipSelector`` pattern in ``selector.py`` so the rest of
    the codebase can treat both as services.
    """

    def analyze_clip_hook(
        self,
        clip_path: str | Path,
        transcript_text: str,
        start_sec: float,
        clip_duration: float,
        rms_map: dict | None = None,
        avg_rms: float | None = None,
    ) -> dict[str, Any]:
        return analyze_clip_hook(
            clip_path=clip_path,
            transcript_text=transcript_text,
            start_sec=start_sec,
            clip_duration=clip_duration,
            rms_map=rms_map,
            avg_rms=avg_rms,
        )

    def analyze_multiple_clips(
        self,
        clip_data_list: list[dict],
        rms_map: dict | None = None,
        avg_rms: float | None = None,
    ) -> list[dict]:
        return analyze_multiple_clips(
            clip_data_list=clip_data_list,
            rms_map=rms_map,
            avg_rms=avg_rms,
        )
