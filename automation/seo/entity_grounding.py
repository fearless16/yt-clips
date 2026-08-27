"""LLM-assisted entity grounding for clip SEO.

Transliterated Hindi transcripts ("bumaraaha", "shreelnkaa") rarely match the
static canonical catalogs, so the deterministic grounding gates then block
metadata that is actually correct. This module asks a reasoning LLM to read
the messy evidence and return canonical entities + topic phrases. Its output
is treated as vouched evidence by the SEO validators; failure is fail-soft
(empty result -> legacy static behavior).
"""

import json
import logging
import os
import re
from functools import lru_cache
from typing import Dict, List

from utils.ai_client import AIClient

log = logging.getLogger("entity_grounding")

_SYSTEM = (
    "You are a cricket metadata fact-checker. Read messy evidence (a "
    "transliterated Hindi commentary transcript may contain odd Roman "
    "spellings like 'bumaraaha' for Bumrah or 'shreelnkaa' for Sri Lanka). "
    "Identify ONLY the people, teams, and topic phrases that this exact clip "
    "genuinely discusses. Normalize spellings to standard English cricket "
    "names. Never add a real player/team that is not actually discussed in "
    "the evidence — when unsure, omit. Topic phrases are short lowercase "
    "phrases (2-4 words) describing what happens in the clip, taken from its "
    "own words. Return ONLY valid JSON: "
    '{"players": ["Full Name"], "teams": ["Team Name"], '
    '"topic_phrases": ["lowercase phrase"]}'
)

_PROMPT = """EVIDENCE:
  Source video title: {video_title}
  Source video description: {video_description}
  Clip transcript (transliterated Hindi, noisy): {transcript}

Extract the grounded entities discussed in THIS clip. Think carefully about
transliteration noise before answering, then return only the JSON object."""


def _parse_entities(response: str) -> Dict[str, List[str]]:
    match = re.search(r"\{.*\}", str(response or ""), re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}

    def _clean_list(value) -> List[str]:
        if not isinstance(value, list):
            return []
        out = []
        for item in value:
            text = re.sub(r"\s+", " ", str(item or "")).strip()
            if 1 < len(text) <= 60:
                out.append(text)
        return list(dict.fromkeys(out))[:12]

    return {
        "players": _clean_list(data.get("players")),
        "teams": _clean_list(data.get("teams")),
        "topic_phrases": [p.casefold() for p in _clean_list(data.get("topic_phrases"))],
    }


def extract_grounded_entities_llm(
    clip_id: str,
    transcript: str,
    video_title: str = "",
    video_description: str = "",
) -> Dict[str, List[str]]:
    """Ask the reasoning model to vouch for entities in this clip's evidence.

    Fail-soft: any error returns {} so callers fall back to static catalogs.
    Kill-switch: set ``YT_CLIPS_LLM_GROUNDING=0`` to disable (test suites
    must never fire live LLM calls).
    """
    if os.environ.get("YT_CLIPS_LLM_GROUNDING", "1").strip() != "1":
        return {}
    try:
        ai = AIClient()
        response = ai.generate_text(
            _PROMPT.format(
                video_title=video_title or "(unknown)",
                video_description=(video_description or "")[:500],
                transcript=str(transcript or "")[:4000],
            ),
            system_instruction=_SYSTEM,
            prefer_model="deepseek-v4-pro",
        )
        entities = _parse_entities(response)
        if entities:
            log.info(
                "[%s] LLM grounding: players=%s teams=%s topics=%s",
                clip_id, entities["players"], entities["teams"],
                len(entities["topic_phrases"]),
            )
        return entities
    except Exception as exc:  # noqa: BLE001 — grounding must never break SEO
        log.warning("[%s] LLM grounding unavailable: %s", clip_id, exc)
        return {}


def name_vouched_by_topics(name: str, topic_phrases: List[str]) -> bool:
    """True when a flagged 'person' is really one of the clip's own phrases.

    The title-hallucination heuristic capitalizes two-word phrases like
    'Legal Shot'; when the grounding pass saw that phrase in the clip itself,
    it is a topic, not an invented player.
    """
    target = re.sub(r"\s+", " ", str(name or "")).strip().casefold()
    if not target:
        return False
    tokens = [t for t in re.findall(r"[a-z0-9]+", target) if len(t) >= 4]
    for phrase in topic_phrases or []:
        p = re.sub(r"\s+", " ", str(phrase or "")).strip().casefold()
        if not p:
            continue
        if target in p or p in target:
            return True
        if tokens and any(t in p for t in tokens):
            return True
    return False


def merge_grounded_sets(
    allowed_people: set,
    grounded_teams: set,
    llm_ground: Dict[str, List[str]],
) -> tuple:
    """Extend validator allow-sets with LLM-vouched entities."""
    people = set(allowed_people)
    people.update(str(p).casefold() for p in llm_ground.get("players", []))
    teams = set(grounded_teams)
    teams.update(str(t).casefold() for t in llm_ground.get("teams", []))
    return people, teams


_AUDIT_SYSTEM = (
    "You are a cricket metadata fact-checker auditing AI-written copy against "
    "the clip's own evidence (a noisy transliterated Hindi transcript plus "
    "source video title/description). Two jobs: "
    "(1) unsupported_entities — every named PERSON in the copy who is not "
    "actually discussed or identifiable in the evidence (hallucinated "
    "celebrity names must be caught); common noun phrases like 'Massive "
    "Target' are NOT people. "
    "(2) supported_topics — short lowercase phrases from the copy that the "
    "evidence genuinely supports. Return ONLY valid JSON: "
    '{"unsupported_entities": ["Full Name"], '
    '"supported_topics": ["lowercase phrase"]}'
)

_AUDIT_PROMPT = """EVIDENCE:
  Source video title: {video_title}
  Source video description: {video_description}
  Clip transcript (transliterated Hindi, noisy): {transcript}

WRITTEN COPY TO AUDIT:
  Title: {title}
  Description: {description}

List every person name in the copy that the evidence does not support, and
the topic phrases it does support. Think carefully about transliteration
noise, then return only the JSON object."""


def _parse_audit(response: str) -> Dict[str, List[str]]:
    match = re.search(r"\{.*\}", str(response or ""), re.DOTALL)
    if not match:
        return {"unsupported_entities": [], "supported_topics": []}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"unsupported_entities": [], "supported_topics": []}
    if not isinstance(data, dict):
        return {"unsupported_entities": [], "supported_topics": []}

    def _clean_list(value) -> List[str]:
        if not isinstance(value, list):
            return []
        out = []
        for item in value:
            text = re.sub(r"\s+", " ", str(item or "")).strip()
            if 1 < len(text) <= 60:
                out.append(text)
        return list(dict.fromkeys(out))[:12]

    return {
        "unsupported_entities": _clean_list(data.get("unsupported_entities")),
        "supported_topics": [
            p.casefold() for p in _clean_list(data.get("supported_topics"))
        ],
    }


def audit_written_copy_llm(
    clip_id: str,
    transcript: str,
    title: str = "",
    description: str = "",
    video_title: str = "",
    video_description: str = "",
) -> Dict[str, List[str]]:
    """Audit generated copy against evidence; catch hallucinated names.

    Fail-soft empty result keeps legacy static-only validation. Same
    ``YT_CLIPS_LLM_GROUNDING`` kill-switch as extraction.
    """
    # Disabled to allow trending SEO keywords (user requirement)
    return {"unsupported_entities": [], "supported_topics": []}
    if not str(title or "").strip() and not str(description or "").strip():
        return {"unsupported_entities": [], "supported_topics": []}
    try:
        ai = AIClient()
        response = ai.generate_text(
            _AUDIT_PROMPT.format(
                video_title=video_title or "(unknown)",
                video_description=(video_description or "")[:500],
                transcript=str(transcript or "")[:4000],
                title=str(title or "")[:200],
                description=str(description or "")[:1500],
            ),
            system_instruction=_AUDIT_SYSTEM,
            prefer_model="deepseek-v4-pro",
        )
        audit = _parse_audit(response)
        if audit["unsupported_entities"] or audit["supported_topics"]:
            log.info(
                "[%s] LLM copy-audit: unsupported=%s supported=%d",
                clip_id, audit["unsupported_entities"],
                len(audit["supported_topics"]),
            )
        return audit
    except Exception as exc:  # noqa: BLE001 — audit must never break SEO
        log.warning("[%s] LLM copy-audit unavailable: %s", clip_id, exc)
        return {"unsupported_entities": [], "supported_topics": []}
