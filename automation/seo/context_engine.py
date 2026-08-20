"""Build a provenance-aware evidence pack for cricket Shorts SEO."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from .cricket_context import correct_cricket_spelling, find_canonical_entities


_LIVE_QUERY = re.compile(r"\b(?:live|livestream|live stream|live score)\b", re.I)
_TOPIC_TERMS = (
    "coach", "captain", "six", "four", "wicket", "yorker", "century",
    "innings", "batting", "bowling", "selection", "debate", "analysis",
    "reaction", "test", "odi", "t20", "ipl", "world cup", "series",
)
_QUERY_SUFFIXES = (
    "cricket discussion",
    "cricket analysis",
    "fan reaction",
    "explained in hindi",
    "hinglish cricket opinion",
    "shorts discussion",
)


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", correct_cricket_spelling(str(value or ""))).strip()


def _clean_query(value: object) -> str:
    """Preserve real autocomplete wording, including audience aliases."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _dedupe(values: Iterable[str], limit: int = 15) -> List[str]:
    seen = set()
    result = []
    for value in values:
        clean = re.sub(r"\s+", " ", str(value or "")).strip()
        key = clean.casefold()
        if not clean or key in seen or _LIVE_QUERY.search(clean):
            continue
        seen.add(key)
        result.append(clean)
        if len(result) >= limit:
            break
    return result


def _query_anchors(text: str, entities: Dict[str, List[str]]) -> set[str]:
    anchors = set()
    for entity in entities["players"] + entities["teams"]:
        anchors.update(re.findall(r"[a-z0-9]+", entity.casefold()))
    low = text.casefold()
    anchors.update(term for term in _TOPIC_TERMS if term in low)
    return anchors


def build_grounded_search_queries(
    video_title: str,
    video_description: str,
    clip_transcript: str,
    suggestions: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return 8-15 queries tied to source/clip entities and current suggest.

    Autocomplete is accepted only when it shares a concrete anchor with the
    local evidence.  If the network is unavailable, deterministic combinations
    preserve the same grounding contract without inventing match facts.
    """
    title = _clean(video_title)
    description = _clean(video_description)
    transcript = _clean(clip_transcript)
    evidence = " ".join((title, description, transcript))
    entities = find_canonical_entities(evidence)
    anchors = _query_anchors(evidence, entities)

    accepted = []
    for suggestion in suggestions or []:
        clean = _clean_query(suggestion)
        words = set(re.findall(r"[a-z0-9]+", clean.casefold()))
        if words & anchors:
            accepted.append(clean)

    subjects = entities["players"] + entities["teams"]
    subject = subjects[0] if subjects else "cricket"
    team_context = entities["teams"][0] if entities["teams"] else ""
    low = evidence.casefold()
    topic = next((term for term in _TOPIC_TERMS if term in low), "cricket")
    local = [
        f"{subject} {topic}",
        f"{subject} {topic} {team_context}",
        f"{subject} {topic} debate",
        f"{subject} {topic} analysis",
        f"{subject} cricket opinion",
        f"should {subject} {topic} {team_context}",
        f"{subject} {team_context} discussion",
        f"{subject} {topic} explained",
    ]
    local.extend(f"{subject} {suffix}" for suffix in _QUERY_SUFFIXES)

    queries = _dedupe([*accepted, *local], limit=15)
    return queries[:15]


def build_cricket_evidence_pack(
    *,
    video_title: str,
    video_description: str,
    clip_transcript: str,
    ocr_entities: Optional[Dict] = None,
    research_context: Optional[Dict] = None,
) -> Dict:
    """Fuse local and fetched evidence while keeping provenance explicit."""
    title = _clean(video_title)
    description = _clean(video_description)
    transcript = _clean(clip_transcript)
    research = research_context or {}
    ocr = ocr_entities or {}
    match_facts = [
        _clean(fact) for fact in research.get("match_facts", [])
        if str(fact).strip()
    ]
    ocr_text = " ".join(
        str(item)
        for key in ("scoreboard", "player_names", "on_screen_text")
        for item in (ocr.get(key) or [])
    )
    grounding_text = " ".join((
        title,
        description,
        transcript,
        _clean(ocr_text),
        " ".join(match_facts),
    ))
    approved = build_grounded_search_queries(
        title,
        description,
        transcript,
        research.get("search_queries") or [],
    )
    return {
        "source_video": {"title": title, "description": description},
        "clip_transcript": transcript,
        "ocr": ocr,
        "grounded_entities": find_canonical_entities(grounding_text),
        "match_facts": match_facts,
        "current_topics": [str(topic) for topic in research.get("topics", []) if str(topic).strip()],
        "approved_search_queries": approved,
        "sources": [source for source in research.get("sources", []) if isinstance(source, dict)],
    }
