"""Build a provenance-aware evidence pack for cricket Shorts SEO."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from .cricket_context import correct_cricket_spelling, find_canonical_entities


_LIVE_QUERY = re.compile(r"\b(?:live|livestream|live stream|live score)\b", re.I)
_TOPIC_TERMS = (
    "coach", "captaincy", "captain", "six", "four", "wicket", "yorker", "century",
    "innings", "batting", "bowling", "selection", "debate", "analysis",
    "reaction", "test", "odi", "t20", "ipl", "world cup", "series",
)

_HINDI_TOPIC_TERMS = (
    ("कप्तानी", "captaincy"),
    ("कोच", "coach"),
    ("वर्ल्ड कप", "world cup"),
    ("विकेट", "wicket"),
    ("छक्का", "six"),
    ("चौका", "four"),
    ("टेस्ट", "test"),
    ("वनडे", "odi"),
    ("सीरीज", "series"),
)
_QUERY_SUFFIXES = (
    "cricket analysis",
    "fan reaction",
    "explained",
    "match review",
    "key moment",
    "turning point",
    "shorts",
)


def _local_query_combos(subject: str, topic: str, team_context: str,
                         secondary: List[str]) -> List[str]:
    """Deterministic long-tail combos grounded in the exact clip premise.

    Avoid phrases such as "full highlights" or "live" unless the source itself
    supports them. Those queries can attract the wrong viewer and wreck retention.
    """
    subject = re.sub(r"\s+", " ", str(subject or "")).strip()
    topic = re.sub(r"\s+", " ", str(topic or "cricket")).strip()
    team_context = re.sub(r"\s+", " ", str(team_context or "")).strip()
    combos = [
        f"{subject} {topic}",
        f"{subject} {topic} {team_context}".rstrip(),
        f"{subject} {topic} reaction",
        f"{subject} {topic} analysis",
        f"{subject} {topic} explained",
        f"{subject} cricket discussion",
        f"{subject} match moment",
        f"{subject} shorts",
    ]
    if team_context:
        combos.extend([
            f"{subject} vs {team_context} {topic}",
            f"{subject} {team_context} cricket",
        ])
    for name in secondary[:4]:
        combos.extend([
            f"{name} {topic}",
            f"{name} {topic} reaction",
            f"{name} cricket analysis",
        ])
    return combos


def _clean(
    value: object,
    player_names: Optional[Iterable[str]] = None,
    player_aliases: Optional[Dict[str, str]] = None,
) -> str:
    return re.sub(
        r"\s+", " ",
        correct_cricket_spelling(
            str(value or ""), player_names=player_names,
            player_aliases=player_aliases,
        ),
    ).strip()


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


def _topic_from_text(text: str) -> str:
    low = text.casefold()
    for marker, topic in _HINDI_TOPIC_TERMS:
        if marker in low:
            return topic
    return next((term for term in _TOPIC_TERMS if term in low), "")


def build_grounded_search_queries(
    video_title: str,
    video_description: str,
    clip_transcript: str,
    suggestions: Optional[Iterable[str]] = None,
    player_names: Optional[Iterable[str]] = None,
    player_aliases: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Return 8-15 queries tied to source/clip entities and current suggest.

    Autocomplete is accepted only when it shares a concrete anchor with the
    local evidence.  If the network is unavailable, deterministic combinations
    preserve the same grounding contract without inventing match facts.
    """
    runtime_players = list(player_names or [])
    title = _clean(video_title, runtime_players, player_aliases)
    description = _clean(video_description, runtime_players, player_aliases)
    transcript = _clean(clip_transcript, runtime_players, player_aliases)
    evidence = " ".join((title, description, transcript))
    entities = find_canonical_entities(evidence, runtime_players)
    anchors = _query_anchors(evidence, entities)

    accepted = []
    evidence_players = set(entities["players"])
    evidence_teams = set(entities["teams"])
    for suggestion in suggestions or []:
        clean = _clean_query(suggestion)
        words = set(re.findall(r"[a-z0-9]+", clean.casefold()))
        if not (words & anchors):
            continue
        # Autocomplete is discovery evidence, not factual evidence. A suggestion
        # that introduces another canonical player/team can poison the vidIQ seed
        # even though it shares a broad anchor such as "India". Reject it here.
        suggested_entities = find_canonical_entities(clean, runtime_players)
        if set(suggested_entities["players"]) - evidence_players:
            continue
        if set(suggested_entities["teams"]) - evidence_teams:
            continue
        accepted.append(clean)

    clip_entities = find_canonical_entities(transcript, runtime_players)
    # The exact spoken player is usually the best Shorts search anchor. Fall back
    # to a clip team, then source-level entities. This keeps vidIQ research from
    # starting with a broad team query when the clip is really about one player.
    candidates = [
        *clip_entities["players"],
        *clip_entities["teams"],
        *entities["players"],
        *entities["teams"],
    ]
    candidates = list(dict.fromkeys(candidates))
    subject = next(
        (name for name in candidates if _is_canonical_subject(name, evidence)),
        "cricket",
    )
    secondary = [
        name for name in candidates
        if name != subject and _is_canonical_subject(name, evidence)
    ][:4]
    team_pool = list(dict.fromkeys([*clip_entities["teams"], *entities["teams"]]))
    team_context = next((team for team in team_pool if team != subject), "")
    topic = _topic_from_text(transcript) or _topic_from_text(evidence) or "cricket"
    local = _local_query_combos(subject, topic, team_context, secondary)
    local.extend(f"{subject} {suffix}" for suffix in _QUERY_SUFFIXES)

    # Clip-built queries lead so the first seed sent to vidIQ is about the exact
    # spoken moment. Autocomplete remains useful enrichment, but never owns seed 1.
    queries = _dedupe([*local, *accepted], limit=20)
    return queries[:20]


def _is_canonical_subject(name: str, evidence: str) -> bool:
    """Only roster-backed proper names qualify as secondary query subjects.

    A 'player' discovered as a capitalized bigram (e.g. 'Massive Target')
    is rejected when its tokens are common words rather than a roster name.
    """
    tokens = re.findall(r"[A-Za-z]+", str(name or ""))
    if not tokens:
        return False
    common = {
        "massive", "target", "huge", "total", "live", "match", "today",
        "day", "test", "score", "runs", "wickets", "highlights", "review",
    }
    return not all(token.casefold() in common for token in tokens)


def build_cricket_evidence_pack(
    *,
    video_title: str,
    video_description: str,
    clip_transcript: str,
    ocr_entities: Optional[Dict] = None,
    research_context: Optional[Dict] = None,
) -> Dict:
    """Fuse local and fetched evidence while keeping provenance explicit."""
    research = research_context or {}
    runtime_players = [
        _clean(name) for name in research.get("player_names", [])
        if str(name).strip()
    ]
    runtime_aliases = {
        str(alias).casefold(): str(player)
        for alias, player in (research.get("player_aliases") or {}).items()
    }
    title = _clean(video_title, runtime_players, runtime_aliases)
    description = _clean(video_description, runtime_players, runtime_aliases)
    transcript = _clean(clip_transcript, runtime_players, runtime_aliases)
    ocr = ocr_entities or {}
    match_facts = [
        _clean(fact, runtime_players, runtime_aliases)
        for fact in research.get("match_facts", [])
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
        _clean(ocr_text, runtime_players, runtime_aliases),
        " ".join(match_facts),
    ))
    approved = build_grounded_search_queries(
        title,
        description,
        transcript,
        research.get("search_queries") or [],
        runtime_players,
        runtime_aliases,
    )
    return {
        "source_video": {"title": title, "description": description},
        "clip_transcript": transcript,
        "ocr": ocr,
        "grounded_entities": find_canonical_entities(grounding_text, runtime_players),
        "player_names": runtime_players,
        "player_aliases": runtime_aliases,
        "match_facts": match_facts,
        "current_topics": [str(topic) for topic in research.get("topics", []) if str(topic).strip()],
        "approved_search_queries": approved,
        "sources": [source for source in research.get("sources", []) if isinstance(source, dict)],
    }
