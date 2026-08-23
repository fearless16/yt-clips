"""InstaEvidencePack builder — REAL-ONLY policy enforcement point (PLAN.md).

Every caption token must trace to a verified evidence item. Sources:
  - Cricbuzz match facts via trends.fetch_verified_match_context
  - canonical roster via entity grounding catalogs + Cricbuzz player names
  - OUR shorts_intelligence.db outcomes (learner_top_captions)
  - YT-suggest strings + caller-approved search queries -> seed_phrases,
    each explicitly labeled {"seed": True} proxy corpus
  - REAL hashtag volume via GraphClient hashtag endpoints, only when a
    client is supplied AND validate_hashtags=True (post App-Review mode).

Every source failure degrades gracefully: the field becomes an empty list.
This module never raises on external-source failure.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from automation.seo.cricket_context import find_canonical_entities
from automation.seo.trends import (
    _research_query,
    fetch_verified_match_context,
    fetch_youtube_suggestions,
)

log = logging.getLogger("insta_evidence")

FORBIDDEN = [
    "invented player names",
    "generic tags without volume evidence",
]

MEGA_TAGS = frozenset({
    "reels", "viral", "explore", "shorts", "fyp", "trending",
})

MAX_SEED_PHRASES = 25
MAX_ROSTER = 20
MAX_HASHTAG_PROBES = 12
LEARNER_CAPTION_LIMIT = 8
_MAX_TAG_SLUG_LEN = 30


def _clean_phrase(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _tag_slug(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "", _clean_phrase(value)).lower()
    return slug[:_MAX_TAG_SLUG_LEN]


def _load_learner_top_captions(limit: int = LEARNER_CAPTION_LIMIT) -> list:
    """Top outcome titles from OUR shorts_intelligence.db; fail-soft."""
    try:
        from shorts_intelligence.config import RuntimeConfig
        from shorts_intelligence.store import ShortsStore

        config = RuntimeConfig.from_yaml()
        db_path = Path(config.db_path)
        if not db_path.exists():
            return []
        with ShortsStore(db_path, channel_id=config.channel_id) as store:
            rows = store.training_rows()
        rows.sort(
            key=lambda r: (
                float(r.get("outcome_score") or 0.0),
                str(r.get("captured_at") or ""),
            ),
            reverse=True,
        )
        out: list[str] = []
        seen = set()
        for row in rows:
            title = _clean_phrase(row.get("title"))
            key = title.casefold()
            if not title or key in seen:
                continue
            seen.add(key)
            out.append(title)
            if len(out) >= limit:
                break
        return out
    except Exception as exc:
        log.warning("learner captions unavailable: %s", exc)
        return []


def _fetch_match_facts(video_title: str, video_description: str,
                       transcript: str) -> tuple:
    """One live Cricbuzz probe -> (facts, player_names); fail-soft."""
    try:
        query = _research_query(video_title, video_description, transcript)
        context = fetch_verified_match_context(query) or {}
        facts = [f for f in (context.get("facts") or []) if _clean_phrase(f)]
        players = [p for p in (context.get("player_names") or [])
                   if _clean_phrase(p)]
        return facts, players
    except Exception as exc:
        log.warning("match facts unavailable: %s", exc)
        return [], []


def _build_roster(grounded_players, match_player_names,
                  canonical_players) -> list:
    out: list[str] = []
    seen = set()
    for name in [*(grounded_players or []), *(match_player_names or []),
                 *(canonical_players or [])]:
        clean = _clean_phrase(name)
        key = clean.casefold()
        if not clean or key in seen:
            continue
        seen.add(key)
        out.append(clean)
        if len(out) >= MAX_ROSTER:
            break
    return out


def _canonical_players(video_title: str, video_description: str,
                       transcript: str) -> list:
    try:
        combined = " ".join(filter(None, (
            video_title or "", video_description or "", transcript or "",
        )))
        return [p for p in find_canonical_entities(combined)["players"]
                if _clean_phrase(p)]
    except Exception as exc:
        log.warning("canonical roster unavailable: %s", exc)
        return []


def _collect_seed_phrases(approved_search_queries, video_title: str,
                          video_description: str, transcript: str) -> list:
    phrases: list[str] = []
    seen = set()

    def add(value) -> None:
        phrase = _clean_phrase(value)
        if not (1 < len(phrase) <= 80):
            return
        key = phrase.casefold()
        if key in seen:
            return
        seen.add(key)
        phrases.append(phrase)

    for query in approved_search_queries or []:
        add(query)
    try:
        seed_query = (_research_query(video_title, video_description,
                                      transcript) or "").strip() or "cricket"
        suggestions = fetch_youtube_suggestions(seed_query)
    except Exception as exc:
        log.warning("YT suggest unavailable: %s", exc)
        suggestions = []
    for suggestion in suggestions or []:
        if len(phrases) >= MAX_SEED_PHRASES:
            break
        add(suggestion)
    return [{"phrase": phrase, "seed": True} for phrase in phrases]


def _hashtag_candidates(teams, roster, seed_phrases) -> list:
    candidates: list[str] = []
    seen = set()
    for value in [*teams, *roster, *[s["phrase"] for s in seed_phrases]]:
        slug = _tag_slug(value)
        if not slug or slug in seen or slug in MEGA_TAGS:
            continue
        seen.add(slug)
        candidates.append(slug)
    return candidates


def _validate_candidate_tags(client, teams, roster, seed_phrases) -> list:
    validated = []
    probes = 0
    for slug in _hashtag_candidates(teams, roster, seed_phrases):
        if probes >= MAX_HASHTAG_PROBES:
            break
        probes += 1
        try:
            result = client.hashtag_search(slug) or {}
            items = [i for i in (result.get("data") or [])
                     if isinstance(i, dict)]
            hashtag_id = str(items[0].get("id") or "") if items else ""
            if not hashtag_id:
                continue
            velocity = int(client.hashtag_recent_velocity(hashtag_id))
            engagement = float(client.hashtag_top_engagement(hashtag_id))
        except Exception as exc:
            log.warning("hashtag probe %r failed: %s", slug, exc)
            continue
        if velocity <= 0:
            continue
        validated.append({
            "slug": slug,
            "volume": velocity,
            "engagement": engagement,
        })
    validated.sort(key=lambda e: (-e["volume"], -e["engagement"]))
    return [
        {
            "tag": f"#{entry['slug']}",
            "recent_volume_24h": entry["volume"],
            "source": "ig-hashtag-api",
        }
        for entry in validated
    ]


def build_insta_evidence_pack(
    video_title: str,
    video_description: str,
    transcript: str,
    *,
    match_facts=None,
    teams=None,
    grounded_players=None,
    approved_search_queries=None,
    client=None,
    validate_hashtags=False,
) -> dict:
    """Build the PLAN.md evidence pack; every failure degrades to [] ."""
    if match_facts is not None:
        if isinstance(match_facts, dict):
            facts = [f for f in (match_facts.get("facts") or [])
                     if _clean_phrase(f)]
            match_players = [p for p in (match_facts.get("player_names") or [])
                             if _clean_phrase(p)]
        else:
            facts = [_clean_phrase(f) for f in match_facts
                     if _clean_phrase(f)]
            match_players = []
    else:
        facts, match_players = _fetch_match_facts(video_title,
                                                  video_description,
                                                  transcript)

    provided_teams = [_clean_phrase(t) for t in teams or () if _clean_phrase(t)]
    try:
        canonical = _canonical_players(video_title, video_description,
                                       transcript)
    except Exception as exc:
        log.warning("canonical roster unavailable: %s", exc)
        canonical = []

    try:
        learner = _load_learner_top_captions()
    except Exception as exc:
        log.warning("learner captions unavailable: %s", exc)
        learner = []
    seeds = _collect_seed_phrases(approved_search_queries, video_title,
                                  video_description, transcript)

    validated = []
    if client is not None and validate_hashtags:
        try:
            validated = _validate_candidate_tags(
                client, provided_teams,
                _build_roster(grounded_players, match_players, canonical),
                seeds,
            )
        except Exception as exc:
            log.warning("hashtag validation unavailable: %s", exc)
            validated = []

    return {
        "match_facts": facts,
        "roster": _build_roster(grounded_players, match_players, canonical),
        "learner_top_captions": learner,
        "seed_phrases": seeds,
        "validated_hashtags": validated,
        "forbidden": list(FORBIDDEN),
    }


def validate_hashtag_pool(tags, pack) -> tuple:
    """Split tags into (clean_tags, rejected) against the evidence pack.

    A tag survives ONLY when it is present in pack["validated_hashtags"] OR
    traceable into a labeled seed phrase. Mega-tags are always rejected
    (M2 verdict 3), even if seeded or API-validated.
    """
    clean: list = []
    rejected: list = []
    seen_clean = set()
    seen_rejected = set()

    validated = {
        _tag_slug(entry.get("tag"))
        for entry in (pack.get("validated_hashtags") or [])
        if isinstance(entry, dict)
    }
    seeds = []
    for entry in pack.get("seed_phrases") or []:
        if isinstance(entry, dict):
            slug = _tag_slug(entry.get("phrase"))
            if slug:
                seeds.append(slug)

    for tag in tags or []:
        norm = _tag_slug(tag)
        if not norm:
            continue
        if norm in MEGA_TAGS:
            if norm not in seen_rejected:
                seen_rejected.add(norm)
                rejected.append(tag)
            continue
        survives = norm in validated or any(norm == s or norm in s
                                            for s in seeds)
        bucket = clean if survives else rejected
        seen_bucket = seen_clean if survives else seen_rejected
        if norm not in seen_bucket:
            seen_bucket.add(norm)
            bucket.append(tag)
    return clean, rejected
