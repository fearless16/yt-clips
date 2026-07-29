"""CricketHeuristics — zero-LLM cricket-specific content scoring.

Pre-filters topics by:
  - Entity density (player/team name frequency)
  - Event keyword density (six, wicket, boundary, etc.)
  - Sentiment intensity (reaction word + punctuation analysis)
  - Commentary intensity markers

Returns per-topic scores that SemanticContentExpert uses.
"""

import re
from typing import Any

from utils.logger import get_logger

from automation.clip_selection.keywords import CRICKET_PLAYERS, CRICKET_EVENTS, EMOTION_WORDS, CRICKET_TEAMS, CRICKET_PHRASES

log = get_logger("cricket_heuristics")


def _words(text: str) -> set[str]:
    return set(re.findall(r'\b\w+\b', text.lower()))


def _phrase_hits(text_lower: str, phrases: list[str]) -> int:
    return sum(1 for p in phrases if p in text_lower)


# ── Scoring functions ───────────────────────────────────────────────

def entity_density(text: str) -> dict[str, Any]:
    """Score entity density (player + team mentions per word)."""
    word_set = _words(text)
    player_hits = len(word_set & CRICKET_PLAYERS)
    team_hits = sum(1 for t in CRICKET_TEAMS if t in text.lower())
    total_words = len(text.split()) or 1
    density = (player_hits + team_hits) / total_words
    return {
        "player_hits": player_hits,
        "team_hits": team_hits,
        "density": round(density, 4),
        "score": min(100, (player_hits * 12 + team_hits * 8) / max(total_words / 10, 1)),
    }


def event_intensity(text: str) -> dict[str, Any]:
    """Score cricket event keyword density."""
    word_set = _words(text)
    event_hits = len(word_set & CRICKET_EVENTS)
    phrase_hits = _phrase_hits(text.lower(), CRICKET_PHRASES)
    total = event_hits + phrase_hits
    return {
        "event_hits": event_hits,
        "phrase_hits": phrase_hits,
        "total_events": total,
        "score": min(100, total * 15),
    }


def sentiment_intensity(text: str) -> dict[str, Any]:
    """Score emotional intensity from reaction words + punctuation."""
    word_set = _words(text)
    emotion_hits = len(word_set & EMOTION_WORDS)
    excl_count = text.count("!")
    ques_count = text.count("?")
    intensity = emotion_hits * 8 + excl_count * 5 + ques_count * 3
    return {
        "emotion_hits": emotion_hits,
        "exclamations": excl_count,
        "questions": ques_count,
        "intensity": intensity,
        "score": min(100, intensity),
    }


def topic_salience_score(text: str) -> dict[str, Any]:
    """Combined salience score for a topic (no LLM)."""
    ed = entity_density(text)
    ei = event_intensity(text)
    si = sentiment_intensity(text)

    raw_score = ed["score"] * 0.30 + ei["score"] * 0.40 + si["score"] * 0.30
    score = min(100, raw_score)
    return {
        "score": round(score, 2),
        "entity_density": ed,
        "event_intensity": ei,
        "sentiment_intensity": si,
    }


def score_topic(topic: dict[str, Any]) -> dict[str, Any]:
    """Score a single topic dict (output of TopicSegmenter)."""
    text = topic.get("text", "")
    duration = topic.get("end", 0) - topic.get("start", 0)
    salience = topic_salience_score(text)

    result = {
        "topic_id": topic.get("topic_id", 0),
        "salience_score": salience["score"],
        "entity_density": salience["entity_density"],
        "event_intensity": salience["event_intensity"],
        "sentiment_intensity": salience["sentiment_intensity"],
        "duration": round(duration, 1),
    }
    return result


def score_all_topics(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score all topics from TopicSegmenter."""
    return [score_topic(t) for t in topics]
