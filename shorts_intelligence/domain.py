"""Conservative cricket-domain gate with no player-name roster."""

from __future__ import annotations

import re
from dataclasses import dataclass


_CRICKET_ANCHORS = (
    "cricket", "ipl", "wpl", "psl", "bbl", "t20", "odi", "test match",
    "test cricket", "wicket", "innings", "batsman", "batter", "bowler",
    "bowling", "batting", "scorecard", "lbw", "stumps", "powerplay",
    "run chase", "playing xi", "pitch report", "toss", "century",
    "half century", "world test championship",
    "zing bails", "team india", "chennai super kings", "delhi capitals", "csk",
)

_CRICKET_CONTEXT = (
    "runs", "overs", "six", "boundary", "captaincy", "coach", "series",
    "match", "team india", "australia", "england", "pakistan",
    "bangladesh", "west indies", "sri lanka", "new zealand",
)

_NON_CRICKET_ANCHORS = (
    "wwe", "wrestling", "brock lesnar", "goldberg", "gta", "grand theft auto",
    "fifa", "football", "soccer", "pokemon", "minecraft", "free fire",
    "pubg", "gameplay", "hell in a cell", "survivor series",
    "hockey", "travel", "vlog", "jio fiber", "jio airfiber", "broadband",
)


def _contains(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text))


@dataclass(frozen=True, slots=True)
class DomainDecision:
    """Domain label plus an auditable confidence and reason list."""

    status: str
    confidence: float
    reasons: tuple[str, ...]


class CricketDomainGate:
    """Label cricket conservatively without assuming a fixed player catalog."""

    def classify(
        self,
        title: str,
        description: str = "",
        *,
        trusted_cricket_origin: bool = False,
    ) -> DomainDecision:
        """Classify text as cricket, non-cricket, or unknown."""
        text = f"{title} {description}".casefold()
        positive = tuple(term for term in _CRICKET_ANCHORS if _contains(text, term))
        context = tuple(term for term in _CRICKET_CONTEXT if _contains(text, term))
        negative = tuple(term for term in _NON_CRICKET_ANCHORS if _contains(text, term))

        if negative and not positive:
            return DomainDecision("non_cricket", min(0.99, 0.85 + 0.03 * len(negative)), negative)
        if positive:
            confidence = min(0.99, 0.82 + 0.04 * len(positive) + 0.01 * len(context))
            return DomainDecision("cricket", confidence, positive + context[:3])
        if trusted_cricket_origin and not negative:
            return DomainDecision("cricket", 0.9, ("trusted_cricket_origin",))
        return DomainDecision("unknown", 0.0, ())
