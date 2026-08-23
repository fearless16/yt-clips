"""Content-angle classifier for cricket Shorts candidates.

The channel's own outcome data shows WHY a clip is watched matters more than
which metadata wraps it: moments with stakes and story beats outperform
debate/banter chatter. The angle is stamped on every candidate so the LLM
arbiter can weigh it, and so shorts_intelligence learns which angles win.
"""

import re
from functools import lru_cache

Moment = "moment"
Comedy = "comedy"
Debate = "debate"
News = "news"

_MOMENT_RE = re.compile(
    r"\b(six|sixer|four|boundary|wicket|wickets|out|caught|catch|bowled|"
    r"lbw|stumps|yorker|bouncer|run\s?out|runout|hundred|century|\bfifty\b|"
    r"\bhalf century\b|hat\s?tri?ck|appeal|review|drs|maiden|over boundary|"
    r"\bno ball\b|free hit|maiden over|chakka|chauka|shikar)\b",
    re.I,
)
_COMEDY_RE = re.compile(
    r"\b(funny|comedy|joke|laughing|laugh|hilarious|meme|banter|prank|"
    r"troll|mazaak|hasa|hasi|mastii?|masti)\b",
    re.I,
)
_DEBATE_RE = re.compile(
    r"\b(i think|i feel|in my opinion|opinion|debate|argument|should be|"
    r"should he|should they|drop him|drop him from|greatest of all time|"
    r"\bgoat\b|overrated|underrated|better than|versus the best|agree or "
    r"not|hot take|controversial)\b",
    re.I,
)
_NEWS_RE = re.compile(
    r"\b(returns?|returning|comeback|announced|announcement|squad|selected|"
    r"selection|dropped from squad|ruled out|injur(?:y|ed)|signed|rested|"
    r"captain named|vice captain)\b",
    re.I,
)


@lru_cache(maxsize=4096)
def classify_content_type(text: str) -> str:
    """Label a clip transcript as moment | comedy | debate | news.

    Precedence: comedy beats debate when both appear (banter with opinions is
    still entertainment); a hard game event always wins as ``moment`` because
    stakes on screen outrank talk about them. Falls back to ``debate``
    (talk-heavy chatter) only when nothing stronger matches.
    """
    lowered = str(text or "")
    if _MOMENT_RE.search(lowered):
        return Moment
    if _COMEDY_RE.search(lowered):
        return Comedy
    if _NEWS_RE.search(lowered):
        return News
    if _DEBATE_RE.search(lowered):
        return Debate
    return Debate
