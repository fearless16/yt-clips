"""Feature extraction from raw clip data for scoring."""

import re
from typing import Any


_HOOK_WORDS = [
    "wait", "watch", "incredible", "amazing", "shocking",
    "you won't believe", "omg", "mind blown", "this is why", "the reason",
]

_PAYOFF_INDICATORS = [
    "finally", "there it is", "got it", "yes", "nailed it", "perfect",
]

_PROFANITY_SET: set[str] = {
    "shit", "damn", "fuck", "ass", "bitch", "bastard", "crap", "dick",
}


def _count_words_in_text(text: str, words: list[str]) -> int:
    count = 0
    text_lower = text.lower()
    for word in words:
        if re.search(rf'\b{re.escape(word)}\b', text_lower):
            count += 1
    return count


def _has_profanity(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return any(w in _PROFANITY_SET for w in words)


class FeatureExtractor:
    """Extracts structured features from raw clip data.

    Required clip_data fields: clip_id, duration_s, transcript, title.
    """

    def extract(self, clip_data: dict[str, Any]) -> dict[str, Any]:
        if "duration_s" not in clip_data:
            raise ValueError("missing required field: duration_s")
        if "transcript" not in clip_data:
            raise ValueError("missing required field: transcript")
        if "title" not in clip_data:
            raise ValueError("missing required field: title")

        transcript = clip_data.get("transcript", "")
        title = clip_data.get("title", "")
        duration_s = clip_data["duration_s"]
        combined = f"{transcript} {title}"

        hook_words_count = _count_words_in_text(combined, _HOOK_WORDS)
        payoff_indicators = _count_words_in_text(combined, _PAYOFF_INDICATORS)
        profanity_flag = _has_profanity(combined)

        if duration_s < 30:
            length_category = "short"
        elif duration_s <= 60:
            length_category = "medium"
        else:
            length_category = "long"

        return {
            "hook_words_count": hook_words_count,
            "payoff_indicators": payoff_indicators,
            "profanity_flag": profanity_flag,
            "duration_s": duration_s,
            "length_category": length_category,
        }
