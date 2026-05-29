"""
transcript_postproc.py — Centralized, cricket-aware transcript post-processing.

Used by BOTH the local Whisper engine (``transcribe.py``) and the remote
transcript fetcher (``automation/transcript.py``) so EVERY transcript source
(api / vtt / local_whisper) gets the same correction + validation. Previously
only the local Whisper path was corrected.

Design choices:
  * **Spelling-only, context-guarded lexicon.** We fix clear mishearings of
    cricket names (e.g. "koli" -> "Kohli") but deliberately do NOT expand to
    full names (that is an SEO concern, see ``automation/seo/cricket_context``)
    and do NOT touch tokens that are ordinary English words (no "sky"->"SKY",
    no bare "stark"->"Starc") to avoid corrupting commentary.
  * **LLM-correction validation.** Output of the optional LLM pass is validated
    per-segment (index must exist, length must be sane, not emptied) before it
    is applied — a misbehaving model can no longer silently rewrite/translate
    the transcript.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Spelling-only corrections. Keys are common Whisper mishearings; values are the
# canonical *surname/short* spelling (NOT the expanded full name). Entries are
# chosen to be unambiguous cricket terms that are NOT normal English words, so a
# blunt word-boundary replace is safe.
CRICKET_SPELLING: Dict[str, str] = {
    # Players (mishearings only — unambiguous)
    "coaly": "Kohli",
    "koli": "Kohli",
    "kohly": "Kohli",
    "bumra": "Bumrah",
    "bumrah": "Bumrah",
    "boomrah": "Bumrah",
    "doni": "Dhoni",
    "dhony": "Dhoni",
    "rohith": "Rohit",
    "hardic": "Hardik",
    "pandiya": "Pandya",
    "pandya": "Pandya",
    "jadeja": "Jadeja",
    "jaddu": "Jadeja",
    "ashwin": "Ashwin",
    "siraj": "Siraj",
    "shami": "Shami",
    "rizvan": "Rizwan",
    "rizwaan": "Rizwan",
    "babar": "Babar",
    "shaheen": "Shaheen",
    "gambhir": "Gambhir",
    "yashasvi": "Yashasvi",
    "jaiswal": "Jaiswal",
    "suryakumar": "Suryakumar",
    # Venues / terms
    "wankede": "Wankhede",
    "wankhede": "Wankhede",
    "chinaswamy": "Chinnaswamy",
    "chinnaswamy": "Chinnaswamy",
    "chepak": "Chepauk",
}

# Tokens we explicitly REFUSE to "correct" because they are ordinary English
# words that the old lexicon wrongly mapped (documented to prevent regressions).
_REFUSED_FALSE_POSITIVES = {"sky", "stark", "head", "root", "hope", "young", "salt"}

_COMPILED = [
    (re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE), v)
    for k, v in sorted(CRICKET_SPELLING.items(), key=lambda kv: -len(kv[0]))
    if k not in _REFUSED_FALSE_POSITIVES
]


def correct_text(text: str) -> Tuple[str, int]:
    """Apply the spelling-only cricket lexicon to *text*.

    Returns ``(corrected_text, num_substitutions)``. Case-insensitive match,
    canonical-cased replacement, word-boundary guarded.
    """
    if not text:
        return text, 0
    corrected = text
    n = 0
    for pattern, replacement in _COMPILED:
        corrected, count = pattern.subn(replacement, corrected)
        n += count
    return corrected, n


def correct_segments(segments: List[dict]) -> Tuple[List[dict], int]:
    """Apply :func:`correct_text` to each segment's ``text`` in place.

    Returns ``(segments, total_substitutions)``.
    """
    total = 0
    for seg in segments or []:
        new_text, n = correct_text(seg.get("text", ""))
        if n:
            seg["text"] = new_text
            total += n
    return segments, total


def validate_and_apply_llm_corrections(
    segments: List[dict],
    corrected_map: Dict[int, str],
    max_growth_ratio: float = 3.0,
) -> Tuple[List[dict], int, int]:
    """Validate per-index LLM corrections before applying them.

    A correction for index *i* is applied only if:
      * index *i* exists in *segments*,
      * the corrected text is non-empty after strip,
      * its length is not absurdly larger than the original (guards against the
        model appending commentary/translation): ``len <= original*ratio + 20``.

    Returns ``(segments, applied, rejected)``.
    """
    applied = 0
    rejected = 0
    for idx, new_text in (corrected_map or {}).items():
        if not isinstance(idx, int) or idx < 0 or idx >= len(segments):
            rejected += 1
            continue
        candidate = (new_text or "").strip()
        if not candidate:
            rejected += 1
            continue
        original = segments[idx].get("text", "") or ""
        if len(candidate) > len(original) * max_growth_ratio + 20:
            rejected += 1
            continue
        segments[idx]["text"] = candidate
        applied += 1
    return segments, applied, rejected
