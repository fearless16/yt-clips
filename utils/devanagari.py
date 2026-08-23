"""Devanagari -> Roman (Hinglish) transliteration.

Whisper transcribes Hindi commentary in Devanagari script, but every
downstream signal in this repo — emotion/cricket keyword lists, the
content-angle classifier, SEO grounding aliases — operates on Roman-Hinglish
text. Transliterate once at transcript load so the whole system sees the same
script the channel actually speaks.

Scheme: pragmatic huntergian-free Hinglish (schwa dropped word-finally,
aspiration preserved: छ->chh, ठ->th), matching how the audience types.
"""

import re

# Independent vowels
_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}

# Consonants (inherent 'a' handled by vowel-sign logic below)
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l", "क़": "q", "ख़": "kh",
    "ग़": "g", "ज़": "z", "फ़": "f", "ड़": "r", "ढ़": "rh",
}

# Vowel signs (matras)
_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ृ": "ri",
    "ॉ": "o", "ॅ": "e", "ॊ": "o",
    "ं": "n", "ँ": "n", "ः": "h", "़": "",
}

_NUKTA_ABBREVIATIONS = "\u093C"
_KEEP = {"्"}  # virama handled explicitly


def _is_consonant(ch: str) -> bool:
    return ch in _CONSONANTS


def _transliterate_word(word: str) -> str:
    out: list[str] = []
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        if ch == "्":  # virama: suppress inherent vowel
            i += 1
            continue
        if ch in _VOWELS:
            out.append(_VOWELS[ch])
            i += 1
            continue
        if _is_consonant(ch):
            base = _CONSONANTS[ch]
            nxt = word[i + 1] if i + 1 < n else ""
            # implicit 'a' unless virama or matra follows
            if nxt == "्":
                out.append(base)
                i += 2
                continue
            if nxt in _MATRAS:
                m = _MATRAS[nxt]
                # anusvara-style signs attach without inherent 'a'
                out.append(base + m)
                i += 2
                # trailing chandrabindu/visarga may follow matra
                while i < n and word[i] in ("ं", "ँ", "ः"):
                    out.append(_MATRAS[word[i]])
                    i += 1
                continue
            out.append(base + "a")
            i += 1
            continue
        if ch in _MATRAS:  # stray matra (word start) — rare
            out.append(_MATRAS[ch])
            i += 1
            continue
        out.append(ch)  # latin/digits/punctuation untouched
        i += 1
    return "".join(out)


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(str(text or "")))


def to_roman(text: str) -> str:
    """Transliterate every Devanagari token; leave Latin text untouched."""
    src = str(text or "")
    if not _DEVANAGARI_RE.search(src):
        return src

    def _sub(match: re.Match) -> str:
        return _transliterate_word(match.group(0))

    return _DEVANAGARI_RE.sub(_sub, src)
