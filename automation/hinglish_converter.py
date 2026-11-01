"""hinglish_converter.py — High-fidelity Hindi to Hinglish transliteration.

Converts Devanagari Hindi text (e.g., from YouTube auto-generated captions)
into natural, conversational, modern Roman-script Hinglish
(e.g., "Ye kya hai yaar", "Aur bhai kya scene", "Picchle match me jo nar-sanhaar hua tha").

Uses local high-speed LLM (via llama-swap at 127.0.0.1:8081/v1) with a fast,
robust rule-based phonetic fallback if LLM is unavailable.
"""

import json
import logging
import re
import urllib.request
import urllib.error
from typing import List, Dict, Optional

log = logging.getLogger("hinglish_converter")

# Common cricket & conversational Hindi terms mapped for clean, standard Hinglish
COMMON_TERMS = {
    "नरसंहार": "nar-sanhaar",
    "पलटवार": "palat-waar",
    "ताजातन": "taaza",
    "ताजा": "taaza",
    "एपिसोड": "episode",
    "क्रिकेट": "cricket",
    "मैच": "match",
    "न्यूजीलैंड": "New Zealand",
    "इंडिया": "India",
    "भारतीय": "Bhartiya",
    "खिलाड़ी": "khilaadi",
    "विकेट": "wicket",
    "गेंदबाज": "gendbaaz",
    "बल्लेबाज": "ballebaaz",
    "छक्का": "chhakka",
    "चौका": "chauka",
    "ओवर": "over",
    "रन": "run",
    "कप्तान": "captain",
    "इंतजार": "intezaar",
    "हाजिर": "haazir",
    "स्वस्थ": "swasth",
    "मस्त": "mast",
    "टेलीकास्ट": "telecast",
    "रिपीट": "repeat",
    "ब्रेक": "break",
    "दिन": "din",
    "घड़ी": "ghadi",
    "साल": "saal",
    "आज": "aaj",
    "कल": "kal",
    "भाई": "bhai",
    "यार": "yaar",
}

# Devanagari to Roman phonetic mapping table for offline fallback
_CONSONANTS = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
    'ष': 'sh', 'स': 's', 'ह': 'h', 'क़': 'q', 'ख़': 'kh',
    'ग़': 'gh', 'ज़': 'z', 'ड़': 'r', 'ढ़': 'rh', 'फ़': 'f'
}

_VOWELS = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u',
    'ऊ': 'oo', 'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o',
    'औ': 'au', 'अं': 'an', 'अः': 'ah'
}

_MATRAS = {
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    'ं': 'n', 'ँ': 'n', 'ः': 'h', '्': ''
}


def rule_based_transliterate(text: str) -> str:
    """Fast, deterministic Devanagari to Hinglish transliteration fallback."""
    if not text:
        return ""

    # Replace known vocabulary first
    words = text.split()
    converted_words = []
    for w in words:
        clean_w = re.sub(r'[^\w\s]', '', w)
        if clean_w in COMMON_TERMS:
            converted_words.append(COMMON_TERMS[clean_w])
            continue

        res = []
        i = 0
        n = len(w)
        while i < n:
            ch = w[i]
            # Check nukta combinations
            if i + 1 < n and w[i + 1] == '़':
                combo = ch + '़'
                if combo in _CONSONANTS:
                    ch = combo
                    i += 1

            if ch in _VOWELS:
                res.append(_VOWELS[ch])
            elif ch in _CONSONANTS:
                cons = _CONSONANTS[ch]
                # Look ahead for matra or halant
                if i + 1 < n and w[i + 1] in _MATRAS:
                    matra = _MATRAS[w[i + 1]]
                    res.append(cons + matra)
                    i += 1
                elif i + 1 < n and w[i + 1] == '्':
                    # Halant (half consonant)
                    res.append(cons)
                    i += 1
                else:
                    # Inherent 'a' if not at the end of word
                    if i + 1 == n:
                        res.append(cons)
                    else:
                        res.append(cons + 'a')
            elif ch in _MATRAS:
                res.append(_MATRAS[ch])
            else:
                res.append(ch)
            i += 1

        converted_words.append("".join(res))

    out = " ".join(converted_words)
    # Common post-cleanup
    out = re.sub(r'(\w)\1{2,}', r'\1\1', out)
    return out


def llm_to_hinglish_batch(
    texts: List[str],
    api_url: str = "http://127.0.0.1:8081/v1/chat/completions",
    model: str = "qwen3.8-27b",
    timeout: int = 15,
) -> Optional[List[str]]:
    """Batch convert Hindi sentences to natural, conversational Hinglish using local LLM."""
    if not texts:
        return []

    numbered = "\n".join(f"{i+1}. {t.strip()}" for i, t in enumerate(texts))
    prompt = (
        "You are an expert in natural, viral Hinglish subtitles for YouTube Shorts.\n"
        "Convert each of the following Hindi sentences (in Devanagari) into conversational, "
        "modern Hinglish (Roman/English alphabet, Hindi language, WhatsApp/Shorts style like 'Ye kya hai', 'Aur bhai kya scene').\n"
        "Guidelines:\n"
        "- Maintain exact meaning, high energy, and conversational cricket commentary tone.\n"
        "- Use standard cricket spellings: match, over, wicket, run, team, India, etc.\n"
        "- Return EXACTLY the same number of numbered lines. No conversational filler, no explanations.\n\n"
        f"Hindi input:\n{numbered}\n\n"
        "Hinglish output (numbered 1 to N):"
    )

    req_body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    try:
        req = urllib.request.Request(
            api_url,
            data=json.dumps(req_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()

        # Parse numbered list
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        results = []
        for line in lines:
            # Strip number prefix e.g. "1. " or "1) "
            clean = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if clean:
                results.append(clean)

        if len(results) == len(texts):
            return results
        log.warning("LLM returned %d lines for %d inputs, using fallback alignment", len(results), len(texts))
        if len(results) > 0:
            # Pad or slice if close
            while len(results) < len(texts):
                results.append(rule_based_transliterate(texts[len(results)]))
            return results[:len(texts)]
    except Exception as e:
        log.warning("LLM transliteration call failed: %s; falling back to rule-based", e)

    return None


def convert_segments_to_hinglish(
    segments: List[Dict],
    batch_size: int = 10,
    api_url: str = "http://127.0.0.1:8081/v1/chat/completions",
    model: str = "qwen3.8-27b",
) -> List[Dict]:
    """Convert segments with Hindi text into Hinglish segments.

    Adds 'hinglish' key to each segment while keeping original 'text', 'start', 'end'.
    """
    out_segments = []
    total = len(segments)

    for i in range(0, total, batch_size):
        chunk = segments[i:i + batch_size]
        texts = [s.get("text", "") for s in chunk]

        # Try LLM batch first
        hinglish_texts = llm_to_hinglish_batch(texts, api_url=api_url, model=model)
        if not hinglish_texts:
            # Rule based fallback
            hinglish_texts = [rule_based_transliterate(t) for t in texts]

        for s, h_text in zip(chunk, hinglish_texts):
            s_copy = dict(s)
            s_copy["hinglish"] = h_text
            out_segments.append(s_copy)

    return out_segments
