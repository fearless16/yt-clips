"""
cricket_context.py — Cricket player names, team names, venues, and corrections.
"""
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set

# Spelling corrections for common Whisper audio transcript errors
CRICKET_SPELLING_CORRECTIONS = {
    # Players
    "yuvi": "Yuvraj Singh",
    "yuvraj": "Yuvraj Singh",
    "coaly": "Kohli",
    "koli": "Kohli",
    "virat koli": "Virat Kohli",
    "bumra": "Bumrah",
    "bumrah": "Jasprit Bumrah",
    "doni": "Dhoni",
    "dhoni": "MS Dhoni",
    "dhony": "MS Dhoni",
    "stark": "Starc",
    "mitchell stark": "Mitchell Starc",
    "shami": "Mohammed Shami",
    "sky": "Suryakumar Yadav",
    "surya kumar": "Suryakumar Yadav",
    "hardic": "Hardik",
    "hardik pandiya": "Hardik Pandya",
    "pandiya": "Pandya",
    "babar azam": "Babar Azam",
    "babar": "Babar Azam",
    "rizwan": "Mohammad Rizwan",
    "shaheen": "Shaheen Afridi",
    "rohit": "Rohit Sharma",
    "hitman": "Rohit Sharma",
    "gill": "Shubman Gill",
    "jaiswal": "Yashasvi Jaiswal",
    "pant": "Rishabh Pant",
    "kl rahul": "KL Rahul",
    "rahul": "KL Rahul",
    "iyer": "Shreyas Iyer",
    "rinku": "Rinku Singh",
    "axar": "Axar Patel",
    "siraj": "Mohammed Siraj",
    "kuldeep": "Kuldeep Yadav",
    "chahal": "Yuzvendra Chahal",
    "ashwin": "Ravichandran Ashwin",
    "jadeja": "Ravindra Jadeja",
    "rutherford": "Sherfane Rutherford",
    "narine": "Sunil Narine",
    "russell": "Andre Russell",
    "stoinis": "Marcus Stoinis",
    "pooran": "Nicholas Pooran",
    "de kock": "Quinton de Kock",
    "klassen": "Heinrich Klaasen",
    "cummins": "Pat Cummins",
    "abhishek": "Abhishek Sharma",
    "samson": "Sanju Samson",
    "parag": "Riyan Parag",
    "boult": "Trent Boult",
    "chahal": "Yuzvendra Chahal",
    
    # Teams
    "rcb": "Royal Challengers Bengaluru",
    "csk": "Chennai Super Kings",
    "mi": "Mumbai Indians",
    "kkr": "Kolkata Knight Riders",
    "srh": "Sunrisers Hyderabad",
    "rr": "Rajasthan Royals",
    "dc": "Delhi Capitals",
    "lsg": "Lucknow Super Giants",
    "gt": "Gujarat Titans",
    "pbks": "Punjab Kings",
    "india": "India",
    "pakistan": "Pakistan",
    "australia": "Australia",
    "england": "England",
    "south africa": "South Africa",
    "new zealand": "New Zealand",
    "ireland": "Ireland",
    "पाकिस्तान": "Pakistan",
    "इंग्लैंड": "England",
    "ऑस्ट्रेलिया": "Australia",
    "साउथ अफ्रीका": "South Africa",
    "दक्षिण अफ्रीका": "South Africa",
    "इंडिया": "India",
    "भारत": "India",
    "न्यूजीलैंड": "New Zealand",
    "आयरलैंड": "Ireland",
    "श्रीलंका": "Sri Lanka",
    "श्री लंका": "Sri Lanka",
    
    # Venues
    "wankhede": "Wankhede Stadium, Mumbai",
    "eden gardens": "Eden Gardens, Kolkata",
    "chinnaswamy": "M. Chinnaswamy Stadium, Bengaluru",
    "chepauk": "M. A. Chidambaram Stadium, Chennai",
    "dharamsala": "HPCA Stadium, Dharamshala",
    "narendra modi": "Narendra Modi Stadium, Ahmedabad",
    
    # Tournaments
    # Never inject a season year: source metadata/match evidence owns the date.
    "ipl": "IPL",
    "t20": "T20",
    "odi": "ODI",
    "wct20": "T20 World Cup",
}

# Canonical player names for SEO tag enrichment
CRICKET_PLAYERS: Set[str] = {
    "Yuvraj Singh",
    "Virat Kohli", "Rohit Sharma", "Jasprit Bumrah", "MS Dhoni", "Hardik Pandya",
    "Suryakumar Yadav", "Rishabh Pant", "Shubman Gill", "Yashasvi Jaiswal",
    "Ravindra Jadeja", "KL Rahul", "Shreyas Iyer", "Rinku Singh", "Axar Patel",
    "Mohammed Shami", "Mohammed Siraj", "Kuldeep Yadav", "Yuzvendra Chahal",
    "Ravichandran Ashwin", "Sanju Samson", "Abhishek Sharma", "Riyan Parag",
    "Ruturaj Gaikwad", "Shivam Dube", "Arshdeep Singh", "Harshal Patel",
    "Babar Azam", "Mohammad Rizwan", "Shaheen Afridi", "Naseem Shah",
    "Haris Rauf", "Shadab Khan", "Fakhar Zaman", "Iftikhar Ahmed",
    "Mitchell Starc", "Pat Cummins", "Travis Head", "Glenn Maxwell",
    "Marcus Stoinis", "Mitchell Marsh", "Adam Zampa", "Josh Hazlewood",
    "Sunil Narine", "Andre Russell", "Nicholas Pooran", "Quinton de Kock",
    "Heinrich Klaasen", "Trent Boult", "Jos Buttler", "Phil Salt",
    "Sherfane Rutherford", "Rashid Khan", "Kane Williamson", "Daryl Mitchell"
}

# Canonical team names
CRICKET_TEAMS: Set[str] = {
    "Chennai Super Kings", "Royal Challengers Bengaluru", "Mumbai Indians",
    "Kolkata Knight Riders", "Sunrisers Hyderabad", "Rajasthan Royals",
    "Delhi Capitals", "Lucknow Super Giants", "Gujarat Titans", "Punjab Kings",
    "India", "Pakistan", "Australia", "England", "South Africa", "New Zealand",
    "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan", "Ireland",
}

def _runtime_player_corrections(player_names: Iterable[str]) -> Dict[str, str]:
    """Build conservative aliases from the verified players for this video.

    Runtime match/source evidence is the catalog; the global list is only a
    seed.  First/last names are expanded only when unique inside that runtime
    catalog. Short all-caps tokens (``UV``, ``AI``) are deliberately excluded.
    """
    names = list(dict.fromkeys(
        re.sub(r"\s+", " ", str(name or "")).strip()
        for name in player_names
        if str(name or "").strip()
    ))
    alias_targets: Dict[str, Set[str]] = defaultdict(set)
    for name in names:
        parts = re.findall(r"[A-Za-z][A-Za-z.'-]*", name)
        alias_targets[name.casefold()].add(name)
        if len(parts) >= 2:
            for token in (parts[0], parts[-1]):
                if len(token) >= 4:
                    alias_targets[token.casefold()].add(name)
    return {
        alias: next(iter(targets))
        for alias, targets in alias_targets.items()
        if len(targets) == 1
    }


def correct_cricket_spelling(
    text: str,
    player_names: Optional[Iterable[str]] = None,
    player_aliases: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve aliases against static seeds plus the verified runtime roster."""
    if not text:
        return text
    corrections = dict(CRICKET_SPELLING_CORRECTIONS)
    runtime_names = list(player_names or [])
    runtime = _runtime_player_corrections(runtime_names)

    # Runtime ambiguity beats a static guess. For example, if two verified
    # players are named Rahul, plain "Rahul" must remain unresolved.
    runtime_alias_targets: Dict[str, Set[str]] = defaultdict(set)
    for name in runtime_names:
        parts = re.findall(r"[A-Za-z][A-Za-z.'-]*", str(name))
        if len(parts) >= 2:
            for token in (parts[0], parts[-1]):
                if len(token) >= 4:
                    runtime_alias_targets[token.casefold()].add(str(name).strip())
    for alias, targets in runtime_alias_targets.items():
        if len(targets) > 1:
            corrections.pop(alias, None)
    corrections.update(runtime)
    contextual_aliases = {
        "Tim Southee": ("सऊदी", "साउदी", "टिम साउदी"),
        "Tom Latham": ("टॉम लेदम", "टॉम लैथम", "लेदम"),
    }
    runtime_keys = {str(name).casefold() for name in runtime_names}
    for canonical, aliases_for_player in contextual_aliases.items():
        if canonical.casefold() in runtime_keys:
            corrections.update({alias: canonical for alias in aliases_for_player})
    corrections.update({
        str(alias).casefold(): str(player).strip()
        for alias, player in (player_aliases or {}).items()
        if len(str(alias).strip()) >= 4 and str(player).strip()
    })
    # Identity entries make the longest canonical phrase win before a short
    # alias inside it ("Yuvraj Singh" must not become "Yuvraj Singh Singh").
    for canonical in CRICKET_SPELLING_CORRECTIONS.values():
        corrections.setdefault(canonical.lower(), canonical)
    aliases = sorted(corrections, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z0-9_\u0900-\u097f])(?:"
        + "|".join(re.escape(alias) for alias in aliases)
        + r")(?![A-Za-z0-9_\u0900-\u097f])",
        re.IGNORECASE,
    )
    return pattern.sub(
        lambda match: corrections[match.group(0).lower()],
        text,
    )


_STRONG_CRICKET_TERMS = {
    "cricket", "ipl", "bbl", "psl", "t20", "odi", "test match",
    "wicket", "bowled", "lbw", "stumped", "batsman", "batter", "bowler",
    "yorker", "googly", "doosra", "powerplay", "run rate", "super over",
    "century", "half-century", "hattrick", "innings", "crease", "over",
    "six", "four", "chauka", "chhakka", "sixer", "boundary",
}

# YouTube's Hindi captions are usually Devanagari even when the streamer mixes
# Hindi and English.  Keeping these domain words separate avoids weakening the
# Latin-word boundary checks above while allowing the cricket-only gate to see
# the actual commentary.
_STRONG_HINDI_CRICKET_TERMS = {
    "क्रिकेट", "विकेट", "रन", "गेंद", "बॉल", "बल्लेबाज", "गेंदबाज",
    "ओवर", "पारी", "छक्का", "चौका", "बाउंड्री", "एलबीडब्ल्यू", "कैच",
    "रन रेट", "टेस्ट मैच", "टी20", "वनडे",
}

_NON_PLAYER_NAME_PHRASES = {
    "team india", "india coach", "world cup", "test match", "super kings",
    "royal challengers", "rajasthan royals", "mumbai indians",
    "sunrisers hyderabad", "delhi capitals", "punjab kings",
    "gujarat titans", "latest news", "match highlights", "cricket shorts",
}


def discover_grounded_player_names(
    source_text: str,
    current_search_texts: Iterable[str],
) -> List[str]:
    """Discover non-static players confirmed by current query-specific results.

    A name is admitted when it appears in the source plus one current result,
    or independently in at least two current results. This lets aliases such as
    ``Cheeku`` resolve through live search evidence without trusting one title.
    """
    source_low = str(source_text or "").casefold()
    counts: Dict[str, int] = defaultdict(int)
    display: Dict[str, str] = {}
    generic_tokens = {
        "live", "match", "test", "day", "score", "cricket", "today",
        "highlights", "commentary", "stream", "analysis", "discussion",
        "kya", "ka", "ki", "ke", "mein", "se", "aur", "hai", "raha",
        "runs", "run", "rate", "ahead", "short", "shorts", "boundary",
        "boundaries", "tez", "wahi", "phir", "har", "baar", "purana",
        "khilaf", "sahi", "wali", "fan", "reaction", "explained",
        "stadium", "pitch", "secret", "chhupa",
        "liye", "mushkil", "pak", "eng", "bowling", "batting", "wicket",
        "captaincy", "captain", "coach", "four", "six", "lead", "pressure",
        "world", "cup", "start", "bada", "mod", "door", "sach",
        "domination", "khatam", "sawal",
    }
    team_names = {team.casefold() for team in CRICKET_TEAMS}
    pattern = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][A-Za-z.'-]{2,}\b")
    for text in current_search_texts:
        in_result = set()
        for match in pattern.findall(str(text or "")):
            clean = re.sub(r"\s+", " ", match).strip(" .'\"")
            key = clean.casefold()
            tokens = set(re.findall(r"[a-z]+", key))
            if (
                key in _NON_PLAYER_NAME_PHRASES
                or key in team_names
                or tokens & generic_tokens
                or key in in_result
            ):
                continue
            in_result.add(key)
            display.setdefault(key, clean)
            counts[key] += 1
    grounded = [
        display[key]
        for key, count in counts.items()
        if count >= 2 or (count >= 1 and key in source_low)
    ]
    return sorted(grounded, key=lambda name: (-counts[name.casefold()], name))


def discover_grounded_player_aliases(
    source_text: str,
    current_search_texts: Iterable[str],
    player_names: Iterable[str],
) -> Dict[str, str]:
    """Link source nicknames only when current results co-mention a player."""
    stop = {
        "india", "cricket", "coach", "mentor", "match", "team", "player",
        "banao", "bana", "highlights", "shorts", "latest", "today",
    }
    source_tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]+", source_text or "")
        if len(token) >= 4 and token.casefold() not in stop
    }
    alias_targets: Dict[str, Set[str]] = defaultdict(set)
    result_lows = [str(text or "").casefold() for text in current_search_texts]
    for player in player_names:
        player_key = str(player).casefold()
        player_tokens = set(re.findall(r"[a-z]+", player_key))
        for alias in source_tokens - player_tokens:
            if any(
                player_key in result
                and re.search(r"\b" + re.escape(alias) + r"\b", result)
                for result in result_lows
            ):
                alias_targets[alias].add(str(player))
    return {
        alias: next(iter(targets))
        for alias, targets in alias_targets.items()
        if len(targets) == 1
    }
_WEAK_CRICKET_TERMS = {"shot", "coach", "captain", "team", "target", "chase"}


def _contains_hindi_term(text: str, term: str) -> bool:
    """Match a Hindi cricket term across scripts.

    Commentary transcripts may arrive as Devanagari (raw Whisper) or as
    Roman-Hinglish (post-transliteration). Match the Devanagari term as a
    token in Devanagari text, and its transliterated Roman form in Roman
    text. Using the same ``to_roman`` scheme on both sides keeps the forms
    consistent by construction.
    """
    if re.search(
        r"(?<![\u0900-\u097f])" + re.escape(term) + r"(?![\u0900-\u097f])",
        text,
    ):
        return True
    from utils.devanagari import to_roman

    term_roman = to_roman(term).casefold()
    if not term_roman or term_roman == str(term).casefold():
        return False
    text_roman = to_roman(text).casefold()
    return bool(re.search(r"\b" + re.escape(term_roman) + r"\b", text_roman))


def _cricket_relevance_score(text: str) -> int:
    """Return a conservative cricket relevance score for one text fragment."""
    if not text:
        return 0
    corrected = correct_cricket_spelling(text)
    low = corrected.lower()
    entities = find_canonical_entities(corrected)
    score = 3 if entities["players"] else 0
    score += 2 if entities["teams"] and any(
        term in low for term in _STRONG_CRICKET_TERMS
    ) else 0
    score += min(3, sum(1 for term in _STRONG_CRICKET_TERMS if re.search(
        r"\b" + re.escape(term) + r"\b", low
    )))
    score += min(3, sum(
        1 for term in _STRONG_HINDI_CRICKET_TERMS
        if _contains_hindi_term(low, term)
    ))
    score += min(1, sum(1 for term in _WEAK_CRICKET_TERMS if re.search(
        r"\b" + re.escape(term) + r"\b", low
    )))
    if re.search(r"\b\d{1,3}/\d{1,2}\b", low):
        score += 2
    return score


def is_cricket_content(text: str, source_context: str = "") -> bool:
    """Hard cricket-only gate with source context for ambiguous clip phrases."""
    clip_score = _cricket_relevance_score(text)
    source_score = _cricket_relevance_score(source_context)
    return clip_score >= 2 or (clip_score >= 1 and source_score >= 2)


def find_canonical_entities(
    text: str,
    player_names: Optional[Iterable[str]] = None,
) -> Dict[str, List[str]]:
    """Find canonical players/teams mentioned in *text* (post-correction).

    Wires the canonical name sets into SEO enrichment so tags/hashtags can be
    grounded in verified cricket entities rather than raw transcript tokens.

    Returns a dict ``{"players": [...], "teams": [...]}`` of canonical names
    found (case-insensitive substring match on full or last name).
    """
    if not text:
        return {"players": [], "teams": []}
    low = text.lower()
    player_catalog = set(CRICKET_PLAYERS)
    player_catalog.update(
        re.sub(r"\s+", " ", str(name or "")).strip()
        for name in (player_names or [])
        if str(name or "").strip()
    )
    surname_counts: Dict[str, int] = {}
    for player_name in player_catalog:
        surname = player_name.split()[-1].lower()
        surname_counts[surname] = surname_counts.get(surname, 0) + 1
    players = []
    for name in sorted(player_catalog, key=len, reverse=True):
        # A surname is safe only when it identifies exactly one known player.
        # "Singh", "Sharma" and "Yadav" must never fan out into fake entities.
        last = name.split()[-1].lower()
        unique_last_name = surname_counts.get(last) == 1
        if name.lower() in low or (
            unique_last_name
            and len(last) > 3
            and re.search(r"\b" + re.escape(last) + r"\b", low)
        ):
            players.append(name)
    teams = []
    for name in sorted(CRICKET_TEAMS, key=len, reverse=True):
        if name.lower() in low:
            teams.append(name)
    # De-dup while preserving order.
    return {
        "players": list(dict.fromkeys(players)),
        "teams": list(dict.fromkeys(teams)),
    }
