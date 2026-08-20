"""
cricket_context.py — Cricket player names, team names, venues, and corrections.
"""
import re
from typing import Dict, List, Set

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
    
    # Venues
    "wankhede": "Wankhede Stadium, Mumbai",
    "eden gardens": "Eden Gardens, Kolkata",
    "chinnaswamy": "M. Chinnaswamy Stadium, Bengaluru",
    "chepauk": "M. A. Chidambaram Stadium, Chennai",
    "dharamsala": "HPCA Stadium, Dharamshala",
    "narendra modi": "Narendra Modi Stadium, Ahmedabad",
    
    # Tournaments
    "ipl": "IPL 2026",
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
    "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan"
}

def correct_cricket_spelling(text: str) -> str:
    """Resolve cricket aliases/mishearings in one pass without recursive expansion."""
    if not text:
        return text
    corrections = dict(CRICKET_SPELLING_CORRECTIONS)
    # Identity entries make the longest canonical phrase win before a short
    # alias inside it ("Yuvraj Singh" must not become "Yuvraj Singh Singh").
    for canonical in CRICKET_SPELLING_CORRECTIONS.values():
        corrections.setdefault(canonical.lower(), canonical)
    aliases = sorted(corrections, key=len, reverse=True)
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(alias) for alias in aliases) + r")\b",
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
_WEAK_CRICKET_TERMS = {"shot", "coach", "captain", "team", "target", "chase"}


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


def find_canonical_entities(text: str) -> Dict[str, List[str]]:
    """Find canonical players/teams mentioned in *text* (post-correction).

    Wires the canonical name sets into SEO enrichment so tags/hashtags can be
    grounded in verified cricket entities rather than raw transcript tokens.

    Returns a dict ``{"players": [...], "teams": [...]}`` of canonical names
    found (case-insensitive substring match on full or last name).
    """
    if not text:
        return {"players": [], "teams": []}
    low = text.lower()
    surname_counts: Dict[str, int] = {}
    for player_name in CRICKET_PLAYERS:
        surname = player_name.split()[-1].lower()
        surname_counts[surname] = surname_counts.get(surname, 0) + 1
    players = []
    for name in sorted(CRICKET_PLAYERS, key=len, reverse=True):
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
