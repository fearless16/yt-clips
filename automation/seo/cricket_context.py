"""
cricket_context.py — Cricket player names, team names, venues, and corrections.
"""
import re
from typing import Dict, List, Set

# Spelling corrections for common Whisper audio transcript errors
CRICKET_SPELLING_CORRECTIONS = {
    # Players
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
    """Replace misheard/lowercase cricket names with canonical spelling."""
    import re
    corrected = text
    # Sort keys by length descending to replace longer phrases first (e.g. 'mitchell stark' before 'stark')
    for misheard in sorted(CRICKET_SPELLING_CORRECTIONS.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(misheard) + r"\b"
        replacement = CRICKET_SPELLING_CORRECTIONS[misheard]
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    return corrected


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
    players = []
    for name in sorted(CRICKET_PLAYERS, key=len, reverse=True):
        # Match the full canonical name or its last token (e.g. "Bumrah").
        last = name.split()[-1].lower()
        if name.lower() in low or (len(last) > 3 and re.search(r"\b" + re.escape(last) + r"\b", low)):
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
