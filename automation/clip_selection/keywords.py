"""Shared cricket keyword sets — single source of truth.

Imported by agents.py, cricket_heuristics.py, pipeline.py.
"""

CRICKET_PLAYERS = {
    "virat", "kohli", "rohit", "sharma", "dhoni", "msd", "sachin", "tendulkar",
    "bumrah", "jasprit", "hardik", "pandya", "rahul", "ishan",
    "kishan", "surya", "sky", "yadav", "gill", "shubman", "iyer", "shreyas",
    "jaddu", "jadeja", "ashwin", "ravi", "kuldeep", "chahal", "shami",
    "siraj", "arshdeep", "bhuvi", "bhuvneshwar", "umesh",
    "babar", "rizwan", "shaheen", "fakhar", "naseem", "shadab",
    "stokes", "buttler", "root", "bairstow", "woakes", "archer",
    "warner", "smith", "maxwell", "starc", "cummins", "hazlewood",
    "kane", "williamson", "conway", "phillips", "southee", "boult",
    "rabada", "miller", "nortje", "jansen", "markram",
    "rashid", "gurbaz", "mujeeb", "nabi", "zadran",
    "malinga", "mathews", "shakib", "mustafizur", "tamim", "mushfiqur",
}

CRICKET_EVENTS = {
    "six", "four", "wicket", "catch", "boundary", "century", "fifty",
    "hattrick", "stumping", "lbw", "bowled",
    "wide", "over", "inning", "strike", "maiden", "duck", "collapse",
    "chase", "target", "required", "win", "lost", "final", "semifinal",
    "playoff", "qualifier", "tie", "drs",
    "out", "stump", "castled", "delivery", "batsman", "bowler",
    "yorker", "bouncer", "googly", "doosra", "fulltoss", "slower",
    "reverse", "sweep", "cover", "drive", "pull", "hook", "cut",
    "flick", "glance", "timber", "knocked", "clean", "middle",
    "on", "off", "leg", "point", "slip", "gully", "third",
    "sixer", "fourr", "chhakka", "chauka",
}

CRICKET_TEAMS = {
    "india", "australia", "england", "south africa", "pakistan",
    "bangladesh", "afghanistan", "sri lanka", "zimbabwe",
    "ireland", "scotland", "netherlands", "west indies",
    "new zealand",
}

CRICKET_VENUES = {
    "wankhede", "lords", "mcg", "eden", "scg",
    "chinnaswamy", "chepauk", "kotla", "mohali",
}

COMMENTARY_WORDS = {
    "crowd", "eruption", "roar", "stadium", "ground",
    "atmosphere", "electric", "vibrant", "thunderous",
}

CRICKET_PHRASES = [
    "run out", "no ball", "super over", "free hit", "power play",
    "death over", "last over", "world cup", "ipl final",
    "test match", "one day", "t20 world",
    "takes catch", "direct hit", "run a ball",
    "player of the match", "man of the match",
]

EMOTION_WORDS = {
    "oh", "wow", "what", "no", "yes", "whoa", "insane", "crazy", "bro",
    "holy", "damn", "unbelievable", "incredible", "amazing", "brilliant",
    "superb", "fantastic", "massive", "clutch", "huge", "destroyed",
    "killed", "smashed", "demolished", "dominated", "thrashing",
    "kya", "arre", "bhai", "yaar", "baap", "pagal", "gajab",
    "khatarnak", "shandar", "dhamaakedaar", "zabardast", "bawaal",
    "oho", "accha", "haan", "nahi", "chhakka", "chauka", "sixer",
    "jeet", "machaa", "dekho", "khatam",
    "berserk", "wild", "pandemonium", "carnage", "riot",
    "stunning", "sensational", "magnificent", "extraordinary",
    "unreal", "speechless", "historic", "legendary", "iconic",
    "majestic", "devastating", "ruthless", "monstrous",
    "absolute", "beauty", "class", "vintage", "magic",
    "woww", "ohh", "ahh", "woah",
}

PAYOFF_WORDS = {
    "out", "gone", "taken", "bowled", "caught", "stumped",
    "six", "four", "boundary", "century", "fifty", "win", "victory",
    "final", "champion", "record", "history", "hattrick",
    "castled", "timber", "knocked", "clean",
}

PAYOFF_PHRASES = ["got him", "clean bowled", "middle stump", "goes for a walk"]

SERIES_WORDS = {
    "final", "semifinal", "playoff", "qualifier", "cup",
    "trophy", "championship", "ipl", "bbl",
    "psl", "test", "odi", "t20",
}

RARE_EVENT_TERMS = {
    "hattrick", "century", "record", "history",
    "unbelievable", "craziest", "biggest",
    "longest", "fastest", "slowest",
    "controversy", "fight", "argument", "angry", "confrontation",
    "comeback", "upset", "shock", "stunner",
}

RARE_EVENT_PHRASES = ["first time", "never seen", "massive six"]

CONTROVERSY_WORDS = {
    "controversy", "fight", "angry", "argument",
    "abuse", "sledging", "send off", "drama",
}

CROWD_WORDS = {"crowd", "audience", "stadium", "fans", "roar", "cheer"}

REACTION_PHRASES = [
    "kya baat", "oh ho", "are yaar", "kya shot", "maine kya",
    "haan haan", "arre arre", "are bhai", "kya hua", "yeh kya",
    "oh my god", "oh god", "what a", "kya cheez", "baap re",
    "nahi yaar", "haan bhai", "oho ho", "gajab ka", "chhakka maar",
    "dhamaakedaar shot", "what a shot", "what a six", "what a catch",
]
