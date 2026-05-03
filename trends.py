"""
trends.py — Version 4: Real-time Trend Analysis for Cricket.
"""
import random
from typing import List

# In a production environment, we could use 'pytrends' or 'requests' to scrape
# For now, we use a curated, dynamic list of trending Cricket hooks and search terms.
# This makes the content look human and high-energy.

VIRAL_HOOKS = [
    "UNBELIEVABLE! 😱", "Did you see that?! 🔥", "Pure Class. 🙌", 
    "Absolute Destruction! ⚡", "The GOAT strikes again! 🐐",
    "MAGIC Moment! ✨", "Wait for the end... 😲", "Cricket at its best! 🏏",
    "Insane Skills! 💎", "Total Dominance. 💪"
]

TRENDING_HASHTAGS = [
    "#Cricket", "#CricketShorts", "#IPL", "#T20WorldCup",
    "#CricketFever", "#CricketLovers", "#ViralShorts", "#Sports",
    "#CricketHighlights", "#CricketMoments", "#Trending", "#MustWatch",
]

def get_trending_context():
    """
    Returns a random viral hook and a set of trending hashtags.
    """
    return {
        "hook": random.choice(VIRAL_HOOKS),
        "tags": random.sample(TRENDING_HASHTAGS, 3)
    }

def humanize_title(keywords: List[str]):
    """
    Combines keywords into a high-CTR, human-looking title.
    """
    hook = random.choice(VIRAL_HOOKS)
    if not keywords:
        return f"{hook} Cricket Highlights #Shorts"
    
    main_topic = " ".join([k.capitalize() for k in keywords[:2]])
    
    templates = [
        f"{hook} {main_topic} was INSANE!",
        f"{main_topic}: You won't believe this! {hook}",
        f"This is why we love Cricket! {main_topic} {hook}",
        f"{main_topic} - Absolute Masterclass! 🔥",
    ]
    return random.choice(templates)
