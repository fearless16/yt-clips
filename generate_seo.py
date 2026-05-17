#!/usr/bin/env python3
"""Generate SEO metadata for all 10 clips using LLM API calls."""
import json, os, sys, time, re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
)
MODEL = "deepseek/deepseek-v4-flash"

SCORECARD = """
Match: Gujarat Titans vs Sunrisers Hyderabad, 56th Match, IPL 2026
Venue: Narendra Modi Stadium, Ahmedabad
Date: May 12, 2026
Toss: SRH won toss, elected to field first
Result: GT won by 82 runs

GT Innings: 168/5 (20 overs)
- Shubman Gill (c): 5 off 7
- Jos Buttler: 7 off 11
- Nishant Sindhu: 22 off 14 (3 fours, 1 six)
- Sai Sudharsan: 61 off 44 (5 fours, 2 sixes) - Anchor knock
- Washington Sundar: 50 off 33 (7 fours, 1 six) - Crucial late impetus
- Jason Holder: 11* off 10

SRH Innings: 86 all out (14.5 overs)
- Travis Head: 0 off 4 (Golden duck - dismissed by Siraj in first over)
- Abhishek Sharma: Bowled by Rabada after hitting a six
- Ishan Kishan: 11 off 7
- Smaran Ravichandran: 9
- Pat Cummins: 19 off 9 (1 four, 2 sixes) - Brief resistance
- Praful Hinge: Stumped by Buttler off Rashid Khan

Key Bowling (GT):
- Kagiso Rabada: 3/28 (Fiery spell, dismissals: Abhishek, Ishan Kishan, Ravichandran)
- Mohammed Siraj: Key wicket of Travis Head
- Jason Holder: 3/20
- Prasidh Krishna: 2/23
- Rashid Khan: 1/3 (0.5 overs)

Key Facts:
- GT's 5th consecutive win
- GT rose to #1 on points table
- GT wore special lavender jerseys for cancer awareness
- Rabada leads Powerplay wicket chart (13 wickets this season)
- GT has won 6 of 7 completed games against SRH
"""

# Hindi transcripts for each clip
CLIPS = {
    "clip1": {
        "transcript": "Commentators joking about live stream, notifications, social media. Cricket live commentary fun moments.",
        "context": "Opening banter between commentators during live stream"
    },
    "clip2": {
        "transcript": "Ishan Kishan batting aggressively - back to back boundaries. Explosive power hitting from SRH's wicketkeeper-batter.",
        "context": "Ishan Kishan attacking batting during SRH chase"
    },
    "clip3": {
        "transcript": "Beautiful batting display - timing and placement. The batsman is in supreme form hitting gorgeous shots all around the ground.",
        "context": "Sai Sudharsan's elegant batting during GT innings"
    },
    "clip4": {
        "transcript": "Dramatic wicket falls! Indian cricket team moment - huge turning point in the match. Crowd goes wild.",
        "context": "Key wicket moment - likely Travis Head or Abhishek Sharma dismissal"
    },
    "clip5": {
        "transcript": "Intense running between wickets! Quick singles and doubles. Brilliant cricket awareness between the wickets.",
        "context": "Sundar/Sudharsan running hard between wickets during GT innings"
    },
    "clip6": {
        "transcript": "Classic cricket shots and power hitting. Reminiscent of legendary players. IPL nostalgia moment with amazing strokeplay.",
        "context": "Washington Sundar's crucial 50 off 33 in death overs"
    },
    "clip7": {
        "transcript": "Wicketkeeper catches and brilliant fielding! The match intensity reaches new level. Edges and catches flying.",
        "context": "Rabada's fiery spell with wickets tumbling - Ishan Kishan/Smaran Ravichandran dismissals"
    },
    "clip8": {
        "transcript": "Stadium erupts! Electric atmosphere. Fans going absolutely crazy. Pure IPL energy at Ahmedabad.",
        "context": "GT fans celebrating as SRH wickets tumble"
    },
    "clip9": {
        "transcript": "The art of cricket batting - beautiful strokeplay. Cricket shot masterclass on display.",
        "context": "Sai Sudharsan's 61 off 44 - masterclass innings"
    },
    "clip10": {
        "transcript": "Match analysis and commentary discussion. RCB connection mentioned. Test match vs IPL comparison.",
        "context": "Post-match analysis comparing batting styles, IPL playoff implications"
    }
}

SYSTEM_PROMPT = (
    "You are an elite YouTube Shorts SEO expert for Indian cricket. "
    "Your goal: Maximize CTR (Click-Through Rate) and watch time. "
    "Return ONLY valid JSON — no markdown, no explanation."
)

def make_prompt(clip_id, clip_data):
    return f"""MATCH SCORECARD:
{SCORECARD}

CLIP: {clip_id}
Hindi Commentary Transcript: {clip_data['transcript']}
Clip Context: {clip_data['context']}

══ STEP 1: TRANSCRIPTION CORRECTION ════════════════════════════════════════
Fix any misspelled cricket names using the Scorecard above.
Use CORRECTED names in ALL output fields below.

══ STEP 2: GENERATE SEO METADATA ═══════════════════════════════════════════

TITLE (max 100 chars, MAX 1 emoji):
  Pick ONE proven formula:
    1. SHOCK: "GT vs SRH: <unexpected moment> | IPL 2026"
    2. STAR: "IPL 2026: <star player> <action> <result>"
    3. CLOSE: "GT vs SRH: <close call> - did they?! | IPL 2026"
    4. NUMBERS: "IPL 2026: <score> runs in <overs> - game changer!"
  RULES:
    - MUST include "GT vs SRH" or "Gujarat Titans" or "Sunrisers Hyderabad"
    - Include "IPL 2026" and individual player name + action
    - NEVER generic like "Cricket Amazing!"

DESCRIPTION (follow EXACTLY this template structure):
🔴 GT vs SRH Live Match | Gujarat Titans vs Sunrisers Hyderabad Live | IPL 2026 Live Commentary

Welcome to the ultimate IPL 2026 live match experience!
Tonight's blockbuster clash features Gujarat Titans taking on Sunrisers Hyderabad at Narendra Modi Stadium, Ahmedabad.

🏏 Match Details
Match: GT vs SRH, Match 56, IPL 2026
Venue: Narendra Modi Stadium, Ahmedabad
Date: May 12, 2026
Result: GT won by 82 runs

🔥 LIVE MATCH UPDATE
Gujarat Titans posted 168/5 in 20 overs with crucial knocks from Sai Sudharsan (61 off 44) and Washington Sundar (50 off 33). Sunrisers Hyderabad were then bundled out for just 86 in 14.5 overs after a devastating Powerplay bowling display from Kagiso Rabada and Mohammed Siraj.

⚡ Key Highlights
• [5-6 specific highlights from THIS clip's moment]

🎙️ Live Hindi Commentary | Ball-by-Ball Updates | Live Reactions

#GTvsSRH #IPL2026 #GTLive #SRHLive #IPLMatchToday #LiveCricket #IPLlive #GT #SRH #CricketLive #LiveScore #IPLScore #IPL2026Live #HindiCommentary #CricketFans #TATAIPL2026

SEARCH TERMS (maximize the 500 char budget with 25-35 tags):
  Structure your tags in this priority order:
    Tier 1 — Player + action + tournament (highest search intent)
    Tier 2 — Team matchup phrases
    Tier 3 — Hindi search patterns (massive Indian search volume)
    Tier 4 — Moment-specific
    Tier 5 — Broad but relevant

  RULES:
    - Mix Hindi and English (Hinglish)
    - NO generic single words
    - Each tag must be a SEARCH PHRASE (2-5 words)
    - Aim for 25-35 tags to maximize the 500 char limit

Return ONLY valid JSON — no markdown, no explanation:
{{
  "clip_id": "{clip_id}",
  "title": "<corrected title>",
  "description": "<full structured description following the template>",
  "hashtags": ["#GTvsSRH", "#IPL2026", "#GTLive", "#...", "#Shorts"],
  "search_terms": ["<term1>", "<term2>", "..."],
  "tags": ["<tag1>", "<tag2>", "..."]
}}"""


def parse_json_response(text):
    """Extract JSON from LLM response."""
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    return None


def generate_clip_seo(clip_id, clip_data, retries=3):
    prompt = make_prompt(clip_id, clip_data)
    
    for attempt in range(retries):
        try:
            print(f"  [{clip_id}] Calling LLM (attempt {attempt+1})...")
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            text = resp.choices[0].message.content
            result = parse_json_response(text)
            if result and "title" in result:
                print(f"  [{clip_id}] SUCCESS - title: {result['title'][:60]}...")
                return result
            else:
                print(f"  [{clip_id}] No valid JSON, retrying...")
        except Exception as e:
            print(f"  [{clip_id}] Error: {e}")
        if attempt < retries - 1:
            wait = 5 * (2 ** attempt)
            print(f"  [{clip_id}] Waiting {wait}s...")
            time.sleep(wait)
    
    print(f"  [{clip_id}] ALL ATTEMPTS FAILED")
    return None


def main():
    out_dir = Path("/tmp/seo_llm_output")
    out_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    for clip_id, clip_data in CLIPS.items():
        print(f"\n{'='*50}")
        print(f"Generating SEO for {clip_id}...")
        print(f"{'='*50}")
        
        result = generate_clip_seo(clip_id, clip_data)
        if result:
            result["clip_id"] = clip_id
            all_results[clip_id] = result
            
            path = out_dir / f"{clip_id}_metadata.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  Saved to {path}")
        
        # Rate limit: 2s between calls
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"RESULTS: {len(all_results)}/{len(CLIPS)} clips generated")
    for clip_id, r in all_results.items():
        print(f"  {clip_id}: {r['title'][:70]}...")
    
    return all_results


if __name__ == "__main__":
    main()
