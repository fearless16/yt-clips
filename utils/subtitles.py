import json
from pathlib import Path
from typing import List, Dict, Tuple

def convert_to_ass_time(seconds: float) -> str:
    """Convert float seconds to ASS time format (H:MM:SS.cs)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def generate_ass_subtitles(segments: List[Dict], output_ass_path: str, start_time: float, end_time: float) -> bool:
    """
    Generate an ASS subtitle file for a specific clip segment.
    Uses word-level timestamps to highlight the current spoken word.
    """
    if not segments:
        return False

    # Mobile-friendly styles
    # Font: Roboto/Arial, Size: large, MarginV: lifted to avoid bottom UI elements
    # Alignment: 2 (Bottom Center)
    # Highlight color: Yellow (&H00FFFF& in ASS BGR format)
    # Primary color: White
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,85,&H00FFFFFF,&H000000FF,&H00000000,&H99000000,-1,0,0,0,100,100,0,0,1,6,3,2,60,60,400,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    events = []
    
    for seg in segments:
        # Check if segment overlaps with our clip window
        if seg["end"] <= start_time or seg["start"] >= end_time:
            continue
            
        words = seg.get("words", [])
        if not words:
            # Fallback if no word-level timestamps
            seg_start = max(0, seg["start"] - start_time)
            seg_end = min(end_time - start_time, seg["end"] - start_time)
            
            s_ass = convert_to_ass_time(seg_start)
            e_ass = convert_to_ass_time(seg_end)
            text = seg.get("text", "").strip().replace("\n", "\\N")
            events.append(f"Dialogue: 0,{s_ass},{e_ass},Default,,0,0,0,,{text}")
            continue
            
        # Process word by word for Karaoke/Highlight effect
        # We will generate a distinct dialogue line for each word's duration
        # where that word is highlighted.
        
        # Filter words within the clip range
        clip_words = []
        for w in words:
            if w["end"] > start_time and w["start"] < end_time:
                clip_words.append(w)
                
        if not clip_words:
            continue
            
        # To make it look like a continuous sentence where the current word turns yellow
        sentence = [w["word"].strip() for w in clip_words]
        
        for i, current_word in enumerate(clip_words):
            w_start = max(0, current_word["start"] - start_time)
            w_end = min(end_time - start_time, current_word["end"] - start_time)
            
            if w_end <= w_start:
                continue
                
            s_ass = convert_to_ass_time(w_start)
            e_ass = convert_to_ass_time(w_end)
            
            # Format the sentence: before_words + HIGHLIGHTED_WORD + after_words
            formatted_words = []
            for j, word_text in enumerate(sentence):
                if j == i:
                    # Highlight color (Yellow: BGR -> Cyan-ish in ASS? No, Yellow is &H0000FFFF&)
                    # Let's use bright yellow: \c&H00FFFF& ... \c&HFFFFFF&
                    formatted_words.append(f"{{\\c&H00FFFF&}}{word_text}{{\\c&HFFFFFF&}}")
                else:
                    formatted_words.append(word_text)
                    
            text_line = " ".join(formatted_words)
            events.append(f"Dialogue: 0,{s_ass},{e_ass},Default,,0,0,0,,{text_line}")

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(events))
        f.write("\n")
        
    return True
