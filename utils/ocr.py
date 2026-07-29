"""ocr.py — Keyframe-based OCR for cricket on-screen text extraction.

Extracts scoreboard overlays, player names, match context text
from video keyframes. Uses EasyOCR when available, otherwise
operates in degraded mode with clear markers.

Architecture:
  Video → keyframe sampling → EasyOCR → entity dedup → merge with transcript

Usage:
  from utils.ocr import extract_ocr_entities
  entities = extract_ocr_entities("path/to/video.mp4")
  # returns {"scoreboard": [...], "player_names": [...], "on_screen_text": [...]}
"""

import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("ocr", cfg.get("logging", {}).get("log_file", "logs/pipeline.log"),
                 cfg.get("logging", {}).get("level", "INFO"))

_OCR_AVAILABLE = False
_OCR_READER = None

try:
    import easyocr
    _OCR_AVAILABLE = True
except ImportError:
    pass

KNOWN_TEAMS = {
    "csk", "chennai", "mi", "mumbai", "rcb", "bangalore", "kkr", "kolkata",
    "srh", "hyderabad", "dc", "delhi", "pbks", "punjab", "rr", "rajasthan",
    "lsg", "lucknow", "gt", "gujarat", "india", "pakistan", "australia",
    "england", "new zealand", "south africa", "sri lanka", "west indies",
    "bangladesh", "afghanistan", "zimbabwe",
}
KNOWN_PLAYERS = {
    "kohli", "virat", "rohit", "sharma", "bumrah", "dhoni", "ms dhoni",
    "sky", "suryakumar", "yadav", "gill", "shubman", "pant", "rishabh",
    "iyer", "shreyas", "jadeja", "ravindra", "ashwin", "rahul", "kl rahul",
    "hardik", "pandya", "boult", "trent", "maxwell", "glenn", "watson",
    "warner", "david", "buttler", "jos", "starc", "mitchell", "cummins",
    "pat", "smith", "steve", "williamson", "kane", "root", "joe",
    "shami", "siraj", "arshdeep", "chahal", "kuldeep", "bishnoi",
    "rinku", "singh", "padikkal", "patidar", "dube", "shivam",
    "jadhav", "kedar", "rayudu", "ambati", "rinku",
}


def is_ocr_available() -> bool:
    return _OCR_AVAILABLE


def _get_reader():
    global _OCR_READER
    if _OCR_READER is None and _OCR_AVAILABLE:
        try:
            _OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
            log.info("EasyOCR reader initialized (CPU mode)")
        except Exception as e:
            log.warning("EasyOCR init failed: %s", e)
            return None
    return _OCR_READER


def _scene_change_score(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Histogram-based scene change detection (0=identical, 1=completely different)."""
    hb = cv2.compareHist(
        cv2.calcHist([frame_a], [0], None, [32], [0, 256]),
        cv2.calcHist([frame_b], [0], None, [32], [0, 256]),
        cv2.HISTCMP_BHATTACHARYYA,
    )
    hg = cv2.compareHist(
        cv2.calcHist([frame_a], [1], None, [32], [0, 256]),
        cv2.calcHist([frame_b], [1], None, [32], [0, 256]),
        cv2.HISTCMP_BHATTACHARYYA,
    )
    hr = cv2.compareHist(
        cv2.calcHist([frame_a], [2], None, [32], [0, 256]),
        cv2.calcHist([frame_b], [2], None, [32], [0, 256]),
        cv2.HISTCMP_BHATTACHARYYA,
    )
    return (hb + hg + hr) / 3.0


def _extract_keyframes(video_path: str, max_frames: int = 50,
                       scene_thresh: float = 0.35) -> List[np.ndarray]:
    """Extract keyframes from video using scene-change detection + uniform sampling.

    Returns:
        List of keyframe images (RGB uint8 arrays).
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / max(fps, 1)

        interval = max(1, int(fps * max(2.0, duration / max_frames)))

        frames = []
        prev_frame = None

        pos = 0
        while pos < total_frames and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is None:
                frames.append(frame)
                prev_frame = frame
                pos += interval
                continue

            score = _scene_change_score(prev_frame, frame)
            if score > scene_thresh:
                frames.append(frame)
                prev_frame = frame

            pos += interval
    finally:
        cap.release()
    return frames


def _dedup_texts(texts: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Deduplicate OCR results by text content similarity."""
    seen = set()
    result = []
    for text, conf in texts:
        key = text.lower().strip()
        if key and key not in seen:
            seen.add(key)
            result.append((text, conf))
    return result


def _extract_cricket_entities(texts: List[Tuple[str, float]]) -> Dict[str, List[str]]:
    """Classify detected text into cricket-specific categories."""
    scoreboard = []
    player_names = []
    general_text = []

    for text, conf in texts:
        low = text.lower()
        # Score patterns: "198/4", "287/6", "49.3", "Need 8 off 3"
        if re.search(r'\d{1,3}/\d', low) or re.search(r'need \d+ off', low):
            scoreboard.append(text)
        elif re.search(r'\d+\.\d+\s*(overs|ov)', low):
            scoreboard.append(text)
        elif re.search(r'(target|require|need)\s*\d+', low):
            scoreboard.append(text)

        # Player name patterns
        words = low.split()
        for w in words:
            if w in KNOWN_PLAYERS:
                player_names.append(text)
                break

        # Team names
        for team in KNOWN_TEAMS:
            if team in low:
                scoreboard.append(text)
                break

        if text not in scoreboard and text not in player_names:
            general_text.append(text)

    return {
        "scoreboard": list(dict.fromkeys(scoreboard)),
        "player_names": list(dict.fromkeys(player_names)),
        "on_screen_text": list(dict.fromkeys(general_text)),
    }


def extract_ocr_entities(video_path: str, max_frames: int = 50) -> Dict[str, List[str]]:
    """Extract on-screen text entities from video using OCR.

    Args:
        video_path: Path to video file.
        max_frames: Maximum number of keyframes to OCR (default 50).

    Returns:
        Dict with keys: scoreboard, player_names, on_screen_text.
        Each value is a list of detected text strings.
        Always returns the expected structure — empty lists if OCR unavailable.
    """
    if not _OCR_AVAILABLE:
        log.info("OCR not available (easyocr not installed). Returning empty entities.")
        return {"scoreboard": [], "player_names": [], "on_screen_text": []}

    reader = _get_reader()
    if reader is None:
        return {"scoreboard": [], "player_names": [], "on_screen_text": []}

    keyframes = _extract_keyframes(video_path, max_frames=max_frames)
    if not keyframes:
        log.warning("No keyframes extracted from %s", video_path)
        return {"scoreboard": [], "player_names": [], "on_screen_text": []}

    all_texts = []
    for frame in keyframes:
        try:
            # Convert BGR to RGB for easyocr
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = reader.readtext(rgb)
            for bbox, text, conf in results:
                text = text.strip()
                if len(text) > 1 and conf > 0.3:
                    all_texts.append((text, conf))
        except Exception as e:
            log.warning("OCR frame error: %s", e)

    all_texts = _dedup_texts(all_texts)
    entities = _extract_cricket_entities(all_texts)
    log.info("OCR extracted %d entities from %d keyframes (scoreboard: %d, players: %d)",
             sum(len(v) for v in entities.values()), len(keyframes),
             len(entities["scoreboard"]), len(entities["player_names"]))
    return entities
