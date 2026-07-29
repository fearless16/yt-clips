"""All 7 selection agents. Each is a stateless scorer that returns a dict
with ``score`` (0-100) and ``reasoning`` (short string).

Agents operate on ``candidate`` (start/end/text/dimension_scores) and
``context`` which carries the full transcript, RMS map, match info, etc.
"""

import re
from typing import Any

from automation.clip_selection.agent_base import Agent
from automation.clip_selection.keywords import (
    CRICKET_PLAYERS,
    CRICKET_EVENTS,
    CRICKET_PHRASES,
    EMOTION_WORDS,
    PAYOFF_WORDS,
    PAYOFF_PHRASES,
    COMMENTARY_WORDS,
    CRICKET_VENUES,
    CROWD_WORDS,
    CONTROVERSY_WORDS,
    RARE_EVENT_TERMS,
    RARE_EVENT_PHRASES,
    SERIES_WORDS,
)


def _words(text: str) -> set[str]:
    return set(re.findall(r'\b\w+\b', text.lower()))


def _count_overlap(words_set: set[str], keywords: set[str]) -> int:
    return len(words_set & keywords)


def _phrase_hits(text_lower: str, phrases: list[str]) -> int:
    return sum(1 for p in phrases if p in text_lower)


def _collapse_repeats(w: str) -> str:
    result = []
    i = 0
    while i < len(w):
        run_start = i
        while i < len(w) and w[i] == w[run_start]:
            i += 1
        run_len = i - run_start
        if run_len >= 3:
            result.append(w[run_start])
        else:
            result.extend(w[run_start:i])
    return ''.join(result)


def _fuzzy_words(text: str) -> set[str]:
    raw = set(re.findall(r'\b\w+\b', text.lower()))
    normalized = {_collapse_repeats(w) for w in raw}
    return raw | normalized


def _rms_range(rms_map: dict, t_start: float, t_end: float) -> list[float]:
    return [rms_map.get(int(t), 0.0) for t in range(int(t_start), int(t_end) + 1)]


def _avg_rms(rms_map: dict, t_start: float, t_end: float) -> float:
    vals = _rms_range(rms_map, t_start, t_end)
    return sum(vals) / len(vals) if vals else 0.0


# ── Agent 1: Hook Expert ──────────────────────────────────────────────────

class HookExpert(Agent):
    name = "hook_expert"
    weight = 0.30

    def score(self, candidate: dict, context: dict) -> dict[str, Any]:
        start = candidate["start"]
        end = candidate["end"]
        text = candidate.get("text", "")
        rms_map = context.get("rms_map", {})
        avg_rms = context.get("avg_rms", 0.0)
        max_rms = context.get("max_rms", 0.0)

        hook_end = min(start + 3, end)
        hook_text = text[:100]
        hook_rms = _avg_rms(rms_map, start, hook_end)

        score = 30.0
        reasons = []

        # Audio spike in first 3s
        if avg_rms > 0 and hook_rms > avg_rms * 1.5:
            score += 25
            reasons.append("loud_audio_spike")
        elif avg_rms > 0 and hook_rms > avg_rms * 1.2:
            score += 12
            reasons.append("moderate_audio_spike")

        # Reaction word opener
        first_word = hook_text.split()[0].lower().strip(".!?,") if hook_text.split() else ""
        if first_word in EMOTION_WORDS:
            score += 20
            reasons.append(f"reaction_opener={first_word}")

        # Punctuation excitement
        if "!" in hook_text[:50]:
            score += 10
            reasons.append("exclamation")
        if "?" in hook_text[:50]:
            score += 5
            reasons.append("question_hook")

        # Commentary intensity in hook window
        hook_words = _fuzzy_words(hook_text)
        emotion_hits = _count_overlap(hook_words, EMOTION_WORDS)
        score += min(emotion_hits * 5, 15)
        if emotion_hits > 0:
            reasons.append(f"emotion_words={emotion_hits}")

        return {
            "score": min(100, score),
            "reasoning": "; ".join(reasons) if reasons else "no_strong_hook",
            "hook_rms_ratio": round(hook_rms / avg_rms, 2) if avg_rms > 0 else 0,
        }


# ── Agent 2: Emotion Expert ────────────────────────────────────────────────

class EmotionExpert(Agent):
    name = "emotion_expert"
    weight = 0.15

    def score(self, candidate: dict, context: dict) -> dict[str, Any]:
        text = candidate.get("text", "")
        start = candidate["start"]
        end = candidate["end"]
        rms_map = context.get("rms_map", {})
        avg_rms = context.get("avg_rms", 0.0)
        max_rms = context.get("max_rms", 0.0)

        score = 20.0
        reasons = []

        # Energy level
        seg_rms_values = _rms_range(rms_map, start, end)
        if seg_rms_values and avg_rms > 0:
            energy_ratio = sum(seg_rms_values) / len(seg_rms_values) / avg_rms
            score += min(energy_ratio * 15, 30)
            if energy_ratio > 1.3:
                reasons.append(f"high_energy={energy_ratio:.1f}x")

        # Spikes (sudden loud moments)
        if max_rms > 0:
            spike_count = sum(1 for v in seg_rms_values if v > max_rms * 0.85)
            score += min(spike_count * 5, 20)
            if spike_count >= 2:
                reasons.append(f"spikes={spike_count}")

        # Emotion word density
        words = _fuzzy_words(text)
        emotion_hits = _count_overlap(words, EMOTION_WORDS)
        score += min(emotion_hits * 4, 20)
        if emotion_hits >= 3:
            reasons.append(f"emotion_words={emotion_hits}")

        # Building intensity (second half > first half energy)
        mid = len(seg_rms_values) // 2
        if len(seg_rms_values) >= 4:
            first_half = seg_rms_values[:mid]
            second_half = seg_rms_values[mid:]
            avg_first = sum(first_half) / len(first_half) if first_half else 0
            avg_second = sum(second_half) / len(second_half) if second_half else 0
            if avg_first > 0 and avg_second > avg_first * 1.3:
                score += 10
                reasons.append("building_intensity")

        return {
            "score": min(100, score),
            "reasoning": "; ".join(reasons) if reasons else "low_emotion",
        }


# ── Agent 3: Cricket Context Expert ─────────────────────────────────────────

class CricketContextExpert(Agent):
    name = "cricket_context"
    weight = 0.08

    def score(self, candidate: dict, context: dict) -> dict[str, Any]:
        text = candidate.get("text", "")
        start = candidate["start"]
        end = candidate["end"]
        match_context = context.get("match_context", {})

        score = 20.0
        reasons = []

        words = _fuzzy_words(text)
        text_lower = text.lower()

        # Player mentions
        player_matched = words & CRICKET_PLAYERS
        player_hits = len(player_matched)
        score += min(player_hits * 8, 25)
        if player_hits > 0:
            reasons.append(f"players={player_hits}")

        # Entity bias: count entity-level (not word-level) matches
        entity_bias = context.get("entity_bias", {})
        top_player_hits = 0
        for name, _ in entity_bias.get("top_players", []):
            name_parts = name.lower().split()
            if any(p in words for p in name_parts if len(p) > 1):
                top_player_hits += 1
        if top_player_hits:
            score += min(top_player_hits * 6, 15)
            reasons.append(f"high_perf_players={top_player_hits}")
        avoid_player_hits = 0
        for name in entity_bias.get("avoid_players", []):
            name_parts = name.lower().split()
            if any(p in words for p in name_parts if len(p) > 1):
                avoid_player_hits += 1
        if avoid_player_hits:
            score -= min(avoid_player_hits * 4, 8)
            reasons.append(f"low_perf_players={avoid_player_hits}")

        # Entity bias: boost/hit teams that historically get more/less views
        top_team_hits = 0
        for name, _ in entity_bias.get("top_teams", []):
            name_parts = name.lower().split()
            if any(p in words for p in name_parts if len(p) > 1):
                top_team_hits += 1
        if top_team_hits:
            score += min(top_team_hits * 5, 10)
            reasons.append(f"high_perf_teams={top_team_hits}")
        avoid_team_hits = 0
        for name in entity_bias.get("avoid_teams", []):
            name_parts = name.lower().split()
            if any(p in words for p in name_parts if len(p) > 1):
                avoid_team_hits += 1
        if avoid_team_hits:
            score -= min(avoid_team_hits * 3, 6)
            reasons.append(f"low_perf_teams={avoid_team_hits}")

        # Key action terms (single words + multi-word phrases)
        action_hits = _count_overlap(words, CRICKET_EVENTS)
        action_hits += _phrase_hits(text_lower, CRICKET_PHRASES)
        score += min(action_hits * 6, 20)
        if action_hits > 0:
            reasons.append(f"actions={action_hits}")

        # Match context boost (e.g., if candidate falls in known highlight window)
        if match_context:
            for highlight_time in match_context.get("highlight_timestamps", []):
                if abs(start - highlight_time) < 5:
                    score += 15
                    reasons.append("matches_known_highlight")
                    break

        # Tournament / series mention
        series_hits = _count_overlap(words, SERIES_WORDS)
        series_hits += _phrase_hits(text_lower, ["world cup"])
        if series_hits:
            score += 10
            reasons.append(f"series_context={series_hits}")

        # Rivalry boost
        rivalry_pairs = [
            ("india", "pakistan"), ("india", "australia"),
            ("india", "england"), ("australia", "england"),
        ]
        for team_a, team_b in rivalry_pairs:
            if team_a in text_lower and team_b in text_lower:
                score += 10
                reasons.append(f"rivalry={team_a}_vs_{team_b}")
                break

        return {
            "score": min(100, score),
            "reasoning": "; ".join(reasons) if reasons else "no_cricket_context",
        }


# ── Agent 4: Viral Potential Expert ─────────────────────────────────────────

class ViralPotentialExpert(Agent):
    name = "viral_potential"
    weight = 0.13

    def score(self, candidate: dict, context: dict) -> dict[str, Any]:
        text = candidate.get("text", "")
        start = candidate["start"]
        end = candidate["end"]
        rms_map = context.get("rms_map", {})
        max_rms = context.get("max_rms", 0.0)

        score = 10.0
        reasons = []

        words = _fuzzy_words(text)

        # Rare/high-impact events
        rare_hits = _count_overlap(words, RARE_EVENT_TERMS)
        rare_hits += _phrase_hits(text.lower(), RARE_EVENT_PHRASES)
        score += min(rare_hits * 10, 30)
        if rare_hits > 0:
            reasons.append(f"rare_event={rare_hits}")

        # Audio eruption (crowd/commentator peak)
        if max_rms > 0:
            seg_values = _rms_range(rms_map, start, end)
            eruption_count = sum(1 for v in seg_values if v > max_rms * 0.9)
            if eruption_count >= 2:
                score += min(eruption_count * 6, 20)
                reasons.append(f"eruption_peaks={eruption_count}")

        # Repeat keywords (obsession hooks)
        word_list = re.findall(r'\b\w+\b', text.lower())
        for w in set(word_list):
            if len(w) > 2 and word_list.count(w) >= 3:
                score += 10
                reasons.append(f"repeated_keyword={w}")
                break

        # Controversy signals
        if _count_overlap(words, CONTROVERSY_WORDS):
            score += 15
            reasons.append("controversy")

        # Crowd reaction (textual)
        if _count_overlap(words, CROWD_WORDS):
            score += 10
            reasons.append("crowd_reaction")

        return {
            "score": min(100, score),
            "reasoning": "; ".join(reasons) if reasons else "low_viral_potential",
        }


# ── Agent 5: Viewer Psychology Expert ──────────────────────────────────────

class ViewerPsychologyExpert(Agent):
    name = "viewer_psychology"
    weight = 0.09

    def score(self, candidate: dict, context: dict) -> dict[str, Any]:
        text = candidate.get("text", "")
        start = candidate["start"]
        end = candidate["end"]
        duration = end - start

        score = 20.0
        reasons = []

        # Cliffhanger effect — ends with open question/tension
        text_end = text.rstrip()[-30:] if len(text) >= 30 else text
        if text_end.endswith("?") or "what" in text_end.lower():
            score += 15
            reasons.append("cliffhanger")
        if text_end.endswith("!"):
            score += 8
            reasons.append("exclamation_end")

        # Identity validation (fan bias triggers)
        identity_words = {
            "king", "goat", "legend", "best", "greatest", "champion",
            "our", "we", "us", "india", "team india",
        }
        words = _fuzzy_words(text)
        identity_hits = _count_overlap(words, identity_words)
        score += min(identity_hits * 5, 15)
        if identity_hits > 0:
            reasons.append(f"identity_trigger={identity_hits}")

        # Social sharing triggers
        share_words = {"must watch", "share", "tag", "show this", "you won't believe",
                       "wait for it", "watch till end", "you have to see"}
        text_lower = text.lower()
        for phrase in share_words:
            if phrase in text_lower:
                score += 10
                reasons.append(f"share_trigger={phrase}")
                break

        # Reward prediction — payoff expected
        payoff_hints = {"wait", "watch", "see", "check", "look", "here it comes"}
        if _count_overlap(words, payoff_hints):
            score += 8
            reasons.append("reward_anticipation")

        # Duration psychology: 15-30s ideal for retention
        if 15 <= duration <= 30:
            score += 10
            reasons.append("ideal_duration")
        elif 10 <= duration <= 45:
            score += 5
            reasons.append("good_duration")

        return {
            "score": min(100, score),
            "reasoning": "; ".join(reasons) if reasons else "weak_psychology",
        }


# ── Agent 6: Retention Expert ──────────────────────────────────────────────

class RetentionExpert(Agent):
    name = "retention_expert"
    weight = 0.05

    def score(self, candidate: dict, context: dict) -> dict[str, Any]:
        text = candidate.get("text", "")
        start = candidate["start"]
        end = candidate["end"]
        duration = end - start
        rms_map = context.get("rms_map", {})
        avg_rms = context.get("avg_rms", 0.0)

        score = 30.0
        reasons = []

        # Duration sweet spot for Shorts retention
        if 18 <= duration <= 25:
            score += 25
            reasons.append("duration_sweet_spot")
        elif 12 <= duration <= 35:
            score += 15
            reasons.append("duration_good")
        elif duration > 50:
            score -= 20
            reasons.append("too_long")

        # Pacing — no dead air
        word_count = len(text.split())
        wpm = (word_count / duration) * 60 if duration > 0 else 0
        if 120 <= wpm <= 200:
            score += 15
            reasons.append(f"good_pace={int(wpm)}wpm")
        elif 80 <= wpm < 120:
            score += 5
            reasons.append(f"ok_pace={int(wpm)}wpm")
        elif wpm < 60:
            score -= 10
            reasons.append(f"slow_pace={int(wpm)}wpm")

        # Payoff in last 3 seconds (audio energy + text)
        payoff_start = max(start, end - 3)
        payoff_rms = _avg_rms(rms_map, payoff_start, end)
        if avg_rms > 0 and payoff_rms > avg_rms * 1.3:
            score += 15
            reasons.append("strong_payoff_energy")

        last_words = text.rstrip()[-40:] if len(text) >= 40 else text
        payoff_hits = _count_overlap(_fuzzy_words(last_words), PAYOFF_WORDS)
        payoff_hits += _phrase_hits(last_words.lower(), PAYOFF_PHRASES)
        if payoff_hits:
            score += 10
            reasons.append("text_payoff")
        if last_words.rstrip().endswith("!"):
            score += 5
            reasons.append("exclamation_payoff")

        # Repetition (good for retention — viewer stays to confirm)
        word_list = re.findall(r'\b\w+\b', text.lower())
        for w in set(word_list):
            if len(w) > 2 and word_list.count(w) >= 3:
                score += 5
                reasons.append(f"repetition={w}")
                break

        return {
            "score": min(100, score),
            "reasoning": "; ".join(reasons) if reasons else "average_retention",
        }


# ── Agent 8: Semantic Content Expert ────────────────────────────────────────

class SemanticContentExpert(Agent):
    name = "semantic_content"
    weight = 0.20

    def score(self, candidate: dict, context: dict) -> dict[str, Any]:
        text = candidate.get("text", "")
        start = candidate["start"]
        end = candidate["end"]
        duration = end - start

        score = 25.0
        reasons = []

        # ── Topic segmentation + heuristic scores ──────────────────────
        topics = context.get("topics", [])
        heuristics = context.get("topic_heuristics", {})
        if topics:
            overlapping = [
                t for t in topics
                if t["start"] < end and t["end"] > start
            ]
            if overlapping:
                reasons.append(f"topic_overlap={len(overlapping)}")
                best_topic = max(overlapping, key=lambda t: t.get("salience_score", 0))
                topic_score = best_topic.get("salience_score", 0)
                score += topic_score * 0.3
                if topic_score > 60:
                    reasons.append(f"high_salience_topic={topic_score:.0f}")

                # Heuristic scores keyed by topic boundaries
                topic_key = f"{best_topic['start']:.1f}-{best_topic['end']:.1f}"
                h = heuristics.get(topic_key, {})
                if h:
                    salience = h.get("salience_score", 0)
                    score += salience * 0.25
                    if salience > 60:
                        reasons.append(f"heuristic_salience={salience:.0f}")
                    ed = h.get("entity_density", {})
                    if ed.get("player_hits", 0) >= 2:
                        score += 8
                        reasons.append(f"multi_player={ed['player_hits']}")
                    if ed.get("team_hits", 0) >= 1:
                        score += 4
                        reasons.append("team_mention")

        # ── Boundary quality (starts/ends at natural topic boundaries) ─
        if topics:
            aligned_start = any(abs(t["start"] - start) < 2.0 for t in topics)
            aligned_end = any(abs(t["end"] - end) < 2.0 for t in topics)
            if aligned_start and aligned_end:
                score += 12
                reasons.append("topic_aligned")
            elif aligned_start or aligned_end:
                score += 6
                reasons.append("partial_topic_alignment")

        # ── Self-contained heuristic ────────────────────────────────────
        words = _fuzzy_words(text)
        text_lower = text.lower()

        # Pronouns / references to prior context = less self-contained
        context_refs = {"he", "she", "they", "that", "this", "then",
                        "earlier", "before", "previous", "so",
                        "therefore", "thus", "hence", "after"}
        ref_hits = _count_overlap(words, context_refs)
        if ref_hits >= 3:
            score -= 6
            reasons.append(f"context_refs={ref_hits}")

        # Proper nouns / named entities = more self-contained
        entity_hits = _count_overlap(words, CRICKET_PLAYERS)
        if entity_hits >= 1:
            score += 5
            reasons.append("has_named_entity")

        # Duration check for topic completeness
        if 10 <= duration <= 45:
            score += 8
            reasons.append("good_topic_duration")
        elif duration > 60:
            score -= 8
            reasons.append("too_long_for_topic")

        return {
            "score": min(100, score),
            "reasoning": "; ".join(reasons) if reasons else "neutral_semantic",
        }


# ── Agent 7: Brutal Rejection Agent ────────────────────────────────────────

class BrutalRejectionAgent(Agent):
    name = "brutal_rejection"
    weight = 0.0  # Rejection agent doesn't contribute positive score

    def score(self, candidate: dict, context: dict) -> dict[str, Any]:
        text = candidate.get("text", "")
        start = candidate["start"]
        end = candidate["end"]
        duration = end - start
        rms_map = context.get("rms_map", {})
        avg_rms = context.get("avg_rms", 0.0)
        max_rms = context.get("max_rms", 0.0)

        rejection_score = 0.0
        reasons = []

        # Too short — can't build any hook
        if duration < 6:
            rejection_score += 30
            reasons.append("too_short")

        # Too long — will have dead air in Shorts
        if duration > 55:
            rejection_score += 25
            reasons.append("too_long_for_shorts")

        # Low energy throughout
        seg_values = _rms_range(rms_map, start, end)
        if seg_values and avg_rms > 0:
            avg_seg_rms = sum(seg_values) / len(seg_values)
            if avg_seg_rms < avg_rms * 0.7:
                rejection_score += 25
                reasons.append("low_energy_segment")

        # No emotional content
        words = _fuzzy_words(text)
        if _count_overlap(words, EMOTION_WORDS) == 0:
            rejection_score += 15
            reasons.append("no_emotion_words")

        # No cricket relevance
        if (_count_overlap(words, CRICKET_PLAYERS) == 0
                and _count_overlap(words, CRICKET_EVENTS) == 0):
            rejection_score += 10
            reasons.append("no_cricket_content")

        # Repetitive/generic
        if len(text.split()) < 5:
            rejection_score += 15
            reasons.append("too_few_words")

        # Dull hook — no audio spike, no reaction word
        hook_end = min(start + 3, end)
        hook_rms = _avg_rms(rms_map, start, hook_end)
        if avg_rms > 0 and hook_rms < avg_rms * 1.1:
            first_word = text.split()[0].lower().strip(".!?,") if text.split() else ""
            if first_word not in EMOTION_WORDS:
                rejection_score += 20
                reasons.append("dull_hook")

        # Repeated content (same as another candidate)
        candidate_texts = context.get("all_candidate_texts", set())
        if text in candidate_texts:
            rejection_score += 20
            reasons.append("duplicate_content")

        # Very high silence ratio
        word_count = len(text.split())
        est_speech = word_count * 0.35
        silence = max(0, duration - est_speech)
        silence_ratio = silence / duration if duration > 0 else 1
        if silence_ratio > 0.5:
            rejection_score += 15
            reasons.append(f"high_silence={silence_ratio:.0%}")

        return {
            "score": min(100, rejection_score),
            "reasoning": "; ".join(reasons) if reasons else "pass",
            "should_reject": rejection_score >= 40,
        }


# ── All agents ─────────────────────────────────────────────────────────────

ALL_AGENTS: list[Agent] = [
    HookExpert(),
    EmotionExpert(),
    CricketContextExpert(),
    ViralPotentialExpert(),
    ViewerPsychologyExpert(),
    RetentionExpert(),
    BrutalRejectionAgent(),
    SemanticContentExpert(),
]
