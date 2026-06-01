"""PreferenceEngine — derives content preferences from learned state."""

from automation.memory.decision_store import DecisionStore
from automation.memory.event_models import EventType
from automation.learner.learner import Learner


class PreferenceEngine:
    def __init__(self, learner: Learner, decision_store: DecisionStore | None = None) -> None:
        self._learner = learner
        self._decision_store = decision_store

    def compute_preferences(self) -> dict:
        hook_weight_str = self._learner.get_state("hook_weight")
        payoff_weight_str = self._learner.get_state("payoff_weight")
        hook_weight = float(hook_weight_str) if hook_weight_str else 1.0
        payoff_weight = float(payoff_weight_str) if payoff_weight_str else 1.0

        if hook_weight > payoff_weight:
            preferred_duration = "short"
        else:
            preferred_duration = "medium"

        total_weight = hook_weight + payoff_weight
        if total_weight > 2.5:
            preferred_pacing = "fast"
        else:
            preferred_pacing = "normal"

        topic_fatigue_penalty = 0.0
        if self._decision_store is not None:
            rejected = self._decision_store.get_events(event_type=EventType.rejected)
            topic_fatigue_penalty = min(len(rejected) * 0.05, 0.3)

        return {
            "preferred_duration": preferred_duration,
            "preferred_pacing": preferred_pacing,
            "topic_fatigue_penalty": topic_fatigue_penalty,
            "hook_weight": hook_weight,
            "payoff_weight": payoff_weight,
        }

    def get_recommendation(self, clip_data: dict) -> dict:
        clip_id = clip_data.get("clip_id", "unknown")
        preferences = self.compute_preferences()
        duration = clip_data.get("duration_s", 30.0)

        recommended = True
        reason = "no issues detected"

        if preferences["preferred_duration"] == "short" and duration >= 30:
            recommended = False
            reason = f"duration {duration}s exceeds preferred short format"
        elif preferences["preferred_duration"] == "medium" and duration < 30:
            recommended = False
            reason = f"duration {duration}s is too short for preferred medium format"

        return {
            "clip_id": clip_id,
            "recommended": recommended,
            "reason": reason,
        }
