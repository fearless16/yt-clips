"""ReplayEngine — rebuilds learned state from event store for recovery and verification."""

from automation.memory.decision_store import DecisionStore, LearnedStateStore
from automation.learner.learner import Learner
from automation.learner.policy_updater import PolicyUpdater


class ReplayEngine:
    def __init__(
        self,
        decision_store: DecisionStore,
        learned_state: LearnedStateStore,
        learner: Learner,
    ) -> None:
        self._store = decision_store
        self._state = learned_state
        self._learner = learner
        self._updater = PolicyUpdater(learner)

    def replay(self) -> int:
        events = self._store.get_all_events()
        for event in events:
            self._updater.update_from_event(event)
        return len(events)

    def replay_since(self, timestamp: str) -> int:
        events = self._store.get_all_events()
        filtered = [e for e in events if e.timestamp > timestamp]
        for event in filtered:
            self._updater.update_from_event(event)
        return len(filtered)

    def verify(self) -> bool:
        saved = dict(self._state.get_all())
        self._state.clear()
        self.replay()
        new_state = self._state.get_all()
        if new_state == saved:
            return True
        self._state.clear()
        for key, val in saved.items():
            self._state.set(key, val)
        return False
