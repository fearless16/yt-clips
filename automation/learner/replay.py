"""ReplayEngine — rebuilds learned state from event store for recovery and verification.

Updated for idempotent replay using ProcessedEventLog.
Skips already-processed events to prevent replay inflation.
"""

from automation.memory.decision_store import DecisionStore, LearnedStateStore
from automation.learner.learner import Learner
from automation.learner.policy_updater import PolicyUpdater


class ReplayEngine:
    def __init__(
        self,
        decision_store: DecisionStore,
        learned_state: LearnedStateStore,
        learner: Learner,
        dispatcher=None,
        processed_log=None,
    ) -> None:
        self._store = decision_store
        self._state = learned_state
        self._learner = learner
        self._updater = PolicyUpdater(learner)
        self._dispatcher = dispatcher
        self._log = processed_log

    def replay(self) -> int:
        events = self._store.get_all_events()
        processed = 0
        for event in events:
            if self._dispatcher and self._log:
                count = self._dispatcher.dispatch(event)
                processed += count
            else:
                self._updater.update_from_event(event)
                processed += 1
        return processed

    def replay_since(self, timestamp: str) -> int:
        events = self._store.get_all_events()
        filtered = [e for e in events if e.timestamp > timestamp]
        processed = 0
        for event in filtered:
            if self._dispatcher and self._log:
                count = self._dispatcher.dispatch(event)
                processed += count
            else:
                self._updater.update_from_event(event)
                processed += 1
        return processed

    def replay_force(self, learner: str | None = None) -> int:
        """Force replay by clearing processed log first.

        Args:
            learner: Optional learner name to clear. If None, clears all.

        Returns:
            Number of events processed
        """
        if self._log:
            if learner:
                self._log.clear_learner(learner)
            else:
                self._log.clear_all()
        return self.replay()

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
