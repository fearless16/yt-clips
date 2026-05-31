"""knowledge.py — Structured knowledge base with facts and relations.

Builds on PersistentMemory to store typed facts with relationships
between entities. Supports queries by relation, entity, and type.
"""

import time
from typing import Any, Optional

from self_learner.memory import PersistentMemory


class Fact:
    """A single atomic fact with optional relation.

    Attributes:
        subject: The primary entity (e.g., ``"pipeline"``).
        relation: The relationship (e.g., ``"has_status"``).
        object: The value or related entity.
        confidence: Confidence score (0-1).
        timestamp: When the fact was recorded.
        source: Origin identifier.
    """

    __slots__ = ("subject", "relation", "object", "confidence",
                 "timestamp", "source")

    def __init__(self, subject: str, relation: str, object: Any, *,
                 confidence: float = 1.0,
                 timestamp: Optional[float] = None,
                 source: str = "knowledge"):
        self.subject = subject
        self.relation = relation
        self.object = object
        self.confidence = confidence
        self.timestamp = timestamp or time.time()
        self.source = source

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(
            subject=d["subject"],
            relation=d["relation"],
            object=d["object"],
            confidence=d.get("confidence", 1.0),
            timestamp=d.get("timestamp"),
            source=d.get("source", "knowledge"),
        )

    def __repr__(self) -> str:
        return (f"Fact({self.subject} --[{self.relation}]--> "
                f"{self.object}, {self.confidence:.2f})")


_FACT_KEY_PREFIX = "fact:"
_RELATION_INDEX_KEY = "_relation_index"


class KnowledgeBase:
    """Typed knowledge store backed by PersistentMemory.

    Args:
        memory: A ``PersistentMemory`` instance. Creates a default one
                if not provided.

    Facts are stored as individual memories with key prefix ``"fact:"``.
    A relation index maps each relation to the set of fact keys.
    """

    def __init__(self, memory: Optional[PersistentMemory] = None):
        self._memory = memory or PersistentMemory()
        self._rebuild_index()

    # -- index maintenance ---------------------------------------------------

    def _rebuild_index(self):
        index: dict[str, set[str]] = {}
        for mem in self._memory.recall_all(source="knowledge"):
            key = mem["key"]
            if not key.startswith(_FACT_KEY_PREFIX):
                continue
            fact = Fact.from_dict(mem["value"])
            index.setdefault(fact.relation, set()).add(key)
        self._save_index(index)

    def _save_index(self, index: dict[str, set[str]]):
        serializable = {rel: list(keys) for rel, keys in index.items()}
        self._memory.remember(
            _RELATION_INDEX_KEY, serializable,
            source="_internal",
        )

    def _update_index(self, relation: str, fact_key: str):
        raw = self._memory.recall_value(_RELATION_INDEX_KEY)
        index: dict[str, set[str]] = {
            rel: set(keys) for rel, keys in (raw or {}).items()
        }
        index.setdefault(relation, set()).add(fact_key)
        self._save_index(index)

    def _remove_from_index(self, relation: str, fact_key: str):
        raw = self._memory.recall_value(_RELATION_INDEX_KEY)
        if raw is None:
            return
        index: dict[str, set[str]] = {
            rel: set(keys) for rel, keys in raw.items()
        }
        keys = index.get(relation)
        if keys:
            keys.discard(fact_key)
            if not keys:
                del index[relation]
            self._save_index(index)

    # -- public API ----------------------------------------------------------

    def add_fact(self, fact: Fact) -> None:
        """Store a fact in the knowledge base.

        Args:
            fact: The ``Fact`` to store.
        """
        fact_key = _FACT_KEY_PREFIX + f"{fact.subject}:{fact.relation}:{int(time.time() * 1000000)}"
        self._memory.remember(
            fact_key, fact.to_dict(),
            confidence=fact.confidence,
            source="knowledge",
        )
        self._update_index(fact.relation, fact_key)

    def get_facts(self, subject: Optional[str] = None,
                  relation: Optional[str] = None) -> list[Fact]:
        """Retrieve facts, optionally filtered by *subject* and/or *relation*.

        Args:
            subject: Filter by subject entity.
            relation: Filter by relation type.

        Returns:
            List of matching ``Fact`` objects.
        """
        facts: list[Fact] = []
        for mem in self._memory.recall_all(source="knowledge"):
            key = mem["key"]
            if not key.startswith(_FACT_KEY_PREFIX):
                continue
            fact = Fact.from_dict(mem["value"])
            if subject is not None and fact.subject != subject:
                continue
            if relation is not None and fact.relation != relation:
                continue
            facts.append(fact)
        return facts

    def query(self, relation: str) -> list[Fact]:
        """Quick lookup of all facts for a given *relation*.

        Uses the relation index for O(1) key resolution.
        """
        raw = self._memory.recall_value(_RELATION_INDEX_KEY)
        index: dict[str, list[str]] = raw or {}
        fact_keys: list[str] = index.get(relation, [])
        facts: list[Fact] = []
        for fk in fact_keys:
            mem = self._memory.recall(fk)
            if mem is None:
                continue
            facts.append(Fact.from_dict(mem["value"]))
        return facts

    def remove_fact(self, fact: Fact) -> bool:
        """Remove a specific fact from the knowledge base.

        Args:
            fact: The fact to remove.

        Returns:
            ``True`` if the fact was found and removed.
        """
        for mem in self._memory.recall_all(source="knowledge"):
            key = mem["key"]
            if not key.startswith(_FACT_KEY_PREFIX):
                continue
            existing = Fact.from_dict(mem["value"])
            if (existing.subject == fact.subject
                    and existing.relation == fact.relation
                    and existing.object == fact.object
                    and existing.timestamp == fact.timestamp):
                removed = self._memory.forget(key)
                if removed:
                    self._remove_from_index(fact.relation, key)
                return removed
        return False

    def facts_count(self) -> int:
        """Return the total number of stored facts."""
        count = 0
        for mem in self._memory.recall_all(source="knowledge"):
            if mem["key"].startswith(_FACT_KEY_PREFIX):
                count += 1
        return count

    def clear(self) -> None:
        """Remove all facts and the relation index."""
        for mem in self._memory.recall_all(source="knowledge"):
            self._memory.forget(mem["key"])
        self._memory.forget(_RELATION_INDEX_KEY)

    def close(self):
        self._memory.close()
