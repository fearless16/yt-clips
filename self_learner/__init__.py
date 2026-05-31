"""self_learner — persistent-memory learning engine.

Zero FaceOS deps. Built for autonomous learning with SQLite-backed memory.

Submodules:
    memory      SQLite-backed persistent key-value store with metadata
    knowledge   Knowledge base with facts, patterns, and relationships
    learner     Learning engine: observe -> extract -> predict
    runner      CLI entry point for one-shot and daemon modes

Usage:
    from self_learner import Learner
    learner = Learner()
    learner.observe("pipeline_run", {"duration": 120, "success": True})
    pred = learner.predict("pipeline_run")
"""

from self_learner.memory import PersistentMemory
from self_learner.knowledge import KnowledgeBase
from self_learner.learner import Learner

VERSION = "1.0.0"

__all__ = [
    "PersistentMemory",
    "KnowledgeBase",
    "Learner",
    "VERSION",
]
