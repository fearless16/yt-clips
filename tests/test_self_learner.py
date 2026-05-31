"""Tests for self_learner module — 100% coverage target."""

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def memory(tmp_db):
    from self_learner.memory import PersistentMemory
    mem = PersistentMemory(db_path=tmp_db)
    yield mem
    mem.close()


@pytest.fixture
def knowledge(memory):
    from self_learner.knowledge import KnowledgeBase
    kb = KnowledgeBase(memory=memory)
    yield kb
    kb.close()


@pytest.fixture
def learner(knowledge, memory):
    from self_learner.learner import Learner
    l = Learner(memory=memory, knowledge=knowledge)
    yield l
    l.close()


@pytest.fixture
def populated_learner(learner):
    for i in range(5):
        learner.observe("pipeline_run",
                        {"duration": 100 + i * 10, "success": True})
    return learner


# ═══════════════════════════════════════════════════════════════════════════
# memory.py — PersistentMemory
# ═══════════════════════════════════════════════════════════════════════════

class TestPersistentMemory:

    def test_init_default_path(self):
        from self_learner.memory import PersistentMemory
        mem = PersistentMemory()
        assert mem._db_path == "self_learner.db"
        mem.close()
        if os.path.exists("self_learner.db"):
            os.unlink("self_learner.db")

    def test_init_custom_path(self, tmp_db):
        from self_learner.memory import PersistentMemory
        mem = PersistentMemory(db_path=tmp_db)
        assert mem._db_path == tmp_db
        mem.close()

    def test_remember_and_recall(self, memory):
        memory.remember("key1", {"a": 1, "b": "hello"}, confidence=0.9,
                        source="test", ttl=None)
        result = memory.recall("key1")
        assert result is not None
        assert result["value"] == {"a": 1, "b": "hello"}
        assert result["confidence"] == 0.9
        assert result["source"] == "test"
        assert result["ttl"] is None
        assert isinstance(result["timestamp"], float)

    def test_recall_missing(self, memory):
        assert memory.recall("nonexistent") is None

    def test_recall_expired(self, memory):
        memory.remember("exp", "val", ttl=0.01)
        time.sleep(0.02)
        assert memory.recall("exp") is None

    def test_recall_value(self, memory):
        memory.remember("k", {"nested": "data"})
        assert memory.recall_value("k") == {"nested": "data"}
        assert memory.recall_value("missing") is None

    def test_forget_existing(self, memory):
        memory.remember("k", "v")
        assert memory.forget("k") is True
        assert memory.recall("k") is None

    def test_forget_missing(self, memory):
        assert memory.forget("nope") is False

    def test_recall_all(self, memory):
        memory.remember("a", 1, source="src1")
        memory.remember("b", 2, source="src2")
        all_mem = memory.recall_all()
        assert len(all_mem) == 2

    def test_recall_all_filter_source(self, memory):
        memory.remember("a", 1, source="src1")
        memory.remember("b", 2, source="src2")
        src1 = memory.recall_all(source="src1")
        assert len(src1) == 1
        assert src1[0]["key"] == "a"

    def test_recall_all_expired_pruned(self, memory):
        memory.remember("a", 1, ttl=0.01)
        memory.remember("b", 2, ttl=0.01)
        time.sleep(0.02)
        all_mem = memory.recall_all()
        assert len(all_mem) == 0

    def test_size(self, memory):
        assert memory.size() == 0
        memory.remember("a", 1)
        assert memory.size() == 1
        memory.remember("b", 2)
        assert memory.size() == 2

    def test_clear(self, memory):
        memory.remember("a", 1)
        memory.remember("b", 2)
        memory.clear()
        assert memory.size() == 0

    def test_prune(self, memory):
        memory.remember("a", 1, ttl=0.01)
        memory.remember("b", 2)
        time.sleep(0.02)
        pruned = memory.prune()
        assert pruned == 1
        assert memory.size() == 1

    def test_context_manager(self, tmp_db):
        from self_learner.memory import PersistentMemory
        with PersistentMemory(db_path=tmp_db) as mem:
            mem.remember("k", "v")
            assert mem.recall("k") is not None
        assert mem._conn is None

    def test_close_idempotent(self, memory):
        memory.close()
        memory.close()
        assert memory._conn is None

    def test_connect_already_connected(self, memory):
        memory._connect()
        assert memory._conn is not None

    def test_remember_overwrites(self, memory):
        memory.remember("k", "v1")
        memory.remember("k", "v2")
        assert memory.recall_value("k") == "v2"

    def test_thread_safety(self, tmp_db):
        from self_learner.memory import PersistentMemory
        mem = PersistentMemory(db_path=tmp_db)
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    mem.remember(f"k{n}_{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,))
                   for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        all_mem = mem.recall_all()
        assert len(all_mem) == 40
        mem.close()

    def test_serialize_non_json_value(self, memory):
        memory.remember("path", Path("/tmp/test"), source="test")
        result = memory.recall("path")
        assert result is not None
        assert "/tmp/test" in str(result["value"])

    def test_expired_recall_does_not_return_stale(self, memory):
        memory.remember("x", 99, ttl=0.01)
        time.sleep(0.02)
        assert memory.recall("x") is None
        assert memory.recall_value("x") is None


# ═══════════════════════════════════════════════════════════════════════════
# knowledge.py — KnowledgeBase
# ═══════════════════════════════════════════════════════════════════════════

class TestFact:

    def test_constructor(self):
        from self_learner.knowledge import Fact
        f = Fact("cpu", "has_temp", 75.0, confidence=0.8,
                 timestamp=100.0, source="sensor")
        assert f.subject == "cpu"
        assert f.relation == "has_temp"
        assert f.object == 75.0
        assert f.confidence == 0.8
        assert f.timestamp == 100.0
        assert f.source == "sensor"

    def test_default_timestamp(self):
        from self_learner.knowledge import Fact
        f = Fact("a", "b", "c")
        assert f.timestamp is not None
        assert abs(time.time() - f.timestamp) < 1

    def test_to_dict(self):
        from self_learner.knowledge import Fact
        f = Fact("x", "y", "z", confidence=0.5, timestamp=1.0, source="s")
        d = f.to_dict()
        assert d == {"subject": "x", "relation": "y", "object": "z",
                     "confidence": 0.5, "timestamp": 1.0, "source": "s"}

    def test_from_dict(self):
        from self_learner.knowledge import Fact
        d = {"subject": "x", "relation": "y", "object": "z",
             "confidence": 0.5, "timestamp": 1.0, "source": "s"}
        f = Fact.from_dict(d)
        assert f.subject == "x"
        assert f.relation == "y"
        assert f.object == "z"

    def test_from_dict_defaults(self):
        from self_learner.knowledge import Fact
        d = {"subject": "x", "relation": "y", "object": "z"}
        f = Fact.from_dict(d)
        assert f.confidence == 1.0
        assert f.source == "knowledge"
        assert f.timestamp is not None

    def test_repr(self):
        from self_learner.knowledge import Fact
        f = Fact("cpu", "has_temp", 75.0, confidence=0.85)
        r = repr(f)
        assert "cpu" in r
        assert "has_temp" in r
        assert "0.85" in r


class TestKnowledgeBase:

    def test_init_default_memory(self):
        from self_learner.knowledge import KnowledgeBase
        kb = KnowledgeBase()
        assert kb._memory is not None
        kb.close()

    def test_init_custom_memory(self, memory):
        from self_learner.knowledge import KnowledgeBase
        kb = KnowledgeBase(memory=memory)
        assert kb._memory is memory
        kb.close()

    def test_add_and_get_facts(self, knowledge):
        from self_learner.knowledge import Fact
        f1 = Fact("cpu", "has_temp", 75.0)
        f2 = Fact("gpu", "has_temp", 68.0)
        knowledge.add_fact(f1)
        knowledge.add_fact(f2)
        facts = knowledge.get_facts()
        assert len(facts) == 2

    def test_get_facts_filter_subject(self, knowledge):
        from self_learner.knowledge import Fact
        knowledge.add_fact(Fact("cpu", "has_temp", 75.0))
        knowledge.add_fact(Fact("gpu", "has_temp", 68.0))
        facts = knowledge.get_facts(subject="cpu")
        assert len(facts) == 1
        assert facts[0].subject == "cpu"

    def test_get_facts_filter_relation(self, knowledge):
        from self_learner.knowledge import Fact
        knowledge.add_fact(Fact("cpu", "has_temp", 75.0))
        knowledge.add_fact(Fact("cpu", "has_load", 0.5))
        facts = knowledge.get_facts(relation="has_load")
        assert len(facts) == 1
        assert facts[0].relation == "has_load"

    def test_get_facts_filter_both(self, knowledge):
        from self_learner.knowledge import Fact
        knowledge.add_fact(Fact("cpu", "has_temp", 75.0))
        knowledge.add_fact(Fact("gpu", "has_temp", 68.0))
        facts = knowledge.get_facts(subject="cpu", relation="has_temp")
        assert len(facts) == 1

    def test_get_facts_no_match(self, knowledge):
        facts = knowledge.get_facts(subject="nonexistent")
        assert facts == []

    def test_query_by_relation(self, knowledge):
        from self_learner.knowledge import Fact
        f1 = Fact("cpu", "has_temp", 75.0)
        f2 = Fact("gpu", "has_temp", 68.0)
        knowledge.add_fact(f1)
        knowledge.add_fact(f2)
        results = knowledge.query("has_temp")
        assert len(results) == 2

    def test_query_nonexistent_relation(self, knowledge):
        results = knowledge.query("nonexistent")
        assert results == []

    def test_remove_fact_existing(self, knowledge):
        from self_learner.knowledge import Fact
        f = Fact("cpu", "has_temp", 75.0, timestamp=100.0)
        knowledge.add_fact(f)
        assert knowledge.facts_count() == 1
        removed = knowledge.remove_fact(f)
        assert removed is True
        assert knowledge.facts_count() == 0

    def test_remove_fact_not_found(self, knowledge):
        from self_learner.knowledge import Fact
        f = Fact("cpu", "nonexistent", "val", timestamp=1.0)
        removed = knowledge.remove_fact(f)
        assert removed is False

    def test_facts_count(self, knowledge):
        from self_learner.knowledge import Fact
        assert knowledge.facts_count() == 0
        knowledge.add_fact(Fact("a", "b", 1))
        assert knowledge.facts_count() == 1

    def test_clear(self, knowledge):
        from self_learner.knowledge import Fact
        knowledge.add_fact(Fact("a", "b", 1))
        knowledge.add_fact(Fact("c", "d", 2))
        knowledge.clear()
        assert knowledge.facts_count() == 0

    def test_close(self, knowledge):
        knowledge.close()
        assert knowledge._memory._conn is None

    def test_rebuild_index_with_existing_data(self, tmp_db):
        from self_learner.memory import PersistentMemory
        from self_learner.knowledge import Fact, KnowledgeBase
        mem = PersistentMemory(db_path=tmp_db)
        kb = KnowledgeBase(memory=mem)
        kb.add_fact(Fact("x", "rel", "y"))
        kb.close()

        mem2 = PersistentMemory(db_path=tmp_db)
        kb2 = KnowledgeBase(memory=mem2)
        assert kb2.facts_count() == 1
        assert len(kb2.query("rel")) == 1
        kb2.close()
        mem2.close()

    def test_remove_from_index_no_index(self, knowledge):
        from self_learner.knowledge import Fact
        f = Fact("a", "b", 1, timestamp=100.0)
        removed = knowledge.remove_fact(f)
        assert removed is False

    def test_remove_fact_updates_index(self, knowledge):
        from self_learner.knowledge import Fact
        f = Fact("cpu", "has_temp", 75.0, timestamp=200.0)
        knowledge.add_fact(f)
        assert len(knowledge.query("has_temp")) == 1
        knowledge.remove_fact(f)
        assert len(knowledge.query("has_temp")) == 0

    def test_get_facts_ignores_non_fact_keys(self, knowledge):
        knowledge._memory.remember("not_a_fact", {"data": 1})
        facts = knowledge.get_facts()
        assert len(facts) == 0


# ═══════════════════════════════════════════════════════════════════════════
# learner.py — Learner, Observation, Pattern, Prediction
# ═══════════════════════════════════════════════════════════════════════════

class TestObservation:

    def test_constructor(self):
        from self_learner.learner import Observation
        obs = Observation("pipeline_run", {"dur": 120}, timestamp=1.0,
                          source="sensor")
        assert obs.event_type == "pipeline_run"
        assert obs.attributes == {"dur": 120}
        assert obs.timestamp == 1.0
        assert obs.source == "sensor"

    def test_default_timestamp(self):
        from self_learner.learner import Observation
        obs = Observation("evt", {"k": "v"})
        assert abs(time.time() - obs.timestamp) < 1

    def test_default_source(self):
        from self_learner.learner import Observation
        obs = Observation("evt", {"k": "v"})
        assert obs.source == "learner"


class TestPattern:

    def test_constructor(self):
        from self_learner.learner import Pattern
        p = Pattern("pipeline_run", {"dur": 120}, confidence=0.8,
                    sample_count=5, last_updated=100.0)
        assert p.event_type == "pipeline_run"
        assert p.attributes == {"dur": 120}
        assert p.confidence == 0.8
        assert p.sample_count == 5
        assert p.last_updated == 100.0

    def test_defaults(self):
        from self_learner.learner import Pattern
        p = Pattern("evt", {"k": "v"})
        assert p.confidence == 0.5
        assert p.sample_count == 1


class TestPrediction:

    def test_constructor(self):
        from self_learner.learner import Prediction
        pr = Prediction("pipeline_run", {"dur": 120}, confidence=0.9,
                        supporting_patterns=5, timestamp=100.0)
        assert pr.event_type == "pipeline_run"
        assert pr.predicted_attributes == {"dur": 120}
        assert pr.confidence == 0.9
        assert pr.supporting_patterns == 5
        assert pr.timestamp == 100.0

    def test_defaults(self):
        from self_learner.learner import Prediction
        pr = Prediction("evt", {"k": "v"})
        assert pr.confidence == 0.0
        assert pr.supporting_patterns == 0


class TestLearner:

    def test_init_defaults(self):
        from self_learner.learner import Learner
        l = Learner()
        assert l._memory is not None
        assert l._knowledge is not None
        l.close()

    def test_init_custom(self, memory, knowledge):
        from self_learner.learner import Learner
        l = Learner(memory=memory, knowledge=knowledge)
        assert l._memory is memory
        assert l._knowledge is knowledge
        l.close()

    def test_observe_single(self, learner):
        obs = learner.observe("pipeline_run",
                              {"duration": 120, "success": True})
        assert obs.event_type == "pipeline_run"
        assert obs.attributes == {"duration": 120, "success": True}
        assert learner.stats()["total_observations"] == 1

    def test_observe_no_pattern_before_2(self, learner):
        learner.observe("event_a", {"val": 1})
        assert learner.stats()["total_patterns"] == 0

    def test_observe_creates_pattern_after_2(self, learner):
        learner.observe("event_a", {"val": 1})
        learner.observe("event_a", {"val": 2})
        assert learner.stats()["total_patterns"] == 1

    def test_observe_multiple_refines_pattern(self, populated_learner):
        stats = populated_learner.stats()
        assert stats["total_observations"] == 5
        assert stats["total_patterns"] == 1

    def test_observe_non_dict_object(self, learner):
        learner.observe("event_a", {"val": 1})
        learner._refine_patterns("event_a")
        assert learner.stats()["total_patterns"] == 0

    def test_predict_with_pattern(self, populated_learner):
        pred = populated_learner.predict("pipeline_run")
        assert pred is not None
        assert pred.event_type == "pipeline_run"
        assert "duration" in pred.predicted_attributes
        assert pred.confidence > 0
        assert pred.supporting_patterns > 0

    def test_predict_no_pattern(self, learner):
        pred = learner.predict("nonexistent")
        assert pred is None

    def test_predict_best_pattern_selected(self, learner):
        learner.observe("evt", {"x": 1})
        learner.observe("evt", {"x": 2})
        pred = learner.predict("evt")
        assert pred is not None
        assert "x" in pred.predicted_attributes

    def test_insights_with_patterns(self, populated_learner):
        insights = populated_learner.insights()
        assert len(insights) >= 1
        assert any("pipeline_run" in ins for ins in insights)
        assert any("confidence" in ins for ins in insights)

    def test_insights_no_patterns(self, learner):
        assert learner.insights() == []

    def test_insights_non_dict_object(self, learner):
        from self_learner.learner import Fact
        learner._knowledge.add_fact(
            Fact("evt", "is_pattern", {"not": "dict"})
        )
        learner.insights()

    def test_stats_after_observations(self, populated_learner):
        stats = populated_learner.stats()
        assert "total_observations" in stats
        assert "total_patterns" in stats
        assert "memory_size" in stats
        assert "facts_count" in stats

    def test_stats_empty(self, learner):
        stats = learner.stats()
        assert stats["total_observations"] == 0
        assert stats["total_patterns"] == 0

    def test_close(self, learner):
        learner.close()
        assert learner._memory._conn is None

    def test_refine_with_categorical_attrs(self, learner):
        learner.observe("status", {"state": "ok"})
        learner.observe("status", {"state": "error"})
        learner.observe("status", {"state": "ok"})
        pats = learner._knowledge.get_facts(relation="is_pattern")
        assert len(pats) == 1

    def test_refine_with_mixed_attrs(self, learner):
        learner.observe("mix", {"val": 1, "tag": "a"})
        learner.observe("mix", {"val": 2, "tag": "b"})
        pats = learner._knowledge.get_facts(relation="is_pattern")
        assert len(pats) == 1
        obj = pats[0].object
        assert "val" in obj["attributes"]
        assert "tag" in obj["attributes"]

    def test_refine_pattern_replaces_old(self, learner):
        learner.observe("evt", {"x": 1})
        learner.observe("evt", {"x": 2})
        assert learner.stats()["total_patterns"] == 1
        learner.observe("evt", {"x": 3})
        pats = learner._knowledge.get_facts(relation="is_pattern")
        assert len(pats) == 1

    def test_predict_nondict_object(self, learner):
        from self_learner.learner import Fact
        learner._knowledge.add_fact(
            Fact("evt", "is_pattern", "not_a_dict")
        )
        pred = learner.predict("evt")
        assert pred is None

    def test_predict_nondict_obj_nested(self, learner):
        from self_learner.learner import Fact
        learner._knowledge.add_fact(
            Fact("evt", "is_pattern", {"attributes": None, "confidence": 0.5})
        )
        pred = learner.predict("evt")
        assert pred is not None
        assert pred.predicted_attributes is None
        assert pred.confidence == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# runner.py — Runner
# ═══════════════════════════════════════════════════════════════════════════

class TestRunner:

    def test_help(self):
        from self_learner.runner import Runner
        runner = Runner()
        rc = runner.run(["-h"])
        assert rc == 0

    def test_unknown_command(self):
        from self_learner.runner import Runner
        runner = Runner()
        rc = runner.run(["nonexistent"])
        assert rc == 1

    def test_observe(self, learner):
        from self_learner.runner import Runner
        runner = Runner(learner=learner)
        rc = runner.run(["observe", "evt", '{"val": 42}'])
        assert rc == 0
        assert learner.stats()["total_observations"] == 1

    def test_observe_invalid_json(self, learner):
        from self_learner.runner import Runner
        runner = Runner(learner=learner)
        rc = runner.run(["observe", "evt", "not-json"])
        assert rc == 1

    def test_observe_missing_args(self, learner):
        from self_learner.runner import Runner
        runner = Runner(learner=learner)
        rc = runner.run(["observe"])
        assert rc == 1

    def test_predict(self, populated_learner):
        from self_learner.runner import Runner
        runner = Runner(learner=populated_learner)
        rc = runner.run(["predict", "pipeline_run"])
        assert rc == 0

    def test_predict_no_args(self, learner):
        from self_learner.runner import Runner
        runner = Runner(learner=learner)
        rc = runner.run(["predict"])
        assert rc == 1

    def test_predict_no_pattern(self, learner):
        from self_learner.runner import Runner
        runner = Runner(learner=learner)
        rc = runner.run(["predict", "nonexistent"])
        assert rc == 1

    def test_insights(self, populated_learner):
        from self_learner.runner import Runner
        runner = Runner(learner=populated_learner)
        rc = runner.run(["insights"])
        assert rc == 0

    def test_insights_empty(self, learner):
        from self_learner.runner import Runner
        runner = Runner(learner=learner)
        rc = runner.run(["insights"])
        assert rc == 0

    def test_stats(self, populated_learner):
        from self_learner.runner import Runner
        runner = Runner(learner=populated_learner)
        rc = runner.run(["stats"])
        assert rc == 0

    def test_default_learner(self):
        from self_learner.runner import Runner
        runner = Runner()
        assert runner._learner is not None
        runner._learner.close()

    def test_daemon_valid_line(self, learner):
        from self_learner.runner import Runner
        runner = Runner(learner=learner)
        from io import StringIO
        old_stdin = sys.stdin
        sys.stdin = StringIOSim(['{"event_type": "test", "attributes": {"x": 1}}'])
        old_stdout = sys.stdout
        sys.stdout = StringIOSim([])
        try:
            rc = runner.run(["daemon"])
        except SystemExit:
            pass
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout

    def test_daemon_invalid_json(self, learner):
        from self_learner.runner import Runner
        runner = Runner(learner=learner)
        from io import StringIO
        old_stdin = sys.stdin
        sys.stdin = StringIOSim(["not json"])
        old_stderr = sys.stderr
        sys.stderr = StringIOSim([])
        old_stdout = sys.stdout
        sys.stdout = StringIOSim([])
        try:
            rc = runner.run(["daemon"])
        except SystemExit:
            pass
        finally:
            sys.stdin = old_stdin
            sys.stderr = old_stderr
            sys.stdout = old_stdout

    def test_daemon_missing_event_type(self, learner):
        from self_learner.runner import Runner
        runner = Runner(learner=learner)
        from io import StringIO
        old_stdin = sys.stdin
        sys.stdin = StringIOSim(['{"attributes": {"x": 1}}'])
        old_stderr = sys.stderr
        sys.stderr = StringIOSim([])
        old_stdout = sys.stdout
        sys.stdout = StringIOSim([])
        try:
            rc = runner.run(["daemon"])
        except SystemExit:
            pass
        finally:
            sys.stdin = old_stdin
            sys.stderr = old_stderr
            sys.stdout = old_stdout

    def test_daemon_keyboard_interrupt(self, learner):
        from self_learner.runner import Runner
        runner = Runner(learner=learner)
        from io import StringIO
        old_stdin = sys.stdin
        sys.stdin = StringIOSim([])
        old_stdout = sys.stdout
        sys.stdout = StringIOSim([])
        try:
            runner.run(["daemon"])
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout

    def test_daemon_empty_line(self, learner):
        from self_learner.runner import Runner
        runner = Runner(learner=learner)
        old_stdin = sys.stdin
        sys.stdin = StringIOSim([""])
        old_stdout = sys.stdout
        sys.stdout = StringIOSim([])
        try:
            rc = runner.run(["daemon"])
        except SystemExit:
            pass
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout

    def test_main_function_runs(self):
        old_argv = sys.argv
        sys.argv = ["runner.py", "-h"]
        try:
            from self_learner.runner import main
            main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv


class StringIOSim:
    def __init__(self, lines):
        self._lines_iter = iter(lines)
        self.buffer = ""
    def readline(self):
        try:
            return next(self._lines_iter) + "\n"
        except StopIteration:
            raise KeyboardInterrupt()
    def __iter__(self):
        return self
    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line
    def write(self, s):
        self.buffer += s
    def flush(self):
        pass
    def close(self):
        pass


class TestRunnerDefaultLearner:
    def test_default_learner_cleanup(self):
        from self_learner.runner import Runner
        r = Runner()
        assert r._learner is not None
        r._learner.close()


# ═══════════════════════════════════════════════════════════════════════════
# Edge-case coverage tests — hit every remaining uncovered line
# ═══════════════════════════════════════════════════════════════════════════

class TestCoverageEdgeCases:
    """Targets specific uncovered lines for 100% coverage."""

    def test_init_py_no_side_effects(self):
        """__init__.py does not mutate sys.path."""
        import subprocess
        code = "import sys; before = list(sys.path); from self_learner import VERSION; assert VERSION == '1.0.0'; assert sys.path == before, 'sys.path was mutated'"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_knowledge_rebuild_index_non_fact_key(self, memory, tmp_db):
        """knowledge.py:91 — non-fact key in _rebuild_index."""
        from self_learner.knowledge import Fact, KnowledgeBase
        memory.remember("not_a_fact", {"data": 1}, source="knowledge")
        kb = KnowledgeBase(memory=memory)
        kb.add_fact(Fact("a", "b", 1))
        new_kb = KnowledgeBase(memory=memory)
        assert new_kb.facts_count() == 1
        new_kb.close()

    def test_knowledge_remove_from_index_no_index(self, memory, tmp_db):
        """knowledge.py:114 — _remove_from_index with no index."""
        from self_learner.knowledge import Fact, KnowledgeBase
        kb = KnowledgeBase(memory=memory)
        f = Fact("a", "b", 1, timestamp=100.0)
        kb.add_fact(f)
        kb._memory.forget("_relation_index")
        removed = kb.remove_fact(f)
        assert removed is True

    def test_knowledge_get_facts_skips_non_fact_key(self, memory):
        """knowledge.py:156 — get_facts skips non-fact prefix keys."""
        from self_learner.knowledge import Fact, KnowledgeBase
        kb = KnowledgeBase(memory=memory)
        kb._memory.remember("other_stuff", {"x": 1}, source="knowledge")
        kb.add_fact(Fact("a", "b", 1))
        facts = kb.get_facts()
        assert len(facts) == 1
        kb.close()

    def test_knowledge_query_missing_fact_key(self, memory):
        """knowledge.py:177 — query skips forgotten fact keys."""
        from self_learner.knowledge import Fact, KnowledgeBase
        kb = KnowledgeBase(memory=memory)
        f = Fact("a", "b", 1)
        kb.add_fact(f)
        all_facts = kb.get_facts()
        fact_key = None
        for mem in kb._memory.recall_all(source="knowledge"):
            if mem["key"].startswith("fact:"):
                fact_key = mem["key"]
                break
        assert fact_key is not None
        kb._memory.forget(fact_key)
        results = kb.query("b")
        assert len(results) == 0
        kb.close()

    def test_knowledge_remove_fact_skips_non_fact_key(self, memory):
        """knowledge.py:193 — remove_fact skips non-fact prefix keys."""
        from self_learner.knowledge import Fact, KnowledgeBase
        kb = KnowledgeBase(memory=memory)
        kb._memory.remember("not_a_fact", {"x": 1}, source="knowledge")
        f = Fact("a", "b", 1, timestamp=200.0)
        removed = kb.remove_fact(f)
        assert removed is False
        kb.close()

    def test_learner_refine_non_dict_object(self, learner):
        """learner.py:146 — skip non-dict observation objects."""
        from self_learner.learner import Fact
        learner._knowledge.add_fact(
            Fact("evt", "was_observed", "not_a_dict", source="learner")
        )
        learner._knowledge.add_fact(
            Fact("evt", "was_observed", "also_not_dict", source="learner")
        )
        learner._refine_patterns("evt")
        assert learner.stats()["total_patterns"] == 1

    def test_learner_single_numeric_pattern(self, learner):
        """learner.py:158-159 — single numeric value pattern path."""
        learner.observe("evt", {"shared": 1, "only_once": 42})
        learner.observe("evt", {"shared": 2})
        pats = learner._knowledge.get_facts(relation="is_pattern")
        assert len(pats) == 1
        obj = pats[0].object
        attrs = obj.get("attributes", {})
        assert "shared" in attrs
        assert "only_once" in attrs

    def test_learner_insights_non_dict_object(self, learner):
        """learner.py:231 — skip non-dict pattern objects in insights."""
        from self_learner.learner import Fact
        learner._knowledge.add_fact(
            Fact("evt", "is_pattern", "string_instead_of_dict")
        )
        insights = learner.insights()
        assert len(insights) == 0

    def test_runner_main_via_module(self):
        """runner.py:175 — __main__ block execution."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "self_learner.runner", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0

    def test_learner_single_numeric_observe(self, learner):
        """learner.py:159 — single numeric attr pattern creation."""
        learner.observe("singleton", {"cpu": 50})
        learner.observe("singleton", {"cpu": 50})
        pats = learner._knowledge.get_facts(relation="is_pattern")
        found = False
        for p in pats:
            if p.subject == "singleton":
                found = True
        assert found
