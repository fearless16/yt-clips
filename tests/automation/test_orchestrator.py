"""Tests for the orchestrator and CLI modules."""

import pytest
import json
from unittest.mock import MagicMock

from automation.memory.event_models import EventType, ClipEvent
from automation.memory.decision_store import DecisionStore
from automation.scoring.scoring import ClipScorer


class TestOrchestratorInit:
    def test_initializes_with_default_components(self):
        from automation.orchestrator import Orchestrator
        orch = Orchestrator()
        assert orch._decision_store is not None
        assert orch._clip_scorer is None
        assert orch._clip_ranker is None

    def test_initializes_with_custom_components(self):
        store = DecisionStore()
        scorer = ClipScorer()
        from automation.scoring.ranker import ClipRanker
        ranker = ClipRanker(scorer)
        from automation.orchestrator import Orchestrator
        orch = Orchestrator(
            decision_store=store, clip_scorer=scorer, clip_ranker=ranker,
        )
        assert orch._decision_store is store
        assert orch._clip_scorer is scorer
        assert orch._clip_ranker is ranker


class TestOrchestratorRunPipeline:
    def test_returns_dict_with_required_keys(self):
        from automation.orchestrator import Orchestrator
        orch = Orchestrator()
        result = orch.run_pipeline("https://youtube.com/watch?v=test")
        assert isinstance(result, dict)
        assert "url" in result
        assert "stages_completed" in result
        assert "events_emitted" in result
        assert "errors" in result
        assert result["url"] == "https://youtube.com/watch?v=test"
        assert isinstance(result["stages_completed"], list)
        assert isinstance(result["events_emitted"], int)
        assert isinstance(result["errors"], list)

    def test_emits_events_to_decision_store(self):
        store = DecisionStore()
        from automation.orchestrator import Orchestrator
        orch = Orchestrator(decision_store=store)
        orch.run_pipeline("https://youtube.com/watch?v=test")
        assert store.count() > 0
        events = store.get_all_events()
        assert any(e.event_type == EventType.candidate_created for e in events)

    def test_respects_stage_skip_flags(self):
        store = DecisionStore()
        from automation.orchestrator import Orchestrator
        orch = Orchestrator(decision_store=store)
        stages = {s: False for s in (
            "download", "transcribe", "score", "rank",
            "export", "seo", "upload", "cleanup",
        )}
        result = orch.run_pipeline(
            "https://youtube.com/watch?v=test", stages=stages,
        )
        assert result["stages_completed"] == []
        assert result["events_emitted"] > 0

    def test_pipeline_continues_on_stage_failure(self):
        store = DecisionStore()
        from automation.orchestrator import Orchestrator
        orch = Orchestrator(decision_store=store)
        original_run_stage = orch._run_stage

        def failing_run_stage(name, data):
            if name == "score":
                raise RuntimeError("Score stage failed")
            return original_run_stage(name, data)

        orch._run_stage = failing_run_stage
        result = orch.run_pipeline("https://youtube.com/watch?v=test")
        assert "score" not in result["stages_completed"]
        assert len(result["errors"]) == 1
        assert result["errors"][0]["stage"] == "score"
        assert "download" in result["stages_completed"]
        assert "export" in result["stages_completed"]
        assert "cleanup" in result["stages_completed"]


class TestOrchestratorEmitEvent:
    def test_emit_event_stores_valid_clip_event(self):
        store = DecisionStore()
        from automation.orchestrator import Orchestrator
        orch = Orchestrator(decision_store=store)
        orch.emit_event("clip-abc", "candidate_created", {"source": "test"})
        assert store.count() == 1
        event = store.get_all_events()[0]
        assert isinstance(event, ClipEvent)
        assert event.clip_id == "clip-abc"
        assert event.event_type == EventType.candidate_created


class TestOrchestratorRunStage:
    def test_score_stage_uses_clip_scorer(self):
        scorer = MagicMock(spec=ClipScorer)
        scorer.score.return_value = {
            "clip_id": "clip-abc", "score": 0.75,
            "features": {}, "breakdown": {},
        }
        from automation.orchestrator import Orchestrator
        orch = Orchestrator(clip_scorer=scorer)
        result = orch._run_stage("score", {"clip_id": "clip-abc"})
        scorer.score.assert_called_once_with({"clip_id": "clip-abc"})
        assert result["score"] == 0.75

    def test_unknown_stage_returns_data_with_stage_name(self):
        from automation.orchestrator import Orchestrator
        orch = Orchestrator()
        result = orch._run_stage("unknown_stage", {"clip_id": "clip-abc"})
        assert isinstance(result, dict)
        assert "unknown_stage" in result
        assert result["clip_id"] == "clip-abc"


class TestOrchestratorGetPipelineStatus:
    def test_returns_dict(self):
        from automation.orchestrator import Orchestrator
        orch = Orchestrator()
        status = orch.get_pipeline_status()
        assert isinstance(status, dict)
        assert "pipeline_running" in status
        assert "last_run" in status
        assert "total_events" in status
        assert status["pipeline_running"] is False
        assert status["last_run"] is None
        assert status["total_events"] == 0


class TestDecisionStoreAfterPipeline:
    def test_has_events_after_pipeline_run(self):
        store = DecisionStore()
        from automation.orchestrator import Orchestrator
        orch = Orchestrator(decision_store=store)
        orch.run_pipeline("https://youtube.com/watch?v=test")
        assert store.count() > 0
        events = store.get_all_events()
        assert len(events) == store.count()


class TestSetupArgparse:
    def test_returns_argument_parser(self):
        from automation.cli import setup_argparse
        parser = setup_argparse()
        assert parser is not None
        assert hasattr(parser, "parse_args")


class TestMain:
    def test_version_returns_0(self, capsys):
        from automation.cli import main
        ret = main(["--version"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "3.0.0" in captured.out

    def test_memory_report_returns_0(self, capsys):
        from automation.cli import main
        ret = main(["--memory-report"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Memory: OK" in captured.out

    def test_override_emits_manual_override_event(self, monkeypatch, capsys):
        store = DecisionStore()
        monkeypatch.setattr(
            "automation.memory.decision_store.DecisionStore", lambda: store,
        )
        from automation.cli import main
        ret = main(["--override", "keep", "--override-clip-id", "clip-123"])
        assert ret == 0
        events = store.get_all_events()
        assert len(events) == 1
        assert events[0].event_type == EventType.manual_override

    def test_valid_url_runs_pipeline(self, monkeypatch, capsys):
        from automation.orchestrator import PipelineResult
        called = {}

        def mock_run(url, **kwargs):
            called["url"] = url
            called["kwargs"] = kwargs
            r = PipelineResult()
            r.exported = []
            r.uploaded_count = 0
            r.failures = []
            r.total_seconds = 0.1
            return r

        monkeypatch.setattr("automation.orchestrator.run", mock_run)
        from automation.cli import main
        ret = main(["https://youtube.com/watch?v=test"])
        assert ret == 0
        assert called["url"] == "https://youtube.com/watch?v=test"

    def test_returns_1_on_error(self, capsys):
        from automation.cli import main
        ret = main(["--invalid-flag"])
        assert ret == 1

    def test_dry_run_does_not_run_pipeline(self, monkeypatch, capsys):
        pipeline_called = []

        class MockOrchestrator:
            def __init__(self, **kwargs):
                pass

            def run_pipeline(self, **kwargs):
                pipeline_called.append(True)
                return {}

        monkeypatch.setattr(
            "automation.orchestrator.Orchestrator", MockOrchestrator,
        )
        monkeypatch.setattr(
            "automation.memory.decision_store.DecisionStore",
            lambda: DecisionStore(),
        )
        from automation.cli import main
        ret = main(["--dry-run", "https://youtube.com/watch?v=test"])
        assert ret == 0
        assert len(pipeline_called) == 0
