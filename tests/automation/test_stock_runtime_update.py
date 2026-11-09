"""The accepted stock-pair recoveries must run in production, not shell snippets."""

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest


def test_garbled_primary_is_redecoded_before_whole_source_rediscovery(tmp_path, monkeypatch):
    from automation import editorial_planner as planner
    from tests.automation.test_transcript_curation import material, proposal, final, config
    from shorts_intelligence.completeness import completeness_value
    speech = material()[:2]
    cues = [{key: value for key, value in s.items() if key != "words"} for s in speech]
    first_review = final((2,), ())
    first_review["rejected"] = [{"candidate_id": 1, "reason": "Garbled ASR in retained speech"}]
    responses = [{"proposals": [proposal(0), proposal(1)]}, first_review, final((1, 2), ())]
    ai = Mock()
    ai.generate_json.side_effect = [json.dumps(r) for r in responses]
    monkeypatch.setattr(planner, "_get_ai", lambda: ai)
    calls = []

    def asr(_source, start, _end, **kwargs):
        calls.append((start, kwargs))
        evidence = copy.deepcopy(speech[0 if start == 10 else 1])
        if start == 10 and kwargs["model_name"] == "medium":
            evidence["words"][5]["word"] = "garbled"
        return [evidence]

    monkeypatch.setattr("automation.speech_alignment.transcribe_range", asr)
    context = {}
    result = planner.curate_clips(cues, [], "source.mp4", config(), context, tmp_path / "curation.json")
    assert len(result) == 2
    assert all(completeness_value(c) is True for c in result)
    assert [kw["model_name"] for _, kw in calls] == ["medium", "medium", "large-v3-turbo"]
    assert [kw["context_seconds"] for _, kw in calls] == [3, 3, 3]
    payloads = [json.loads(c.args[0]) for c in ai.generate_json.call_args_list]
    assert [p["stage"] for p in payloads] == ["discovery", "final_review", "final_review"]
    assert "garbled" in payloads[1]["candidates"][0]["text"]
    assert "garbled" not in payloads[2]["candidates"][0]["text"]
    assert result[0]["speech_recovery"]["prior_alignment_attempts"][0]["model"] == "medium"
    assert context["editorial_decision"]["speech_recoveries"][0]["status"] == "TEXT_REVIEWED"
    # Same source, same evidence: cached ASR AND decisions, no manual script.
    ai.generate_json.side_effect = AssertionError("No new provider calls on resume")
    calls.clear()
    again = planner.curate_clips(cues, [], "source.mp4", config(), {}, tmp_path / "curation.json")
    assert again == result and not calls


@pytest.mark.parametrize("reason,supplied,model,expected", [
    ("Missing payoff, incomplete thought", False, "medium", 0),
    ("Garbled ASR", True, "medium", 0),
    ("Garbled ASR", False, "large-v3-turbo", 0),
    ("Garbled ASR", False, "medium", 1),
])
def test_speech_recovery_never_overwrites_supplied_or_retries_weak_content(
        tmp_path, monkeypatch, reason, supplied, model, expected):
    from automation import editorial_planner as planner
    from tests.automation.test_transcript_curation import config
    original = {"start": 10, "end": 20, "source_cue_ids": [1]}
    if supplied:
        original["speech_segments"] = [{"words": []}]
    aligned = [{**original, "alignment_attempts": [{"model": model}]}]
    align = Mock(side_effect=ValueError("still invalid"))
    monkeypatch.setattr(planner, "align_candidate", align)
    decision = {}
    assert planner._recover_speech_rejections({"rejected": [{"candidate_id": 1, "reason": reason}]},
        aligned, [original], "source", config(), tmp_path, decision) == []
    assert align.call_count == expected


def test_failed_speech_recoveries_still_consume_the_bounded_budget(tmp_path, monkeypatch):
    from automation import editorial_planner as planner
    originals = [{"start": i * 30, "end": i * 30 + 10, "source_cue_ids": [i]} for i in range(5)]
    aligned = [{**p, "alignment_attempts": [{"model": "medium"}]} for p in originals]
    align = Mock(side_effect=ValueError("bad speech"))
    monkeypatch.setattr(planner, "align_candidate", align)
    decision = {}
    planner._recover_speech_rejections({"rejected": [{"candidate_id": i+1, "reason": "Garbled ASR"}
        for i in range(5)]}, aligned, originals, "source", {}, tmp_path, decision)
    assert align.call_count == len(decision["speech_recoveries"]) == 2


def test_title_cover_format_repair_is_one_strict_call_and_preserves_other_fields(monkeypatch):
    from automation.seo import seo
    original = {"title": "Kohli की smooth drive", "cover_text": "A very smooth Kohli drive for four",
        "description": "Virat Kohli plays a smooth drive for four runs during Prajjwal's original cricket watch-along commentary.",
        "tags": ["Virat Kohli"], "hashtags": ["#Shorts"], "search_terms": ["Kohli smooth drive"]}
    ai = Mock()
    ai.generate_json.return_value = json.dumps({"title": "Kohli Ki Smooth Drive, Seedha Chaar!", "cover_text": "SMOOTH DRIVE"})
    monkeypatch.setattr(seo, "_get_ai", lambda: ai)
    before = copy.deepcopy(original)
    fixed = seo._repair_packaging_format(original, "Kohli plays a smooth drive for four", True)
    assert original == before
    assert seo._validate_seo_quality(fixed)
    for key in ("description", "tags", "hashtags", "search_terms"):
        assert fixed[key] == before[key]
    assert fixed["format_repair"]["status"] == "PASS"
    ai.generate_json.assert_called_once()
    assert ai.generate_json.call_args.kwargs["timeout"] == 60
    assert ai.generate_json.call_args.kwargs["reasoning_effort"] == "low"


@pytest.mark.parametrize("response", ["not json", '{"title":"अब भी हिंदी","cover_text":"FOUR"}',
    '{"title":"Kohli smooth drive","cover_text":"too many words for this cover"}',
    '{"title":"Kohli smooth drive","cover_text":"FOUR","description":"invented"}'])
def test_format_repair_failure_never_becomes_generic_fallback(monkeypatch, response):
    from automation.seo import seo
    original = {"title": "Kohli की drive", "description": "Source grounded original cricket commentary."}
    ai = Mock()
    ai.generate_json.return_value = response
    monkeypatch.setattr(seo, "_get_ai", lambda: ai)
    assert seo._repair_packaging_format(original, "Kohli drive", True) == original
    assert not seo._validate_seo_quality(original)
    ai.generate_json.assert_called_once()


def test_valid_packaging_never_spends_a_format_repair_call(monkeypatch):
    from automation.seo import seo
    ai = Mock()
    monkeypatch.setattr(seo, "_get_ai", lambda: ai)
    good = {"title": "Kohli Ki Smooth Drive!", "cover_text": "SMOOTH DRIVE"}
    assert seo._repair_packaging_format(good, "Kohli drive", True) is good
    ai.generate_json.assert_not_called()


def test_late_seo_worker_cannot_overwrite_deadline_or_owner_metadata(tmp_path, monkeypatch):
    import export
    started, finish = threading.Event(), threading.Event()
    control = {"lock": threading.Lock(), "cancelled": threading.Event()}

    def generate(**kwargs):
        started.set()
        assert finish.wait(5), "test release missing"
        result = {"title": "Late result", "description": "Late provider output"}
        Path(kwargs["output_dir"], "clip1_metadata.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr("automation.seo.seo.generate_seo_for_exported_clip", generate)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(export._seo_worker, "clip1", {"text": "source speech"}, tmp_path,
                                 {"_seo_control": control})
        try:
            assert started.wait(5)
            with control["lock"]:
                control["cancelled"].set()
                marker = tmp_path / "clip1_seo_failed.json"
                marker.write_text('{"error":"deadline"}')
                metadata = tmp_path / "clip1_metadata.json"
                metadata.write_text('{"title":"Owner recovered copy"}')
        finally:
            finish.set()
        assert future.result(timeout=5)["_seo_failed"]
    assert json.loads(marker.read_text())["error"] == "deadline"
    assert json.loads(metadata.read_text())["title"] == "Owner recovered copy"
    assert not list(tmp_path.glob(".seo-*"))


def test_ontime_seo_worker_promotes_metadata_and_removes_stale_marker(tmp_path, monkeypatch):
    import export
    control = {"lock": threading.Lock(), "cancelled": threading.Event()}
    marker = tmp_path / "clip1_seo_failed.json"
    marker.write_text('{"error":"prior attempt"}')

    def generate(**kwargs):
        result = {"title": "Kohli smooth drive", "description": "Grounded copy"}
        Path(kwargs["output_dir"], "clip1_metadata.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr("automation.seo.seo.generate_seo_for_exported_clip", generate)
    result = export._seo_worker("clip1", {"text": "source speech"}, tmp_path, {"_seo_control": control})
    assert json.loads((tmp_path / "clip1_metadata.json").read_text()) == result
    assert not marker.exists()
    assert not list(tmp_path.glob(".seo-*"))


@pytest.mark.parametrize("url", ["https://youtu.be/wRP_KO9vP5g?t=10",
    "https://www.youtube.com/watch?v=wRP_KO9vP5g", "https://youtube.com/live/wRP_KO9vP5g",
    "https://www.youtube.com/shorts/wRP_KO9vP5g"])
def test_scoped_source_id_only_accepts_real_youtube_paths(url):
    from automation.source_runtime import youtube_source_id
    assert youtube_source_id(url) == "wRP_KO9vP5g"


@pytest.mark.parametrize("url", ["https://example.com/watch?v=wRP_KO9vP5g",
    "https://youtu.be/../../video.mp4", "https://youtu.be/local", "file:///watch?v=wRP_KO9vP5g"])
def test_bad_source_scope_fails_before_mutating_configuration(url):
    from automation.source_runtime import source_runtime
    from utils.config import load_config
    before = copy.deepcopy(load_config())
    with pytest.raises(ValueError, match="real YouTube"):
        with source_runtime(url):
            pytest.fail("Invalid scope cannot execute")
    assert load_config() == before


def test_source_scope_routes_all_existing_modules_and_restores_after_failure():
    import export
    from automation import orchestrator
    from automation.clip_selection import pipeline
    from automation.source_runtime import source_runtime
    from utils.config import load_config
    shared = load_config()
    old_paths, old_runner = shared["paths"], orchestrator._CONFIG
    with pytest.raises(RuntimeError, match="fixture failure"):
        with source_runtime("https://youtu.be/wRP_KO9vP5g") as cfg:
            assert cfg is shared is export.cfg is pipeline.cfg is orchestrator._cfg()
            for key in ("input", "transcripts", "highlights", "shorts"):
                assert Path(cfg["paths"][key]) == Path(old_paths[key]) / "wRP_KO9vP5g"
            assert cfg["export"]["gameplay_manifest"] == shared["export"]["gameplay_manifest"]
            with pytest.raises(RuntimeError, match="serialized"):
                with source_runtime("https://youtu.be/H8OjNZ8XLbI"):
                    pytest.fail("Nested scope cannot rewrite shared config")
            raise RuntimeError("fixture failure")
    assert shared["paths"] is old_paths
    assert orchestrator._CONFIG is old_runner


def test_cli_source_scope_uses_one_pipeline_call_without_upload(monkeypatch):
    from types import SimpleNamespace
    from automation import cli, orchestrator
    from utils.config import load_config
    calls = []
    original = copy.deepcopy(load_config())

    def run(**kwargs):
        calls.append((copy.deepcopy(orchestrator._cfg()), kwargs))
        return SimpleNamespace(failures=[])

    monkeypatch.setattr(orchestrator, "run", run)
    assert cli.main(["https://youtu.be/wRP_KO9vP5g", "--source-scoped"]) == 0
    assert len(calls) == 1
    assert Path(calls[0][0]["paths"]["input"]).name == "wRP_KO9vP5g"
    assert not calls[0][1]["auto_upload"]
    assert load_config() == original
