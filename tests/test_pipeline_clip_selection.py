"""Focused tests for root pipeline clip-selection and learner wiring."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml


def test_pipeline_uses_clip_selector_and_stable_match_key_when_enabled():
    import pipeline

    config = {"clip_selection": {"enabled": True}}
    with patch(
        "automation.clip_selection.pipeline.detect_highlights",
        return_value=[{"id": "clip1"}],
    ) as selector, patch("highlight.detect_highlights") as legacy:
        result = pipeline._detect_highlights(
            "transcript.json",
            "video.mp4",
            "highlights.yaml",
            "https://youtu.be/AbC123xyz98?t=15",
            config,
        )

    assert result == [{"id": "clip1"}]
    selector.assert_called_once_with(
        "transcript.json",
        "video.mp4",
        "highlights.yaml",
        match_key="AbC123xyz98",
    )
    legacy.assert_not_called()


def test_pipeline_uses_legacy_selector_when_disabled():
    import pipeline

    config = {"clip_selection": {"enabled": False}}
    with patch(
        "automation.clip_selection.pipeline.detect_highlights",
    ) as selector, patch(
        "highlight.detect_highlights",
        return_value=[{"id": "legacy"}],
    ) as legacy:
        result = pipeline._detect_highlights(
            "transcript.json",
            "video.mp4",
            "highlights.yaml",
            "https://youtu.be/AbC123xyz98",
            config,
        )

    assert result == [{"id": "legacy"}]
    legacy.assert_called_once_with("transcript.json", "video.mp4", "highlights.yaml")
    selector.assert_not_called()


def test_pipeline_persists_only_successful_exports_with_batch_ids(tmp_path):
    import pipeline

    highlights_path = tmp_path / "highlights.yaml"
    highlights_path.write_text(yaml.safe_dump({
        "clip1": {
            "score": 81.0,
            "agent_scores": {"hook_expert": {"score": 90}},
            "rejection_reasons": [],
        },
        "clip2": {
            "final_score": 76.0,
            "agent_scores": {"emotion_expert": {"score": 85}},
        },
    }), encoding="utf-8")
    batch_dir = tmp_path / "shorts" / "2026-08-20_120000"
    exported = [batch_dir / "clip1.mp4"]
    learner = MagicMock()

    pipeline._persist_clip_selections(
        str(highlights_path), exported, "India vs Australia", learner,
    )

    assert learner.save_clip_selection.call_count == 1
    assert learner.save_clip_selection.call_args_list[0].kwargs == {
        "clip_id": "2026-08-20_120000/clip1",
        "video_title": "India vs Australia",
        "selected_rank": 1,
        "final_score": 81.0,
        "agent_scores": {"hook_expert": {"score": 90}},
        "rejection_reasons": [],
    }


def test_pipeline_reads_actual_video_title_from_metadata(tmp_path):
    import json
    import pipeline

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "video_metadata.json").write_text(json.dumps({
        "title": "India vs Australia Final",
    }), encoding="utf-8")

    assert pipeline._load_video_title(
        {"paths": {"input": str(input_dir)}}, input_dir / "video.mp4",
    ) == "India vs Australia Final"


def test_pipeline_records_upload_for_same_collision_free_clip_id(tmp_path):
    import pipeline

    clip_path = tmp_path / "shorts" / "2026-08-20_120000" / "clip1.mp4"
    learner = MagicMock()

    pipeline._record_upload_performance(clip_path, "youtube-id-123", learner)

    learner.update_performance.assert_called_once_with(
        clip_id="2026-08-20_120000/clip1",
        youtube_video_id="youtube-id-123",
        views=0,
    )


def test_none_upload_id_is_failure_and_not_success(tmp_path):
    import pipeline

    failures = []
    learner = MagicMock()
    clip = tmp_path / "batch" / "clip1.mp4"

    success = pipeline._accept_upload_result(clip, None, failures, learner)

    assert success is False
    assert failures == [{
        "phase": "YouTube Upload",
        "clip_id": "clip1.mp4",
        "error": "upload returned no video ID",
    }]
    learner.update_performance.assert_not_called()


def test_real_upload_id_is_success(tmp_path):
    import pipeline

    failures = []
    learner = MagicMock()
    clip = tmp_path / "batch" / "clip1.mp4"

    success = pipeline._accept_upload_result(
        clip, "youtube-id-123", failures, learner,
    )

    assert success is True
    assert failures == []
    learner.update_performance.assert_called_once()


def test_dedup_commits_only_terminal_success_stems(tmp_path):
    import pipeline
    from automation.clip_selection.dedupe import load_previous_windows

    highlights = tmp_path / "video.yaml"
    highlights.write_text(yaml.safe_dump({
        "clip1": {"start_sec": 10.0, "end_sec": 30.0},
        "clip2": {"start_sec": 40.0, "end_sec": 60.0},
    }), encoding="utf-8")
    history_dir = tmp_path / "history"
    config = {
        "paths": {"highlights": str(history_dir)},
        "clip_selection": {"enabled": True, "dedup_enabled": True},
    }

    pipeline._commit_successful_dedup(
        str(highlights), {"clip1"}, "match-key", config,
    )

    assert load_previous_windows(history_dir / "match-key.dedupe_history.yaml") == [
        {"start": 10.0, "end": 30.0},
    ]


def test_local_placeholder_uses_metadata_youtube_id(tmp_path):
    import json
    import pipeline

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video = input_dir / "video.mp4"
    video.write_bytes(b"local-video")
    (input_dir / "video_metadata.json").write_text(json.dumps({
        "url": "https://youtu.be/AbC123xyz98",
    }), encoding="utf-8")
    config = {"paths": {"input": str(input_dir)}}

    assert pipeline._resolve_match_key(
        "https://youtu.be/local", video, config,
    ) == "AbC123xyz98"


def test_local_placeholder_uses_content_fingerprint_without_metadata(tmp_path):
    import pipeline

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    first = input_dir / "first.mp4"
    second = input_dir / "second.mp4"
    first.write_bytes(b"first-video-content")
    second.write_bytes(b"second-video-content")
    config = {"paths": {"input": str(input_dir)}}

    first_key = pipeline._resolve_match_key("https://youtu.be/local", first, config)
    second_key = pipeline._resolve_match_key("https://youtu.be/local", second, config)

    assert first_key.startswith("local-")
    assert first_key == pipeline._resolve_match_key(
        "https://youtu.be/local", first, config,
    )
    assert first_key != second_key


def test_local_placeholder_disables_dedup_without_metadata_or_video(tmp_path):
    import pipeline

    config = {"paths": {"input": str(tmp_path)}}
    assert pipeline._resolve_match_key(
        "https://youtu.be/local", tmp_path / "missing.mp4", config,
    ) is None


def test_export_batch_ids_are_collision_resistant():
    import export

    first = export._new_export_batch_id()
    second = export._new_export_batch_id()

    assert first != second
    assert len(first) > len("2026-08-20_120000")


def test_clip_pipeline_rejects_stale_adaptive_weight_schema(tmp_path):
    import json
    from automation.clip_selection.arbiter import _DEFAULT_WEIGHTS
    import automation.clip_selection.pipeline as clip_pipeline

    weights_path = tmp_path / "clip_selection_weights.json"
    weights_path.write_text(json.dumps({
        "hook_expert": 0.35,
        "emotion_expert": 0.20,
        "viral_potential": 0.15,
        "cricket_context": 0.10,
        "viewer_psychology": 0.10,
        "retention_expert": 0.05,
        "technical_quality": 0.05,
    }), encoding="utf-8")

    assert clip_pipeline._load_adaptive_weights(weights_path) is None

    weights_path.write_text(json.dumps(_DEFAULT_WEIGHTS), encoding="utf-8")
    assert clip_pipeline._load_adaptive_weights(weights_path) == _DEFAULT_WEIGHTS
