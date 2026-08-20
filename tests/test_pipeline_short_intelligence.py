from pathlib import Path

import yaml

import pipeline


def test_pipeline_bridge_records_only_exported_clips(tmp_path, monkeypatch):
    highlights_path = tmp_path / "highlights.yaml"
    highlights_path.write_text(yaml.safe_dump({
        "clip1": {"start": 1, "end": 10, "text": "cricket thought"},
        "clip2": {"start": 20, "end": 30, "text": "not exported"},
    }), encoding="utf-8")
    exported = [tmp_path / "batch" / "clip1.mp4"]
    captured = {}

    def fake_record(config, highlights, paths):
        captured["config"] = config
        captured["highlights"] = highlights
        captured["paths"] = paths
        return 1

    monkeypatch.setattr("shorts_intelligence.bridge.record_exported", fake_record)

    count = pipeline._record_short_intelligence_exports(
        {"channel": {"id": "id"}, "shorts_intelligence": {"enabled": True}},
        str(highlights_path),
        exported,
    )

    assert count == 1
    assert [item["id"] for item in captured["highlights"]] == ["clip1", "clip2"]
    assert captured["paths"] == exported


def test_successful_upload_is_linked_to_short_intelligence(monkeypatch):
    linked = {}
    monkeypatch.setattr(
        "shorts_intelligence.bridge.record_upload",
        lambda config, clip_id, video_id: linked.update(
            config=config, clip_id=clip_id, video_id=video_id
        ) or True,
    )
    clip = Path("shorts/batch/clip1.mp4")
    config = {"channel": {"id": "id"}, "shorts_intelligence": {"enabled": True}}

    assert pipeline._accept_upload_result(clip, "youtube-id", [], intelligence_config=config)
    assert linked["clip_id"] == "batch/clip1"
    assert linked["video_id"] == "youtube-id"
