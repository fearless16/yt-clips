"""Transcript display gaps must never independently recut rendered media."""

import json
from unittest.mock import MagicMock

import export


def test_export_keeps_rendered_timeline_and_precise_boundaries(monkeypatch, tmp_path):
    monkeypatch.setitem(export.cfg["export"], "composition", "portrait-caption-v1")
    observed = []

    def render(source, start, end, output, *_args, **_kwargs):
        from pathlib import Path
        observed.append((start, end))
        Path(output).write_bytes(b"original encoded media")
        return output

    monkeypatch.setattr(export, "export_clip", render)
    monkeypatch.setattr(export, "_validate_av_sync", lambda *_: True)
    monkeypatch.setattr(export, "_get_best_encoder", lambda: "libx264")
    monkeypatch.setattr(export, "analyze_clip", lambda *_args, **_kwargs: {
        "export_strategy": {"should_drop": False}, "layout": {"layout_type": "solo"}})
    monkeypatch.setitem(export.cfg, "paths", {**export.cfg["paths"], "shorts": str(tmp_path / "shorts")})
    monkeypatch.setitem(export.cfg["export"], "captions", {"enabled": False, "require_word_timing": False})
    monkeypatch.setitem(export.cfg["export"], "super_resolution", False)
    monkeypatch.setitem(export.cfg["export"], "max_workers", 1)
    calls = MagicMock()
    monkeypatch.setattr(export.subprocess, "run", calls)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({"segments": [
        {"start": 10.25, "end": 11, "text": "Kohli"},
        {"start": 15, "end": 16, "text": "hits a six"},
    ]}))
    outputs = export.export_all({"clip1": {"start": "00:00:10", "end": "00:00:20",
                                          "start_sec": 10.25, "end_sec": 20.75,
                                          "text": "Kohli hits a six"}},
                                "source.mp4", transcript_path=str(transcript), generate_seo=False)
    assert observed == [(10.25, 20.75)]
    assert len(outputs) == 1
    assert outputs[0].read_bytes() == b"original encoded media"
    calls.assert_not_called()
