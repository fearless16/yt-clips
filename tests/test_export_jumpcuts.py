import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import export
import json
import subprocess
import os

def test_export_applies_jumpcuts_to_remove_silence(monkeypatch, tmp_path):
    def mock_export_clip(*args, **kwargs):
        p = tmp_path / "mock.mp4"
        p.write_text("dummy")
        return str(p)
        
    monkeypatch.setattr(export, "export_clip", mock_export_clip)
    monkeypatch.setattr(export, "_validate_av_sync", lambda *args, **kwargs: True)
    
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(export.subprocess, "run", mock_run)
    monkeypatch.setattr(export.os, "replace", lambda src, dst: None)
    monkeypatch.setattr(export, "_get_best_encoder", lambda: "libx264")
    
    monkeypatch.setattr(export, "analyze_clip", lambda *args, **kwargs: {})
    monkeypatch.setattr(export, "_is_exportable_cricket_highlight", lambda *args, **kwargs: True)
    
    highlights_dict = {
        "test_clip_1": {
            "start": 10.0,
            "end": 20.0,
            "text": "Hello world",
        }
    }
    
    transcript_segments = [
        {"start": 10.0, "end": 11.0, "text": "Hello"},
        {"start": 15.0, "end": 16.0, "text": "world"},
    ]
    transcript_file = tmp_path / "transcript.json"
    transcript_file.write_text(json.dumps({"segments": transcript_segments}))
    
    with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
        def mock_map(func, iterable):
            return [func(item) for item in iterable]
        mock_executor.return_value.__enter__.return_value.map = mock_map
        
        original_cfg = export.cfg.get
        def mock_cfg(key, default=None):
            if key == "paths": return {"exports": str(tmp_path)}
            return original_cfg(key, default)
        monkeypatch.setattr(export.cfg, "get", mock_cfg)
        
        export.export_all(
            highlights=highlights_dict,
            video_path="dummy.mp4",
            transcript_path=str(transcript_file),
            generate_seo=False
        )
        
    assert mock_run.called, "subprocess.run should have been called to apply jumpcuts"
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    
    filter_complex_idx = cmd.index("-filter_complex") + 1
    filter_complex = cmd[filter_complex_idx]
    
    assert "trim=start=0.000:end=1.200" in filter_complex, "First segment with padding should be kept"
    assert "trim=start=4.800:end=6.200" in filter_complex, "Second segment with padding should be kept"
    assert "concat=n=2" in filter_complex, "It should concatenate exactly the two segments"

