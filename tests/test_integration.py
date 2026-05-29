import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json


def test_pipeline_integration_success_flow(tmp_path):
    import sys
    import utils.config
    utils.config._config_cache.clear()
    sys.modules.pop("pipeline", None)
    
    mock_config = {
        "paths": {
            "input": str(tmp_path / "input"),
            "transcripts": str(tmp_path / "transcripts"),
            "highlights": str(tmp_path / "highlights"),
            "shorts": str(tmp_path / "shorts"),
            "temp": str(tmp_path / "temp")
        },
        "download": {
            "output_filename": "video.mp4"
        },
        "testing": {
            "enabled": False
        },
        "logging": {
            "level": "INFO",
            "log_file": str(tmp_path / "pipeline.log")
        },
        "youtube": {
            "category_id": "22",
            "enforce_shorts_eligibility": True,
            "shorts_max_seconds": 180.0,
            "self_declared_made_for_kids": False,
            "schedule_interval_hours": 2,
            "privacy_status": "public"
        }
    }
    
    for p in mock_config["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
        
    video_file = Path(mock_config["paths"]["input"]) / "video.mp4"
    video_file.write_bytes(b"dummy mp4 video content")
    
    def mock_transcribe(video_path, output_path):
        print(f"\n[DEBUG] mock_transcribe called: video={video_path}, output={output_path}")
        with open(output_path, "w") as f:
            json.dump({"segments": [{"start": 0.0, "end": 2.0, "text": "hello"}]}, f)
            
    def mock_detect_highlights(transcript_path, video_path, output_path):
        print(f"\n[DEBUG] mock_detect_highlights called: transcript={transcript_path}, video={video_path}, output={output_path}")
        with open(output_path, "w") as f:
            f.write("highlights:\n  - start: 0.0\n    end: 2.0\n")
        return [{"start": 0.0, "end": 2.0}]
            
    def mock_process_all_seo(highlights_path, export_dir):
        meta_path = Path(export_dir) / "clip1_metadata.json"
        meta_path.write_text(json.dumps({"title": "Test Title"}))

    def mock_export_all(highlights_path, video_path, **kwargs):
        clip_path = Path(mock_config["paths"]["shorts"]) / "clip1.mp4"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        clip_path.write_bytes(b"dummy clip content")
        return [clip_path]
        
    with patch("utils.config.load_config", return_value=mock_config), \
         patch("download.download", return_value=video_file), \
         patch("transcribe.transcribe", side_effect=mock_transcribe), \
         patch("highlight.detect_highlights", side_effect=mock_detect_highlights), \
         patch("video_analyzer.analyze_video", return_value={"summary": {"face_detection_rate": 80.0, "avg_quality": 0.9}}), \
         patch("export.export_all", side_effect=mock_export_all), \
         patch("seo.process_all_seo", side_effect=mock_process_all_seo) as mock_seo, \
         patch("thumbnail.process_all_thumbnails") as mock_thumb, \
         patch("sync.sync_to_drive") as mock_sync, \
         patch("upload.upload_video", return_value="uploaded_id_789") as mock_upload:
             
        from pipeline import run as run_pipeline
        run_pipeline(
            url="https://youtu.be/dummyurl",
            skip_download=False,
            skip_transcribe=False,
            skip_highlight=False,
            skip_export=False,
            skip_sync=False,
            skip_seo=False,
            auto_sync=True,
            auto_upload=True,
            skip_tests=True
        )
        
        
        mock_seo.assert_called_once()
        mock_thumb.assert_called_once()
        mock_sync.assert_called_once()
        mock_upload.assert_called_once()
        
        transcript_file = Path(mock_config["paths"]["transcripts"]) / "video.json"
        assert transcript_file.exists()
        
        highlight_file = Path(mock_config["paths"]["highlights"]) / "video.yaml"
        assert highlight_file.exists()
