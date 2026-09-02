import pytest
from unittest.mock import patch, MagicMock
import upload
import json

def test_upload_respects_notify_subscribers_config(monkeypatch, tmp_path):
    monkeypatch.setattr("time.sleep", lambda x: None)
    
    original_cfg_get = upload.cfg.get
    def mock_cfg_get(key, default=None):
        if key == "youtube":
            return {"notify_subscribers": False}
        return original_cfg_get(key, default)
        
    monkeypatch.setattr(upload.cfg, "get", mock_cfg_get)
    
    mock_youtube = MagicMock()
    mock_insert = MagicMock()
    # next_chunk returns (status, response)
    mock_insert.next_chunk.return_value = (None, {"id": "test_id"})
    
    mock_youtube.videos.return_value.insert.return_value = mock_insert
    
    monkeypatch.setattr(upload, "get_authenticated_service", lambda *args, **kwargs: mock_youtube)
    monkeypatch.setattr(upload, "MediaFileUpload", MagicMock())
    
    # Mock video validation so it passes
    monkeypatch.setattr(upload, "_validate_shorts_video", lambda *args, **kwargs: True)
    monkeypatch.setattr(upload, "_probe_video", lambda *args, **kwargs: {"duration": 10.0, "width": 1080, "height": 1920})
    
    metadata = {
        "title": "Dummy Title",
        "description": "Dummy Description",
        "tags": ["dummy"]
    }
    
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(metadata))
    
    vid_path = tmp_path / "dummy.mp4"
    vid_path.write_text("dummy")
    
    upload.upload_video(
        video_path=str(vid_path),
        metadata_path=str(meta_path),
        privacy="private"
    )
        
    assert mock_youtube.videos.return_value.insert.called
    kwargs = mock_youtube.videos.return_value.insert.call_args[1]
    assert "notifySubscribers" in kwargs
    assert kwargs["notifySubscribers"] is False

