import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json

from upload import (
    _validate_shorts_video,
    _ensure_shorts_metadata,
    upload_video,
)

def test_validate_shorts_video():
    with patch("upload._probe_video", return_value={"width": 1080, "height": 1920, "duration": 30.0}):
        assert _validate_shorts_video(Path("dummy.mp4")) is True

    with patch("upload._probe_video", return_value={"width": 1920, "height": 1080, "duration": 30.0}):
        assert _validate_shorts_video(Path("dummy.mp4")) is False

    with patch("upload._probe_video", return_value={"width": 1080, "height": 1920, "duration": 200.0}):
        assert _validate_shorts_video(Path("dummy.mp4")) is False

def test_ensure_shorts_metadata():
    title = "Epic Cricket Play"
    description = "Check out this shot!"
    tags = ["cricket", "ipl"]
    
    clean_title, clean_desc, clean_tags = _ensure_shorts_metadata(title, description, tags)
    
    assert clean_title == "Epic Cricket Play"
    assert "#Shorts" in clean_desc
    assert "shorts" in clean_tags
    
    long_tags = ["a" * 100] * 6
    _, _, clean_long_tags = _ensure_shorts_metadata(title, description, long_tags)
    assert len(clean_long_tags) < 6

def test_upload_video_success_path(tmp_path):
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"dummy video")
    
    metadata_file = tmp_path / "metadata.json"
    metadata_content = {
        "title": "Super Kohli Shot",
        "description": "Kohli plays a great shot",
        "tags": ["Kohli", "cricket"],
        "search_terms": ["Kohli highlights", "IPL highlights"]
    }
    metadata_file.write_text(json.dumps(metadata_content))
    
    mock_service = MagicMock()
    mock_insert = MagicMock()
    mock_service.videos().insert.return_value = mock_insert
    
    mock_status = MagicMock()
    mock_status.progress.return_value = 1.0
    mock_insert.next_chunk.return_value = (mock_status, {"id": "uploaded_vid_123"})
    
    with patch("upload._validate_shorts_video", return_value=True), \
         patch("upload.get_authenticated_service", return_value=mock_service), \
         patch("upload.MediaFileUpload"), \
         patch("upload.Path.exists", return_value=True):
             
        vid_id = upload_video(str(video_file), str(metadata_file), privacy="public")
        
        assert vid_id == "uploaded_vid_123"
        
        mock_service.videos().insert.assert_called_once()
        _, kwargs = mock_service.videos().insert.call_args
        body = kwargs["body"]
        
        assert body["snippet"]["title"] == "Super Kohli Shot"
        assert "#Shorts" in body["snippet"]["description"]
        assert body["status"]["privacyStatus"] == "public"
        assert body["status"]["selfDeclaredMadeForKids"] is False
        assert body["status"]["containsSyntheticMedia"] is False
        assert kwargs["notifySubscribers"] is True

def test_upload_video_retry_on_transient_error(tmp_path):
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"dummy video")
    
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps({"title": "Retry Test"}))
    
    mock_service = MagicMock()
    mock_insert = MagicMock()
    mock_service.videos().insert.return_value = mock_insert
    
    from googleapiclient.errors import HttpError
    mock_resp = MagicMock()
    mock_resp.status = 503
    http_error = HttpError(resp=mock_resp, content=b"Service Unavailable")
    
    mock_status = MagicMock()
    mock_status.progress.return_value = 1.0
    mock_insert.next_chunk.side_effect = [http_error, (mock_status, {"id": "success_id"})]
    
    with patch("upload._validate_shorts_video", return_value=True), \
         patch("upload.get_authenticated_service", return_value=mock_service), \
         patch("upload.MediaFileUpload"), \
         patch("upload.Path.exists", return_value=True), \
         patch("time.sleep") as mock_sleep:
             
        vid_id = upload_video(str(video_file), str(metadata_file), privacy="public")
        
        assert vid_id == "success_id"
        assert mock_insert.next_chunk.call_count == 2
        mock_sleep.assert_called_once_with(5)

def test_upload_video_token_rotation_on_quota(tmp_path):
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"dummy")
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps({"title": "Rotation Test"}))
    
    mock_service1 = MagicMock()
    mock_service2 = MagicMock()
    
    mock_insert1 = MagicMock()
    mock_service1.videos().insert.return_value = mock_insert1
    mock_insert1.next_chunk.side_effect = Exception("quotaExceeded on this token")
    
    mock_insert2 = MagicMock()
    mock_service2.videos().insert.return_value = mock_insert2
    mock_status = MagicMock()
    mock_status.progress.return_value = 1.0
    mock_insert2.next_chunk.return_value = (mock_status, {"id": "second_token_vid_id"})
    
    with patch("upload._validate_shorts_video", return_value=True), \
         patch("upload.get_authenticated_service") as mock_auth_service, \
         patch("upload.MediaFileUpload"), \
         patch("upload.Path.exists", return_value=True), \
         patch("json.load") as mock_json_load:
             
        mock_json_load.side_effect = [
            {"title": "Rotation Test"},
            [{"token": "token1"}, {"token": "token2"}]
        ]
        
        mock_auth_service.side_effect = [mock_service1, mock_service2]
        
        vid_id = upload_video(str(video_file), str(metadata_file), privacy="public")
        
        assert vid_id == "second_token_vid_id"
        assert mock_auth_service.call_count == 2
