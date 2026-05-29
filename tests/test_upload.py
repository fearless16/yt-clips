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



# ─── feat/youtube-upload reliability ──────────────────────────────────────────

import upload as upload_mod
from upload import _truncate_bytes, _limit_youtube_tags, _assignable_category_id


def test_truncate_bytes_respects_byte_limit_with_multibyte():
    # Devanagari + emoji are multi-byte; char slicing would overshoot 5000 bytes.
    text = "क" * 3000 + "🔥" * 100  # well over 5000 bytes
    out = _truncate_bytes(text, 5000)
    assert len(out.encode("utf-8")) <= 5000
    # Valid decode (cut on a boundary, no replacement chars).
    assert "\ufffd" not in out


def test_truncate_bytes_passthrough_when_small():
    assert _truncate_bytes("hello", 5000) == "hello"


def test_limit_tags_accounts_for_quote_overhead():
    # Multi-word tags incur +2 (quotes). 20 tags of "kohli six" (=9 +2 +1 sep).
    tags = ["kohli six"] * 100
    out = _limit_youtube_tags(tags, max_chars=480)
    # Budget must include quote overhead — total stays within the real cap.
    total = sum(len(t) + (2 if " " in t else 0) for t in out) + (len(out) - 1)
    assert total <= 480


def test_assignable_category_id_valid_and_fallback():
    upload_mod._CATEGORY_CACHE.clear()
    svc = MagicMock()
    svc.videoCategories().list().execute.return_value = {
        "items": [
            {"id": "17", "snippet": {"assignable": True}},   # Sports
            {"id": "24", "snippet": {"assignable": True}},   # Entertainment
            {"id": "29", "snippet": {"assignable": False}},  # Nonprofits (not assignable)
        ]
    }
    assert _assignable_category_id(svc, "17", region="IN") == "17"
    upload_mod._CATEGORY_CACHE.clear()
    # Non-assignable desired -> falls back to Sports(17).
    assert _assignable_category_id(svc, "29", region="IN") == "17"


def test_assignable_category_id_api_error_returns_desired():
    upload_mod._CATEGORY_CACHE.clear()
    svc = MagicMock()
    svc.videoCategories().list().execute.side_effect = RuntimeError("api down")
    assert _assignable_category_id(svc, "17", region="IN") == "17"


def test_upload_body_synthetic_media_from_metadata(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"x")
    metadata_file = tmp_path / "m.json"
    metadata_file.write_text(json.dumps({
        "title": "AI Edit", "description": "d",
        "contains_synthetic_media": True,
    }))

    mock_service = MagicMock()
    mock_insert = MagicMock()
    mock_service.videos().insert.return_value = mock_insert
    mock_status = MagicMock(); mock_status.progress.return_value = 1.0
    mock_insert.next_chunk.return_value = (mock_status, {"id": "vid"})
    # categoryId validation returns a clean assignable set.
    mock_service.videoCategories().list().execute.return_value = {
        "items": [{"id": "17", "snippet": {"assignable": True}}]
    }
    upload_mod._CATEGORY_CACHE.clear()

    captured = {}
    with patch("upload._validate_shorts_video", return_value=True), \
         patch("upload.get_authenticated_service", return_value=mock_service), \
         patch("upload.MediaFileUpload") as mock_media, \
         patch("upload.Path.exists", return_value=True):
        upload_video(str(video_file), str(metadata_file), privacy="public")

    _, kwargs = mock_service.videos().insert.call_args
    assert kwargs["body"]["status"]["containsSyntheticMedia"] is True
    # Finite, resumable chunk size (not the old chunksize=-1 single-shot).
    # The FIRST MediaFileUpload call is the video upload (a later call uploads
    # the thumbnail positionally with no kwargs), so inspect call_args_list[0].
    _, mk = mock_media.call_args_list[0]
    assert mk.get("resumable") is True
    assert mk.get("chunksize", -1) > 0
