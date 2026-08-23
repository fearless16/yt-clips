"""tests/automation/test_instagram_upload_post.py — Upload-Post adapter."""

import json
from unittest.mock import MagicMock

import pytest

from automation.instagram.upload_post_client import (
    KILL_SWITCH_ENV,
    BASE_URL,
    UploadPostClient,
    UploadPostError,
    _idempotency_key,
)


def _client():
    return UploadPostClient("key-123", "cricket-profile")


def _resp(payload, status_code=200, content_type="application/json"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.headers = {"content-type": content_type}
    if not isinstance(payload, dict):
        raise ValueError()
    return resp


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")


class TestGuards:
    def test_requires_api_key(self):
        with pytest.raises(ValueError):
            UploadPostClient("", "p")

    def test_requires_profile(self):
        with pytest.raises(ValueError):
            UploadPostClient("k", "")

    def test_kill_switch_blocks(self, monkeypatch):
        monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
        c = _client()
        with pytest.raises(RuntimeError):
            c.publish_reel(__file__, "cap")
        with pytest.raises(RuntimeError):
            c.verify_credentials()


class TestIdempotencyKey:
    def test_stable_across_calls(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"abc")
        assert _idempotency_key(clip) == _idempotency_key(clip)

    def test_changes_when_file_changes(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"abc")
        first = _idempotency_key(clip)
        import os
        os.utime(clip, (2000000000, 2000000000))
        assert _idempotency_key(clip) != first


class TestSyncPublish:
    def test_happy_path_posts_multipart(self, tmp_path, live):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00\x01vid")
        session = MagicMock()
        session.post.return_value = _resp({
            "success": True,
            "results": {"instagram": {
                "success": True,
                "url": "https://instagram.com/reel/ABC/",
                "container_id": "1789x",
            }},
        })
        client = UploadPostClient("key", "prof", session=session)
        result = client.publish_reel(clip, "Hook line #IndvSl")

        assert result == {
            "media_id": "1789x",
            "permalink": "https://instagram.com/reel/ABC/"}
        kwargs = session.post.call_args.kwargs
        assert session.post.call_args.args[0] == f"{BASE_URL}/api/upload"
        assert kwargs["headers"]["Authorization"] == "Apikey key"
        assert kwargs["headers"]["Idempotency-Key"] == \
            _idempotency_key(clip)
        data = kwargs["data"]
        assert data["user"] == "prof"
        assert data["platform[]"] == "instagram"
        assert data["media_type"] == "REELS"
        assert data["title"].startswith("Hook line")

    def test_audio_name_passed(self, tmp_path, live):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"x")
        session = MagicMock()
        session.post.return_value = _resp({
            "success": True,
            "results": {"instagram": {"success": True,
                                      "post_id": "180z",
                                      "url": "https://ig.com/p/x/"}},
        })
        client = UploadPostClient("key", "prof", session=session)
        client.publish_reel(clip, "cap", audio_name="Moment – Chan")
        assert session.post.call_args.kwargs["data"]["audio_name"] == \
            "Moment – Chan"

    def test_ig_failure_inside_results_raises(self, tmp_path, live):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"x")
        session = MagicMock()
        session.post.return_value = _resp({
            "success": True,
            "results": {"instagram": {"success": False,
                                      "error": "Expired token"}},
        })
        with pytest.raises(UploadPostError, match="Expired token"):
            UploadPostClient("k", "p",
                             session=session).publish_reel(clip, "c")


class TestAuthAndQuotaErrors:
    def test_401_raises(self, tmp_path, live):
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"x")
        session = MagicMock()
        session.post.return_value = _resp(
            {"success": False, "message": "Invalid or expired token"},
            status_code=401)
        with pytest.raises(UploadPostError, match="API key"):
            UploadPostClient("bad", "p", session=session) \
                .publish_reel(clip, "c")

    def test_429_quota_raises_with_usage(self, tmp_path, live):
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"x")
        session = MagicMock()
        session.post.return_value = _resp({
            "success": False,
            "message": "This upload would exceed your monthly limit.",
            "usage": {"count": 10, "limit": 10},
        }, status_code=429)
        with pytest.raises(UploadPostError, match="10/10"):
            UploadPostClient("k", "p", session=session) \
                .publish_reel(clip, "c")


class TestAsyncFallback:
    def test_polls_until_instagram_result(self, tmp_path, live,
                                          monkeypatch):
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"x")
        session = MagicMock()
        session.post.return_value = _resp({
            "success": True,
            "message": "Upload initiated successfully in background.",
            "request_id": "req1",
            "total_platforms": 1,
        })
        session.get.side_effect = [
            _resp({"status": "processing"}),
            _resp({"results": {"instagram": {
                "success": True, "url": "https://ig/reel/Z/",
                "post_id": "181q"}}}),
        ]
        sleeps = []
        monkeypatch.setattr(
            "automation.instagram.upload_post_client.time.sleep",
            lambda s: sleeps.append(s))
        client = UploadPostClient("k", "p", session=session)
        result = client.publish_reel(clip, "c", poll_interval_s=0.01)
        assert result["media_id"] == "181q"
        assert len(sleeps) == 1
        poll_kwargs = session.get.call_args.kwargs
        assert poll_kwargs["params"] == {"request_id": "req1"}

    def test_poll_timeout_raises(self, tmp_path, live, monkeypatch):
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"x")
        session = MagicMock()
        session.post.return_value = _resp({
            "success": True, "request_id": "r2"})
        session.get.return_value = _resp({"status": "processing"})
        monkeypatch.setattr(
            "automation.instagram.upload_post_client.time.sleep",
            lambda s: None)
        with pytest.raises(UploadPostError, match="timed out"):
            UploadPostClient("k", "p", session=session).publish_reel(
                clip, "c", poll_interval_s=0, timeout_s=0.5)

    def test_terminal_failed_status(self, tmp_path, live, monkeypatch):
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"x")
        session = MagicMock()
        session.post.return_value = _resp({
            "success": True, "request_id": "r3"})
        session.get.return_value = _resp({
            "status": "failed", "error": "IG rejected the media"})
        monkeypatch.setattr(
            "automation.instagram.upload_post_client.time.sleep",
            lambda s: None)
        with pytest.raises(UploadPostError, match="rejected the media"):
            UploadPostClient("k", "p", session=session).publish_reel(
                clip, "c", poll_interval_s=0)


class TestVerifyCredentials:
    def test_me_ok(self, live):
        session = MagicMock()
        session.get.return_value = _resp({
            "success": True, "email": "e@x.com", "plan": "Default"})
        out = UploadPostClient("k", "p", session=session) \
            .verify_credentials()
        assert out["plan"] == "Default"

    def test_me_invalid(self, live):
        session = MagicMock()
        session.get.return_value = _resp(
            {"success": False, "message": "Invalid"}, status_code=401)
        with pytest.raises(UploadPostError, match="credential check"):
            UploadPostClient("k", "p", session=session).verify_credentials()


class TestAsyncListShape:
    """Status endpoint returns results as a LIST of platform entries."""

    def test_list_results_recognized(self, tmp_path, live, monkeypatch):
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"x")
        session = MagicMock()
        session.post.return_value = _resp({
            "success": True, "request_id": "r9"})
        session.get.return_value = _resp({
            "status": "completed", "completed": 1, "total": 1,
            "results": [{"platform": "instagram", "success": True,
                         "message": "Published",
                         "post_url": "https://instagram.com/reel/LIVE1/",
                         "post_id": "182live"}],
        })
        monkeypatch.setattr(
            "automation.instagram.upload_post_client.time.sleep",
            lambda s: None)
        out = UploadPostClient("k", "p", session=session).publish_reel(
            clip, "c", poll_interval_s=0)
        assert out == {"media_id": "182live",
                       "permalink": "https://instagram.com/reel/LIVE1/"}

    def test_success_without_ids_accepted(self, tmp_path, live):
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"x")
        session = MagicMock()
        session.post.return_value = _resp({
            "success": True,
            "results": {"instagram": {"success": True,
                                      "message": "Published"}},
        })
        out = UploadPostClient("k", "p", session=session).publish_reel(
            clip, "c", poll_interval_s=0)
        assert out["permalink"] == ""
        assert out["media_id"].startswith("up_")
