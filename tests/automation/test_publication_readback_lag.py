"""C-UPLOAD-READBACK: videos.list may lag videos.update; poll before failing."""
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from automation import youtube_delivery as yd

VIDEO_ID = "lagVideo001"
PUBLISH_AT = (datetime.now(timezone.utc) + timedelta(days=1)).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _service(reads):
    """Status reads return `reads` in order (last one repeats)."""
    calls = {"n": 0}

    def listed(**_):
        status = reads[min(calls["n"], len(reads) - 1)]
        calls["n"] += 1
        return Mock(execute=Mock(return_value={"items": [
            {"id": VIDEO_ID, "status": dict(status), "snippet": {"channelId": "ch"}}]}))

    videos = Mock()
    videos.list.side_effect = listed
    videos.update.return_value = Mock(execute=Mock(return_value={"id": VIDEO_ID}))
    return Mock(videos=Mock(return_value=videos)), calls


def _receipt():
    return {"video_id": VIDEO_ID, "channel_id": "ch", "status": "THUMBNAIL_VERIFIED",
            "intent": {"privacy": "scheduled", "publish_at": PUBLISH_AT, "notify_subscribers": False}}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(yd.time, "sleep", sleeps.append)
    return sleeps


STAGED = {"privacyStatus": "private"}
SCHEDULED = {"privacyStatus": "private", "publishAt": PUBLISH_AT}


def test_lagging_readback_verifies_after_retry(tmp_path, no_sleep):
    # pre-update read, then two stale post-update reads, then the scheduled state
    service, calls = _service([STAGED, STAGED, STAGED, SCHEDULED])
    result = yd.finish_publication(service, tmp_path / "clip1.mp4", _receipt())
    assert result["status"] == "VERIFIED"
    assert calls["n"] == 4 and len(no_sleep) == 2
    service.videos().update.assert_called_once()


def test_persistent_mismatch_still_raises_bounded(tmp_path, no_sleep):
    service, calls = _service([STAGED])
    with pytest.raises(ValueError, match="does not yet match.*privacy='private'"):
        yd.finish_publication(service, tmp_path / "clip1.mp4", _receipt())
    assert calls["n"] == 1 + yd.PUBLICATION_READBACK_ATTEMPTS
    assert len(no_sleep) == yd.PUBLICATION_READBACK_ATTEMPTS - 1
    service.videos().update.assert_called_once()
