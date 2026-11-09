"""Real local media handles, stubbed YouTube service; never a network upload."""
import json
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("outcome", ["success", "insert-error", "execute-error", "thumbnail-error"])
def test_upload_closes_video_and_thumbnail_handles(tmp_path, monkeypatch, outcome):
    import upload
    from PIL import Image

    monkeypatch.setitem(upload.cfg, "release_review", {"required": False})
    monkeypatch.setitem(upload.cfg, "thumbnail", {"enabled": True})
    monkeypatch.setattr("automation.youtube_delivery.wait_for_processing", lambda *_: None)
    monkeypatch.setattr("automation.youtube_delivery.verify_thumbnail", lambda *_: {"method": "stubbed CDN"})
    monkeypatch.setattr(upload, "_validate_shorts_video", lambda *_: True)
    monkeypatch.setattr(upload, "_assignable_category_id", lambda *_a, **_k: "17")
    service = Mock()
    service.videos().insert().execute.return_value = {"id": "fixture-id"}
    from tests.youtube_fakes import add_video_status_api
    add_video_status_api(service, "fixture-id", upload.cfg["youtube"].get("channel_id"))
    service.thumbnails().set().execute.return_value = {
        "items": [{"shortsAspectRatio": {"url": "https://i.ytimg.com/vi/fixture-id/custom.jpg"}}]}
    if outcome == "insert-error":
        service.videos().insert.side_effect = RuntimeError("fixture insert")
    if outcome == "execute-error":
        service.videos().insert().execute.side_effect = RuntimeError("fixture execute")
    if outcome == "thumbnail-error":
        service.thumbnails().set().execute.side_effect = RuntimeError("fixture thumbnail")
    monkeypatch.setattr(upload, "get_authenticated_service", lambda: service)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"synthetic video bytes")
    metadata = tmp_path / "clip_metadata.json"
    metadata.write_text(json.dumps({"title": "Ground bhi dekho", "description": "Grounded fixture", "tags": []}))
    cover = tmp_path / "clip_thumb.jpg"
    Image.new("RGB", (360, 640), "navy").save(cover, format="JPEG")
    media_type = upload.MediaFileUpload
    bodies = []

    def media(*args, **kwargs):
        body = media_type(*args, **kwargs)
        bodies.append(body)
        return body

    monkeypatch.setattr(upload, "MediaFileUpload", media)
    try:
        if outcome == "thumbnail-error":
            with pytest.raises(upload.DeliveryIncompleteError) as exc:
                upload.upload_video(str(video), str(metadata), privacy="private")
            assert exc.value.video_id == "fixture-id"
        else:
            result = upload.upload_video(str(video), str(metadata), privacy="private")
            assert result == (None if outcome in {"insert-error", "execute-error"} else "fixture-id")
        assert bodies and all(body.stream().closed for body in bodies)
        video.unlink()  # Windows must not retain an SDK-owned lock.
        cover.unlink()
    finally:
        for body in bodies:
            body.stream().close()
