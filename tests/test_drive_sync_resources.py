"""Real local media handles, fake Drive requests: Windows files must be releasable."""

from unittest.mock import Mock

import pytest

import sync


@pytest.mark.parametrize("fail", [False, True])
def test_drive_upload_closes_owned_media_even_when_request_retains_it(tmp_path, fail):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"SYNTHETIC upload fixture")
    service = Mock()
    if fail:
        service.files().create().execute.side_effect = RuntimeError("Test network failure")
        with pytest.raises(RuntimeError, match="Test network failure"):
            sync._upload_file(service, path, "test-folder")
    else:
        service.files().create().execute.return_value = {"id": "test-id"}
        assert sync._upload_file(service, path, "test-folder") == "test-id"
    media = service.files().create.call_args.kwargs["media_body"]
    assert media.stream().closed
    path.unlink()  # Real Windows deletion while the fake request still owns media.


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_db_backup_closes_owned_media_after_create_or_update(tmp_path, monkeypatch, existing, fail):
    path = tmp_path / "shorts_intelligence.db"
    path.write_bytes(b"SYNTHETIC DB bytes, not production analytics")
    service = Mock()
    service.files().list().execute.return_value = {"files": [{"id": "test-id"}] if existing else []}
    method = service.files().update if existing else service.files().create
    if fail:
        method().execute.side_effect = RuntimeError("Test request failure")
    monkeypatch.setattr(sync, "DB_FILES", [str(path)])
    monkeypatch.setattr("utils.drive_auth.get_drive_service", lambda: service)
    monkeypatch.setattr("utils.drive_auth.find_or_create_folder", lambda *_: "test-folder")
    sync.sync_db_to_drive()
    media = method.call_args.kwargs["media_body"]
    assert media.stream().closed
    path.unlink()
