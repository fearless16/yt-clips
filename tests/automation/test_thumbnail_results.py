"""Thumbnail failures are per-clip results, never successful batch silence."""

import json

import pytest

import thumbnail


def seed_clip(directory, stem):
    clip = directory / f"{stem}.mp4"
    clip.write_bytes(b"video")
    clip.with_name(f"{stem}_metadata.json").write_text(json.dumps({
        "title": "Cricket discussion", "description": "Grounded context", "cover_text": "TEST MATCH",
    }))
    return clip


@pytest.mark.parametrize("outcome", ["exception", "false", "missing", "empty", "partial"])
def test_batch_reports_generation_failures_and_continues(tmp_path, monkeypatch, outcome):
    monkeypatch.setitem(thumbnail.cfg, "thumbnail", {"enabled": True})
    seed_clip(tmp_path, "bad")
    seed_clip(tmp_path, "good")

    def generate(self, video, metadata, output):
        from pathlib import Path
        if Path(video).stem == "bad":
            if outcome == "partial":
                Path(output).write_bytes(b"partial")
            if outcome in ("exception", "partial"):
                raise ValueError("Cover text needs review")
            if outcome == "empty":
                Path(output).touch()
            return outcome != "false"
        Path(output).write_bytes(b"fixture cover")
        return True

    monkeypatch.setattr(thumbnail.ThumbnailGenerator, "generate_for_clip", generate)
    result = thumbnail.process_all_thumbnails(str(tmp_path))
    assert len(result.failures) == 1
    assert "bad.mp4" in result.failures[0]
    assert result.generated == [tmp_path / "good_thumb.jpg"]
    assert not (tmp_path / "bad_thumb.jpg").exists()
    assert not list(tmp_path.glob(".bad-thumbnail-*"))


def test_existing_cover_and_failed_seo_are_preserved_as_distinct_outcomes(tmp_path, monkeypatch):
    monkeypatch.setitem(thumbnail.cfg, "thumbnail", {"enabled": True})
    clip = seed_clip(tmp_path, "existing")
    invalid = seed_clip(tmp_path, "failed")
    invalid.with_name("failed_seo_failed.json").write_text("{}")
    cover = clip.with_name("existing_thumb.jpg")
    cover.write_bytes(b"owner cover")
    monkeypatch.setattr(thumbnail.ThumbnailGenerator, "generate_for_clip",
                        lambda *_: pytest.fail("must not regenerate"))
    result = thumbnail.process_all_thumbnails(str(tmp_path))
    assert result.preserved == [cover]
    assert result.skipped == [invalid]
    assert result.failures == []
    assert cover.read_bytes() == b"owner cover"


def test_disabled_thumbnails_report_no_attempts(tmp_path, monkeypatch):
    monkeypatch.setitem(thumbnail.cfg, "thumbnail", {"enabled": False})
    seed_clip(tmp_path, "clip")
    monkeypatch.setattr(thumbnail, "ThumbnailGenerator", lambda: pytest.fail("disabled"))
    result = thumbnail.process_all_thumbnails(str(tmp_path))
    assert not (result.generated or result.preserved or result.skipped or result.failures)


def test_empty_existing_cover_is_reported_without_overwriting_owner_file(tmp_path, monkeypatch):
    monkeypatch.setitem(thumbnail.cfg, "thumbnail", {"enabled": True})
    seed_clip(tmp_path, "empty")
    cover = tmp_path / "empty_thumb.jpg"
    cover.touch()
    monkeypatch.setattr(thumbnail.ThumbnailGenerator, "generate_for_clip",
                        lambda *_: pytest.fail("explicit rebuild required"))
    result = thumbnail.process_all_thumbnails(str(tmp_path))
    assert result.generated == result.preserved == []
    assert len(result.failures) == 1
    assert "Existing cover is empty" in result.failures[0]
    assert cover.read_bytes() == b""


def test_generation_cannot_overwrite_cover_staged_while_it_was_rendering(tmp_path, monkeypatch):
    from PIL import Image
    from automation.youtube_delivery import stage_thumbnail, sha256
    monkeypatch.setitem(thumbnail.cfg, "thumbnail", {"enabled": True})
    clip = seed_clip(tmp_path, "portrait")
    approved = tmp_path / "approved.jpg"
    Image.new("RGB", (360, 640), "navy").save(approved)
    digest = sha256(approved)

    def generate(self, video, metadata, output):
        stage_thumbnail(video, approved, expected_sha256=digest)
        Image.new("RGB", (360, 640), "red").save(output)
        return True

    monkeypatch.setattr(thumbnail.ThumbnailGenerator, "generate_for_clip", generate)
    result = thumbnail.process_all_thumbnails(str(tmp_path))
    cover = tmp_path / "portrait_thumb.jpg"
    assert sha256(cover) == digest
    assert result.preserved == [cover]
    assert not result.generated
    assert not result.failures


def test_generator_cannot_install_cover_while_upload_holds_delivery_lock(tmp_path, monkeypatch):
    from automation.youtube_delivery import delivery_lock
    from pathlib import Path
    monkeypatch.setitem(thumbnail.cfg, "thumbnail", {"enabled": True})
    clip = seed_clip(tmp_path, "portrait")
    def generate(self, video, metadata, output):
        Path(output).write_bytes(b"fixture generated cover")
        return True
    monkeypatch.setattr(thumbnail.ThumbnailGenerator, "generate_for_clip", generate)
    with delivery_lock(clip):
        result = thumbnail.process_all_thumbnails(str(tmp_path))
    assert not (tmp_path / "portrait_thumb.jpg").exists()
    assert result.failures and "locked" in result.failures[0]


def test_discarded_source_cover_cannot_replace_approved_cover_evidence(tmp_path, monkeypatch):
    from pathlib import Path
    from PIL import Image, ImageDraw
    from automation.youtube_delivery import stage_thumbnail, sha256
    monkeypatch.setitem(thumbnail.cfg, "thumbnail", {"enabled": True})
    clip = seed_clip(tmp_path, "portrait")
    source = tmp_path / "source.raw"
    source.write_bytes(b"fixture source; decoder is stubbed")
    clip.with_name("portrait_edit.json").write_text(json.dumps({
        "source_path": str(source), "source_start": 0, "source_end": 10}))
    approved = tmp_path / "approved.jpg"
    Image.new("RGB", (360, 640), "navy").save(approved)
    plan = tmp_path / "portrait_cover.json"
    plan.write_text('{"owner":"approved cover evidence"}')
    original = plan.read_bytes()

    def extract(cmd, **kwargs):
        frame = Image.new("RGB", (320, 180), "gray")
        ImageDraw.Draw(frame).rectangle((30, 30, 120, 160), fill="white")
        frame.save(Path(cmd[-1]))
    def compose(*args):
        stage_thumbnail(clip, approved, expected_sha256=sha256(approved))
        return Image.new("RGB", (360, 640), "red"), {"fixture": True}
    monkeypatch.setattr(thumbnail.subprocess, "run", extract)
    monkeypatch.setattr("automation.cover_design.compose_cover", compose)
    result = thumbnail.process_all_thumbnails(str(tmp_path))
    assert result.preserved == [tmp_path / "portrait_thumb.jpg"]
    assert plan.read_bytes() == original
