"""Contract for per-clip packaging (automation/seo/seo.py, cancer register C4)."""
import json

import pytest

from automation.artifacts import atomic_json
from automation.seo import seo

TRANSCRIPT = "Arre Kohli ne kya six maara yaar, seedha stadium ke bahar! India ab jeet jayega."
TITLE = "India vs New Zealand live commentary"


class FakeAI:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def generate_json(self, prompt, system, **kwargs):
        self.prompts.append(json.loads(prompt))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer)


def good(**overrides):
    return {"title": "Kohli ka six seedha stadium ke bahar 🏏", "cover_text": "Kohli Six Gone Big",
            "description": "Kohli smashes a huge six and the commentary loses it. #INDvsNZ",
            "tags": ["Kohli six", "India vs New Zealand", "Kohli six", "cricket reaction"],
            "hashtags": ["#INDvsNZ", "Kohli"], **overrides}


@pytest.fixture
def ai(monkeypatch):
    def install(*answers):
        fake = FakeAI(*answers)
        monkeypatch.setattr(seo, "_get_ai", lambda: fake)
        return fake
    return install


def test_valid_answer_is_repaired_not_blocked(ai):
    fake = ai(good())
    meta = seo.build_metadata("clip1", TRANSCRIPT, TITLE)
    assert meta["seo_method"] == "llm" and len(fake.prompts) == 1
    assert "🏏" not in meta["title"] and len(meta["title"]) <= 60
    assert meta["cover_text"] == "Kohli Six Gone"            # trimmed to 3 words
    assert meta["hashtags"][0] == "#Shorts" and len(meta["hashtags"]) <= 3
    assert meta["tags"].count("Kohli six") == 1                # deduped
    body, _, tail = meta["description"].rpartition("\n\n")
    assert "#" not in body and tail.startswith("#Shorts")
    assert meta["seo_inputs"]["transcript"] == TRANSCRIPT       # retry can rebuild


def test_ungrounded_name_gets_one_corrective_retry(ai):
    fake = ai(good(title="Rohit Sharma ka six"), good())
    meta = seo.build_metadata("clip1", TRANSCRIPT, TITLE)
    assert meta["seo_method"] == "llm"
    assert "rohit" in json.dumps(fake.prompts[1]["fix_these_problems_from_your_last_answer"]).lower()


def test_repeated_fact_problem_routes_to_template_for_owner(ai):
    ai(good(title="Rohit Sharma ka six"), good(title="Rohit Sharma again"))
    meta = seo.build_metadata("clip1", TRANSCRIPT, TITLE)
    assert meta["seo_method"] == "template" and "rohit" in meta["seo_error"].lower()
    assert "rohit" not in json.dumps(meta).lower().replace(meta["seo_error"].lower(), "")
    assert meta["title"] and 1 <= len(meta["cover_text"].split()) <= 3


def test_provider_failure_is_one_call_with_the_real_reason(ai):
    fake = ai(TimeoutError("read timed out"))
    meta = seo.build_metadata("clip1", TRANSCRIPT, TITLE)
    assert len(fake.prompts) == 1
    assert meta["seo_method"] == "template" and meta["seo_error"].startswith("TimeoutError")
    assert meta["metadata_valid"] is True and "India" in meta["title"]


def test_legacy_marker_always_resolves_and_receipts_are_never_repackaged(ai, tmp_path):
    ai(RuntimeError("provider down"))
    atomic_json(tmp_path / "clip1_seo_failed.json", {"transcript": TRANSCRIPT, "video_title": TITLE})
    atomic_json(tmp_path / "clip2_seo_failed.json", {"transcript": TRANSCRIPT})
    atomic_json(tmp_path / "clip2_delivery.json", {"status": "COMPLETE"})
    assert seo.retry_failed_seo(str(tmp_path)) == {"recovered": 0, "total": 1}
    assert not (tmp_path / "clip1_seo_failed.json").exists()       # A4: scripted exit
    assert json.loads((tmp_path / "clip1_metadata.json").read_text())["seo_method"] == "template"
    assert (tmp_path / "clip2_seo_failed.json").exists()             # uploaded clip untouched


def test_template_upgrades_when_the_provider_returns(ai, tmp_path):
    ai(RuntimeError("down"))
    seo.generate_seo_for_exported_clip("clip1", TRANSCRIPT, str(tmp_path), TITLE)
    ai(RuntimeError("still down"))
    assert seo.retry_failed_seo(str(tmp_path)) == {"recovered": 0, "total": 1}
    ai(good())
    assert seo.retry_failed_seo(str(tmp_path), include_templates=False) == {"recovered": 0, "total": 0}
    assert seo.retry_failed_seo(str(tmp_path)) == {"recovered": 1, "total": 1}
    assert json.loads((tmp_path / "clip1_metadata.json").read_text())["seo_method"] == "llm"


def test_template_metadata_needs_the_owner_in_release_review(ai, tmp_path):
    from automation.release_review import prepare_review, release_status, sha256
    clip, source = tmp_path / "clip1.mp4", tmp_path / "source.mp4"
    clip.write_bytes(b"clip")
    source.write_bytes(b"source")
    clip.with_name("clip1_thumb.jpg").write_bytes(b"cover")
    atomic_json(clip.with_name("clip1_qc.json"), {"status": "PASS", "media_sha256": sha256(clip)})
    atomic_json(clip.with_name("clip1_edit.json"), {
        "schema_version": 1, "source_path": str(source), "source_start": 1.0, "source_end": 20.0,
        "completeness": {"status": "PASS", "method": "text_only", "reason": "ok", "source_range": [1.0, 20.0]}})
    ai(RuntimeError("down"))
    seo.generate_seo_for_exported_clip("clip1", TRANSCRIPT, str(tmp_path), TITLE)
    payload = prepare_review(clip, source, mode="auto")
    assert payload["checks"]["metadata_valid"]["status"] == "NEEDS_OWNER"
    assert payload["approval"] is None and release_status(clip)[0] == "REVIEW"


def test_player_named_in_devanagari_speech_is_grounded(ai):
    fake = ai(good(title="Dhoni ka six seedha bahar", cover_text="Dhoni Six",
                   description="Dhoni ka bada six, commentary reaction.", tags=["Dhoni six"],
                   hashtags=["#Dhoni"]))
    meta = seo.build_metadata("clip1", "धोनी ने छक्का मारा यार, बुमराह भी हैरान", TITLE)
    assert meta["seo_method"] == "llm" and len(fake.prompts) == 1
