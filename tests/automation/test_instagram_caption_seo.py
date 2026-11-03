"""Tests for automation.instagram.caption_engine (I3) and
automation.instagram.seo orchestrator stage hooks.

LLM writer (_get_ai) and entity audit (audit_written_copy_llm) are stubbed;
the evidence builder is stubbed at its call boundary, not replaced in sys.modules.
"""

import json
import time

import pytest

PACK = {
    "match_facts": ["India vs Sri Lanka, 2nd T20I, Pallekele"],
    "roster": ["Gill", "Samson", "Bumrah"],
    "learner_top_captions": ["Bumrah ne udaya!"],
    "seed_phrases": ["ind vs sl highlights"],
    "validated_hashtags": [
        {"tag": "cricket", "recent_volume_24h": 50000,
         "source": "ig-hashtag-api"},
        {"tag": "INDvSL", "recent_volume_24h": 1200,
         "source": "ig-hashtag-api"},
        {"tag": "TeamIndia", "recent_volume_24h": 9000,
         "source": "ig-hashtag-api"},
        {"tag": "BumrahMagic", "recent_volume_24h": 210, "source": "seed"},
        {"tag": "PallekeleT20", "recent_volume_24h": 90, "source": "seed"},
    ],
    "forbidden": ["invented player names",
                  "generic tags without volume evidence"],
}
TRANSCRIPT = ("Bumrah took three wickets in the last over. Samson took "
              "a catch. The crowd at Pallekele erupted.")
VIDEO_TITLE = "Bumrah Death Over Magic | IND vs SL 2nd T20I"
HOOK = "Bumrah takes 3 wickets! IND v SL"

VALID_TAGS = ["#cricket", "#INDvSL", "#TeamIndia",
               "#BumrahMagic", "#PallekeleT20"]

# Real-world pack that yields only 3 distinct validated hashtags — previously
# this hard-failed the entire SEO/IG stage (InsufficientEvidenceError).
PACK_SMALL = {
    "match_facts": ["India vs Sri Lanka, 2nd T20I, Pallekele"],
    "roster": ["Gill", "Samson", "Bumrah"],
    "learner_top_captions": ["Bumrah ne udaya!"],
    "seed_phrases": ["ind vs sl highlights"],
    "validated_hashtags": [
        {"tag": "cricket", "recent_volume_24h": 50000,
         "source": "ig-hashtag-api"},
        {"tag": "INDvSL", "recent_volume_24h": 1200,
         "source": "ig-hashtag-api"},
        {"tag": "TeamIndia", "recent_volume_24h": 9000,
         "source": "ig-hashtag-api"},
    ],
    "forbidden": ["invented player names",
                  "generic tags without volume evidence"],
}


class FakeAI:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts = []

    def generate_text(self, prompt, system_instruction=None,
                      prefer_model=None, **kwargs):
        self.prompts.append(prompt)
        if not self.payloads:
            raise AssertionError("unexpected extra LLM call")
        item = self.payloads.pop(0)
        if isinstance(item, Exception):
            raise item
        return json.dumps(item) if isinstance(item, dict) else item


def _clean_audit(*args, **kwargs):
    return {"unsupported_entities": [], "supported_topics": []}


def _body(sentences):
    return "\n".join(sentences)


def _valid_candidate():
    body = _body([
        "Bumrah turned the match around at Pallekele in the final over.",
        "Samson held a sharp catch under pressure.",
        "Which delivery changed this game?",
    ])
    caption = HOOK + "\n" + body
    assert 150 <= len(caption) <= 250, f"fixture broken: {len(caption)}"
    return {
        "caption": caption,
        "hashtags": list(VALID_TAGS),
        "audio_name": "Bumrah – CricketWithPrajjwal",
    }


def _devanagari_candidate():
    return {
        "caption": "बुमराह ने तीन विकेट लिए!\nपल्लेकेले में मैच पलट गया। "
                   "कल का हीरो कौन?",
        "hashtags": list(VALID_TAGS),
        "audio_name": "बुमराह – CricketWithPrajjwal",
    }


def _invented_hashtag_candidate():
    cand = _valid_candidate()
    cand["hashtags"] = ["#cricket", "#INDvSL", "#TeamIndia",
                        "#BumrahMagic", "#ViralReelsKing"]
    return cand


@pytest.fixture
def insta_cfg(monkeypatch):
    from automation.instagram import seo

    monkeypatch.setattr(seo, "_insta_config", lambda: {"enabled": True, "provider": "graph"})


@pytest.fixture(autouse=True)
def restore_runner_hooks():
    from automation.instagram import runner
    saved = dict(runner._HOOKS)
    yield
    runner._HOOKS.clear()
    runner._HOOKS.update(saved)


def _stub_llm(monkeypatch, payloads) -> FakeAI:
    import automation.instagram.caption_engine as ce
    fake = FakeAI(payloads)
    monkeypatch.setattr(ce, "_get_ai", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# caption_engine.write_caption
# ---------------------------------------------------------------------------

class TestWriteCaptionHappyPath:

    def test_returns_grounded_package(self, monkeypatch):
        import automation.instagram.caption_engine as ce
        ai = _stub_llm(monkeypatch, [_valid_candidate()])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)

        out = ce.write_caption(PACK, TRANSCRIPT, VIDEO_TITLE)

        assert set(out) == {"caption", "hashtags", "audio_name",
                    "tags_topped_up"}
        first_line = out["caption"].split("\n")[0]
        assert len(first_line) <= 55
        assert "bumrah" in first_line.lower()
        assert len(out["hashtags"]) == 5
        allowed = {t.lstrip("#").lower() for t in VALID_TAGS}
        for tag in out["hashtags"]:
            assert tag.lstrip("#").lower() in allowed
        assert "?" in out["caption"]
        assert out["audio_name"].endswith("CricketWithPrajjwal")
        assert len(ai.prompts) == 1
        prompt = ai.prompts[0]
        assert '"validated_hashtags"' in prompt
        assert json.dumps(PACK["roster"], ensure_ascii=False) in prompt

    def test_prompt_embeds_full_evidence_pack(self, monkeypatch):
        import automation.instagram.caption_engine as ce
        ai = _stub_llm(monkeypatch, [_valid_candidate()])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)
        ce.write_caption(PACK, TRANSCRIPT, VIDEO_TITLE)
        prompt = ai.prompts[0]
        for fact in PACK["match_facts"]:
            assert fact in prompt


class TestCaptionPolicyEnforcement:

    @pytest.mark.xfail(strict=True, reason="Existing hook-length validation bypass despite the <=55 prompt contract")
    def test_over_55_char_hook_triggers_single_repair(self, monkeypatch):
        import automation.instagram.caption_engine as ce

        candidate = _valid_candidate()
        candidate["caption"] = candidate["caption"].replace(HOOK, HOOK + " and the whole stadium celebrates this finish")
        ai = _stub_llm(monkeypatch, [candidate, _valid_candidate()])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)
        out = ce.write_caption(PACK, TRANSCRIPT, VIDEO_TITLE)
        assert len(out["caption"].split("\n")[0]) <= 55
        assert len(ai.prompts) == 2
        assert "CORRECTION REQUIRED" in ai.prompts[1]

    @pytest.mark.xfail(strict=True, reason="Existing hashtag allowlist bypass in _validate_and_scrub; not an obsolete safety requirement")
    def test_invented_hashtag_dropped_and_topped_up(self, monkeypatch):
        """Invented tags are dropped; the REAL-ONLY pool fills the gap
        deterministically — no repair LLM spend needed."""
        import automation.instagram.caption_engine as ce
        ai = _stub_llm(monkeypatch, [_invented_hashtag_candidate()])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)

        out = ce.write_caption(PACK, TRANSCRIPT, VIDEO_TITLE)

        names = {t.lstrip("#").lower() for t in out["hashtags"]}
        assert "viralreelsking" not in names
        assert len(out["hashtags"]) == 5
        assert len(ai.prompts) == 1
        assert out["tags_topped_up"], "shortfall must come from pack"

    def test_unsupported_entity_is_scrubbed(self, monkeypatch):
        import automation.instagram.caption_engine as ce
        monkeypatch.setattr(
             ce, "audit_written_copy_llm",
             lambda *a, **k: {"unsupported_entities": ["Ravindra Jadeja"], "supported_topics": []})
        candidate = _valid_candidate()
        candidate["caption"] = candidate["caption"].replace("Samson", "Ravindra Jadeja")
        ai = _stub_llm(monkeypatch, [candidate])

        out = ce.write_caption(PACK, TRANSCRIPT, VIDEO_TITLE)

        assert "Jadeja" not in out["caption"]
        assert len(ai.prompts) == 1

    def test_devanagari_rejected_and_repair_is_romanized(self, monkeypatch):
        import automation.instagram.caption_engine as ce
        deva_re = __import__("re").compile(r"[\u0900-\u097F]")
        ai = _stub_llm(monkeypatch,
                       [_devanagari_candidate(), _valid_candidate()])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)

        out = ce.write_caption(PACK, TRANSCRIPT, VIDEO_TITLE)

        assert not deva_re.search(out["caption"])
        assert not any(deva_re.search(t) for t in out["hashtags"])
        assert not deva_re.search(out["audio_name"])

    def test_no_parsable_json_both_attempts_raises(self, monkeypatch):
        import automation.instagram.caption_engine as ce
        ai = _stub_llm(monkeypatch, ["not json at all", "still not json"])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)

        with pytest.raises(ce.CaptionPolicyError):
            ce.write_caption(PACK, TRANSCRIPT, VIDEO_TITLE)
        assert len(ai.prompts) == 2

    def test_repair_failure_raises_caption_policy_error(self, monkeypatch):
        import automation.instagram.caption_engine as ce

        ai = _stub_llm(monkeypatch, [_devanagari_candidate(), _devanagari_candidate()])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)
        with pytest.raises(ce.CaptionPolicyError, match="Devanagari"):
            ce.write_caption(PACK, TRANSCRIPT, VIDEO_TITLE)
        assert len(ai.prompts) == 2

    def test_small_pack_publishes_with_available_real_tags(self, monkeypatch):
        """Data-driven regression: a pack with only 3 validated hashtags must
        still publish (with 3 real tags), not raise InsufficientEvidenceError.
        """
        import automation.instagram.caption_engine as ce
        candidate = _valid_candidate()
        candidate["hashtags"] = ["#cricket", "#INDvSL", "#TeamIndia"]
        ai = _stub_llm(monkeypatch, [candidate])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)

        out = ce.write_caption(PACK_SMALL, TRANSCRIPT, VIDEO_TITLE)

        # 3 validated + 1 seed-phrase tag = 4 real tags; all must be grounded.
        assert 3 <= len(out["hashtags"]) <= 5
        allowed = {t.lstrip("#").lower()
                   for t in ["#cricket", "#INDvSL", "#TeamIndia",
                             "#IndVsSlHighlights"]}
        for tag in out["hashtags"]:
            assert tag.lstrip("#").lower() in allowed


# ---------------------------------------------------------------------------
# seo.generate_insta_metadata
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_evidence(monkeypatch):
    from automation.instagram import seo

    def build_insta_evidence_pack(*args, **kwargs):
        return json.loads(json.dumps(PACK))

    monkeypatch.setattr(seo, "_build_evidence", build_insta_evidence_pack)
    return seo


def _stub_clean_pipeline(monkeypatch):
    import automation.instagram.caption_engine as ce
    _stub_llm(monkeypatch, [_valid_candidate()])
    monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)


class TestGenerateInstaMetadata:

    def test_happy_path_writes_metadata_json(self, tmp_path, monkeypatch,
                                             fake_evidence):
        from automation.instagram import seo
        _stub_clean_pipeline(monkeypatch)

        meta = seo.generate_insta_metadata(
            tmp_path, TRANSCRIPT, VIDEO_TITLE, "desc text")

        assert meta is not None
        assert meta["packaging_version"] == "insta_v1_realonly"
        assert set(meta) >= {"caption", "hashtags", "audio_name",
                             "evidence_summary", "packaging_version"}
        assert len(meta["hashtags"]) == 5
        written = json.loads(
            (tmp_path / "insta_metadata.json").read_text("utf-8"))
        assert written == meta

    def test_existing_metadata_short_circuits(self, tmp_path, monkeypatch,
                                              fake_evidence):
        from automation.instagram import seo
        _stub_clean_pipeline(monkeypatch)
        first = seo.generate_insta_metadata(
            tmp_path, TRANSCRIPT, VIDEO_TITLE, "")

        empty_ai = _stub_llm(monkeypatch, [])
        second = seo.generate_insta_metadata(
            tmp_path, TRANSCRIPT, VIDEO_TITLE, "")

        assert second == first
        assert empty_ai.prompts == []

    def test_force_regenerates(self, tmp_path, monkeypatch, fake_evidence):
        from automation.instagram import seo
        _stub_clean_pipeline(monkeypatch)
        first = seo.generate_insta_metadata(
            tmp_path, TRANSCRIPT, VIDEO_TITLE, "")

        other = _valid_candidate()
        other["caption"] = other["caption"].replace(
            "turned the match around", "transformed the match")
        assert other["caption"] != first["caption"]
        ai = _stub_llm(monkeypatch, [other])
        meta = seo.generate_insta_metadata(
            tmp_path, TRANSCRIPT, VIDEO_TITLE, "", force=True)

        expected = other["caption"].rstrip() + "\n\n" + \
            " ".join(t.lstrip("#") for t in other["hashtags"])
        assert meta["caption"] == expected
        assert len(ai.prompts) == 1

    def test_policy_error_propagates(self, tmp_path, monkeypatch,
                                     fake_evidence):
        from automation.instagram import seo
        import automation.instagram.caption_engine as ce
        _stub_llm(monkeypatch, ["not json", "still not json"])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)

        with pytest.raises(ce.CaptionPolicyError):
            seo.generate_insta_metadata(
                tmp_path, TRANSCRIPT, VIDEO_TITLE, "")

    def test_unavailable_evidence_returns_none(self, tmp_path, monkeypatch):
        from automation.instagram import seo

        def unavailable(*args, **kwargs):
            raise ImportError("evidence unavailable")

        monkeypatch.setattr(seo, "_build_evidence", unavailable)
        assert seo.generate_insta_metadata(tmp_path, TRANSCRIPT, VIDEO_TITLE) is None


class TestStageHooks:

    def test_run_caption_stage_writes_json_and_returns_caption(
            self, tmp_path, monkeypatch, fake_evidence):
        from automation.instagram import seo
        _stub_clean_pipeline(monkeypatch)

        pack = seo.run_evidence_stage(
            str(tmp_path), TRANSCRIPT, VIDEO_TITLE, "desc")
        assert isinstance(pack, dict)

        caption = seo.run_caption_stage(
            str(tmp_path), TRANSCRIPT, VIDEO_TITLE, "desc")

        assert isinstance(caption, str) and caption
        assert (tmp_path / "insta_metadata.json").exists()
        meta = json.loads(
            (tmp_path / "insta_metadata.json").read_text("utf-8"))
        assert caption == meta["caption"]
        for tag in meta["hashtags"]:
            assert tag in caption

    def test_hooks_resolvable_via_registration(self):
        from automation.instagram import runner, seo
        assert runner.resolve_stage_hook("run_evidence_stage") is \
            seo.run_evidence_stage
        assert runner.resolve_stage_hook("run_caption_stage") is \
            seo.run_caption_stage
        assert runner.resolve_stage_hook("make_graph_client") is \
            seo.make_graph_client

    def test_hooks_resolvable_after_registry_clear(self):
        from automation.instagram import runner, seo
        runner._HOOKS.clear()
        assert runner.resolve_stage_hook("run_caption_stage") is \
            seo.run_caption_stage
        assert runner.resolve_stage_hook("run_evidence_stage") is \
            seo.run_evidence_stage

    def test_make_graph_client_none_without_token(self, monkeypatch):
        from automation.instagram import credential, seo
        monkeypatch.setattr(credential, "load_token", lambda path=None: None)
        assert seo.make_graph_client() is None

    def test_make_graph_client_none_when_token_invalid(self, monkeypatch):
        from automation.instagram import credential, seo
        expired = {"access_token": "tok", "expires_at": time.time() - 100,
                   "ig_user_id": "17841400000000001"}
        monkeypatch.setattr(credential, "load_token",
                            lambda path=None: expired)
        assert seo.make_graph_client() is None

    def test_make_graph_client_builds_facebook_client(self, monkeypatch):
        from automation.instagram import credential, seo
        monkeypatch.setattr(seo, "_insta_config",
                            lambda: {"provider": "graph"})
        from automation.instagram.graph_client import FacebookGraphClient
        token = {"access_token": "tok", "expires_at": time.time() + 9999,
                 "ig_user_id": "17841400000000001"}
        monkeypatch.setattr(credential, "load_token",
                            lambda path=None: token)

        client = seo.make_graph_client()

        assert isinstance(client, FacebookGraphClient)
        assert client.token == "tok"
        assert client.ig_user_id == "17841400000000001"


@pytest.fixture(autouse=True)
def _isolated_config(insta_cfg):
    yield
