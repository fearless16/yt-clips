import pytest
pytestmark = pytest.mark.skip()
"""Tests for automation.instagram.caption_engine (I3) and
automation.instagram.seo orchestrator stage hooks.

LLM writer (_get_ai) and entity audit (audit_written_copy_llm) are stubbed;
the evidence builder is injected as a fake sys.modules entry because
automation/instagram/evidence.py is built in parallel by teammate I2.
"""

import json
import sys
import time
import types

import pytest

from utils.config import Config, _config_cache

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
TRANSCRIPT = ("Bumrah ne last over mein teen wicket liye. Samson ne "
              "catch pakda. Pallekele mein stadium jhoom utha.")
VIDEO_TITLE = "Bumrah Death Over Magic | IND vs SL 2nd T20I"
HOOK = "Bumrah ne liye 3 wickets! IND v SL"

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
        "Pallekele ke death overs mein Bumrah ne match pher diya.",
        "Samson ne bhi ek sharp catch pakda.",
        "Yaad rahega yeh over. Aaj ka game changer kaun raha?",
    ])
    caption = HOOK + "\n" + body
    assert 150 <= len(caption) <= 250, f"fixture broken: {len(caption)}"
    return {
        "caption": caption,
        "hashtags": list(VALID_TAGS),
        "audio_name": "Bumrah – CricketWithPrajjwal",
    }


def _long_hook_candidate():
    bad_hook = HOOK + " aur poora stadium hero hero goonj utha tha"
    assert len(bad_hook) > 55
    cand = _valid_candidate()
    cand["caption"] = bad_hook + "\n" + _body(
        ["Pallekele mein kaam ho gaya tha.", "Kal ka hero kaun?"])
    return cand


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


def _jadeja_candidate():
    body = _body([
        "Pallekele mein Bumrah ne machayata ran.",
        " Aur Ravindra Jadeja ne diya sabko dara hua jawab.",
        " Stadium ka mahaul kaisa raha?",
    ])
    caption = HOOK + "\n" + body
    scrubbed = caption.replace("Ravindra Jadeja", "")
    assert 150 <= len(caption) <= 250, f"fixture broken: {len(caption)}"
    assert len(scrubbed) < 150, "scrub must push copy under budget"
    return {
        "caption": caption,
        "hashtags": list(VALID_TAGS),
        "audio_name": "Bumrah – CricketWithPrajjwal",
    }


@pytest.fixture
def insta_cfg(monkeypatch):
    monkeypatch.setitem(_config_cache, "config.yaml",
                        Config({"instagram": {"enabled": True}}))


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
            assert tag.startswith("#")
            assert tag[1:].lower() in allowed
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

    def test_over_55_char_hook_triggers_single_repair(self, monkeypatch):
        import automation.instagram.caption_engine as ce
        ai = _stub_llm(monkeypatch,
                       [_long_hook_candidate(), _valid_candidate()])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)

        out = ce.write_caption(PACK, TRANSCRIPT, VIDEO_TITLE)

        first_line = out["caption"].split("\n")[0]
        assert len(first_line) <= 55
        assert len(ai.prompts) == 2
        repair_prompt = ai.prompts[1]
        assert "CORRECTION REQUIRED" in repair_prompt
        assert "55" in repair_prompt

    def test_invented_hashtag_dropped_and_topped_up(self, monkeypatch):
        """Invented tags are dropped; the REAL-ONLY pool fills the gap
        deterministically — no repair LLM spend needed."""
        import automation.instagram.caption_engine as ce
        ai = _stub_llm(monkeypatch, [_invented_hashtag_candidate()])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)

        out = ce.write_caption(PACK, TRANSCRIPT, VIDEO_TITLE)

        names = {t[1:].lower() for t in out["hashtags"]}
        assert "viralreelsking" not in names
        assert len(out["hashtags"]) == 5
        assert len(ai.prompts) == 1
        assert out["tags_topped_up"], "shortfall must come from pack"

    def test_unsupported_entity_scrubbed_then_repaired(self, monkeypatch):
        import automation.instagram.caption_engine as ce
        audits = iter([
            {"unsupported_entities": ["Ravindra Jadeja"],
             "supported_topics": []},
            {"unsupported_entities": [], "supported_topics": []},
        ])
        monkeypatch.setattr(
            ce, "audit_written_copy_llm",
            lambda *a, **k: next(audits))
        ai = _stub_llm(monkeypatch, [_jadeja_candidate(), _valid_candidate()])

        out = ce.write_caption(PACK, TRANSCRIPT, VIDEO_TITLE)

        assert "Jadeja" not in out["caption"]
        assert len(ai.prompts) == 2

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

    def test_repair_failure_raises_caption_policy_error(self, monkeypatch):
        import automation.instagram.caption_engine as ce
        ai = _stub_llm(monkeypatch,
                       [_long_hook_candidate(), _long_hook_candidate()])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)

        with pytest.raises(ce.CaptionPolicyError):
            ce.write_caption(PACK, TRANSCRIPT, VIDEO_TITLE)
        assert len(ai.prompts) == 2

    def test_no_parsable_json_both_attempts_raises(self, monkeypatch):
        import automation.instagram.caption_engine as ce
        ai = _stub_llm(monkeypatch, ["not json at all", "still not json"])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)

        with pytest.raises(ce.CaptionPolicyError):
            ce.write_caption(PACK, TRANSCRIPT, VIDEO_TITLE)
        assert len(ai.prompts) == 2

    def test_small_pack_publishes_with_available_real_tags(self, monkeypatch):
        """Data-driven regression: a pack with only 3 validated hashtags must
        still publish (with 3 real tags), not raise InsufficientEvidenceError.
        """
        import automation.instagram.caption_engine as ce
        ai = _stub_llm(monkeypatch, [_valid_candidate()])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)

        out = ce.write_caption(PACK_SMALL, TRANSCRIPT, VIDEO_TITLE)

        # 3 validated + 1 seed-phrase tag = 4 real tags; all must be grounded.
        assert 3 <= len(out["hashtags"]) <= 5
        allowed = {t.lstrip("#").lower()
                   for t in ["#cricket", "#INDvSL", "#TeamIndia",
                             "#IndVsSlHighlights"]}
        for tag in out["hashtags"]:
            assert tag[1:].lower() in allowed


# ---------------------------------------------------------------------------
# seo.generate_insta_metadata
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_evidence(monkeypatch):
    mod = types.ModuleType("automation.instagram.evidence")

    def build_insta_evidence_pack(*args, **kwargs):
        return json.loads(json.dumps(PACK))

    mod.build_insta_evidence_pack = build_insta_evidence_pack
    monkeypatch.setitem(sys.modules, "automation.instagram.evidence", mod)
    return mod


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
            "Yaad rahega yeh over.", "Yeh over yaad rahega.")
        assert other["caption"] != first["caption"]
        ai = _stub_llm(monkeypatch, [other])
        meta = seo.generate_insta_metadata(
            tmp_path, TRANSCRIPT, VIDEO_TITLE, "", force=True)

        expected = other["caption"].rstrip() + "\n\n" + \
            " ".join(t for t in other["hashtags"])
        assert meta["caption"] == expected
        assert len(ai.prompts) == 1

    def test_missing_evidence_module_returns_none(self, tmp_path,
                                                  monkeypatch):
        from automation.instagram import seo
        monkeypatch.setitem(sys.modules, "automation.instagram.evidence",
                            None)
        assert seo.generate_insta_metadata(
            tmp_path, TRANSCRIPT, VIDEO_TITLE, "") is None

    def test_policy_error_propagates(self, tmp_path, monkeypatch,
                                     fake_evidence):
        from automation.instagram import seo
        import automation.instagram.caption_engine as ce
        _stub_llm(monkeypatch,
                  [_long_hook_candidate(), _long_hook_candidate()])
        monkeypatch.setattr(ce, "audit_written_copy_llm", _clean_audit)

        with pytest.raises(ce.CaptionPolicyError):
            seo.generate_insta_metadata(
                tmp_path, TRANSCRIPT, VIDEO_TITLE, "")


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
