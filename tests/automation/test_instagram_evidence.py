"""TDD tests for automation/instagram/evidence.py (InstaEvidencePack).

Covers:
- Full pack shape per PLAN.md schema
- Graceful degradation: every live source raising -> empty field, never raises
- seed_phrases ONLY from YT-suggest strings + approved_search_queries ("seed": True)
- validate_hashtag_pool: mega-tags always rejected; unvalidated tags rejected;
  seeds survive pre-App-Review mode
- Velocity/engagement wiring when validate_hashtags=True with a fake client
"""
import pytest

import automation.instagram.evidence as evidence
from automation.instagram.evidence import (
    build_insta_evidence_pack,
    validate_hashtag_pool,
)

PACK_KEYS = {
    "match_facts",
    "roster",
    "learner_top_captions",
    "seed_phrases",
    "validated_hashtags",
    "forbidden",
}

_REAL_LOAD_LEARNER = evidence._load_learner_top_captions


class FakeHashtagClient:
    """Real-shaped hashtag methods only — no network."""

    def __init__(self, velocity=4200, engagement=87.5):
        self.velocity = velocity
        self.engagement = engagement
        self.search_calls = []
        self.velocity_calls = []
        self.engagement_calls = []

    def hashtag_search(self, q):
        self.search_calls.append(q)
        return {"data": [{"id": f"id_{q}", "name": q}]}

    def hashtag_recent_velocity(self, hashtag_id):
        self.velocity_calls.append(hashtag_id)
        return 0 if hashtag_id == "id_deadtag" else self.velocity

    def hashtag_top_engagement(self, hashtag_id):
        self.engagement_calls.append(hashtag_id)
        return self.engagement


@pytest.fixture(autouse=True)
def _stub_all_sources(monkeypatch):
    monkeypatch.setattr(
        evidence, "fetch_verified_match_context",
        lambda q: {"facts": ["IND vs AUS: India need 40 runs"],
                   "player_names": ["Jasprit Bumrah"], "source_url": "https://x"},
    )
    monkeypatch.setattr(
        evidence, "fetch_youtube_suggestions",
        lambda q="cricket live": ["bumrah yorker", "cricket live hindi"],
    )
    monkeypatch.setattr(
        evidence, "_load_learner_top_captions",
        lambda limit=10: ["Bumrah ne last over mein match jeeta!"],
    )


class TestFullPackShape:

    def test_pack_has_exact_plan_schema_keys(self):
        pack = build_insta_evidence_pack(
            "Bumrah magic in final over IND vs AUS", "", "what a yorker")
        assert set(pack.keys()) == PACK_KEYS

    def test_live_match_facts_used_when_arg_absent(self):
        pack = build_insta_evidence_pack("title", "", "")
        assert pack["match_facts"] == ["IND vs AUS: India need 40 runs"]

    def test_explicit_match_facts_skip_live_call(self, monkeypatch):
        calls = []

        def boom(q):
            calls.append(q)
            raise AssertionError("live call must not happen")

        monkeypatch.setattr(evidence, "fetch_verified_match_context", boom)
        pack = build_insta_evidence_pack("t", "", "",
                                         match_facts=["given fact"])
        assert pack["match_facts"] == ["given fact"]
        assert calls == []

    def test_roster_unions_grounded_then_match_then_canonical(self):
        pack = build_insta_evidence_pack("t", "", "")
        assert "Jasprit Bumrah" in pack["roster"]
        pack2 = build_insta_evidence_pack(
            "t", "", "", grounded_players=["Rohit Sharma"])
        assert pack2["roster"][0] == "Rohit Sharma"
        assert "Jasprit Bumrah" in pack2["roster"]

    def test_learner_top_captions_flow_into_pack(self):
        pack = build_insta_evidence_pack("t", "", "")
        assert pack["learner_top_captions"] == [
            "Bumrah ne last over mein match jeeta!"]

    def test_seed_phrases_only_from_suggestions_and_approved(self):
        pack = build_insta_evidence_pack(
            "t", "", "", approved_search_queries=["ipl final watchalong"])
        phrases = {item["phrase"].casefold() for item in pack["seed_phrases"]}
        assert "bumrah yorker" in phrases
        assert "cricket live hindi" in phrases
        assert "ipl final watchalong" in phrases
        assert all(item["seed"] is True for item in pack["seed_phrases"])

    def test_validated_hashtags_empty_without_client_or_flag(self):
        client = FakeHashtagClient()
        p1 = build_insta_evidence_pack("t", "", "")
        p2 = build_insta_evidence_pack("t", "", "", client=client,
                                       validate_hashtags=False)
        assert p1["validated_hashtags"] == []
        assert p2["validated_hashtags"] == []
        assert client.search_calls == []

    def test_forbidden_lists_invented_and_generic(self):
        pack = build_insta_evidence_pack("t", "", "")
        joined = " ".join(pack["forbidden"]).casefold()
        assert "invented player names" in joined
        assert "generic tags without volume evidence" in joined


class TestGracefulDegradation:

    @pytest.mark.parametrize("source", [
        "fetch_verified_match_context",
        "fetch_youtube_suggestions",
        "_load_learner_top_captions",
        "find_canonical_entities",
    ])
    def test_source_raising_yields_empty_field_never_raises(self,
                                                            monkeypatch,
                                                            source):
        def boom(*a, **k):
            raise RuntimeError(f"{source} exploded")

        monkeypatch.setattr(evidence, source, boom)
        pack = build_insta_evidence_pack("IND vs AUS", "", "bumrah yorker")
        assert isinstance(pack, dict)
        assert set(pack.keys()) == PACK_KEYS
        if source == "fetch_verified_match_context":
            assert pack["match_facts"] == []
        elif source == "fetch_youtube_suggestions":
            assert pack["seed_phrases"] == []
        elif source == "_load_learner_top_captions":
            assert pack["learner_top_captions"] == []
        else:
            assert "Virat Kohli" not in pack["roster"]
            assert pack["roster"] == ["Jasprit Bumrah"]

    def test_total_network_blackout_still_builds_pack(self, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("offline")

        for name in ("fetch_verified_match_context",
                     "fetch_youtube_suggestions"):
            monkeypatch.setattr(evidence, name, boom)
        pack = build_insta_evidence_pack("t", "", "", client=FakeHashtagClient(),
                                         validate_hashtags=True)
        assert pack["match_facts"] == []
        assert pack["seed_phrases"] == []


class TestValidateHashtagPool:

    def _pack(self, **kw):
        return build_insta_evidence_pack("t", "", "",
                                         approved_search_queries=[
                                             "bumrah yorker"], **kw)

    @pytest.mark.parametrize("mega", [
        "#reels", "#viral", "#explore", "#shorts", "#fyp", "#trending",
        "Reels", "VIRAL",
    ])
    def test_mega_tags_always_rejected_even_if_seeded(self, mega,
                                                      monkeypatch):
        monkeypatch.setattr(
            evidence, "fetch_youtube_suggestions", lambda q="cricket": [mega])
        pack = self._pack()
        clean, rejected = validate_hashtag_pool([mega], pack)
        assert clean == []
        assert rejected == [mega]

    def test_unvalidated_unknown_tag_rejected(self):
        pack = self._pack()
        clean, rejected = validate_hashtag_pool(["#madeupthing"], pack)
        assert clean == []
        assert rejected == ["#madeupthing"]

    def test_seed_traceable_tag_survives_pre_review_mode(self):
        pack = self._pack()
        clean, rejected = validate_hashtag_pool(
            ["#bumrahyorker", "#bumrah"], pack)
        assert set(t.casefold() for t in clean) == {"#bumrahyorker",
                                                    "#bumrah"}
        assert rejected == []

    def test_tag_outside_validated_and_seeds_rejected_when_client_used(
            self):
        client = FakeHashtagClient()
        pack = build_insta_evidence_pack(
            "t", "", "", approved_search_queries=["bumrah yorker"],
            client=client, validate_hashtags=True)
        validated_norms = {v["tag"].lstrip("#").casefold()
                           for v in pack["validated_hashtags"]}
        pool = ["#" + n for n in validated_norms] + ["#randomnonsense"]
        clean, rejected = validate_hashtag_pool(pool, pack)
        assert "#randomnonsense" not in [c.casefold() for c in clean]
        assert "#randomnonsense" in rejected
        assert len(clean) >= 1

    def test_returns_tuple_of_two_lists_on_empty_input(self):
        pack = self._pack()
        result = validate_hashtag_pool([], pack)
        assert result == ([], [])


class TestHashtagValidationWiring:

    def test_validated_entries_carry_real_api_numbers(self):
        client = FakeHashtagClient(velocity=4321, engagement=55.5)
        pack = build_insta_evidence_pack(
            "Jasprit Bumrah vs Mumbai Indians", "", "",
            teams=["Mumbai Indians"], grounded_players=["Jasprit Bumrah"],
            client=client, validate_hashtags=True)
        entries = pack["validated_hashtags"]
        assert entries, "expected at least one validated tag"
        for entry in entries:
            assert entry["recent_volume_24h"] == 4321
            assert entry["source"] == "ig-hashtag-api"
        norms = {e["tag"].lstrip("#").casefold() for e in entries}
        assert any("bumrah" in n for n in norms)
        assert any("mumbaiindians" in n for n in norms)
        assert client.velocity_calls and client.engagement_calls
        assert all(cid.startswith("id_")
                   for cid in client.velocity_calls)

    def test_zero_velocity_tag_dropped_no_volume_evidence(self):
        client = FakeHashtagClient()
        pack = build_insta_evidence_pack(
            "t", "", "", teams=["deadtag"], grounded_players=[],
            client=client, validate_hashtags=True)
        norms = {e["tag"].lstrip("#").casefold() for e in
                 pack["validated_hashtags"]}
        assert "deadtag" not in norms

    def test_flag_off_with_client_never_touches_api(self):
        client = FakeHashtagClient()
        pack = build_insta_evidence_pack("t", "", "", client=client,
                                         validate_hashtags=False)
        assert pack["validated_hashtags"] == []
        assert client.search_calls == []
        assert client.velocity_calls == []
        assert client.engagement_calls == []

    def test_client_raising_degrades_to_empty_validated(self):
        class ExplodingClient(FakeHashtagClient):
            def hashtag_search(self, q):
                raise RuntimeError("api down")

        pack = build_insta_evidence_pack(
            "t", "", "", teams=["mi"], client=ExplodingClient(),
            validate_hashtags=True)
        assert pack["validated_hashtags"] == []


class TestLearnerLoaderRealDbPath:

    def test_missing_db_file_returns_empty_list(self, tmp_path, monkeypatch):
        from shorts_intelligence.config import RuntimeConfig
        cfg = RuntimeConfig.from_mapping({
            "shorts_intelligence": {"channel_id": "UCtest",
                                    "db_path": str(tmp_path / "nope.db")},
        })
        monkeypatch.setattr(
            "shorts_intelligence.config.RuntimeConfig.from_yaml",
            classmethod(lambda cls, path="config.yaml": cfg))
        assert _REAL_LOAD_LEARNER() == []
