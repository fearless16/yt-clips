"""Regression tests for evidence-grounded, runtime cricket entity linking."""


def test_runtime_roster_resolves_player_not_present_in_static_catalog():
    from automation.seo.cricket_context import correct_cricket_spelling

    assert correct_cricket_spelling(
        "Vaibhav ne kya century maari",
        player_names=["Vaibhav Suryavanshi"],
    ) == "Vaibhav Suryavanshi ne kya century maari"


def test_ambiguous_first_name_abstains_instead_of_guessing():
    from automation.seo.cricket_context import correct_cricket_spelling

    assert correct_cricket_spelling(
        "Rahul ko opener banao",
        player_names=["Rahul Tripathi", "Rahul Chahar"],
    ) == "Rahul ko opener banao"


def test_uppercase_short_acronym_is_never_reinterpreted():
    from automation.seo.cricket_context import correct_cricket_spelling

    assert correct_cricket_spelling(
        "UV index high hai, Yuvi ko coach banao",
        player_names=["Yuvraj Singh"],
    ) == "UV index high hai, Yuvraj Singh ko coach banao"


def test_archived_ipl_season_is_not_rewritten_to_current_year():
    from automation.seo.cricket_context import correct_cricket_spelling

    assert correct_cricket_spelling("IPL 2025 final") == "IPL 2025 final"


def test_surname_after_different_first_name_is_not_rewritten_to_catalog_player():
    """A DPL squad list ('Pranav Pant') must not become 'Rishabh Pant'.

    Regression: the live DPL 2026 stream description mentioned squad player
    'Pranav Pant'; the surname alias rewrote it to 'Rishabh Pant', which then
    poisoned trend research (query 'Rishabh Pant T20 cricket'), grounded every
    clip SEO on the wrong player, and got all copy blocked by validators.
    """
    from automation.seo.cricket_context import correct_cricket_spelling

    text = (
        "The South Delhi Superstarz squad features Ayush Badoni, "
        "Pranav Pant and Divansh Rawat."
    )
    corrected = correct_cricket_spelling(text)
    assert "Rishabh Pant" not in corrected
    assert "Pranav Pant" in corrected


def test_bare_surname_still_resolves_when_not_part_of_another_full_name():
    from automation.seo.cricket_context import correct_cricket_spelling

    assert (
        correct_cricket_spelling("pant ne chauka maara")
        == "Rishabh Pant ne chauka maara"
    )


def test_canonical_full_name_passes_through_unchanged():
    from automation.seo.cricket_context import correct_cricket_spelling

    assert correct_cricket_spelling("Rishabh Pant ne six maara") == (
        "Rishabh Pant ne six maara"
    )


def test_phonetic_hinglish_variants_map_to_canonical_terms():
    """whisper language=hi emits pure phonetic Hinglish ('kriketa', 'viketa',
    'maicha'); without these mappings the cricket gate rejects real cricket
    commentary as non-cricket."""
    from automation.seo.cricket_context import (
        _cricket_relevance_score,
        correct_cricket_spelling,
        is_cricket_content,
    )

    assert correct_cricket_spelling("kriketa ke viketa") == "cricket ke wicket"
    assert correct_cricket_spelling("aaj ka maicha") == "aaj ka match"

    whisper_line = (
        "bhaaee kaise usako viketa milate the maara bahuta khaataa hai vo "
        "aura paakistaana vaale sabase buraa maarate the"
    )
    # Standalone fragment with multiple canonical terms clears the hard gate.
    assert is_cricket_content(
        "kriketa ka maicha hai aur usne do viketa liye"
    ) is True
    # Thin fragments pass via the same source-context path the selector uses.
    assert is_cricket_content(whisper_line, "kriketa maicha commentary") is True
    assert _cricket_relevance_score(whisper_line) >= 1


def test_evidence_pack_uses_verified_runtime_player_catalog():
    from automation.seo.context_engine import build_cricket_evidence_pack

    pack = build_cricket_evidence_pack(
        video_title="Rajasthan batting discussion",
        video_description="Vaibhav Suryavanshi changed the game.",
        clip_transcript="Vaibhav ka intent kamaal tha.",
        research_context={
            "player_names": ["Vaibhav Suryavanshi"],
            "match_facts": ["Rajasthan Royals vs Chennai Super Kings"],
        },
    )

    assert pack["clip_transcript"] == "Vaibhav Suryavanshi ka intent kamaal tha."
    assert pack["grounded_entities"]["players"] == ["Vaibhav Suryavanshi"]


def test_current_cricbuzz_page_parser_extracts_runtime_roster():
    from automation.seo.trends import parse_cricbuzz_match_page

    html = r'''
      <h1>Rajasthan Royals vs Chennai Super Kings, 12th Match - Scorecard</h1>
      <script>{\"batName\":\"Vaibhav Suryavanshi\",\"runs\":101,
      \"bowlName\":\"Noor Ahmad\",\"status\":\"RR won by 6 wickets\"}</script>
    '''
    parsed = parse_cricbuzz_match_page(html)

    assert parsed["player_names"] == ["Vaibhav Suryavanshi", "Noor Ahmad"]
    assert parsed["facts"][0].startswith("Rajasthan Royals vs Chennai Super Kings")
    assert "RR won by 6 wickets" in parsed["facts"]


def test_cricbuzz_search_parser_accepts_only_scorecard_result():
    from automation.seo.trends import parse_cricbuzz_search_results

    html = '''
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fbad">Bad</a>
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.cricbuzz.com%2Flive-cricket-scorecard%2F123%2Fa-vs-b">Match</a>
    '''

    assert parse_cricbuzz_search_results(html) == (
        "https://www.cricbuzz.com/live-cricket-scorecard/123/a-vs-b"
    )


def test_current_search_results_can_ground_unknown_player_or_nickname():
    from automation.seo.cricket_context import (
        correct_cricket_spelling,
        discover_grounded_player_aliases,
        discover_grounded_player_names,
    )

    source = "Cheeku ko India ka mentor bana do"
    results = [
        "Virat Kohli reveals why fans call him Cheeku",
        "Cheeku aka Virat Kohli on Team India pressure",
    ]
    players = discover_grounded_player_names(source, results)
    aliases = discover_grounded_player_aliases(source, results, players)

    assert players == ["Virat Kohli"]
    assert aliases == {"cheeku": "Virat Kohli"}
    assert correct_cricket_spelling(source, players, aliases).startswith("Virat Kohli")


def test_search_headlines_do_not_turn_generic_live_phrases_into_players():
    from automation.seo.cricket_context import discover_grounded_player_names

    source = "England vs Pakistan 1st Test live score and commentary"
    results = [
        "England Live Match Today | Pakistan Test Day 3",
        "Live Cricket Score | England vs Pakistan Test Live",
    ]

    assert discover_grounded_player_names(source, results) == []


def test_hinglish_title_phrases_are_not_discovered_as_players():
    from automation.seo.cricket_context import discover_grounded_player_names

    title = "Kya Chal Raha Hai? England Run Rate Ahead"
    assert discover_grounded_player_names(title, [title]) == []
    assert discover_grounded_player_names("Saud Shakeel", ["Saud Shakeel"]) == [
        "Saud Shakeel"
    ]


def test_search_headlines_do_not_turn_team_names_into_players():
    from automation.seo.cricket_context import discover_grounded_player_names

    assert discover_grounded_player_names(
        "New Zealand vs Sri Lanka cricket series",
        [
            "New Zealand vs Sri Lanka Test series",
            "Sri Lanka tour of New Zealand highlights",
        ],
    ) == []


def test_devanagari_team_names_are_canonical_grounded_entities():
    from automation.seo.cricket_context import (
        correct_cricket_spelling,
        find_canonical_entities,
    )

    corrected = correct_cricket_spelling(
        "पाकिस्तान ने ऑस्ट्रेलिया और साउथ अफ्रीका में टेस्ट खेले"
    )

    assert set(find_canonical_entities(corrected)["teams"]) == {
        "Pakistan",
        "Australia",
        "South Africa",
    }


def test_hindi_player_alias_resolves_only_with_verified_runtime_player():
    from automation.seo.cricket_context import correct_cricket_spelling

    text = "सऊदी ने न्यूजीलैंड की कप्तानी क्यों छोड़ी"
    assert correct_cricket_spelling(text).startswith("सऊदी ने New Zealand")
    assert correct_cricket_spelling(
        text,
        player_names=["Tim Southee"],
    ).startswith("Tim Southee ने")


def test_match_team_extraction_supports_international_and_hindi_teams():
    from automation.seo.trends import extract_match_teams

    teams, match_type = extract_match_teams(
        "England vs Pakistan Test: ऑस्ट्रेलिया और साउथ अफ्रीका discussion"
    )

    assert set(teams) == {"England", "Pakistan", "Australia", "South Africa"}
    assert match_type == "test"


def test_other_players_surname_never_canonicalizes_to_catalog_star():
    """DPL squad list says 'Pranav Pant'; the unique-surname shortcut must not
    claim that occurrence for Rishabh Pant."""
    from automation.seo.cricket_context import find_canonical_entities

    text = (
        "New Delhi Tigers squad: Ayush Badoni, Sanat Sangwan, "
        "Pranav Pant and Divansh Rawat."
    )
    assert "Rishabh Pant" not in find_canonical_entities(text)["players"]


def test_bare_surname_still_resolves_to_unique_catalog_player():
    from automation.seo.cricket_context import find_canonical_entities

    assert find_canonical_entities("pant finishes it off in style")[
        "players"
    ] == ["Rishabh Pant"]


def test_full_name_mention_still_resolves():
    from automation.seo.cricket_context import find_canonical_entities

    assert find_canonical_entities("Rishabh Pant keeps wicket")[
        "players"
    ] == ["Rishabh Pant"]


def test_phonetic_country_names_ground_india():
    """Hinglish 'indiyaa'/'bharat' must canonicalize to India so the copy
    gate stops rejecting legitimate national-team mentions."""
    from automation.seo.cricket_context import (
        correct_cricket_spelling,
        find_canonical_entities,
    )

    fixed = correct_cricket_spelling(
        "cricket men kabhee indiyaa achchhaa karataa hai"
    )
    assert "India" in fixed
    assert find_canonical_entities(fixed)["teams"] == ["India"]


def test_bharat_and_hindustaan_map_to_india():
    from automation.seo.cricket_context import correct_cricket_spelling

    assert correct_cricket_spelling("bharat ne jeeta") == "India ne jeeta"
    assert correct_cricket_spelling("hindustaan ka match") == "India ka match"


def test_indiya_does_not_corrupt_inside_words():
    from automation.seo.cricket_context import correct_cricket_spelling

    assert correct_cricket_spelling("Indianapolis") == "Indianapolis"
