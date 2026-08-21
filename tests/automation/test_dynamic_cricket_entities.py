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
