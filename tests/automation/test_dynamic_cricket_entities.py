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
