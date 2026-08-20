from shorts_intelligence.domain import CricketDomainGate


def test_dynamic_cricket_gate_does_not_need_player_roster():
    gate = CricketDomainGate()

    decision = gate.classify(
        title="Yuvi ko T20 coach banao",
        description="AUS vs BAN Test match cricket debate",
    )

    assert decision.status == "cricket"
    assert decision.confidence >= 0.8


def test_non_cricket_short_is_excluded_from_training():
    gate = CricketDomainGate()

    decision = gate.classify(
        title="Goldberg vs Brock Lesnar WWE 2K26",
        description="wrestling gameplay short",
    )

    assert decision.status == "non_cricket"


def test_ambiguous_content_is_unknown_instead_of_force_labeled():
    decision = CricketDomainGate().classify(
        title="What a moment",
        description="You have to see this",
    )

    assert decision.status == "unknown"


def test_cricket_mechanics_and_team_context_resolve_roster_free_titles():
    gate = CricketDomainGate()

    assert gate.classify("Zing Bails are glitching", "DC vs CSK chase").status == "cricket"
    assert gate.classify(
        "SKY controversy",
        "Suryakumar Yadav exit from Team India after World Cup win",
    ).status == "cricket"
    assert gate.classify(
        "The winning shot celebration",
        "CSK vs DC chase reached a nail-biting finish",
    ).status == "cricket"


def test_hockey_travel_and_tech_are_explicitly_non_cricket():
    gate = CricketDomainGate()

    assert gate.classify("Pakistan World Cup hockey record", "India hockey history").status == "non_cricket"
    assert gate.classify("Ooty mini vlog", "India travel shorts").status == "non_cricket"
    assert gate.classify("Jio Fiber exposed", "Broadband internet review").status == "non_cricket"
