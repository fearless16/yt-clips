from shorts_intelligence.policy import SelectionPolicy


RECOMMENDATIONS = [{
    "feature_name": "duration_bucket",
    "feature_value": "under_15",
    "effect": 0.04,
    "effect_lower_bound": 0.01,
    "sample_count": 20,
    "actionable": True,
}]


def test_policy_never_allows_incomplete_thought():
    result = SelectionPolicy(RECOMMENDATIONS, shadow_mode=False).evaluate(
        duration_seconds=12,
        complete_thought=False,
    )

    assert result.eligible is False
    assert result.applied_adjustment == 0


def test_shadow_policy_reports_but_does_not_apply_adjustment():
    result = SelectionPolicy(RECOMMENDATIONS, shadow_mode=True).evaluate(
        duration_seconds=12,
        complete_thought=True,
    )

    assert result.eligible is True
    assert result.shadow_adjustment > 0
    assert result.applied_adjustment == 0


def test_active_policy_applies_bounded_evidence_adjustment():
    result = SelectionPolicy(
        RECOMMENDATIONS,
        shadow_mode=False,
        max_adjustment_points=3.0,
    ).evaluate(duration_seconds=12, complete_thought=True)

    assert result.applied_adjustment == 3.0
