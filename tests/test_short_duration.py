"""Output-duration guard: exported clip must land inside [output_min, output_max].

User rule (current): clips must be UNDER 25s — window is [10, 24]. Selection
normalizes ``speed_factor`` so the EXPORTED (post-speedup) duration lands
inside the configured window. Clip boundaries themselves are never altered
(complete-thought gate).
"""


def test_window_at_or_below_target_runs_at_1x():
    from automation.clip_selection.pipeline import _compute_speed_factor

    assert _compute_speed_factor(24.0, 25.0, 1.6) == 1.0


def test_longer_window_compressed_respecting_cap():
    from automation.clip_selection.pipeline import _compute_speed_factor

    assert _compute_speed_factor(40.0, 25.0, 1.6) == 1.6


def test_compressed_output_is_raised_to_floor():
    """33s @1.32x would export 25.0s (> ceiling) — speed must rise toward the
    24s ceiling."""
    from automation.clip_selection.pipeline import _compute_speed_factor

    speed = _compute_speed_factor(33.0, 25.0, 1.6, output_min=10.0, output_max=24.0)
    assert speed == round(33.0 / 24.0, 2)
    assert 33.0 / speed <= 24.0 + 0.05


def test_floor_never_pushes_speed_below_1x():
    """A short thought cannot be stretched below its natural pace."""
    from automation.clip_selection.pipeline import _compute_speed_factor

    assert _compute_speed_factor(8.0, 25.0, 1.6, output_min=10.0) == 1.0


def test_output_capped_at_ceiling_when_cap_blocks_further_compression():
    """60s @1.6x cap exports 37.5s — cannot fix via speed without editing
    boundaries, so the cap stands (caller logs a warning)."""
    from automation.clip_selection.pipeline import _compute_speed_factor

    assert _compute_speed_factor(60.0, 25.0, 1.6, output_max=24.0) == 1.6


def test_ceiling_compresses_when_headroom_allows():
    """Huge speedup budget must not export under the floor either."""
    from automation.clip_selection.pipeline import _compute_speed_factor

    speed = _compute_speed_factor(120.0, 10.0, 12.0, output_min=10.0, output_max=24.0)
    assert speed == round(120.0 / 10.0, 2)
    assert 10.0 - 0.05 <= 120.0 / speed <= 24.0 + 0.05


def test_yaml_entry_attaches_normalized_speed_factor():
    from automation.clip_selection.pipeline import _build_clip_yaml_entry

    entry = _build_clip_yaml_entry(
        start=2251.0,
        end=2284.0,
        score=42.5,
        text="cricket thought",
        target_duration=25.0,
        max_speedup=1.6,
        output_min=10.0,
        output_max=24.0,
    )
    assert 33.0 / entry["speed_factor"] <= 24.0 + 0.05
