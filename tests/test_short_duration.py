"""Short duration guard — preserve complete thoughts in the shortest useful clip.

TDD: tests written first. Verifies:
1. Pure speed-factor computation (window > target -> speed-up).
2. Config exposes target_duration + max_speedup.
3. Export merges highlight-level speed_factor into export strategy.
4. Both detect_highlights entry points (highlight.py + clip_selection) attach speed_factor to YAML.
"""
import json

import pytest
import yaml


# ── 1. Pure computation ───────────────────────────────────────────────────────

def test_speed_factor_within_target_is_1():
    from highlight import _compute_speed_factor
    assert _compute_speed_factor(15.0, 22.0, 1.6) == 1.0


def test_speed_factor_compresses_over_target():
    from highlight import _compute_speed_factor
    # 30s window, 22s target -> 30/22 = 1.36x
    assert _compute_speed_factor(30.0, 22.0, 1.6) == pytest.approx(1.36, abs=0.01)


def test_speed_factor_caps_at_max_speedup():
    from highlight import _compute_speed_factor
    # 60s window, 22s target -> 2.7x needed but capped at 1.6
    assert _compute_speed_factor(60.0, 22.0, 1.6) == 1.6


def test_speed_factor_zero_duration_safe():
    from highlight import _compute_speed_factor
    assert _compute_speed_factor(0.0, 22.0, 1.6) == 1.0
    assert _compute_speed_factor(-5.0, 22.0, 1.6) == 1.0


def test_speed_factor_degenerate_max_speedup_clamps_to_1():
    from highlight import _compute_speed_factor
    # max_speedup below 1.0 would produce a slow-down; must clamp to 1.0
    assert _compute_speed_factor(30.0, 22.0, 0.8) == 1.0
    assert _compute_speed_factor(60.0, 22.0, 0.5) == 1.0


def test_speed_factor_zero_target_safe():
    from highlight import _compute_speed_factor
    assert _compute_speed_factor(30.0, 0.0, 1.6) == 1.0
    assert _compute_speed_factor(30.0, -5.0, 1.6) == 1.0


def test_clip_selection_speed_factor_matches():
    from automation.clip_selection.pipeline import _compute_speed_factor as csf
    assert csf(25.0, 22.0, 1.6) == pytest.approx(1.14, abs=0.01)
    assert csf(10.0, 22.0, 1.6) == 1.0


# ── 2. Config ────────────────────────────────────────────────────────────────

def test_config_has_target_duration():
    from utils.config import load_config
    hl = load_config().get("highlight", {})
    assert "target_duration" in hl
    assert 20 <= float(hl["target_duration"]) <= 30


def test_config_duration_targets_natural_pace():
    """Channel data: 25-40s natural-pace thoughts win; compression is a fallback."""
    from utils.config import load_config
    hl = load_config().get("highlight", {})
    assert float(hl["max_duration"]) <= 45
    assert float(hl["preferred_duration_min"]) >= 15
    assert 25 <= float(hl["preferred_duration_max"]) <= 45
    assert float(hl["target_duration"]) >= 20


def test_config_has_max_speedup():
    from utils.config import load_config
    hl = load_config().get("highlight", {})
    assert "max_speedup" in hl
    assert 1.2 <= float(hl["max_speedup"]) <= 2.0


# ── 3. Export merges speed_factor ────────────────────────────────────────────

def test_export_merge_highlight_speed_uses_max():
    from export import _merge_highlight_speed, _normalize_speed
    strategy = {"speed_factor": 1.0}
    out = _merge_highlight_speed(strategy, {"speed_factor": 1.4})
    assert _normalize_speed(out.get("speed_factor", 1.0)) == 1.4


def test_export_merge_highlight_speed_keeps_analysis_speed_when_higher():
    from export import _merge_highlight_speed, _normalize_speed
    strategy = {"speed_factor": 1.5}
    out = _merge_highlight_speed(strategy, {"speed_factor": 1.2})
    assert _normalize_speed(out.get("speed_factor", 1.0)) == 1.5


def test_export_merge_highlight_speed_noop_without_key():
    from export import _merge_highlight_speed
    strategy = {"speed_factor": 1.0}
    out = _merge_highlight_speed(strategy, {"start": 10.0, "end": 20.0})
    assert out["speed_factor"] == 1.0


def test_export_merge_highlight_speed_none_info_noop():
    from export import _merge_highlight_speed
    strategy = {"speed_factor": 1.0}
    assert _merge_highlight_speed(strategy, None)["speed_factor"] == 1.0
    assert _merge_highlight_speed(None, None) is not None


def test_export_merge_highlight_speed_string_value():
    from export import _merge_highlight_speed, _normalize_speed
    out = _merge_highlight_speed({}, {"speed_factor": "1.3"})
    assert _normalize_speed(out.get("speed_factor", 1.0)) == 1.3


# ── 4. YAML entries carry speed_factor ───────────────────────────────────────

def test_yaml_window_build_attaches_speed_factor():
    from highlight import _build_clip_yaml_entry
    entry = _build_clip_yaml_entry(
        start=0.0, end=30.0, score=8.5, text="wicket!",
        target_duration=22.0, max_speedup=1.6,
    )
    assert entry["speed_factor"] == pytest.approx(1.36, abs=0.01)
    assert entry["end_sec"] == 30.0
    assert entry["score"] == 8.5


def test_yaml_window_build_within_target_no_speed():
    from highlight import _build_clip_yaml_entry
    entry = _build_clip_yaml_entry(
        start=0.0, end=12.0, score=8.0, text="six!",
        target_duration=22.0, max_speedup=1.6,
    )
    assert entry["speed_factor"] == 1.0


def test_yaml_window_build_clip_selection_attaches_speed():
    from automation.clip_selection.pipeline import _build_clip_yaml_entry as build
    entry = build(
        start=5.0, end=40.0, score=7.0, text="catch!",
        target_duration=22.0, max_speedup=1.6,
    )
    # 35/22 = 1.59 (below cap 1.6, so NOT capped)
    assert entry["speed_factor"] == pytest.approx(1.59, abs=0.01)


# ── 5. End-to-end wiring ─────────────────────────────────────────────────────

def test_detect_highlights_yaml_has_speed_factor(tmp_path, monkeypatch):
    """detect_highlights (highlight.py) writes speed_factor into the YAML."""
    import highlight

    transcript = tmp_path / "clip.json"
    transcript.write_text(
        json.dumps([{"start": 5.0, "end": 29.0, "text": "Huge six from Kohli, the crowd goes absolutely wild"}]),
        encoding="utf-8",
    )
    out_yaml = tmp_path / "clips.yaml"

    monkeypatch.setattr(highlight, "_extract_audio_rms", lambda *a, **k: [(0.0, 1.0)])
    monkeypatch.setattr(highlight, "_get_video_duration", lambda *a: 60.0)
    from utils.config import load_config
    monkeypatch.setattr(highlight, "cfg", load_config())
    monkeypatch.setitem(highlight.cfg["highlight"], "use_ai_refinement", False)

    def fake_parallel(candidates, segments, rms_map, avg_rms, max_rms):
        return [dict(c, weighted_score=50.0) for c in candidates]

    monkeypatch.setattr(highlight, "_parallel_score_candidates", fake_parallel)

    highlights = highlight.detect_highlights(
        transcript_path=str(transcript), video_path="dummy.mp4", output_path=str(out_yaml),
    )

    assert out_yaml.exists()
    data = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert "speed_factor" in data["clip1"]
    # 24s window vs 25s natural target — no speed-up needed anymore.
    assert data["clip1"]["speed_factor"] == pytest.approx(1.0, abs=0.01)
    assert highlights[0]["speed_factor"] == pytest.approx(1.0, abs=0.01)


def test_sanitize_strategy_preserves_merged_speed():
    """The merged speed_factor survives _sanitize_strategy (export path seam)."""
    from export import _merge_highlight_speed, _sanitize_strategy
    strategy = _merge_highlight_speed({}, {"speed_factor": 1.4})
    clean = _sanitize_strategy(strategy)
    assert clean["speed_factor"] == pytest.approx(1.4)
