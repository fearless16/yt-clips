"""Dedupe engine tests — cross-run time-window duplicate detection.

Guards against re-uploading near-identical clips when the same match is
re-processed (same highlight windows get re-selected).
"""

import json
import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from automation.clip_selection.dedupe import (
    load_previous_windows,
    overlaps_any_previous,
    append_windows,
    fraction_republished,
)


# ── fraction_republished ───────────────────────────────────────────

class TestFractionRepublished:
    def test_no_overlap(self):
        assert fraction_republished(0, 10, 20, 30) == 0.0

    def test_exact_same_window(self):
        assert fraction_republished(5, 15, 5, 15) == 1.0

    def test_candidate_inside_previous(self):
        assert fraction_republished(5, 10, 0, 20) == 1.0

    def test_candidate_contains_small_previous(self):
        # 5s published inside a 30s candidate → only 1/6 republished
        assert fraction_republished(0, 30, 10, 15) == 5 / 30

    def test_zero_candidate_duration(self):
        assert fraction_republished(5, 5, 0, 10) == 0.0


# ── load_previous_windows ──────────────────────────────────────────

class TestLoadPreviousWindows:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_previous_windows(tmp_path / "nope.yaml") == []

    def test_corrupt_yaml_returns_empty(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("{ not valid yaml [", encoding="utf-8")
        assert load_previous_windows(p) == []

    def test_empty_yaml_returns_empty(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        assert load_previous_windows(p) == []

    def test_new_format_windows(self, tmp_path):
        p = tmp_path / "highlights.yaml"
        p.write_text(
            "clip1:\n"
            "  start: 00:00:05\n"
            "  end: 00:00:25\n"
            "  start_sec: 5.0\n"
            "  end_sec: 25.0\n"
            "  text: first\n"
            "clip2:\n"
            "  start: 00:01:00\n"
            "  end: 00:01:30\n"
            "  start_sec: 60.0\n"
            "  end_sec: 90.0\n"
            "  text: second\n",
            encoding="utf-8",
        )
        windows = load_previous_windows(p)
        assert windows == [
            {"start": 5.0, "end": 25.0},
            {"start": 60.0, "end": 90.0},
        ]

    def test_old_format_without_sec_keys(self, tmp_path):
        p = tmp_path / "old.yaml"
        p.write_text(
            "clip1:\n"
            "  start: 00:00:05\n"
            "  end: 00:00:25\n"
            "  text: first\n",
            encoding="utf-8",
        )
        windows = load_previous_windows(p)
        assert windows == [{"start": 5.0, "end": 25.0}]

    def test_entries_missing_coords_skipped(self, tmp_path):
        p = tmp_path / "mixed.yaml"
        p.write_text(
            "clip1:\n"
            "  start: 00:00:05\n"
            "  end: 00:00:25\n"
            "clip2:\n"
            "  text: no coords\n",
            encoding="utf-8",
        )
        windows = load_previous_windows(p)
        assert windows == [{"start": 5.0, "end": 25.0}]

    def test_list_format_yaml(self, tmp_path):
        p = tmp_path / "history.yaml"
        p.write_text(
            "- start: 5.0\n"
            "  end: 25.0\n"
            "- start: 60.0\n"
            "  end: 90.0\n",
            encoding="utf-8",
        )
        windows = load_previous_windows(p)
        assert windows == [
            {"start": 5.0, "end": 25.0},
            {"start": 60.0, "end": 90.0},
        ]

    def test_start_sec_zero_is_not_dropped(self, tmp_path):
        p = tmp_path / "zero.yaml"
        p.write_text(
            "clip1:\n"
            "  start_sec: 0.0\n"
            "  end_sec: 25.0\n",
            encoding="utf-8",
        )
        windows = load_previous_windows(p)
        assert windows == [{"start": 0.0, "end": 25.0}]

    def test_non_dict_yaml_returns_empty(self, tmp_path):
        p = tmp_path / "scalar.yaml"
        p.write_text("just a string\n", encoding="utf-8")
        assert load_previous_windows(p) == []

    def test_bool_timestamp_ignored(self, tmp_path):
        p = tmp_path / "bool.yaml"
        p.write_text(
            "clip1:\n"
            "  start_sec: true\n"
            "  end_sec: 25.0\n",
            encoding="utf-8",
        )
        assert load_previous_windows(p) == []

    def test_results_sorted_by_start(self, tmp_path):
        p = tmp_path / "unsorted.yaml"
        p.write_text(
            "clip1:\n"
            "  start_sec: 60.0\n"
            "  end_sec: 90.0\n"
            "clip2:\n"
            "  start_sec: 5.0\n"
            "  end_sec: 25.0\n",
            encoding="utf-8",
        )
        windows = load_previous_windows(p)
        assert windows[0]["start"] == 5.0
        assert windows[1]["start"] == 60.0


# ── overlaps_any_previous ──────────────────────────────────────────

class TestOverlapsAnyPrevious:
    def test_exact_duplicate(self):
        prev = [{"start": 10.0, "end": 30.0}]
        assert overlaps_any_previous(10.0, 30.0, prev, threshold=0.6)

    def test_contained_duplicate(self):
        prev = [{"start": 5.0, "end": 40.0}]
        assert overlaps_any_previous(10.0, 30.0, prev, threshold=0.6)

    def test_high_overlap(self):
        prev = [{"start": 0.0, "end": 20.0}]
        # overlap 0-20 vs 2-22 → 18s shared / 20s min → 0.9
        assert overlaps_any_previous(2.0, 22.0, prev, threshold=0.6)

    def test_low_overlap_rejected(self):
        prev = [{"start": 0.0, "end": 20.0}]
        # overlap 0-20 vs 12-32 → 8s shared / 20s min → 0.4
        assert not overlaps_any_previous(12.0, 32.0, prev, threshold=0.6)

    def test_empty_previous(self):
        assert not overlaps_any_previous(0.0, 10.0, [], threshold=0.6)

    def test_no_overlap(self):
        prev = [{"start": 100.0, "end": 120.0}]
        assert not overlaps_any_previous(0.0, 10.0, prev, threshold=0.6)

    def test_threshold_zero_clamped_positive(self):
        prev = [{"start": 0.0, "end": 10.0}]
        # threshold 0 would match touching-only (ratio 0.0); clamping to
        # positive keeps barely-touching windows from being rejected
        assert not overlaps_any_previous(10.0, 20.0, prev, threshold=0.0)

    def test_garbage_previous_entry_never_matches(self):
        prev = [{"start": "garbage", "end": None}]
        assert not overlaps_any_previous(0.0, 10.0, prev, threshold=0.6)

    def test_bool_timestamp_never_matches(self):
        prev = [{"start": True, "end": 30.0}]
        assert not overlaps_any_previous(0.0, 10.0, prev, threshold=0.6)

    def test_union_of_adjacent_windows_matches(self):
        # 0-100s candidate vs [10-50, 60-90] → 70% unioned coverage.
        # Window-by-window each is only 40%/30% below threshold, so this
        # only passes because overlapping windows are merged first.
        prev = [{"start": 10.0, "end": 50.0}, {"start": 60.0, "end": 90.0}]
        assert overlaps_any_previous(0.0, 100.0, prev, threshold=0.6)

    def test_union_of_overlapping_windows_matches(self):
        # [10-50] and [40-90] overlap → union 10-90 = 80s of a 100s candidate.
        prev = [{"start": 10.0, "end": 50.0}, {"start": 40.0, "end": 90.0}]
        assert overlaps_any_previous(0.0, 100.0, prev, threshold=0.6)

    def test_union_below_threshold_rejected(self):
        # 0-100s candidate vs [10-50, 60-90] covers 70%… wait, below is
        # 0-100 vs [10-40, 45-55] → 30+10 = 40% → below 0.6 threshold.
        prev = [{"start": 10.0, "end": 40.0}, {"start": 45.0, "end": 55.0}]
        assert not overlaps_any_previous(0.0, 100.0, prev, threshold=0.6)

    def test_gap_between_windows_not_republished(self):
        # 0-100 vs [0-40, 80-90] → union 40+10 = 50% → below threshold, and
        # the 40-80 gap is genuinely new content.
        prev = [{"start": 0.0, "end": 40.0}, {"start": 80.0, "end": 90.0}]
        assert not overlaps_any_previous(0.0, 100.0, prev, threshold=0.6)


# ── append_windows ─────────────────────────────────────────────────

class TestAppendWindows:
    def test_appends_and_roundtrips(self, tmp_path):
        p = tmp_path / "history.yaml"
        append_windows(p, [{"start": 5.0, "end": 25.0}])
        append_windows(p, [{"start": 60.0, "end": 90.0}])
        windows = load_previous_windows(p)
        assert windows == [
            {"start": 5.0, "end": 25.0},
            {"start": 60.0, "end": 90.0},
        ]

    def test_dedupes_exact_windows(self, tmp_path):
        p = tmp_path / "history.yaml"
        append_windows(p, [{"start": 5.0, "end": 25.0}])
        append_windows(p, [{"start": 5.0, "end": 25.0}])
        append_windows(p, [{"start": 5.0, "end": 25.0}])
        assert len(load_previous_windows(p)) == 1

    def test_preserves_existing_history_on_rewrite(self, tmp_path):
        p = tmp_path / "history.yaml"
        append_windows(p, [{"start": 5.0, "end": 25.0}])
        append_windows(p, [{"start": 60.0, "end": 90.0}])
        append_windows(p, [{"start": 10.0, "end": 30.0}])
        windows = load_previous_windows(p)
        assert {w["start"] for w in windows} == {5.0, 60.0, 10.0}

    def test_empty_append_keeps_existing(self, tmp_path):
        p = tmp_path / "history.yaml"
        append_windows(p, [{"start": 5.0, "end": 25.0}])
        append_windows(p, [])
        assert len(load_previous_windows(p)) == 1

    def test_creates_parent_dir(self, tmp_path):
        p = tmp_path / "nested" / "deep" / "history.yaml"
        append_windows(p, [{"start": 5.0, "end": 25.0}])
        assert len(load_previous_windows(p)) == 1

    def test_writes_atomic_no_tmp_left_behind(self, tmp_path):
        p = tmp_path / "history.yaml"
        append_windows(p, [{"start": 5.0, "end": 25.0}])
        assert not (tmp_path / "history.tmp").exists()
        assert p.exists()
