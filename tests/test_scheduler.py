from collections import Counter
from datetime import datetime, timedelta

from scheduler import IST, generate_schedule


def test_three_clips_span_the_full_ist_publishing_window():
    slots = generate_schedule(
        3,
        interval_hours=2,
        start_from=datetime(2026, 8, 22, 1, 0, tzinfo=IST),
        day_start_hour=9,
        day_end_hour=21,
        jitter_max_minutes=0,
        spread_across_window=True,
    )

    assert slots[0].hour == 9
    assert 14 <= slots[1].hour <= 15
    assert slots[2].hour == 20
    assert all(slot.tzinfo == IST for slot in slots)


def test_schedule_never_leaks_outside_window_or_overbooks_a_day():
    slots = generate_schedule(
        8,
        interval_hours=2,
        start_from=datetime(2026, 8, 22, 1, 0, tzinfo=IST),
        day_start_hour=9,
        day_end_hour=21,
        jitter_max_minutes=55,
        spread_across_window=True,
    )

    assert len(slots) == 8
    assert slots == sorted(slots)
    assert all(9 <= slot.hour < 21 for slot in slots)
    assert max(Counter(slot.date() for slot in slots).values()) == 6


def test_schedule_uses_next_future_hour_when_day_has_started():
    slots = generate_schedule(
        3,
        interval_hours=2,
        start_from=datetime(2026, 8, 22, 10, 20, tzinfo=IST),
        day_start_hour=9,
        day_end_hour=21,
        jitter_max_minutes=0,
        spread_across_window=True,
    )

    now = datetime(2026, 8, 22, 10, 20, tzinfo=IST)
    assert all(slot > now for slot in slots)
    assert slots[0].hour == 11
    assert slots[-1].hour == 20


def test_naive_start_is_interpreted_as_ist_not_utc():
    slots = generate_schedule(
        1,
        start_from=datetime(2026, 8, 22, 1, 0),
        day_start_hour=9,
        day_end_hour=21,
        jitter_max_minutes=0,
        spread_across_window=True,
    )

    assert slots[0].utcoffset() == timedelta(hours=5, minutes=30)
    assert 9 <= slots[0].hour < 21
