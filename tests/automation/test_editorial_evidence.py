from automation.editorial_evidence import transcript_coverage


def test_coverage_unions_overlapping_cues_and_discloses_gaps():
    result = transcript_coverage([{"start": 2, "end": 6}, {"start": 5, "end": 8}, {"start": 10, "end": 12}], 15)
    assert result["covered_seconds"] == 8
    assert result["untranscribed_intervals"] == [[0, 2], [8, 10], [12, 15]]
    assert result["audio_visual_coverage"] == "NOT CHECKED"


def test_invalid_or_out_of_bounds_cues_are_not_coverage():
    result = transcript_coverage([{"start": float("nan"), "end": 3}, {"start": 4, "end": 2}, {"start": 1, "end": 20}], 10)
    assert result["invalid_cues"] == [0, 1, 2]
    assert result["covered_seconds"] == 0
    assert result["untranscribed_intervals"] == [[0, 10]]
