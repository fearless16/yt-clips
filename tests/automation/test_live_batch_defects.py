"""Regressions for the real 9c12c0fc stock darkness and late SEO contamination."""

import shutil
import subprocess
import json

import pytest


def test_queries_never_use_other_players_from_the_full_stream_description():
    from automation.seo.context_engine import build_grounded_search_queries
    queries = build_grounded_search_queries(
        "India vs New Zealand ODI", "SKY, Arshdeep Singh, Rohit Sharma, Virat Kohli",
        "Virat Kohli is out, caught by Will Young off Jacob Duffy.",
        ["Suryakumar Yadav wicket", "Arshdeep Singh wicket", "Virat Kohli wicket"],
    )
    assert queries[0] == "Virat Kohli wicket"
    assert not any("Suryakumar" in q or "Arshdeep" in q or "Rohit" in q for q in queries)


def test_noisy_hindi_uses_clip_grounding_not_whole_roster_as_query_subject():
    from automation.seo.context_engine import build_cricket_evidence_pack
    pack = build_cricket_evidence_pack(
        video_title="India vs New Zealand ODI", video_description="SKY Arshdeep Singh Virat Kohli Rohit Sharma",
        clip_transcript="रोई शर्पा ने बॉल को पुल कर दिया है छे रनों के लिए",
        research_context={"player_names": ["Suryakumar Yadav", "Arshdeep Singh", "Virat Kohli", "Rohit Sharma"],
            "clip_player_names": ["Rohit Sharma"], "clip_topic_phrases": ["rohit sharma pull shot six"],
            "search_queries": ["Suryakumar Yadav wicket", "Arshdeep Singh wicket", "Rohit Sharma six"]})
    queries = pack["approved_search_queries"]
    assert queries[0] == "Rohit Sharma six"
    assert all(not any(name in q for name in ("Suryakumar", "Arshdeep", "Virat")) for q in queries)
    assert any("six" in q for q in queries)


def test_generic_team_research_cannot_consume_budget_before_exact_clip_names():
    from automation.seo.context_engine import build_grounded_search_queries
    queries = build_grounded_search_queries("India vs New Zealand ODI", "Virat Kohli Rohit Sharma SKY",
        "रोई शर्पा ने छे रनों के लिए पुल किया, तीस रन दो विकेट पर",
        [f"New Zealand wicket {i}" for i in range(20)]+["Rohit Sharma wicket"],
        clip_player_names=["Rohit Sharma"], clip_topic_phrases=["rohit sharma pull shot", "six over midwicket"])
    assert queries[0] == "Rohit Sharma six"
    assert all("Rohit Sharma" in q and "wicket" not in q for q in queries)








def test_guarded_visual_ranges_cannot_cross_dark_transitions_or_join_approvals():
    from automation.gameplay_visual_qc import clean_ranges
    actual = clean_ranges([[10, 20], [30, 40]], [[14, 14.1], [19.9, 20.1], [29.9, 30.1]])
    assert len(actual) == 3
    for got, expected in zip(actual, [[10, 13.7], [14.4, 19.6], [30.4, 40]]):
        assert got == pytest.approx(expected)


def test_selector_uses_screened_intervals_but_preserves_original_approval(tmp_path):
    from tests.test_stock_gameplay import review_library
    from automation.cricket_gameplay import load_render_library, select_gameplay
    manifest, _ = review_library(tmp_path)
    library = load_render_library(manifest)
    library["visual_screening"] = {"stock": {"clean_ranges": [[110, 140], [320, 355]], "version": 1}}
    for i in range(10):
        selection = select_gameplay(library, 27.44, i)
        clean = selection["visual_screening"]["selected_clean_range"]
        assert clean[0] <= selection["start"] < selection["end"] <= clean[1]
        assert selection["approved_range"] in [[100, 180], [300, 400]]
    with pytest.raises(ValueError, match="after visual screening"):
        select_gameplay(library, 38, "cannot loop or retime")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Real FFmpeg required")
def test_every_frame_scan_catches_a_two_frame_black_pitch_with_visible_hud(tmp_path):
    from automation.gameplay_visual_qc import scan_dark_intervals, clean_ranges
    source = tmp_path / "two-frame-transition.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=green:s=320x180:r=30:d=3", "-vf",
        "drawbox=color=black:t=fill:enable='gte(t,1)*lt(t,1.066)',drawbox=x=0:y=0:w=12:h=12:color=white:t=fill",
        "-c:v", "libx264", str(source)], check=True, capture_output=True, timeout=30)
    dark = scan_dark_intervals(source, 0, 3, [0, 0, 320, 180])
    assert len(dark) == 1
    assert dark[0][0] == pytest.approx(1, abs=.04)
    assert dark[0][1] == pytest.approx(1.0667, abs=.04)
    assert len(clean_ranges([[0, 3]], dark)) == 2


def test_visual_scan_failure_is_not_treated_as_clean(monkeypatch):
    from automation.gameplay_visual_qc import scan_dark_intervals
    monkeypatch.setattr("automation.gameplay_visual_qc.subprocess.run", lambda *a, **k:
                        subprocess.CompletedProcess([], 0, "", "frame=0"))
    with pytest.raises(ValueError, match="no frames"):
        scan_dark_intervals("missing.mp4", 0, 3)




@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Real FFmpeg required")
def test_all_black_stock_is_rejected_including_dark_ending(tmp_path):
    from automation.gameplay_visual_qc import scan_dark_intervals, clean_ranges
    source = tmp_path / "black.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=black:s=320x180:r=30:d=1", "-c:v", "libx264", str(source)],
        check=True, capture_output=True, timeout=30)
    dark = scan_dark_intervals(source, 0, 1)
    assert dark and clean_ranges([[0, 1]], dark) == []


@pytest.mark.parametrize("dark", [False, True])
def test_repeated_delivery_qc_rechecks_panel_and_cannot_erase_darkness_result(tmp_path, monkeypatch, dark):
    from automation.delivery_qc import check_delivery
    from automation.cricket_gameplay import sha256
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"synthetic decoder-boundary fixture")
    clip.with_name("clip_gameplay.json").write_text(json.dumps({
        "media_sha256": sha256(clip), "bottom": [44, 1227, 992, 558]}))
    probe = {"streams": [{"codec_type": "video", "width": 1080, "height": 1920, "start_time": "0", "duration": "10"},
                         {"codec_type": "audio", "start_time": "0", "duration": "10"}], "format": {"duration": "10"}}
    scans = []

    def decoder(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, json.dumps(probe), "")
        if "blackdetect=" in " ".join(cmd):
            scans.append(cmd)
            evidence = "frame=300\n"
            if dark:
                evidence += "black_start:9 black_end:9.1 black_duration:0.1\n"
            return subprocess.CompletedProcess(cmd, 0, "", evidence)
        return subprocess.CompletedProcess(cmd, 0, "", '{"input_i":"-16","input_tp":"-2"}')

    monkeypatch.setattr("automation.delivery_qc.subprocess.run", decoder)
    for _ in range(2):
        qc = check_delivery(clip, 10)
        assert qc["status"] == ("FAIL" if dark else "PASS")
        assert qc["gameplay_visual"]["status"] == ("FAIL" if dark else "PASS")
    assert len(scans) == 2
