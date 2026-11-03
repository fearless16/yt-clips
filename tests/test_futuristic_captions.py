import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from automation import caption_engine as engine


def speech():
    return [{"start": 10, "end": 12, "text": "आज नहीं जीते",
             "words": [{"start": 10, "end": 10.5, "word": "आज"},
                       {"start": 10.5, "end": 11, "word": "नहीं"},
                       {"start": 11, "end": 12, "word": "जीते"}]}]


def fake_response(words, model="deepseek-v4-flash"):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = json.dumps({"model": model, "choices": [{"message": {
        "content": json.dumps({"words": words})}}]}).encode()
    return response


def rows():
    return [{"id": i, "roman": text, "uncertain": False, "reason": ""}
            for i, text in enumerate(["aaj", "nahi", "jeete"])]


def test_flash_actual_payload_and_word_mapping(monkeypatch, tmp_path):
    from automation.caption_hinglish import romanize_words_flash
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-not-a-real-key")
    call = Mock(return_value=fake_response(rows()))
    monkeypatch.setattr("urllib.request.urlopen", call)
    words = speech()[0]["words"]
    converted, evidence = romanize_words_flash(words, context="आज नहीं जीते")
    request = call.call_args.args[0]
    body = json.loads(request.data)
    assert body["model"] == "deepseek-v4-flash"
    assert [w["roman"] for w in converted] == ["aaj", "nahi", "jeete"]
    assert [(w["start"], w["end"]) for w in converted] == [(10, 10.5), (10.5, 11), (11, 12)]
    assert evidence["returned_model"] == "deepseek-v4-flash"
    assert evidence["listening_review"] == "NOT CHECKED"
    assert "fixture-not-a-real-key" not in json.dumps(evidence)


@pytest.mark.parametrize("bad", [
    [{"id": 0, "roman": "aaj", "uncertain": False, "reason": ""}],
    [{**r, "id": 0} for r in rows()],
    [{**r, "roman": "<b>invented</b>"} for r in rows()],
    [{**r, "roman": "आज"} for r in rows()],
    [{**r, "uncertain": "false"} for r in rows()],
])
def test_flash_invalid_output_fails_closed(monkeypatch, bad):
    from automation.caption_hinglish import romanize_words_flash
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=fake_response(bad)))
    with pytest.raises(ValueError):
        romanize_words_flash(speech()[0]["words"])


def test_flash_no_model_fallback(monkeypatch):
    from automation.caption_hinglish import romanize_words_flash
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=fake_response(rows(), "deepseek-chat")))
    with pytest.raises(ValueError, match="model"):
        romanize_words_flash(speech()[0]["words"])


def test_retired_model_successor_requires_explicit_option_and_disclosure(monkeypatch):
    from automation.caption_hinglish import romanize_words_flash
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=fake_response(rows(), "deepseek-flash")))
    with pytest.raises(ValueError, match="model"):
        romanize_words_flash(speech()[0]["words"])
    _, evidence = romanize_words_flash(speech()[0]["words"], allow_documented_successor=True)
    assert evidence["returned_model"] == "deepseek-flash"
    assert evidence["exact_original_version_satisfied"] is False
    assert "NOT original V4" in evidence["model_resolution_note"]


def test_flash_cannot_change_number(monkeypatch):
    from automation.caption_hinglish import romanize_words_flash
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=fake_response([
        {"id": 0, "roman": "280", "uncertain": False, "reason": ""}])))
    with pytest.raises(ValueError, match="number"):
        romanize_words_flash([{"word": "260", "start": 0, "end": 1}])


def test_flash_missing_key_never_calls_local(monkeypatch):
    from automation.caption_hinglish import romanize_words_flash
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    call = Mock()
    monkeypatch.setattr("urllib.request.urlopen", call)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        romanize_words_flash(speech()[0]["words"])
    call.assert_not_called()


def test_flash_network_failure_stays_explicit(monkeypatch):
    from automation.caption_hinglish import romanize_words_flash
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    call = Mock(side_effect=TimeoutError("fixture timeout"))
    monkeypatch.setattr("urllib.request.urlopen", call)
    with pytest.raises(TimeoutError):
        romanize_words_flash(speech()[0]["words"])
    assert call.call_count == 1


def test_flash_cannot_drop_negation(monkeypatch):
    from automation.caption_hinglish import romanize_words_flash
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    bad = rows()
    bad[1]["roman"] = "haan"
    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=fake_response(bad)))
    with pytest.raises(ValueError, match="negation"):
        romanize_words_flash(speech()[0]["words"])


def test_active_timed_route_uses_flash_and_preserves_evidence(monkeypatch, tmp_path):
    from automation import caption_hinglish
    converted = [{**w, "roman": roman} for w, roman in zip(speech()[0]["words"], ["aaj", "nahi", "jeete"])]
    call = Mock(return_value=(converted, {"requested_model": "deepseek-v4-flash", "uncertainties": []}))
    monkeypatch.setattr(caption_hinglish, "romanize_words_flash", call)
    render = Mock(return_value=[])
    monkeypatch.setattr(engine, "render_transparent_cards_batch", render)
    engine.prepare_clip_captions(speech(), 10, 12, tmp_path, require_word_timing=True,
                                 hinglish_model="deepseek-v4-flash")
    assert call.call_count == 1
    assert render.call_args.args[0][0].text == "aaj nahi jeete"
    assert json.loads((tmp_path / "caption_processing.json").read_text())["requested_model"] == "deepseek-v4-flash"


def test_design_uses_bundled_font_and_escapes_untrusted_markup():
    from automation.caption_design import caption_document, caption_font
    document = caption_document('<script>bad</script> JEETNE KI KOSHISH')
    assert caption_font().is_file()
    assert '<script>bad</script>' not in document
    assert '&lt;script&gt;' in document
    assert '<linearGradient' in document
    assert 'Oxanium' in document
    assert 'text-transform: uppercase' in document
    assert 'overflow-wrap: anywhere' not in document


@pytest.mark.parametrize("text", ["", "a " * 15, "x" * 90])
def test_design_refuses_empty_or_unreadable_content(text):
    from automation.caption_design import caption_document
    with pytest.raises(ValueError):
        caption_document(text)


def test_real_card_raster_has_transparent_canvas_and_approved_neon_geometry(tmp_path):
    from PIL import Image
    cues = [engine.CaptionCue(0, 1, "jeetne ki koshish", "", []),
            engine.CaptionCue(1, 2, "Australia ke khilaaf", "", [])]
    cards = engine.render_transparent_cards_batch(cues, tmp_path)
    measurements = json.loads((tmp_path / "caption_design.json").read_text())
    for card, bounds in zip(cards, measurements["cards"]):
        with Image.open(card.path) as image:
            assert image.size == (1080, 320)
            assert image.getpixel((0, 0))[3] == 0
            assert image.getpixel((540, 90))[3] > 180
        assert measurements["design"] == "neon-reference-v2"
        assert bounds["font_size"] == 72
        assert bounds["text_offset_y"] == 10
        assert bounds["y"] == 52
        assert bounds["height"] < 240
        assert [line["y"] for line in bounds["lines"]] == pytest.approx([66, 153])
        assert all(line["bottom"] < 234 for line in bounds["lines"])


@pytest.mark.parametrize("text", ["jis rate pe India", "karte hain", "65", "New Zealand 114 baar"])
def test_neon_line_breaks_preserve_every_word_and_order(text):
    from automation.caption_design import caption_lines
    assert " ".join(caption_lines(text)) == text


def test_neon_source_words_cannot_be_reinterpreted_as_template_tokens():
    from automation.caption_design import caption_document
    document = caption_document("SECOND_LINE FIRST_LINE")
    assert '<span class="ink">SECOND_LINE</span>' in document
    assert '<span class="ink">FIRST_LINE</span>' in document


def test_neon_single_word_and_wider_header_fit_without_font_shrink(tmp_path):
    cues = [engine.CaptionCue(0, 1, "65", "", []),
            engine.CaptionCue(1, 2, "New Zealand 114 baar", "", [])]
    engine.render_transparent_cards_batch(cues, tmp_path)
    bounds = json.loads((tmp_path / "caption_design.json").read_text())["cards"]
    assert len(bounds[0]["lines"]) == 1
    assert bounds[0]["lines"][0]["text"] == "65"
    assert bounds[0]["header_width"] is None
    assert bounds[0]["upper_accents_visible"] is False
    assert bounds[1]["header_width"] > 480
    assert bounds[1]["upper_accents_visible"] is False
    assert [b["font_size"] for b in bounds] == [72, 72]
    assert [(b["start"], b["end"]) for b in bounds] == [(0, 1), (1, 2)]


def test_neon_refuses_overflow_without_dropping_or_shrinking_words(tmp_path):
    with pytest.raises(ValueError, match="shorter aligned phrase"):
        engine.render_transparent_cards_batch(
            [engine.CaptionCue(0, 1, "W" * 28, "", [])], tmp_path)


def test_word_cues_can_consume_validated_roman_without_changing_timing():
    segments = speech()
    for word, roman in zip(segments[0]["words"], ["aaj", "nahi", "jeete"]):
        word["roman"] = roman
    cues = engine.word_timed_cues(segments, 10, 12)
    assert cues[0].text == "aaj nahi jeete"
    assert (cues[0].start, cues[0].end) == (0, 2)


def test_solo_reserved_layout_is_moving_blur_not_solid_pad(monkeypatch):
    import export
    monkeypatch.setitem(export.cfg["export"], "reserve_caption_band", True)
    graph = export._build_enhance_stack({"export_strategy": {"use_solo_frame": True,
        "active_crop": {"x": 580, "y": 0, "width": 830, "height": 1040}}}, use_logo=False)
    assert "crop=830:1040:580:0" in graph
    assert "split=2[portrait_bg][portrait_fg]" in graph
    assert "gblur=sigma=" in graph
    assert "[portrait_blur][portrait_fit]overlay=" in graph
    assert "pad=" not in graph
    assert "0x101624" not in graph
    assert "[portrait_fg]scale=972:998" in graph  # 998/1040 < 1; actual face is smaller
    assert "overlay=(W-w)/2:210" in graph


def test_joined_hindi_word_keeps_interval_but_counts_display_words():
    words = [{"word": "joined", "roman": "aaj ke", "start": 0, "end": .5},
             {"word": "match", "roman": "match", "start": .5, "end": 1},
             {"word": "mein", "roman": "mein", "start": 1, "end": 1.5},
             {"word": "nahi", "roman": "nahi", "start": 1.5, "end": 2}]
    cues = engine.word_timed_cues([{"words": words}], 0, 2)
    assert [c.text for c in cues] == ["aaj ke match mein", "nahi"]
    assert [(c.start, c.end) for c in cues] == [(0, 1.5), (1.5, 2)]


def test_final_auxiliary_is_not_a_separate_flashing_caption():
    words = [{"word": text, "roman": text, "start": i * .3, "end": (i + 1) * .3}
             for i, text in enumerate(["jeetne", "ki", "koshish", "karte", "hain"])]
    cues = engine.word_timed_cues([{"words": words}], 0, 1.5)
    assert [c.text for c in cues] == ["jeetne ki koshish", "karte hain"]
    assert (cues[0].start, cues[0].end) == pytest.approx((0, .9))
    assert (cues[1].start, cues[1].end) == pytest.approx((.9, 1.5))


def test_screen_share_never_gets_second_portrait_wrapper(monkeypatch):
    import export
    monkeypatch.setitem(export.cfg["export"], "reserve_caption_band", True)
    graph = export._build_enhance_stack({"export_strategy": {"is_screen_share": True}}, use_logo=False)
    assert "screen_blur" in graph
    assert "portrait_bg" not in graph
    assert "pad=" not in graph


def test_explicit_nonface_overlay_blur_survives_sanitization_and_precedes_split(monkeypatch):
    import export
    monkeypatch.setitem(export.cfg["export"], "reserve_caption_band", True)
    strategy = {"use_solo_frame": True,
        "active_crop": {"x": 410, "y": 0, "width": 1000, "height": 1040},
        "source_overlay_blurs": [{"x": 0, "y": 0, "width": 176, "height": 94}]}
    assert export._sanitize_strategy(strategy)["source_overlay_blurs"] == strategy["source_overlay_blurs"]
    graph = export._build_enhance_stack({"export_strategy": strategy}, use_logo=False)
    assert "crop=176:94:0:0,gblur=sigma=20" in graph
    assert graph.index("gblur=sigma=20") < graph.index("split=2[portrait_bg]")


def test_overlay_blur_cannot_silently_extend_outside_camera_region():
    import export
    with pytest.raises(ValueError, match="overlay blur"):
        export._sanitize_strategy({"active_crop": {"x": 0, "y": 0, "width": 1000, "height": 1040},
            "source_overlay_blurs": [{"x": 900, "y": 0, "width": 200, "height": 90}]})
