"""Packaging proofs are not publishing or speech-certification mechanisms."""
import math

import pytest
from PIL import Image, ImageChops

from automation.packaging_design import clean_cover_frames, compose_opening, compose_picker_cover


def test_gap_candidates_stay_inside_real_gap_not_guessed_time():
    cues = [0] * 454 + [None] * 32 + [1] * 716
    result = clean_cover_frames(cues, fps=30)
    assert result == [462, 470, 477]
    assert all(cues[index] is None for index in result)
    assert clean_cover_frames([0, None, 1], fps=30) == []
    assert clean_cover_frames([0] * 100, fps=30) == []


@pytest.mark.parametrize("fps", [True, 0, -1, math.inf, math.nan, "30"])
def test_gap_candidates_reject_bad_rate(fps):
    with pytest.raises(ValueError):
        clean_cover_frames([None] * 60, fps=fps)


def test_opening_fits_own_band_without_touching_caption_or_picture():
    image, metadata = compose_opening("GROUND BHI DEKHO!")
    assert image.mode == "RGBA" and image.size == (1080, 108)
    assert metadata["font_size"] == 68
    assert metadata["editorial_not_verbatim_subtitle"] is True
    assert metadata["output_y"] == 1248
    assert metadata["output_y"] + metadata["text_bounds"][3] < 1356
    assert image.getbbox()[3] <= 108


@pytest.mark.parametrize("text", ["", "one two three four", "x" * 60, "a\nb", "GROUND\u202eDEKHO"])
def test_design_refuses_empty_long_multiline_or_hidden_control_copy(text):
    with pytest.raises(ValueError):
        compose_opening(text)


def test_picker_cover_uses_actual_frame_and_keeps_picture_and_ui_reserve_exact():
    frame = Image.new("RGB", (1080, 1920), (7, 18, 26))
    frame.paste((145, 75, 25), (0, 0, 1080, 1248))
    before = frame.copy()
    output, metadata = compose_picker_cover(frame, frame_index=1, frame_cues=[0, None, 1],
                                             text="GROUND BHI DEKHO!")
    assert ImageChops.difference(frame, before).getbbox() is None
    assert ImageChops.difference(output.crop((0, 0, 1080, 1248)), frame.crop((0, 0, 1080, 1248))).getbbox() is None
    assert ImageChops.difference(output.crop((0, 1552, 1080, 1920)), frame.crop((0, 1552, 1080, 1920))).getbbox() is None
    assert ImageChops.difference(output, frame).getbbox() is not None
    assert metadata["frame_index"] == 1
    assert metadata["status"] == "APP MOCKUP; not an assigned thumbnail"
    assert metadata["picture_pixels_unchanged"] is True


@pytest.mark.parametrize("index", [-1, 0, 2, 3, 1.0, True])
def test_captioned_missing_and_ambiguous_frame_indices_are_rejected(index):
    with pytest.raises(ValueError):
        compose_picker_cover(Image.new("RGB", (1080, 1920), (7, 18, 26)),
                             frame_index=index, frame_cues=[0, None, 1], text="GROUND BHI DEKHO!")


def test_stale_gap_map_cannot_erase_caption_pixels():
    frame = Image.new("RGB", (1080, 1920), (7, 18, 26))
    frame.paste("white", (100, 1356, 960, 1552))
    with pytest.raises(ValueError, match="blank"):
        compose_picker_cover(frame, frame_index=0, frame_cues=[None], text="GROUND BHI DEKHO!")


def test_non_portrait_frame_cannot_be_silently_stretched():
    with pytest.raises(ValueError, match="1080x1920"):
        compose_picker_cover(Image.new("RGB", (1920, 1080)), frame_index=0,
                             frame_cues=[None], text="GROUND BHI DEKHO!")


def test_even_tiny_stale_caption_pixel_cannot_be_hidden_by_low_average_error():
    frame = Image.new("RGB", (1080, 1920), (7, 18, 26))
    frame.putpixel((540, 1400), (255, 255, 255))
    with pytest.raises(ValueError, match="blank"):
        compose_picker_cover(frame, frame_index=0, frame_cues=[None], text="GROUND BHI DEKHO!")


def test_three_words_still_cannot_overflow_or_force_smaller_type():
    with pytest.raises(ValueError, match="fit"):
        compose_opening("MMMMMMMM MMMMMMMM MMMMMMMM")
