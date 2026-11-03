import pytest
from PIL import Image

from automation.cover_design import compose_cover


def source():
    image = Image.new("RGB", (1920, 1080), (70, 90, 110))
    image.paste((180, 100, 75), (700, 100, 1300, 950))
    return image


def test_cover_has_portrait_canvas_text_bounds_and_no_padding():
    image, evidence = compose_cover(source(), {"x": 580, "y": 0, "width": 830, "height": 1040},
                                    "ITNI JALDI?")
    assert image.size == (1080, 1920)
    assert evidence["design"] == "aurora-v1"
    assert evidence["text"] == "ITNI JALDI?"
    assert evidence["text_bounds"][2] <= 972
    assert evidence["text_bounds"][3] < 1560
    assert image.getpixel((540, 1850)) != (16, 22, 36)
    assert evidence["review_status"] == "NOT CHECKED"


@pytest.mark.parametrize("text", ["one two three four", "x" * 60])
def test_cover_rejects_long_copy_instead_of_truncating(text):
    with pytest.raises(ValueError):
        compose_cover(source(), {"x": 580, "y": 0, "width": 830, "height": 1040}, text)


@pytest.mark.parametrize("crop", [
    {"x": -1, "y": 0, "width": 830, "height": 1040},
    {"x": 580, "y": 0, "width": 2000, "height": 1040},
    {"x": 580, "y": 0, "width": 0, "height": 1040},
])
def test_cover_rejects_out_of_source_crop(crop):
    with pytest.raises(ValueError):
        compose_cover(source(), crop, "ITNI JALDI?")
