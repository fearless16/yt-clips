"""Source-faithful packaging proofs, independent of upload/learning defaults.

An app mockup cannot certify title truth, speech, UI fit or creator approval.
All layout bounds below belong to the creator's source-limited pilot contract,
not a universal YouTube safe zone. No source pixels are synthesized or erased.
"""
import math
import unicodedata

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

from automation.caption_design import caption_font


NAVY = (7, 18, 26)
PICTURE_END = 1248
CAPTION_START = 1356


def _copy(text):
    if not isinstance(text, str) or any(unicodedata.category(c).startswith("C") for c in text):
        raise ValueError("Packaging copy must be visible single-line text")
    words = text.strip().upper().split()
    if not 1 <= len(words) <= 3 or any(len(word) > 24 for word in words):
        raise ValueError("Packaging needs one to three short words")
    return " ".join(words)


def _font(size):
    font = ImageFont.truetype(str(caption_font()), size)
    font.set_variation_by_axes([800])
    return font


def _line(draw, text, top, size, fill, max_width=900):
    font = _font(size)
    box = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    width, height = box[2] - box[0], box[3] - box[1]
    if width > max_width:
        raise ValueError("Packaging copy does not fit; shorten it instead of shrinking type")
    left = (1080 - width) / 2
    draw.text((left - box[0], top - box[1]), text, font=font, fill=fill, stroke_width=1)
    return [left, top, left + width, top + height]


def clean_cover_frames(frame_cues, *, fps):
    """Three review candidates per >=0.5s real gap; not aesthetic winners."""
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Positive finite frame rate required")
    if not isinstance(frame_cues, list):
        raise ValueError("Explicit per-frame cue list required")
    result = []
    start = None
    for index, cue in enumerate([*frame_cues, False]):
        if cue is None and start is None:
            start = index
        elif cue is not None and start is not None:
            length = index - start
            if length >= math.ceil(fps / 2):
                result.extend(sorted({start + int((length - 1) * fraction + .5)
                                      for fraction in (.25, .5, .75)}))
            start = None
    return result


def compose_opening(text):
    """Larger topic hierarchy in the108px between picture and approved card."""
    text = _copy(text)
    image = Image.new("RGBA", (1080, 108))
    draw = ImageDraw.Draw(image)
    label_bounds = _line(draw, "TOPIC", 8, 22, "#a6c2cd")
    bounds = _line(draw, text, 37, 68, "#e9fbff")
    if bounds[3] >= 108:
        raise ValueError("Opening headline exceeds its dedicated band")
    return image, {"design": "ground-package-v4", "text": text, "font_size": 68,
                   "label_bounds": label_bounds, "text_bounds": bounds,
                   "output_y": PICTURE_END, "editorial_not_verbatim_subtitle": True}


def compose_picker_cover(frame, *, frame_index, frame_cues, text):
    """Add only cover lettering to a proven blank encoded caption zone.

    The caller supplies a decoded frame from the pinned video and retains its
    frame index/hash. This pure compositor cannot verify that file provenance.
    """
    text = _copy(text)
    if frame.size != (1080, 1920):
        raise ValueError("Expected an actual1080x1920 encoded frame; no stretching")
    if (type(frame_index) is not int or not isinstance(frame_cues, list)
            or not 0 <= frame_index < len(frame_cues) or frame_cues[frame_index] is not None):
        raise ValueError("Cover frame must be a known caption-free frame index")
    # A stale/misbound frame map must not authorize painting over existing text.
    band = frame.convert("RGB").crop((0, PICTURE_END, 1080, 1552))
    background = Image.new("RGB", band.size, NAVY)
    difference = ImageChops.difference(band, background)
    error = sum(ImageStat.Stat(difference).mean) / 3
    if error > 3 or max(high for low, high in difference.getextrema()) > 12:
        raise ValueError("Cover lettering requires a blank caption zone, not erased captions")
    output = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    words = text.split()
    lines = [" ".join(words[:-1]), words[-1]] if len(words) > 1 else words
    bounds = []
    for index, line in enumerate(lines):
        bounds.append(_line(draw, line, 1278 if index == 0 else 1394,
                            124 if index == 0 else 144, "#ffffff" if index == 0 else "#28f1ff"))
    return output, {"design": "ground-package-v4", "text": text, "frame_index": frame_index,
                    "text_bounds": bounds, "blank_band_mean_rgb_error": error,
                    "picture_pixels_unchanged": True,
                    "status": "APP MOCKUP; not an assigned thumbnail",
                    "app_font_and_position_match": "NOT CHECKED"}
