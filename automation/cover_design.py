"""Source-derived Aurora cover. Composition is not editorial/owner approval."""

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from automation.caption_design import caption_font

# This legacy cover has not been rebuilt or approved with the neon captions.
# Do not mislabel it when the independently versioned caption treatment changes.
DESIGN_VERSION = "aurora-v1"


def _font(size):
    font = ImageFont.truetype(str(caption_font()), size)
    font.set_variation_by_axes([800])
    return font


def compose_cover(source: Image.Image, crop: dict | None, text: str):
    words = text.upper().strip().split()
    if len(words) > 3 or (words and max(map(len, words)) > 24):
        raise ValueError("Cover needs an honest hook of at most three short words")
    image = source.convert("RGB")
    if crop is not None:
        x, y, w, h = (int(crop[k]) for k in ("x", "y", "width", "height"))
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > image.width or y + h > image.height:
            raise ValueError("Cover crop lies outside source")
        image = image.crop((x, y, x + w, y + h))
    background = ImageOps.fit(image, (1080, 1920), method=Image.Resampling.LANCZOS)
    background = ImageEnhance.Brightness(background.filter(ImageFilter.GaussianBlur(72))).enhance(.46)
    background = background.convert("RGBA")
    # Colored light is design decoration; the subject remains an unchanged real
    # frame. No generated expression, player, score or scenery is introduced.
    wash = Image.new("RGBA", background.size)
    painter = ImageDraw.Draw(wash)
    painter.ellipse((-360, 1060, 820, 2320), fill=(0, 154, 205, 70))
    painter.ellipse((700, -300, 1540, 700), fill=(45, 75, 185, 55))
    background = Image.alpha_composite(background, wash.filter(ImageFilter.GaussianBlur(130)))
    foreground = ImageOps.contain(image, (984, 1160), method=Image.Resampling.LANCZOS)
    position = ((1080 - foreground.width) // 2, 86)
    mask = Image.new("L", foreground.size)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, foreground.width - 1, foreground.height - 1), 28, fill=255)
    background.paste(foreground, position, mask)
    draw = ImageDraw.Draw(background)
    x, y = position
    draw.rounded_rectangle((x, y, x + foreground.width - 1, y + foreground.height - 1), 28,
                           outline=(192, 225, 242, 150), width=2)
    text_bounds = None
    font_size = None
    if words:
        if len(words) == 1:
            lines = words
        else:
            split = min(range(1, len(words)),
                        key=lambda i: (abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))), -i))
            lines = [" ".join(words[:split]), " ".join(words[split:])]
        for size in range(142, 103, -2):
            font = _font(size)
            if max(draw.textlength(line, font=font) for line in lines) <= 820:
                font_size = size
                break
        if font_size is None:
            raise ValueError("Cover hook cannot fit at a readable size")
        line_height = font_size + 14
        card_top = 1180
        height = len(lines) * line_height + 42
        card = Image.new("RGBA", background.size)
        cd = ImageDraw.Draw(card)
        cd.rounded_rectangle((72, card_top, 1008, card_top + height), 28,
                             fill=(7, 18, 34, 240), outline=(174, 224, 246, 200), width=2)
        cd.line((218, card_top, 862, card_top), fill=(132, 237, 255, 255), width=4)
        background = Image.alpha_composite(background, card)
        draw = ImageDraw.Draw(background)
        union = []
        for index, line in enumerate(lines):
            length = draw.textlength(line, font=font)
            at = ((1080 - length) / 2, card_top + 21 + index * line_height)
            # `lt` anchors the actual glyph top, avoiding font ascent surprises.
            draw.text(at, line, font=font, anchor="lt",
                      fill="#f5fbff" if index == 0 else "#8beeff", stroke_width=1)
            union.append(draw.textbbox(at, line, font=font, anchor="lt", stroke_width=1))
        text_bounds = [min(b[0] for b in union), min(b[1] for b in union),
                       max(b[2] for b in union), max(b[3] for b in union)]
    return background.convert("RGB"), {"design": DESIGN_VERSION, "text": " ".join(words),
        "source_crop": crop, "foreground_bounds": [*position, foreground.width, foreground.height],
        "text_bounds": text_bounds, "font_size": font_size, "review_status": "NOT CHECKED"}
