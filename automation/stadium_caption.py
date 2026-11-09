"""Accepted 96px Arial Black italic caption ink; no second plaque over the artwork."""

import html

from automation.caption_design import caption_lines
from automation.stadium_layout import stadium_font

DESIGN_VERSION = "stadium-reference-v1"
MEASURE_CAPTION_JS = """() => {
    if (!document.fonts.check('italic 900 96px Reference')) return null;
    const lines = [...document.querySelectorAll('.ink')].map(e => {
        const r=e.getBoundingClientRect(); return {text:e.textContent,x:r.x,y:r.y,
        width:r.width,height:r.height,right:r.right,bottom:r.bottom,center_x:(r.left+r.right)/2};
    });
    if (lines.some(r=>r.x<120 || r.right>960 || r.y<53 || r.bottom>267 || Math.abs(r.center_x-540)>.1)) return null;
    return {x:0,y:48,width:1080,height:224,font_size:96,lines};
}"""


def phrase_fits(text):
    """Conservative font metric for word-boundary grouping; browser is final authority."""
    from PIL import ImageFont
    try:
        lines = caption_lines(text)
    except ValueError:
        return False
    font = ImageFont.truetype(str(stadium_font()),96)
    return all(font.getlength(line.upper())+len(line) <= 830 for line in lines)


def caption_document(text):
    content = "".join('<div class="line"><span class="ink">'+html.escape(line.upper())+
                      '</span></div>' for line in caption_lines(text))
    return f"""<!doctype html><meta charset="utf-8"><style>
    @font-face{{font-family:Reference;src:url('{stadium_font().as_uri()}');font-weight:900}}
    *{{box-sizing:border-box;margin:0}}body{{width:1080px;height:320px;background:transparent}}
    .lines{{position:absolute;top:48px;width:1080px;height:224px;display:flex;
    flex-direction:column;align-items:center;justify-content:center;text-align:center}}
    .line{{font:italic 900 96px/1.04 Reference;white-space:nowrap;color:#f8fcff;
    text-shadow:0 4px 2px #0008;letter-spacing:1px}}.line:nth-child(2){{color:#09e8f3}}
    .ink{{display:inline-block}}</style><div class="lines">{content}</div>"""
