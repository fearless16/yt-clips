"""Reference-style rounded dark glass and bold white type. Entirely local."""

import html
from pathlib import Path

from automation.caption_design import caption_lines, caption_font

DESIGN_VERSION = "portrait-glass-v1"

MEASURE_CAPTION_JS = """() => {
    const frame = document.querySelector('.frame').getBoundingClientRect();
    const lines = [...document.querySelectorAll('.ink')].map(e => {
        const r=e.getBoundingClientRect();
        return {text:e.textContent,x:r.x,y:r.y,width:r.width,height:r.height,
                right:r.right,bottom:r.bottom};
    });
    if (lines.some(r=>r.x<120 || r.right>960 || r.y<42 || r.bottom>248)) return null;
    return {x:frame.x,y:frame.y,width:frame.width,height:frame.height,
            font_size:78,lines};
}"""


def caption_document(text):
    font = Path("C:/Windows/Fonts/ariblk.ttf")
    if not font.is_file():
        font = caption_font()
    lines = caption_lines(text)
    content = "".join('<div class="line"><span class="ink">'+html.escape(line.upper())+'</span></div>'
                      for line in lines)
    return """<!doctype html><html><head><meta charset="utf-8"><style>
@font-face{font-family:Portrait;src:url('FONT_URI');font-weight:900}
*{box-sizing:border-box;margin:0;padding:0}
body{width:1080px;height:320px;background:transparent}
.frame{position:absolute;left:98px;top:32px;width:884px;height:224px;
border:2px solid #dce9e7b0;border-radius:38px;
background:linear-gradient(120deg,#0b2028cf,#081b20db);
display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0}
.line{font:900 78px/1.10 Portrait,sans-serif;letter-spacing:1px;
color:white;white-space:nowrap;text-align:center;text-shadow:0 3px 4px #0008}
.ink{display:inline-block}
</style></head><body><div class="frame">CONTENT</div></body></html>""".replace(
        "FONT_URI", font.as_uri()).replace("CONTENT", content)
