"""Word-timed futuristic center HUD; 320px transport, 240px visible band."""

import html
import re

from automation.caption_design import caption_font, caption_lines

DESIGN_VERSION = "gameplay-hud-v1"
MEASURE_CAPTION_JS = """() => {
    if (!document.fonts.check('800 76px Oxanium')) return null;
    const frame = document.querySelector('.frame').getBoundingClientRect();
    const lines = [...document.querySelectorAll('.ink')].map(e => {
        const r=e.getBoundingClientRect();
        return {text:e.textContent,x:r.x,y:r.y,width:r.width,height:r.height,
                right:r.right,bottom:r.bottom};
    });
    if (lines.some(r=>r.x<116 || r.right>910 || r.y<52 || r.bottom>268)) return null;
    return {x:frame.x,y:frame.y,width:frame.width,height:frame.height,font_size:76,lines};
}"""


def caption_document(text):
    content = "".join('<div class="line"><span class="ink">'+html.escape(line)+'</span></div>'
                      for line in caption_lines(text))
    template = """<!doctype html><html><head><meta charset="utf-8"><style>
@font-face{font-family:Oxanium;src:url('FONT_URI');font-weight:200 800}
*{box-sizing:border-box;margin:0;padding:0}
body{width:1080px;height:320px;background:transparent}
.frame{position:absolute;left:98px;top:40px;width:828px;height:240px;
border:2px solid #22d3ee;border-left:7px solid #22d3ee;border-right:4px solid #8b5cf6;
clip-path:polygon(20px 0,100% 0,100% calc(100% - 20px),calc(100% - 20px) 100%,0 100%,0 20px);
background:linear-gradient(115deg,#07111ff5,#101b31f5);
display:flex;flex-direction:column;justify-content:center;align-items:center;gap:0}
.line{font:800 76px/1.12 Oxanium,sans-serif;letter-spacing:.5px;
color:#f8fafc;white-space:nowrap;text-shadow:0 2px 3px #000,0 0 8px #22d3ee45}
.line:last-child{color:#a5f3fc}.ink{display:inline-block}
</style></head><body><div class="frame">CONTENT</div></body></html>"""
    values = {"FONT_URI": caption_font().as_uri(), "CONTENT": content}
    return re.sub("FONT_URI|CONTENT", lambda m: values[m.group()], template)
