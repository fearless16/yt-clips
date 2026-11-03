"""Creator-approved angular cyan captions, using the bundled local font.

No remote font loading, markup from transcripts or tiny-type auto-fit fallback.
The 1080x320 PNG is a transport canvas, not a visible 320px caption slab.
"""

import html
import re
from pathlib import Path

DESIGN_VERSION = "neon-reference-v2"
CANVAS = (1080, 320)

# Browser-measured fitting widens only the stepped header. It never shrinks the
# approved 72px lettering, drops words, or changes source cue intervals.
MEASURE_CAPTION_JS = """() => {
    if (!document.fonts.check('italic 800 72px Oxanium')) return null;
    const single = document.body.classList.contains('single');
    const first = document.querySelector('.first');
    const second = document.querySelector('.second');
    const topInk = first.querySelector('.ink');
    const bottomInk = second.querySelector('.ink');
    const topWidth = single ? 0 : topInk.getBoundingClientRect().width;
    const bottomWidth = bottomInk.getBoundingClientRect().width;
    if (topWidth > 608 || bottomWidth > 750) return null;
    const headerWidth = Math.max(480, Math.ceil(topWidth + 112));
    const accentsVisible = !single && headerWidth <= 540;
    document.querySelector('.top-accent').style.visibility = accentsVisible ? 'visible' : 'hidden';
    const left = 563 - headerWidth / 2, right = 563 + headerWidth / 2;
    if (!single) {
        first.style.width = (headerWidth + 20) + 'px';
        first.style.left = (556 - (headerWidth + 20) / 2) + 'px';
    }
    const outer = single
        ? 'M126 120H936L964 148L935 248H124L99 221Z'
        : `M126 120H${left-26}L${left} 52H${right}L${right-19} 120H936L964 148L935 248H124L99 221Z`;
    for (const path of document.querySelectorAll('.outer')) path.setAttribute('d', outer);
    document.querySelector('.inner').setAttribute('d', single
        ? 'M131 133H927L947 153L922 234H132L116 216'
        : `M131 133H${left-18}L${left+8} 65H${right-13}M${right-26} 133H927L947 153L922 234H132L116 216`);
    const lines = [...document.querySelectorAll('.line')]
        .filter(e => getComputedStyle(e).display !== 'none').map(e => {
            const ink = e.querySelector('.ink').getBoundingClientRect();
            return {text:e.textContent, x:ink.x, y:ink.y, width:ink.width,
                    height:ink.height, right:ink.right, bottom:ink.bottom};
        });
    if (lines.some(r => r.x < 126 || r.right > 936 || r.bottom > 234)) return null;
    const frame = document.querySelector('.frame').getBBox();
    return {x:frame.x,y:frame.y,width:frame.width,height:frame.height,
            font_size:72,header_width:single ? null : headerWidth,
            text_offset_y:10,upper_accents_visible:accentsVisible,lines};
}"""


def caption_font() -> Path:
    font = Path(__file__).resolve().parents[1] / "assets/fonts/oxanium/Oxanium-Variable.ttf"
    if not font.is_file():
        raise FileNotFoundError("Bundled OFL Oxanium font missing")
    return font


def caption_lines(text: str) -> list[str]:
    words = text.strip().split()
    if not words or len(words) > 8 or max(map(len, words)) > 28:
        raise ValueError("Caption requires a shorter aligned phrase, not smaller type")
    # Balance only at complete word boundaries; the browser measures the result.
    lines = [words]
    if len(words) >= 2:
        cut = min(range(1, len(words)),
                  key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
        lines = [words[:cut], words[cut:]]
    return [" ".join(line) for line in lines]


def caption_document(text: str) -> str:
    lines = caption_lines(text)
    first, second = lines if len(lines) == 2 else ("", lines[0])
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><style>
@font-face{font-family:Oxanium;src:url('FONT_URI');font-weight:200 800}
*{box-sizing:border-box;margin:0;padding:0}
body{width:1080px;height:320px;background:transparent}
.graphic{position:relative;width:1080px;height:320px}
svg{position:absolute;inset:0;width:1080px;height:320px}
.line{position:absolute;text-align:center;font:800 italic 72px/1.06 Oxanium,sans-serif;letter-spacing:1.6px;white-space:nowrap;transform:skewX(-6deg);text-transform: uppercase;text-shadow:0 3px 2px #000,0 0 12px #a5faff40}
.ink{display:inline-block}
.first{left:306px;top:66px;width:500px;color:#f9feff}
.second{left:145px;top:153px;width:790px;color:#28f1ff;text-shadow:0 3px 2px #001219,0 0 14px #02efff66}
.single .first{display:none}.single .second{top:146px}
</style></head><body class="BODY_CLASS"><div class="graphic">
<svg viewBox="0 0 1080 320" aria-hidden="true"><defs>
<linearGradient id="glass" x1="0" y1="0" x2="0.9" y2="1"><stop stop-color="#001923" stop-opacity=".94"/><stop offset=".5" stop-color="#031017" stop-opacity=".91"/><stop offset="1" stop-color="#00232c" stop-opacity=".93"/></linearGradient>
<linearGradient id="edge"><stop stop-color="#04d7e9" stop-opacity=".1"/><stop offset=".22" stop-color="#1ff5ff"/><stop offset=".75" stop-color="#48f8ff"/><stop offset="1" stop-color="#02d7e9" stop-opacity=".28"/></linearGradient>
<filter id="glow" x="-20%" y="-30%" width="140%" height="160%"><feGaussianBlur stdDeviation="4"/></filter>
</defs>
<path class="outer" d="M126 120H297L323 52H803L784 120H936L964 148L935 248H124L99 221Z" fill="none" stroke="#00e9fa" stroke-width="9" opacity=".30" filter="url(#glow)"/>
<path class="outer frame" d="M126 120H297L323 52H803L784 120H936L964 148L935 248H124L99 221Z" fill="url(#glass)" stroke="url(#edge)" stroke-width="2"/>
<path class="inner" d="M131 133H305L331 65H790M777 133H927L947 153L922 234H132L116 216" fill="none" stroke="#33f3ff" stroke-width="1.2" opacity=".45"/>
<path d="M126 161L107 216L126 235H149M895 123H936L956 147L950 170" fill="none" stroke="#44faff" stroke-width="7"/>
<g class="top-accent"><path d="M127 95H226" stroke="#06d9e9" stroke-width="8"/>
<path d="M239 91L233 100M253 91L247 100M267 91L261 100M281 91L275 100" stroke="#54faff" stroke-width="6"/></g>
<path d="M720 264H921" stroke="#08d8e9" stroke-width="2"/>
<path d="M737 260L731 269M751 260L745 269M765 260L759 269M779 260L773 269" stroke="#43f7ff" stroke-width="5"/>
</svg><div class="line first"><span class="ink">FIRST_LINE</span></div><div class="line second"><span class="ink">SECOND_LINE</span></div>
</div></body></html>"""
    replacements = {"FONT_URI": caption_font().as_uri(),
                    "BODY_CLASS": "single" if len(lines) == 1 else "two-line",
                    "FIRST_LINE": html.escape(first), "SECOND_LINE": html.escape(second)}
    # Substitute once, so source words resembling template tokens stay literal.
    return re.sub(r"FONT_URI|BODY_CLASS|FIRST_LINE|SECOND_LINE",
                  lambda match: replacements[match.group()], document)
