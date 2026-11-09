"""Production stadium compositor: accepted geometry, approved gameplay crops and legal footer."""

import json
import math
import subprocess
import textwrap
from pathlib import Path

from automation.cricket_gameplay import STADIUM_COMPOSITION, credit_text
from automation.stadium_layout import (CANVAS, HOST, GAMEPLAY, CAPTIONS, CAPTION_TRANSPORT_Y,
                                      LEGAL, OVERLAY, preflight_assets, stadium_font)


def validate_gameplay_crop(asset):
    crop = asset.get("crop")
    if (not isinstance(crop,list) or len(crop) != 4
            or any(type(v) is not int or v%2 for v in crop)
            or min(crop[:2]) < 0 or min(crop[2:]) <= 0 or crop[2]*9 != crop[3]*16):
        raise ValueError("Stadium gameplay requires an explicitly reviewed even-pixel 16:9 crop in its manifest")
    probe = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height,sample_aspect_ratio","-of","json",asset["path"]],
        check=True,capture_output=True,text=True,timeout=20)
    info = json.loads(probe.stdout)["streams"][0]
    if crop[0]+crop[2] > int(info["width"]) or crop[1]+crop[3] > int(info["height"]):
        raise ValueError("Reviewed gameplay crop exceeds actual source dimensions")
    if info.get("sample_aspect_ratio", "1:1") not in ("1:1", "N/A"):
        raise ValueError("Non-square gameplay pixels require separate crop review")
    return crop


def _legal_image(path, disclaimer):
    from PIL import Image, ImageDraw, ImageFont
    # Keep GAMEPLAY VISUAL intact; publisher text starts below it from frame 1.
    font = ImageFont.truetype(str(stadium_font()),24)
    lines = textwrap.wrap(disclaimer, width=66) if disclaimer else []
    if len(lines) > 2:
        raise ValueError("Publisher disclaimer needs explicit layout review: exceeds legal footer")
    image = Image.new("RGBA",(1080,72),(3,15,23,245))
    draw = ImageDraw.Draw(image)
    for i,line in enumerate(lines):
        width = draw.textlength(line,font=font)
        if width > 1016:
            raise ValueError("Publisher disclaimer exceeds readable footer width")
        draw.text(((1080-width)/2,5+i*30),line,font=font,fill="#f8fcff")
    image.save(path)


def build_stadium_command(source,start,end,output,selection,analysis,cards,scratch,options):
    from automation.gameplay_export import _duration, source_audio_filter
    duration = end-start
    if not math.isfinite(start+end) or start < 0 or duration <= 0:
        raise ValueError("Invalid retained source range")
    if end > _duration(source)+.001 or selection["end"] > _duration(selection["asset"]["path"])+.001:
        raise ValueError("Selected source/gameplay range exceeds actual media duration")
    if abs(selection["end"]-selection["start"]-duration) > .000001:
        raise ValueError("Gameplay duration differs from retained commentary")
    design_assets = preflight_assets()
    crop = validate_gameplay_crop(selection["asset"])
    dims = analysis["source_dimensions"]
    sw,sh = int(dims["width"]),int(dims["height"])
    if sw <= 0 or sh <= 0:
        raise ValueError("Invalid commentary dimensions")
    fps = int(options.get("fps",30))
    if not 1 <= fps <= 60:
        raise ValueError("Gameplay FPS must be 1-60")
    x,y,w,h = HOST
    gx,gy,gw,gh = GAMEPLAY
    cx,cy,cw,ch = crop
    # Full original commentary frame, aspect fitted (no portrait detector ROI).
    graph = (f"color=c=black:s=1080x1920:r={fps}:d={duration:.6f}[bg];"
        f"[0:v]trim=duration={duration:.6f},setpts=PTS-STARTPTS,fps={fps},"
        f"scale={w}:{h}:flags=lanczos:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[host];"
        f"[1:v]trim=duration={duration:.6f},setpts=PTS-STARTPTS,fps={fps},"
        f"crop={cw}:{ch}:{cx}:{cy},scale={gw}:{gh}:flags=lanczos,setsar=1[game];"
        f"[bg][host]overlay={x}:{y}:shortest=1[a];[a][game]overlay={gx}:{gy}:shortest=1[b];"
        "[b][2:v]overlay=0:0:repeatlast=1[art]")
    cmd = ["ffmpeg","-nostdin","-n","-v","error","-xerror","-ss",f"{start:.6f}",
        "-t",f"{duration:.6f}","-i",str(source),"-ss",f"{selection['start']:.6f}",
        "-t",f"{duration:.6f}","-i",selection["asset"]["path"],"-i",str(OVERLAY)]
    label = Path(scratch)/"publisher-disclaimer.png"
    _legal_image(label,selection["disclaimer"])
    cmd += ["-i",str(label)]
    graph += f";[art][3:v]overlay={LEGAL[0]}:{LEGAL[1]}:repeatlast=1[legal]"
    last = "legal"
    for index,card in enumerate(cards,4):
        if not 0 <= card.start < card.end <= duration+.02:
            raise ValueError("Caption outside retained source range")
        cmd += ["-i",card.path]
        graph += (f";[{last}][{index}:v]overlay=0:{CAPTION_TRANSPORT_Y}:repeatlast=1:"
                  f"enable='gte(t,{card.start:.6f})*lt(t,{card.end:.6f})'[c{index}]")
        last = f"c{index}"
    graph += f";[{last}]format=yuv420p[vout]"
    audio_filter, audio_evidence = source_audio_filter(source, start, duration)
    cmd += ["-filter_complex",graph,"-map","[vout]","-map","0:a:0","-af", audio_filter,
        "-c:v","libx264","-preset",str(options.get("encoder_preset","medium")),"-crf",str(options.get("crf",18)),
        "-c:a","aac","-b:a","192k","-t",f"{duration:.6f}","-movflags","+faststart",
        "-map_metadata","-1","-map_chapters","-1",
        "-metadata","artist=Cricket With Prajjwal","-metadata","comment="+credit_text(selection),str(output)]
    return cmd, {"composition":STADIUM_COMPOSITION,"canvas":CANVAS,"top":HOST,"bottom":GAMEPLAY,
        "captions":CAPTIONS,"caption_transport_y":CAPTION_TRANSPORT_Y,"caption_style":"stadium-reference-v1",
        "source_crop":[0,0,sw,sh],"gameplay_crop":crop,"legal_footer":LEGAL,
        "design_assets":design_assets,"selection":selection,
        "fit":"Full source aspect fit; equal 16:9 panels; reviewed gameplay fill crop; identical angular masks",
        "audio":"source input 0 only; no fades; natural speed", "audio_normalization": audio_evidence,
        "verification":"Owner visual/listening/rights review required; footer compliance strip is new to production"}
