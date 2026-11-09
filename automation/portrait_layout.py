"""Full-width host portrait with one shared, face-safe caption rectangle.

The lower continuation is blurred source pixels, not synthesized torso. No
landscape inset, duplicated sharp face, independent motion or timestamp shift.
"""

import math


def caption_geometry(width=1080, height=1920):
    return {"x": round(width * 98 / 1080), "top": round(height * 1484 / 1920),
            "width": round(width * 884 / 1080), "height": round(height * 224 / 1920),
            "transport_y": round(height * 1484 / 1920) - 32}


def plan_portrait(analysis, width=1080, height=1920):
    dimensions = analysis.get("source_dimensions", {})
    strategy = analysis.get("export_strategy", {})
    face = strategy.get("active_crop") or {}
    keys = (dimensions.get("width"), dimensions.get("height"),
            face.get("face_x"), face.get("face_y"), face.get("face_w"), face.get("face_h"))
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in keys):
        return None
    sw, sh, fx, fy, fw, fh = keys
    if (min(sw, sh, fw, fh) <= 0 or min(fx, fy) < 0 or fx+fw > sw or fy+fh > sh
            or strategy.get("guest_cam_on") or strategy.get("guest_cam_off") or fh / sh <= .25):
        return None  # do not turn a genuine presentation/guest into a host-only shot
    samples = [face]
    for sample in analysis.get("face_samples", []):
        values = [sample.get(k) for k in ("face_x", "face_y", "face_w", "face_h")]
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
            continue
        sx, sy, sfw, sfh = values
        if min(sx, sy) >= 0 and min(sfw, sfh) > 0 and sx+sfw <= sw and sy+sfh <= sh:
            samples.append(sample)
    left = min(s["face_x"] for s in samples)
    right = max(s["face_x"]+s["face_w"] for s in samples)
    chin = max(s["face_y"]+s["face_h"] for s in samples)
    fw = max(s["face_w"] for s in samples)
    caption = caption_geometry(width, height)
    top = round(height / 32) // 2 * 2
    clearance = round(height / 30)
    # Keep roughly 56% face width, and constrain the chin ABOVE the card. Source
    # hair may already touch its top; never invent more hair by cropping deeper.
    crop_w = math.ceil(max(fw / .56, chin*width / (caption["top"]-top-clearance),
                           (right-left)*1.15) / 2) * 2
    crop_w = min(int(sw)//2*2, crop_w)
    crop_x = max(0, min(int(sw)-crop_w, int((left+right)/2-crop_w/2))) // 2 * 2
    crop_h = min(int(sh)//2*2, math.ceil(max(sh*.96, chin+20)/2)*2)
    bounds = analysis.get("portrait_source_bounds")
    if bounds is not None:
        bx, by, bw, bh = [bounds[k] for k in ("x", "y", "width", "height")]
        if (any(type(v) is not int or v%2 for v in (bx, by, bw, bh))
                or min(bx, by) < 0 or min(bw, bh) <= 0 or bx+bw > sw or by+bh > sh
                or by != 0 or bw < crop_w or bh < chin or bx > left or bx+bw < right):
            raise ValueError("Reviewed camera bounds cannot contain the portrait safely")
        crop_x = max(bx, min(crop_x, bx+bw-crop_w))
        crop_h = min(crop_h, bh)
    if crop_h < chin:
        raise ValueError("Source lacks a clean portrait below the chin; visual review required")
    picture_h = round(crop_h*width/crop_w) // 2 * 2
    face_bottom = top + chin*width/crop_w
    if face_bottom+clearance > caption["top"] or picture_h+top > height:
        raise ValueError("Cannot fit portrait and face-safe captions without clipping")
    return {"crop": [crop_x, 0, crop_w, crop_h], "picture_width": width,
            "picture_height": picture_h, "picture_top": top,
            "face_bottom": face_bottom, "caption_top": caption["top"],
            "caption": caption, "lower_fill": "blurred source continuation, not generated body"}
