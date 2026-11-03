"""Prepare local listening fixtures, not verified captions or release approvals.

python -m automation.caption_review SOURCE TRANSCRIPT MANIFEST OUTPUT
MANIFEST is a JSON list of {id, start, end, tags?} in source seconds.
Open OUTPUT/index.html, listen, then record corrections in listening.json.
"""

import argparse
import html
import json
import math
import subprocess
import tempfile
from pathlib import Path

from automation.artifacts import atomic_json, validate_clip_id
from automation.release_review import sha256


def create_packet(source: Path, transcript: Path, manifest: list[dict], output: Path) -> dict:
    source, transcript, output = (Path(path).resolve() for path in (source, transcript, output))
    if output.exists():
        raise FileExistsError(f"Preserve existing listening work: {output}")
    if not isinstance(manifest, list) or not 1 <= len(manifest) <= 30:
        raise ValueError("Manifest requires 1–30 listening cases")
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(source),
    ], check=True, capture_output=True, text=True, timeout=60)
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid source duration")
    ids = set()
    for case in manifest:
        if not isinstance(case, dict):
            raise ValueError("Invalid listening case")
        case_id = validate_clip_id(case.get("id"))
        if case_id in ids:
            raise ValueError("Duplicate listening case ID")
        ids.add(case_id)
        start, end = case.get("start"), case.get("end")
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
                   for t in (start, end)) or not 0 <= start < end <= duration or end-start > 60:
            raise ValueError("Invalid listening range; use at most 60 seconds within source")
        if not isinstance(case.get("tags", []), list) or not all(isinstance(t, str) for t in case.get("tags", [])):
            raise ValueError("Tags must be a list of strings")
    original = json.loads(transcript.read_text(encoding="utf-8"))
    cues = original if isinstance(original, list) else original["segments"]
    if not isinstance(cues, list):
        raise ValueError("Transcript requires a cue list")
    invalid_cue_ids = []
    for cue_id, cue in enumerate(cues):
        if not isinstance(cue, dict) or not all(
            isinstance(cue.get(k), (int, float)) and not isinstance(cue[k], bool) and math.isfinite(cue[k])
            for k in ("start", "end")
        ) or not 0 <= cue["start"] < cue["end"]:
            invalid_cue_ids.append(cue_id)
    payload = {
        "schema_version": 1, "status": "NOT CHECKED",
        "purpose": "Listening correction fixtures; not word alignment, audience benchmark or release approval",
        "source": {"path": str(source), "sha256": sha256(source), "duration": duration},
        "transcript": {"path": str(transcript), "sha256": sha256(transcript)},
        "invalid_cue_ids": invalid_cue_ids, "cases": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="caption-listening-", dir=output.parent) as temporary:
        folder = Path(temporary) / "packet"
        folder.mkdir()
        sections = []
        for case in manifest:
            start, end = case["start"], case["end"]
            before, after = max(0, start-2), min(duration, end+2)
            filename = f"{case['id']}.wav"
            audio = folder / filename
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{before:.6f}", "-i", str(source),
                "-t", f"{after-before:.6f}", "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(audio),
            ], check=True, capture_output=True, timeout=180)
            if audio.stat().st_size == 0:
                raise ValueError("Empty listening audio")
            context = [{**cue, "cue_id": i} for i, cue in enumerate(cues)
                       if i not in invalid_cue_ids and cue["end"] > before and cue["start"] < after]
            payload["cases"].append({
                "id": case["id"], "tags": case.get("tags", []),
                "source_range": [start, end], "listening_range": [before, after],
                "audio": {"path": filename, "sha256": sha256(audio)},
                "cue_context": context, "cue_context_scope": "Unverified intersecting cues; not exact-cut speech",
                "status": "NOT CHECKED", "corrected_text": "", "uncertainties": [],
                "reviewer": "", "listened_at": "", "notes": "",
            })
            text = "\n".join(f"[{c['start']:.3f}–{c['end']:.3f}] {c.get('text', '')}" for c in context)
            sections.append(f"<section><h2>{html.escape(case['id'])}: {start}–{end}s</h2>"
                            f"<p>Audio includes context: {before}–{after}s. NOT CHECKED.</p>"
                            f'<audio controls preload="none" src="{filename}"></audio>'
                            f"<pre>{html.escape(text)}</pre></section>")
        atomic_json(folder / "listening.json", payload)
        (folder / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Caption listening review</title>'
            '<style>body{max-width:900px;margin:2rem auto;padding:1rem;font:18px sans-serif}'
            'pre{white-space:pre-wrap}section{border-top:1px solid #aaa;padding:1rem 0}</style>'
            '<h1>Caption listening review — NOT CHECKED</h1>'
            '<p>Listen to every range. Record exact speech for the source range in listening.json; '
            'context outside it is not caption text. Check names, scores, numbers and negations. '
            'Leave unclear speech unresolved; do not guess from cricket knowledge. '
            'Record your name and listening date only after actual listening. '
            'Corrections still require word alignment and full export review. This packet grants no approval.</p>'
            + f"<p>Invalid cue IDs excluded (not repaired): {invalid_cue_ids}</p>"
            + "".join(sections), encoding="utf-8")
        folder.rename(output)  # Never overwrite an existing review directory.
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "transcript", "manifest", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    create_packet(args.source, args.transcript,
                  json.loads(args.manifest.read_text(encoding="utf-8")), args.output)
    print(f"NOT CHECKED: open {args.output / 'index.html'}; nothing approved or uploaded")


if __name__ == "__main__":
    main()
