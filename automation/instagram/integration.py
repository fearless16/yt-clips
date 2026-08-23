"""automation/instagram/integration.py

publish_everywhere — parallel per-clip fan-out across YouTube and
Instagram with independent failure domains (PLAN.md §frozen contracts):

- ThreadPoolExecutor(max_workers=2); each platform's worker captures its
  own exceptions into an {"error": str} entry — a failure or skip on one
  platform NEVER affects the other's result, and nothing propagates.
- skip_instagram resolution order: explicit arg > env
  YT_CLIPS_SKIP_INSTAGRAM=1 > config instagram.enabled == false > False.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SKIP_ENV_VAR = "YT_CLIPS_SKIP_INSTAGRAM"


def resolve_skip_instagram(skip_instagram=None) -> bool:
    if skip_instagram is not None:
        return bool(skip_instagram)
    if os.environ.get(_SKIP_ENV_VAR) == "1":
        return True
    try:
        from utils.config import load_config

        cfg = load_config()
    except Exception:
        return False
    insta_cfg = cfg.get("instagram") or {}
    return not bool(insta_cfg.get("enabled", False))


def _find_clip_video(clip_dir: Path) -> Path:
    videos = sorted(Path(clip_dir).glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no .mp4 in {clip_dir}")
    return videos[0]


def _run_youtube_worker(clip_dir, transcript, video_title,
                        video_description, privacy) -> dict:
    from automation.seo.seo import generate_seo_for_exported_clip
    from upload import upload_video

    clip_dir = Path(clip_dir)
    video_path = _find_clip_video(clip_dir)
    result = generate_seo_for_exported_clip(
        clip_id=video_path.stem,
        transcript=transcript,
        output_dir=str(clip_dir),
        video_title=video_title,
        video_description=video_description,
    )
    if result.get("_seo_failed"):
        raise RuntimeError(
            f"YouTube SEO generation failed for {video_path.stem}")
    metadata_path = clip_dir / f"{video_path.stem}_metadata.json"
    video_id = upload_video(str(video_path), str(metadata_path),
                            privacy=privacy)
    return {"video_id": video_id, "metadata_path": str(metadata_path)}


def _run_instagram_worker(clip_dir, transcript, video_title,
                          video_description):
    from automation.instagram import runner

    return runner.process_instagram_for_clip(
        clip_dir, transcript, video_title, video_description)


def publish_everywhere(clip_dir, transcript, video_title,
                       video_description, *, skip_youtube=False,
                       skip_instagram=None, privacy="public") -> dict:
    """Run YouTube SEO/upload and Instagram publish in parallel.

    Returns {"youtube": {...}|None, "instagram": media_id|None}; a
    platform exception becomes {"error": "<Type>: <msg>"} in its slot.
    """
    result = {"youtube": None, "instagram": None}

    def _youtube():
        try:
            result["youtube"] = _run_youtube_worker(
                clip_dir, transcript, video_title, video_description,
                privacy)
        except Exception as exc:
            result["youtube"] = {"error": f"{type(exc).__name__}: {exc}"}

    def _instagram():
        try:
            result["instagram"] = _run_instagram_worker(
                clip_dir, transcript, video_title, video_description)
        except Exception as exc:
            result["instagram"] = {"error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        if not skip_youtube:
            futures.append(pool.submit(_youtube))
        if not resolve_skip_instagram(skip_instagram):
            futures.append(pool.submit(_instagram))
        for future in futures:
            future.result()
    return result
