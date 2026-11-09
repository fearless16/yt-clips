"""Production geometry/assets for the accepted stadium design; no temp/draft imports."""

import os
import shutil
from pathlib import Path

from automation.cricket_gameplay import sha256

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "assets/gameplay/stadium-reference-v1/overlay.png"
OVERLAY_HASH = "f6a8fcd86c2f4082f12d2341d06bdb253e0ab47b156801315d201227a8efcdd5"
FONT_HASH = "10df702864b1f89cb29ba0d6b97c04228338d16807e13e8d8c74b91aba5e5f23"
CANVAS = [1080,1920]
HOST = [44,322,992,558]
GAMEPLAY = [44,1227,992,558]
CAPTIONS = [0,934,1080,224]
CAPTION_TRANSPORT_Y = 886  # 48 transparent pixels above the approved 224px text plane
LEGAL = [0,1848,1080,72]  # below GAMEPLAY VISUAL; production compliance, separately reviewed


def stadium_font():
    # Arial Black is an installed, licensed Windows font, not redistributed here.
    # No silent substitute changes the approved typeface on another machine.
    font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/ariblk.ttf"
    if not font.is_file() or sha256(font) != FONT_HASH:
        raise ValueError("Approved Arial Black font missing/changed; install licensed original and review before rendering")
    return font


def preflight_assets():
    from PIL import Image
    if not OVERLAY.is_file() or sha256(OVERLAY) != OVERLAY_HASH:
        raise ValueError("Approved stadium overlay missing/changed; restore reviewed production asset")
    with Image.open(OVERLAY) as image:
        if image.size != tuple(CANVAS) or image.mode != "RGBA":
            raise ValueError("Stadium overlay must be 1080x1920 RGBA")
    font = stadium_font()
    return {"overlay": {"path": str(OVERLAY), "sha256": OVERLAY_HASH},
            "font": {"path": str(font), "sha256": FONT_HASH}}


def _check_caption_browser():
    """Real browser boundary; the API-only dry run stubs this probe, not preflight."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ValueError("Stadium captions require Playwright; run python -m pip install -r requirements.txt") from exc
    # Share the renderer lock so concurrent per-clip preflights cannot interfere
    # with a caption browser. Always close the probe, including launch failures.
    from automation.caption_engine import _PLAYWRIGHT_LOCK
    try:
        with _PLAYWRIGHT_LOCK, sync_playwright() as p:
            executable = Path(p.chromium.executable_path)
            if not executable.is_file():
                raise ValueError("Chromium executable is missing")
            browser = p.chromium.launch(headless=True, timeout=15000)
            try:
                page = browser.new_page()
                page.set_default_timeout(5000)
                page.set_content("<!doctype html><body>caption preflight</body>")
                if page.locator("body").inner_text() != "caption preflight":
                    raise ValueError("Chromium page probe failed")
            finally:
                browser.close()
    except Exception as exc:
        raise ValueError("Stadium caption browser unavailable; run python -m playwright install chromium "
                         "and check that Chromium can launch: " + str(exc)) from exc


def preflight_runtime():
    """Check local render dependencies before download/ASR/paid packaging."""
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            raise ValueError(f"Stadium rendering requires {binary} on PATH")
    _check_caption_browser()
