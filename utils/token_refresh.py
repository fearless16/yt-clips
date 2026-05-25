"""
utils/token_refresh.py — Check & refresh OAuth tokens before pushing jobs.

Called by bridge.py before attaching tokens to job payloads.
If a token cannot be refreshed silently, tells the user to run setup_auth.py.
"""

from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("token_refresh", cfg["logging"]["log_file"], cfg["logging"]["level"])

TOKEN_FILES = {
    "yt_token.json": ["https://www.googleapis.com/auth/youtube.upload"],
    "token.json": ["https://www.googleapis.com/auth/drive.file"],
    "yt_analytics_token.json": [
        "https://www.googleapis.com/auth/youtube.readonly",
        "https://www.googleapis.com/auth/yt-analytics.readonly",
    ],
}


def _needs_browser_auth():
    """Flag that gets set if any token needs a full browser OAuth re-auth."""
    needs_browser = False

    for token_file, scopes in TOKEN_FILES.items():
        path = Path(token_file)
        if not path.exists():
            log.warning(f"⚠ {token_file} not found — will need browser auth")
            needs_browser = True
            continue

        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request

            creds = Credentials.from_authorized_user_file(str(path), scopes)

            if creds and creds.valid:
                log.info(f"✓ {token_file} valid (expires {creds.expiry})")
                continue

            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    path.write_text(creds.to_json(), encoding="utf-8")
                    log.info(f"✓ {token_file} refreshed (new expiry: {creds.expiry})")
                    continue
                except Exception as e:
                    log.warning(f"⚠ {token_file} refresh failed: {e}")

            # Token exists but can't be refreshed
            log.warning(f"⚠ {token_file} needs full re-auth (no valid refresh_token)")
            needs_browser = True

        except Exception as e:
            log.warning(f"⚠ {token_file} could not be loaded: {e}")
            needs_browser = True

    return needs_browser


def ensure_fresh_tokens():
    """Check all token files, refresh silently if possible.

    Returns True if all tokens are ready, False if browser auth is needed.
    """
    try:
        import google.auth.transport.requests
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        log.warning("google-auth libraries not installed locally — can't verify tokens")
        return True

    needs_browser = _needs_browser_auth()

    if needs_browser:
        log.warning("─" * 50)
        log.warning("Some tokens need browser-based re-authentication.")
        log.warning("Run this on your Mac:")
        log.warning("  .venv/bin/python setup_auth.py")
        log.warning("Then re-push the job.")
        log.warning("─" * 50)
        return False

    return True
