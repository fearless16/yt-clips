"""
utils/token_refresh.py — Check & refresh OAuth tokens before pushing jobs.

Called by pipeline.py and push_code.py before authenticated operations.
If a token cannot be refreshed silently, tells the user to run setup_auth.py.
"""

from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("token_refresh", cfg["logging"]["log_file"], cfg["logging"]["level"])

TOKEN_FILES = {
    "drive_token.json": ["https://www.googleapis.com/auth/drive.file"],
    "yt_channel_token.json": ["https://www.googleapis.com/auth/youtube.force-ssl"],
    "yt_analytics_token.json": [
        "https://www.googleapis.com/auth/youtube.readonly",
        "https://www.googleapis.com/auth/yt-analytics.readonly",
    ],
}


def _find_tokens_needing_auth() -> list[str]:
    """Return token files that need full browser OAuth re-auth."""
    bad = []

    for token_file, scopes in TOKEN_FILES.items():
        path = Path(token_file)
        if not path.exists():
            log.warning(f"⚠ {token_file} not found — will need browser auth")
            bad.append(token_file)
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
            bad.append(token_file)

        except Exception as e:
            log.warning(f"⚠ {token_file} could not be loaded: {e}")
            bad.append(token_file)

    return bad


def _reauth_browser(token_file: str) -> bool:
    """Open the interactive browser OAuth flow for a single token."""
    try:
        from setup_auth import reauth
        return bool(reauth(token_file, TOKEN_FILES[token_file]))
    except Exception as e:
        log.error(f"❌ Re-auth failed for {token_file}: {e}")
        return False


def ensure_fresh_tokens(auto_reauth: bool = False) -> bool:
    """Check all token files, refresh silently if possible.

    When ``auto_reauth`` is True, tokens that cannot be refreshed trigger an
    interactive browser OAuth flow (setup_auth.reauth) instead of just warning.

    Returns True if all tokens are ready, False otherwise.
    """
    try:
        import google.auth.transport.requests
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        log.warning("google-auth libraries not installed locally — can't verify tokens")
        return True

    bad = _find_tokens_needing_auth()

    if not bad:
        log.info("✓ All OAuth tokens valid")
        return True

    if not auto_reauth:
        log.warning("─" * 50)
        log.warning("Some tokens need browser-based re-authentication:")
        for token_file in bad:
            log.warning(f"  - {token_file}")
        log.warning("Run:")
        log.warning("  python setup_auth.py")
        log.warning("Then re-push the job.")
        log.warning("─" * 50)
        return False

    log.warning("Opening browser to re-authenticate: %s", ", ".join(bad))
    for token_file in list(bad):
        if _reauth_browser(token_file):
            log.info(f"✓ {token_file} re-authenticated")
        else:
            log.error(f"❌ {token_file} still invalid after re-auth")
            return False
    return True
