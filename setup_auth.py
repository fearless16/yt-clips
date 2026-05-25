"""
setup_auth.py — Interactive token setup for YouTube & Google Drive.

Prompts you through browser-based OAuth for each service so you can
log into the correct Google account for each one.

Usage:
  .venv/bin/python setup_auth.py
"""

import json
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request


SECRETS_FILE = "client_secrets.json"
YOUTUBE_TOKEN_FILE = "yt_token.json"
DRIVE_TOKEN_FILE = "token.json"
ANALYTICS_TOKEN_FILE = "yt_analytics_token.json"

YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
ANALYTICS_SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


def _check_secrets():
    path = Path(SECRETS_FILE)
    if not path.exists():
        print(f"ERROR: {SECRETS_FILE} not found in project root.")
        print("Create a Google Cloud OAuth 2.0 client (Desktop app type) and download it.")
        sys.exit(1)
    with open(path) as f:
        secrets = json.load(f)
    web = secrets.get("web", secrets.get("installed", {}))
    cid = web.get("client_id", "")
    if "test" in cid or not cid:
        print(f"ERROR: {SECRETS_FILE} contains stub/placeholder credentials.")
        sys.exit(1)
    print(f"  Client ID: {cid[:35]}...")
    return path


def _run_oauth_flow(secrets_path: Path, scopes: list, token_path: Path, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Token file: {token_path.name}")
    print(f"  Scopes: {scopes}")
    print(f"{'='*60}")

    creds = None
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), scopes)
        except Exception as e:
            print(f"  (could not load existing token: {e})")

    if creds and creds.valid:
        print(f"  ✓ Existing token is still valid.")
        return

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
            print(f"  ✓ Token refreshed and saved to {token_path.name}")
            return
        except Exception as e:
            print(f"  Refresh failed: {e}")

    print(f"  Opening browser for OAuth...")
    print(f"  IMPORTANT: Log into the CORRECT Google account for this service.")
    print()
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), scopes)
    try:
        creds = flow.run_local_server(port=0, open_browser=True)
    except Exception as e:
        print(f"  Browser flow failed: {e}")
        print("  Falling back to manual flow (copy-paste URL)...")
        flow = InstalledAppFlow.from_client_secrets_file(
            str(secrets_path), scopes, redirect_uri="http://localhost:8080/"
        )
        auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
        print(f"\n  Auth URL (open in browser):\n  {auth_url}\n")
        result_url = input("  Paste the full redirect URL here: ").strip()
        if not result_url:
            print("  No URL provided. Skipping.")
            return
        flow.fetch_token(authorization_response=result_url)
        creds = flow.credentials

    token_path.write_text(creds.to_json(), encoding="utf-8")
    print(f"  ✓ {token_path.name} saved ({token_path.stat().st_size} bytes)")
    print(f"    Expires: {creds.expiry}")
    print(f"    Account: {creds.account if hasattr(creds, 'account') and creds.account else 'see token.json'}")

    # Verify token can refresh
    try:
        creds.refresh(Request())
        print(f"    ✓ Token refresh verified")
    except Exception as e:
        print(f"    ⚠ Token refresh failed: {e}")
        print(f"      The token may work for one session but will need re-auth sooner.")


def main():
    print("=" * 60)
    print("  Token Setup for YouTube & Google Drive")
    print("  You will be asked to log into different Google accounts.")
    print("=" * 60)

    secrets_path = _check_secrets()

    while True:
        print(f"\n{'─'*60}")
        print("  What do you want to set up?")
        print(f"{'─'*60}")
        print("  1) YouTube upload token   (yt_token.json)")
        print("     - used for uploading videos to your YouTube channel")
        print("     - log into the Google account that OWNS the channel")
        print()
        print("  2) Google Drive token      (token.json)")
        print("     - used for syncing files to Google Drive")
        print("     - log into the Drive account (may be different from YouTube)")
        print()
        print("  3) YouTube Analytics token (yt_analytics_token.json)")
        print("     - used for reading analytics / trends data")
        print("     - log into your YouTube channel's Google account")
        print()
        print("  4) ALL of the above")
        print("  0) Exit")
        print()

        choice = input("  Enter choice [0-4]: ").strip()
        print()

        if choice == "0":
            print("  Done.")
            break
        elif choice == "1":
            _run_oauth_flow(secrets_path, YOUTUBE_SCOPES, Path(YOUTUBE_TOKEN_FILE), "YouTube Upload Token")
        elif choice == "2":
            _run_oauth_flow(secrets_path, DRIVE_SCOPES, Path(DRIVE_TOKEN_FILE), "Google Drive Token")
        elif choice == "3":
            _run_oauth_flow(secrets_path, ANALYTICS_SCOPES, Path(ANALYTICS_TOKEN_FILE), "YouTube Analytics Token")
        elif choice == "4":
            _run_oauth_flow(secrets_path, YOUTUBE_SCOPES, Path(YOUTUBE_TOKEN_FILE), "YouTube Upload Token")
            _run_oauth_flow(secrets_path, DRIVE_SCOPES, Path(DRIVE_TOKEN_FILE), "Google Drive Token")
            _run_oauth_flow(secrets_path, ANALYTICS_SCOPES, Path(ANALYTICS_TOKEN_FILE), "YouTube Analytics Token")
        else:
            print("  Invalid choice.")
            continue

        print(f"\n  ✅ {['Skipped', 'Completed'][choice != '0']}")

    print(f"\n{'='*60}")
    print("  Setup complete.")
    print("  Token files created/refreshed in project root:")
    for f in [YOUTUBE_TOKEN_FILE, DRIVE_TOKEN_FILE, ANALYTICS_TOKEN_FILE]:
        p = Path(f)
        status = f"{p.stat().st_size:>6} bytes" if p.exists() else "  MISSING"
        print(f"    {f:<30} {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
