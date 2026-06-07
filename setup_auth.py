"""
setup_auth.py — OAuth re-authentication for YouTube & Drive.
"""
import socket
import time
from pathlib import Path
from wsgiref.simple_server import make_server, WSGIServer, WSGIRequestHandler

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

SCOPES_MAP = {
    "drive_token.json": ["https://www.googleapis.com/auth/drive.file"],
    "yt_channel_token.json": ["https://www.googleapis.com/auth/youtube.force-ssl"],
    "yt_analytics_token.json": [
        "https://www.googleapis.com/auth/youtube.readonly",
        "https://www.googleapis.com/auth/yt-analytics.readonly",
    ],
}

PROMPTS = {
    "drive_token.json": "PERSONAL Google account — for Google Drive access",
    "yt_channel_token.json": "YOUTUBE CHANNEL's Google account — for uploading videos",
    "yt_analytics_token.json": "YOUTUBE CHANNEL's Google account (same as above) — for analytics",
}


class ReuseAddrWSGIServer(WSGIServer):
    allow_reuse_address = True
    daemon_threads = True


def reauth(token_file: str, scopes: list[str]) -> bool:
    path = Path(token_file)
    if path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(path), scopes)
            if creds and creds.valid:
                print(f"[OK]   {token_file} — still valid")
                return True
            if creds and creds.expired and creds.refresh_token:
                from google.auth.transport.requests import Request
                creds.refresh(Request())
                path.write_text(creds.to_json(), encoding="utf-8")
                print(f"[OK]   {token_file} — refreshed")
                return True
        except Exception:
            pass
    else:
        print(f"[NEW]  {token_file} — not found, will create via browser")

    prompt = PROMPTS.get(token_file, f"Authorize for {token_file}")
    print(f"[REAUTH] {token_file} — {prompt}")
    flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", scopes)
    creds = flow.run_local_server(
        port=8080, open_browser=True,
        server_class=ReuseAddrWSGIServer,
    )
    path.write_text(creds.to_json(), encoding="utf-8")
    print(f"[OK]   {token_file} — saved")
    time.sleep(0.5)
    return True


def main():
    if not Path("client_secrets.json").exists():
        print("ERROR: client_secrets.json not found")
        return 1

    for token_file, scopes in SCOPES_MAP.items():
        reauth(token_file, scopes)

    print("\nAll tokens refreshed.")
    return 0


if __name__ == "__main__":
    exit(main())
