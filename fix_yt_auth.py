"""
fix_yt_auth.py — Authenticate youtube.upload scope and write yt_token.json.
"""
import json
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
SECRETS_FILE = "client_secrets.json"
TOKEN_FILE = "yt_token.json"


def main():
    secrets_path = Path(SECRETS_FILE)
    token_path = Path(TOKEN_FILE)

    if not secrets_path.exists():
        print(f"ERROR: {SECRETS_FILE} not found.")
        sys.exit(1)

    with open(secrets_path) as f:
        secrets = json.load(f)
    web = secrets.get("web", secrets.get("installed", {}))
    cid = web.get("client_id", "")
    if "test" in cid or not cid:
        print(f"ERROR: {SECRETS_FILE} contains stub credentials.")
        sys.exit(1)
    print(f"client_id: {cid[:35]}...")
    print(f"SCOPES: {SCOPES}")

    creds = None
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception:
            pass

    if creds and creds.valid:
        print("Token already valid.")
        return

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
            print("Token refreshed successfully.")
            return
        except Exception as e:
            print(f"Token refresh failed: {e}")

    # Try local server OAuth flow
    try:
        print("Starting browser OAuth flow...")
        flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
        creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        print(f"{TOKEN_FILE} written ({token_path.stat().st_size} bytes)")
        return
    except Exception as e:
        print(f"Local server flow failed: {e}")
        print("Falling back to manual auth URL flow...")

    # Manual console flow
    flow = InstalledAppFlow.from_client_secrets_file(
        str(secrets_path), SCOPES,
        redirect_uri="http://localhost:8080/"
    )
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
    print(f"\nAuth URL:\n{auth_url}\n")
    result_url = input("Paste the full redirect URL here: ").strip()
    if not result_url:
        print("No URL provided. Aborting.")
        sys.exit(1)

    flow.fetch_token(authorization_response=result_url)
    creds = flow.credentials
    token_path.write_text(creds.to_json(), encoding="utf-8")
    print(f"\n{TOKEN_FILE} written ({token_path.stat().st_size} bytes)")
    print(f"  Scopes: {creds.scopes}")
    print(f"  Expiry: {creds.expiry}")
    print("Done.")


if __name__ == "__main__":
    main()
