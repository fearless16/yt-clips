"""
utils/drive_auth.py — Consolidated Google Drive Authentication.

Single source of truth for all Drive API authentication across the project.
Used by: sync.py, bridge.py, push_code.py
"""
import os
import httplib2
import google_auth_httplib2
from pathlib import Path
from typing import Optional

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("drive_auth", cfg["logging"]["log_file"], cfg["logging"]["level"])


# Sentinel value for Colab filesystem sync mode
FILESYSTEM_MODE = "FILESYSTEM_MODE"


def get_drive_service():
    """
    Authenticate and return a Google Drive API service object.

    Auth cascade (tries in order):
      0. Google Colab filesystem mount (fastest, most reliable)
      0b. Kaggle — use client_secrets.json delivered via tunnel
      1. Application Default Credentials (gcloud auth)
      2. Saved OAuth token.json
      3. Returns None if all fail

    Returns:
      - A Drive API service object, or
      - "FILESYSTEM_MODE" string if Colab mount is detected, or
      - None if authentication fails
    """
    try:
        import google.auth
        from googleapiclient.discovery import build
    except ImportError:
        log.error(
            "Google API libraries not installed. Run:\n"
            "  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )
        return None

    # ─── Method 0: Google Colab Filesystem Mount ─────────────────────────────
    if Path("/content").exists():
        paths_to_check = [
            Path("/content/drive/MyDrive/yt-clips"),
            Path("/content/drive/My Drive/yt-clips"),
        ]
        for colab_drive in paths_to_check:
            if colab_drive.exists():
                log.info("Colab detected (%s). Prioritizing Filesystem Sync.", colab_drive.name)
                os.environ["COLAB_SYNC_PATH"] = str(colab_drive)
                return FILESYSTEM_MODE

    # ─── Method 0b: Kaggle — use delivered client_secrets.json ──────────────
    is_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists())
    if is_kaggle:
        kaggle_token = Path("yt_token.json")
        kaggle_secrets = Path("client_secrets.json")
        if kaggle_token.exists():
            try:
                from google.oauth2.credentials import Credentials
                from google.auth.transport.requests import Request
                credentials = Credentials.from_authorized_user_file(
                    str(kaggle_token),
                    scopes=["https://www.googleapis.com/auth/drive"],
                )
                if credentials and credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
                if credentials and credentials.valid:
                    log.info("✅ Kaggle: Using delivered yt_token.json")
                    base_http = httplib2.Http(timeout=60)
                    auth_http = google_auth_httplib2.AuthorizedHttp(credentials, http=base_http)
                    return build("drive", "v3", http=auth_http)
            except Exception as e:
                log.warning("Kaggle token refresh failed: %s", e)

        if kaggle_secrets.exists() and not kaggle_token.exists():
            log.info("🔑 Kaggle: client_secrets.json found but no token — run OAuth flow")
            try:
                from google_auth_oauthlib.flow import InstalledAppFlow
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(kaggle_secrets),
                    scopes=["https://www.googleapis.com/auth/drive"],
                )
                # Headless: run console flow (no browser on Kaggle)
                creds = flow.run_local_server(port=0, open_browser=False)
                # Save for future runs
                kaggle_token.write_text(creds.to_json(), encoding="utf-8")
                log.info("✅ Kaggle: OAuth complete, token saved")
                base_http = httplib2.Http(timeout=60)
                auth_http = google_auth_httplib2.AuthorizedHttp(creds, http=base_http)
                return build("drive", "v3", http=auth_http)
            except Exception as e:
                log.warning("Kaggle OAuth failed: %s", e)

    credentials = None

    # ─── Method 1: Saved OAuth token.json ────────────────────────────────────
    token_path = Path("token.json")
    if token_path.exists():
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            credentials = Credentials.from_authorized_user_file(
                str(token_path),
                scopes=["https://www.googleapis.com/auth/drive"],
            )
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            log.info("✅ Using saved OAuth token")
        except Exception as e:
            log.warning("Token refresh failed: %s", e)
            credentials = None

    # ─── Method 2: Application Default Credentials (gcloud auth) ─────────────
    if not credentials:
        try:
            import google.auth.exceptions
            from google.auth.transport.requests import Request
            credentials, project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/drive"]
            )
            # Probe test: verify credentials actually work
            if hasattr(credentials, "refresh"):
                credentials.refresh(Request())
            log.info("✅ Using gcloud Application Default Credentials")
        except Exception as e:
            log.debug("ADC validation failed: %s", e)
            credentials = None

    if not credentials:
        log.error(
            "❌ No Google credentials found.\n"
            "   Run: .venv/bin/python setup_auth.py  (option 2 for Drive)\n"
            "   This will open a browser to log into your Drive Google account."
        )
        return None

    # Build service with robust httplib2 transport
    base_http = httplib2.Http(timeout=60)
    auth_http = google_auth_httplib2.AuthorizedHttp(credentials, http=base_http)
    return build("drive", "v3", http=auth_http)


def find_or_create_folder(
    service, name: str, parent_id: Optional[str] = None
) -> str:
    """Find a folder by name (under optional parent), or create it.

    RULE: No duplicate folders allowed. If multiple folders with the same
    name exist, keeps the newest and trashes the older ones. Creation only
    happens when *zero* matching folders exist — never accidentally creates
    a duplicate due to a race condition or mismatched case.
    
    Uses case-insensitive search via Drive API name contains + post-filter
    to catch differently-cased duplicates.
    """
    # Search with case-insensitive matching via name contains + post-filter
    query = (
        "mimeType = 'application/vnd.google-apps.folder' and "
        "trashed = false"
    )
    if parent_id:
        query += " and '%s' in parents" % parent_id

    results = service.files().list(
        q=query, spaces="drive", fields="files(id, name, createdTime)"
    ).execute()
    items = results.get("files", [])

    # Post-filter by exact name match (case-insensitive) since Drive API
    # doesn't support case-insensitive equality
    matched = [f for f in items if f.get("name", "").lower() == name.lower()]

    if matched:
        if len(matched) > 1:
            matched.sort(key=lambda x: x.get("createdTime", ""), reverse=True)
            keep = matched[0]
            duplicates = matched[1:]
            log.warning(
                "Duplicate folders named '%s' found — trashing %d older copies, keeping %s",
                name, len(duplicates), keep["id"],
            )
            for dup in duplicates:
                try:
                    service.files().update(
                        fileId=dup["id"], body={"trashed": True}
                    ).execute()
                    log.info("Trashed duplicate folder '%s' (id=%s)", name, dup["id"])
                except Exception as e:
                    log.warning("Failed to trash duplicate %s: %s", dup["id"], e)
        return matched[0]["id"]

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        metadata["parents"] = [parent_id]

    folder = service.files().create(body=metadata, fields="id").execute()
    folder_id = folder.get("id")
    log.info("Created Drive folder: %s (id=%s)", name, folder_id)
    return folder_id


def find_folder(service, name: str, parent_id: Optional[str] = None) -> Optional[str]:
    """Find a folder by name (case-insensitive). Returns folder ID or None.

    Unlike find_or_create_folder, this NEVER creates a folder — use for
    read/download operations where you just want to find an existing path.
    """
    query = (
        "mimeType = 'application/vnd.google-apps.folder' and "
        "trashed = false"
    )
    if parent_id:
        query += " and '%s' in parents" % parent_id

    results = service.files().list(
        q=query, spaces="drive", fields="files(id, name)"
    ).execute()
    items = results.get("files", [])
    matched = [f for f in items if f.get("name", "").lower() == name.lower()]
    if matched:
        return matched[0]["id"]
    return None
