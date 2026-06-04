"""
push_code.py — Push tokens and .env to Google Drive.
Code is synced via git + tunnel; Drive only stores credentials.

Persistent files (e.g. self_learner.db) use a separate list with locked
upload semantics: NEVER delete, only create-once / update-in-place. If the
local copy is missing, restore from Drive rather than treating absence as
"the new state".
"""
import sys
import json
import hashlib
from pathlib import Path
from typing import Optional

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("push_code", cfg["logging"]["log_file"], cfg["logging"]["level"])

CACHE_FILE = Path(".push_cache.json")

TOKEN_FILES = [
    "drive_token.json",
    "yt_channel_token.json",
    "yt_analytics_token.json",
    "client_secrets.json",
    ".env",
]

# Persistent files: never deleted, only create-once / update-in-place.
# If missing locally but present on Drive, restore (don't push "missing").
PERSISTENT_FILES = [
    "self_learner.db",
]

try:
    from googleapiclient.http import MediaFileUpload
except ImportError:
    log.error("Missing libraries. Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    sys.exit(1)


def get_md5(fname: Path) -> str:
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _load_cache() -> dict:
    """Load local MD5 cache."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    """Save local MD5 cache."""
    try:
        CACHE_FILE.write_text(json.dumps(cache))
    except Exception:
        pass


def _find_file_by_name(service, name: str, parent_id: str) -> Optional[str]:
    """Search Drive for a file by name in the given folder. Returns file ID or None."""
    try:
        q = f"name='{name}' and '{parent_id}' in parents and trashed=false"
        resp = service.files().list(q=q, fields="files(id)", pageSize=10).execute()
        files = resp.get("files", [])
        if files:
            return files[0]["id"]
    except Exception:
        pass
    return None


def _upload_one(service, file_path: Path, target_folder_id: str,
                drive_file_id: str = None) -> bool:
    """Upload a single file (create or update). Prevents Drive (1) duplicates."""
    media = MediaFileUpload(str(file_path), mimetype="application/json",
                            resumable=True)
    if not drive_file_id:
        drive_file_id = _find_file_by_name(service, file_path.name, target_folder_id)
    if drive_file_id:
        service.files().update(fileId=drive_file_id, media_body=media).execute()
    else:
        file_metadata = {"name": file_path.name, "parents": [target_folder_id]}
        service.files().create(body=file_metadata, media_body=media).execute()
    return True


def push(**_kwargs) -> bool:
    """Push tokens and .env to Google Drive."""
    try:
        from utils.token_refresh import ensure_fresh_tokens
        ensure_fresh_tokens()
    except Exception as e:
        log.warning(f"Token refresh check skipped: {e}")

    try:
        from utils.drive_auth import get_drive_service, find_or_create_folder, FILESYSTEM_MODE

        service = get_drive_service()
        if not service:
            log.error("Authentication failed — cannot push.")
            return False

        if service == FILESYSTEM_MODE:
            log.info("Colab filesystem mode — tokens already on Drive mount.")
            return False

        folder_id = find_or_create_folder(service, "yt-clips")

        # ── Persistent files: locked, never deleted, restore if missing locally ──
        for fname in PERSISTENT_FILES:
            fp = Path(fname)
            _sync_persistent(service, fp, folder_id)

        files_to_sync = [Path(f) for f in TOKEN_FILES if Path(f).exists()]
        if not files_to_sync:
            log.warning("No token/.env files found to push.")
            return True

        log.info(f"Checking {len(files_to_sync)} credential files...")

        local_cache = _load_cache()
        local_hashes = {}
        changed = []
        for fp in sorted(files_to_sync):
            h = get_md5(fp)
            local_hashes[str(fp)] = h
            if local_cache.get(str(fp)) != h:
                changed.append(fp)

        if not changed:
            log.info("No changes detected. All tokens up to date.")
            return True

        # List existing Drive files
        existing = {}
        page_token = None
        while True:
            resp = service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken,files(id,name,md5Checksum)",
                pageToken=page_token, pageSize=100,
            ).execute()
            for f in resp.get("files", []):
                existing[f["name"]] = {"id": f["id"], "md5": f.get("md5Checksum")}
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        to_upload = []
        for fp in changed:
            ex = existing.get(fp.name)
            if ex and ex.get("md5") == local_hashes[str(fp)]:
                continue
            to_upload.append((fp, ex))

        if not to_upload:
            log.info("All changed files already up to date on Drive.")
            _save_cache(local_hashes)
            return True

        log.info(f"Uploading {len(to_upload)} files...")
        for fp, ex in to_upload:
            drive_id = ex["id"] if ex else None
            try:
                _upload_one(service, fp, folder_id, drive_id)
                log.info(f"  {'Updated' if drive_id else 'Created'}: {fp.name}")
            except Exception as e:
                log.warning(f"  Failed: {fp.name}: {e}")

        _save_cache(local_hashes)
        log.info(f"Done! {len(to_upload)} credential files synced.")
        return True

    except Exception as e:
        log.error(f"Error during push: {e}")
        return False


def _sync_persistent(service, fp: Path, folder_id: str) -> bool:
    """Sync a persistent (locked) file: never delete, restore if local is missing.

    Rules:
      - If local file is missing but Drive has it: restore from Drive (safety net).
      - If local exists: push (create-once / update-in-place). No delete ever.
    """
    try:
        drive_file_id = _find_file_by_name(service, fp.name, folder_id)

        if not fp.exists():
            if drive_file_id:
                log.warning(
                    f"[LOCKED] {fp.name} missing locally but exists on Drive — "
                    f"restoring from Drive (NEVER treated as 'delete')."
                )
                # Drive has no native "download to path" for binary files in
                # our stack; signal via return so the caller can handle.
                # In filesystem mode (Colab), this is just a copy.
                return _restore_from_drive(service, drive_file_id, fp)
            else:
                log.info(f"[LOCKED] {fp.name} does not exist locally or on Drive — skip.")
                return True

        # Local file exists: push update (or create on first run).
        media = MediaFileUpload(str(fp), mimetype="application/x-sqlite3",
                                resumable=True)
        if drive_file_id:
            service.files().update(fileId=drive_file_id, media_body=media).execute()
            log.info(f"[LOCKED] Updated: {fp.name} (in-place, no delete)")
        else:
            file_metadata = {
                "name": fp.name,
                "parents": [folder_id],
                "description": "LOCKED — synced by push_code.py. Never delete manually; updated in place.",
            }
            service.files().create(body=file_metadata, media_body=media).execute()
            log.info(f"[LOCKED] Created: {fp.name} (locked, will only be updated in future)")
        return True

    except Exception as e:
        log.error(f"[LOCKED] Sync failed for {fp.name}: {e}")
        return False


def _restore_from_drive(service, drive_file_id: str, dest: Path) -> bool:
    """Restore a locked file from Drive to local path. Used when local copy
    is missing (e.g. accidental `rm`)."""
    try:
        from googleapiclient.http import MediaIoBaseDownload
        import io
        request = service.files().get_media(fileId=drive_file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        dest.write_bytes(fh.getvalue())
        log.info(f"[LOCKED] Restored {dest} ({dest.stat().st_size} bytes) from Drive")
        return True
    except Exception as e:
        log.error(f"[LOCKED] Restore failed for {dest}: {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if push() else 1)
