"""
push_code.py — Fast Code Sync with batch Drive API + local MD5 cache.
Only uploads files that have actually changed.
"""
import sys
import json
import hashlib
from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("push_code", cfg["logging"]["log_file"], cfg["logging"]["level"])

CACHE_FILE = Path(".push_cache.json")

try:
    from googleapiclient.http import MediaFileUpload
except ImportError:
    log.error("Missing libraries. Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    sys.exit(1)


# MIME type map for common file types
MIME_MAP = {
    ".py": "text/x-python",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".json": "application/json",
    ".sh": "text/x-shellscript",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def get_md5(fname: Path) -> str:
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _get_mime_type(file_path: Path) -> str:
    """Get MIME type for a file based on its extension."""
    return MIME_MAP.get(file_path.suffix.lower(), "application/octet-stream")


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


def _batch_list_folder(service, folder_id: str) -> dict:
    """List ALL files in a folder in one paginated call. Returns {name: {id, md5, size}}."""
    result = {}
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken,files(id,name,md5Checksum,size)",
            pageToken=page_token,
            pageSize=1000,
        ).execute()
        for f in resp.get("files", []):
            result[f["name"]] = {
                "id": f["id"],
                "md5": f.get("md5Checksum"),
                "size": int(f.get("size", 0)),
            }
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return result


def _batch_list_all_folders(service, folder_id: str, subfolders: dict) -> dict:
    """List files in root + all subfolders. Returns {folder_name: {name: {id, md5, size}}}."""
    result = {"__root__": _batch_list_folder(service, folder_id)}
    for sub_name, sub_id in subfolders.items():
        result[sub_name] = _batch_list_folder(service, sub_id)
    return result


def _upload_one(service, file_path: Path, target_folder_id: str,
                drive_file_id: str = None) -> bool:
    """Upload a single file (create or update)."""
    mimetype = _get_mime_type(file_path)
    media = MediaFileUpload(str(file_path), mimetype=mimetype, resumable=True)
    if drive_file_id:
        service.files().update(fileId=drive_file_id, media_body=media).execute()
    else:
        file_metadata = {"name": file_path.name, "parents": [target_folder_id]}
        service.files().create(body=file_metadata, media_body=media).execute()
    return True


def push(include_data: bool = False) -> bool:
    """Sync local code files to Google Drive with batch API + local cache."""
    try:
        from utils.drive_auth import get_drive_service, find_or_create_folder, FILESYSTEM_MODE

        service = get_drive_service()
        if not service:
            log.error("Authentication failed — cannot push code.")
            return False

        if service == FILESYSTEM_MODE:
            log.info("🚀 Colab filesystem mode — use direct file copy instead of push_code.")
            return False

        # 1. Get or Create the 'yt-clips' folder
        folder_id = find_or_create_folder(service, "yt-clips")

        # 2. Build file list to sync
        root = Path(".")
        files_to_sync = (
            list(root.glob("*.py"))
            + list(root.glob("*.yaml"))
            + list(root.glob("*.sh"))
            + list(root.glob("*.txt"))
            + list(root.glob(".env"))
            + list(root.glob("utils/*.py"))
            + list(root.glob("utils/*.yaml"))
            + list(root.glob("transcripts/*.json"))
            + list(root.glob("highlights/*.yaml"))
            + list(root.glob("photos/*.png"))
            + list(root.glob("photos/*.jpg"))
            + list(root.glob("photos/*.jpeg"))
        )

        if include_data:
            log.info("📦 Including input data (video) in sync...")
            files_to_sync += list(root.glob("input/*.mp4"))
            files_to_sync += list(root.glob("input/*.json"))

        for special_file in [
            "channel_logo.png", "client_secrets.json", "yt_token.json",
            "remote_job.json", "colab_url.txt",
        ]:
            p = Path(special_file)
            if p.exists():
                files_to_sync.append(p)

        files_to_sync = list(set(files_to_sync))
        files_to_sync = [f for f in files_to_sync if f.exists()]

        log.info(f"🔄 Checking {len(files_to_sync)} files for changes...")

        # 3. Compute local MD5s and compare with cache
        local_cache = _load_cache()
        local_hashes = {}
        changed_files = []
        for fp in sorted(files_to_sync):
            h = get_md5(fp)
            local_hashes[str(fp)] = h
            if local_cache.get(str(fp)) != h:
                changed_files.append(fp)

        if not changed_files:
            log.info("✅ No changes detected (local cache hit). Skipped all.")
            return True

        log.info(f"  {len(changed_files)} files changed locally.")

        # 4. Resolve subfolder IDs (pre-create them)
        subfolder_map = {}  # folder_name -> folder_id
        for fp in changed_files:
            if len(fp.parts) > 1:
                subfolder_name = fp.parts[0]
                if subfolder_name not in subfolder_map:
                    subfolder_map[subfolder_name] = find_or_create_folder(
                        service, subfolder_name, folder_id)

        # 5. Batch list Drive contents (1 call per folder instead of N per file)
        drive_files = _batch_list_all_folders(service, folder_id, subfolder_map)

        # 6. Determine what to upload
        to_upload = []
        for fp in changed_files:
            sub = fp.parts[0] if len(fp.parts) > 1 else "__root__"
            drive_folder = drive_files.get(sub, {})
            existing = drive_folder.get(fp.name)
            if existing and existing.get("md5") == local_hashes[str(fp)]:
                continue  # Already up to date on Drive
            to_upload.append((fp, existing))

        if not to_upload:
            log.info("✅ All changed files already up to date on Drive.")
            _save_cache(local_hashes)
            return True

        log.info(f"  Uploading {len(to_upload)} files...")

        # 7. Sequential upload (parallel causes SSL issues with Drive API)
        updated = 0
        created = 0
        for fp, existing in to_upload:
            sub = fp.parts[0] if len(fp.parts) > 1 else "__root__"
            target_id = subfolder_map.get(sub, folder_id)
            drive_id = existing["id"] if existing else None
            try:
                _upload_one(service, fp, target_id, drive_id)
                if existing:
                    updated += 1
                    log.info(f"✅ Updated: {fp.name}")
                else:
                    created += 1
                    log.info(f"✨ Created: {fp.name}")
            except Exception as e:
                log.warning(f"⚠️ Failed: {fp.name}: {e}")

        # 8. Save cache
        _save_cache(local_hashes)

        log.info(
            f"🏁 Sync done! ({created} created, {updated} updated, "
            f"{len(files_to_sync) - len(to_upload)} skipped)"
        )
        return True

    except Exception as e:
        log.error(f"Error during push: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sync local code files to Google Drive.")
    parser.add_argument("--include-data", action="store_true", help="Include input/ video files")
    args = parser.parse_args()
    sys.exit(0 if push(include_data=args.include_data) else 1)
