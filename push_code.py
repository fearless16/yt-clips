"""
push_code.py — Smart Code Sync with MD5 Hash Checking.
Only uploads files that have actually changed.
"""
import sys
import hashlib
from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("push_code", cfg["logging"]["log_file"], cfg["logging"]["level"])

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


def push():
    """Sync local code files to Google Drive with smart change detection."""
    try:
        from utils.drive_auth import get_drive_service, find_or_create_folder, FILESYSTEM_MODE

        service = get_drive_service()
        if not service:
            log.error("Authentication failed — cannot push code.")
            return

        if service == FILESYSTEM_MODE:
            log.info("🚀 Colab filesystem mode — use direct file copy instead of push_code.")
            return

        # 1. Get or Create the 'yt-clips' folder
        folder_id = find_or_create_folder(service, "yt-clips")

        # 2. Build file list to sync (root + subdirectories + data files)
        root = Path(".")
        files_to_sync = (
            list(root.glob("*.py"))
            + list(root.glob("*.yaml"))
            + list(root.glob("*.sh"))
            + list(root.glob("*.txt"))           # requirements.txt, colab_url.txt
            + list(root.glob(".env"))            # API Keys
            + list(root.glob("utils/*.py"))
            + list(root.glob("utils/*.yaml"))     # Any config in utils/
            + list(root.glob("transcripts/*.json"))
            + list(root.glob("highlights/*.yaml"))
        )

        # Add specific files that may exist
        for special_file in ["channel_logo.png", "client_secrets.json", "yt_token.json"]:
            p = Path(special_file)
            if p.exists():
                files_to_sync.append(p)

        # Deduplicate
        files_to_sync = list(set(files_to_sync))

        log.info(f"🔄 Checking {len(files_to_sync)} files for changes...")
        updated_count = 0
        created_count = 0
        skipped_count = 0

        for file_path in sorted(files_to_sync):
            if not file_path.exists():
                continue

            # Determine target folder ID (handle subdirectories like 'utils/')
            target_folder_id = folder_id
            if len(file_path.parts) > 1:
                subfolder_name = file_path.parts[0]
                target_folder_id = find_or_create_folder(service, subfolder_name, folder_id)

            name = file_path.name
            local_md5 = get_md5(file_path)

            # Check if file exists in the target folder on Drive
            results = service.files().list(
                q=f"name = '{name}' and '{target_folder_id}' in parents and trashed = false",
                fields="files(id, md5Checksum)",
            ).execute()
            existing = results.get("files", [])

            mimetype = _get_mime_type(file_path)
            media = MediaFileUpload(str(file_path), mimetype=mimetype)

            if existing:
                drive_md5 = existing[0].get("md5Checksum")
                file_id = existing[0]["id"]

                if local_md5 == drive_md5:
                    log.debug(f"  (Skipped: {name} is up to date)")
                    skipped_count += 1
                    continue

                service.files().update(fileId=file_id, media_body=media).execute()
                log.info(f"✅ Updated: {name}")
                updated_count += 1
            else:
                file_metadata = {"name": name, "parents": [target_folder_id]}
                service.files().create(body=file_metadata, media_body=media).execute()
                log.info(f"✨ Created: {name}")
                created_count += 1

        log.info(
            f"🏁 Sync complete! "
            f"({created_count} created, {updated_count} updated, {skipped_count} unchanged)"
        )

    except Exception as e:
        log.error(f"Error during push: {e}")


if __name__ == "__main__":
    push()
