"""
sync.py — One-click Google Drive sync for exported Shorts.

Uploads all files from the shorts/ directory to Google Drive under:
  yt-clips/shorts/<date-folder>/

Supports:
  1. Google Colab filesystem mount (fastest)
  2. Google Drive API (via shared utils/drive_auth.py)
  3. Manual copy fallback

Usage:
    python sync.py                          # Sync all shorts
    python sync.py --folder shorts/2026-05-03_093000  # Sync specific folder
"""

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Optional, List

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("sync", cfg["logging"]["log_file"], cfg["logging"]["level"])


def _upload_file(service, local_path: Path, parent_folder_id: str) -> str:
    """Upload a single file to a Google Drive folder."""
    from googleapiclient.http import MediaFileUpload

    file_metadata = {
        "name": local_path.name,
        "parents": [parent_folder_id],
    }

    # Determine MIME type
    suffix = local_path.suffix.lower()
    mime_map = {
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".json": "application/json",
        ".yaml": "application/x-yaml",
        ".yml": "application/x-yaml",
        ".txt": "text/plain",
        ".log": "text/plain",
    }
    mimetype = mime_map.get(suffix, "application/octet-stream")

    media = MediaFileUpload(
        str(local_path),
        mimetype=mimetype,
        resumable=True,
        chunksize=10 * 1024 * 1024,  # 10MB chunks for large video files
    )

    log.info("⬆️  Uploading: %s (%.1f MB) …", local_path.name, local_path.stat().st_size / 1_048_576)

    request = service.files().create(
        body=file_metadata, media_body=media, fields="id,name,webViewLink"
    )

    # Upload with progress tracking
    response = None
    last_progress = -10
    last_progress_time = time.monotonic()
    while response is None:
        status, response = request.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            now = time.monotonic()
            if progress >= last_progress + 10 or now - last_progress_time >= 15 or progress >= 100:
                log.info("   Progress: %d%%", progress)
                last_progress = progress
                last_progress_time = now

    file_id = response.get("id")
    link = response.get("webViewLink", "")
    log.info("✅ Uploaded: %s → %s", local_path.name, link or file_id)
    return file_id


def sync_to_drive(
    folder_path: Optional[str] = None,
    sync_all: bool = False,
    dry_run: bool = False,
) -> List[str]:
    """
    Sync exported Shorts to Google Drive.

    Args:
        folder_path: Specific folder to sync (e.g., 'shorts/2026-05-03_093000')
        sync_all:    If True, sync all folders under shorts/
        dry_run:     If True, only log what would be uploaded
    """
    if dry_run:
        log.info("🧪 DRY RUN: No files will be uploaded.")
    shorts_dir = Path("shorts")

    if folder_path:
        folders = [Path(folder_path)]
    elif sync_all or not folder_path:
        if not shorts_dir.exists():
            log.error("No shorts/ directory found. Run the pipeline first.")
            return []
        # Find all date-stamped folders
        folders = sorted(
            [d for d in shorts_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if not folders:
            log.error("No export folders found in shorts/")
            return []
    else:
        # Sync the most recent folder
        if not shorts_dir.exists():
            log.error("No shorts/ directory found.")
            return []
        folders = sorted(
            [d for d in shorts_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if folders:
            folders = [folders[-1]]  # Most recent only

    # Authenticate via shared module
    from utils.drive_auth import get_drive_service, find_or_create_folder, FILESYSTEM_MODE

    service = get_drive_service()
    if not service:
        log.error("Cannot sync — authentication failed.")
        return []

    # ─── Filesystem Mode (Colab Mount) ────────────────────────────────────────
    if service == FILESYSTEM_MODE:
        colab_base = Path(os.environ.get("COLAB_SYNC_PATH", "/content/drive/MyDrive/yt-clips"))
        dest_base = colab_base / "shorts"
        dest_base.mkdir(parents=True, exist_ok=True)
        synced_count = 0
        for folder in folders:
            dest_folder = dest_base / folder.name
            
            # Skip if source and destination are the same physical path
            if folder.resolve() == dest_folder.resolve():
                log.info(f"⏩ Skipping {folder.name} (already on Drive mount)")
                continue
                
            log.info(f"📁 Copying {folder.name} to Drive Mount...")
            shutil.copytree(folder, dest_folder, dirs_exist_ok=True)
            synced_count += 1
        log.info(f"✅ Filesystem sync complete! {synced_count} folders copied to Drive.")
        return ["FS_MODE"]

    # ─── Drive API Mode ──────────────────────────────────────────────────────
    # Create Drive folder hierarchy: yt-clips/shorts/<date-folder>/
    root_folder_id = find_or_create_folder(service, "yt-clips")
    shorts_folder_id = find_or_create_folder(service, "shorts", root_folder_id)

    uploaded_ids: list[str] = []

    for folder in folders:
        if not folder.exists():
            log.warning("Folder not found: %s", folder)
            continue

        # Get all exportable files in the folder
        export_files = sorted(
            list(folder.glob("*.mp4"))
            + list(folder.glob("*.json"))
            + list(folder.glob("*.yaml"))
            + list(folder.glob("*.md"))
        )
        if not export_files:
            log.warning("No exportable files in %s", folder)
            continue

        # Create date-stamped subfolder in Drive
        date_folder_id = find_or_create_folder(service, folder.name, shorts_folder_id)

        log.info("─" * 50)
        log.info("Syncing: %s (%d files)", folder, len(export_files))
        log.info("─" * 50)

        for file_path in export_files:
            try:
                if dry_run:
                    log.info("🧪 [Dry Run] Would upload: %s", file_path.name)
                    uploaded_ids.append("DRY_RUN_ID")
                    continue
                file_id = _upload_file(service, file_path, date_folder_id)
                uploaded_ids.append(file_id)
            except Exception as e:
                log.error("❌ Failed to upload %s: %s", file_path.name, e)

    log.info("═" * 50)
    log.info("🎉 Sync complete! %d files uploaded to Google Drive.", len(uploaded_ids))
    log.info("   Drive path: My Drive/yt-clips/shorts/")
    log.info("═" * 50)

    return uploaded_ids


# ─── Download from Drive ─────────────────────────────────────────────────────

def download_from_drive(
    filenames: Optional[List[str]] = None,
    dest_dir: str = ".",
) -> List[Path]:
    """
    Download files from yt-clips Drive folder to local directory.

    Args:
        filenames: Specific filenames to download (e.g. ["video.mp4", "video.json"]).
                   If None, downloads everything from yt-clips/ root.
        dest_dir:  Local destination directory.

    Returns:
        List of downloaded file paths.
    """
    from utils.drive_auth import get_drive_service, find_or_create_folder, FILESYSTEM_MODE

    service = get_drive_service()
    if not service:
        log.error("Cannot download — Drive API not available (need API auth, not filesystem)")
        return []

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if service == FILESYSTEM_MODE:
        colab_base = Path(os.environ.get("COLAB_SYNC_PATH", "/content/drive/MyDrive/yt-clips"))
        if not colab_base.exists():
            log.error("Drive mount not found at %s", colab_base)
            return []
        source_filenames = filenames if filenames else [f.name for f in colab_base.iterdir() if f.is_file()]
        downloaded = []
        for fname in source_filenames:
            src = colab_base / fname
            if src.exists():
                shutil.copy2(src, dest / fname)
                downloaded.append(str(dest / fname))
                log.info("Copied %s from Drive mount", fname)
            else:
                log.warning("File not found on Drive mount: %s", fname)
        return downloaded

    # Find yt-clips folder on Drive
    folder_id = find_or_create_folder(service, "yt-clips")

    # List files in yt-clips root
    query = f"'{folder_id}' in parents and trashed = false"
    if filenames:
        name_filter = " or ".join(f"name = '{f}'" for f in filenames)
        query += f" and ({name_filter})"

    results = service.files().list(
        q=query, spaces="drive", fields="files(id, name, size)", pageSize=100
    ).execute()
    items = results.get("files", [])

    if not items:
        log.warning("No matching files found on Drive in yt-clips/")
        return []

    downloaded = []
    for item in items:
        file_id = item["id"]
        file_name = item["name"]
        file_size = int(item.get("size", 0))
        local_path = dest / file_name

        if local_path.exists() and local_path.stat().st_size == file_size:
            log.info("⏩ %s already exists (%.1f MB), skipping", file_name, file_size / 1e6)
            downloaded.append(local_path)
            continue

        log.info("📥 Downloading %s (%.1f MB)...", file_name, file_size / 1e6)
        request = service.files().get_media(fileId=file_id)

        from googleapiclient.http import MediaIoBaseDownload
        import io

        with open(local_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    pct = int(status.progress() * 100)
                    if pct % 25 == 0:
                        log.info("   %s: %d%%", file_name, pct)

        log.info("✅ Downloaded %s (%.1f MB)", file_name, local_path.stat().st_size / 1e6)
        downloaded.append(local_path)

    return downloaded


# ─── CLI entry-point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync exported Shorts to Google Drive.",
        epilog="""
Examples:
  # Sync all shorts to Drive
  python sync.py

  # Sync specific folder
  python sync.py --folder shorts/2026-05-03_093000
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--folder", "-f",
        default=None,
        help="Specific folder to sync (default: sync all under shorts/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without uploading anything",
    )
    args = parser.parse_args()

    sync_to_drive(folder_path=args.folder, sync_all=args.folder is None, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
