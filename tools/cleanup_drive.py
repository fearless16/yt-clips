#!/usr/bin/env python3
"""
cleanup_drive.py — Clean up old/orphaned data from Google Drive.

Usage:
    python tools/cleanup_drive.py --dry-run          # Preview what would be removed
    python tools/cleanup_drive.py --all               # Remove ALL yt-clips drive data

Modes:
    --old-shorts DAYS     Remove shorts folders older than DAYS (default: 30)
    --orphan-files        Remove files in root that don't match current structure
    --duplicates          Remove duplicate files/folders (same name, keep newest)
    --all                 Remove EVERYTHING under yt-clips/ on Drive
    --dry-run             Log what would be done without actually deleting
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("cleanup_drive", cfg["logging"]["log_file"], cfg["logging"]["level"])


def _get_service():
    from utils.drive_auth import get_drive_service, FILESYSTEM_MODE
    service = get_drive_service()
    if not service:
        log.error("Drive auth failed.")
        return None
    if service == FILESYSTEM_MODE:
        log.error("Cannot cleanup in filesystem mode (Colab mount).")
        return None
    return service


def _list_all(service, parent_id: str):
    """Recursively list all files/folders under a parent."""
    all_items = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{parent_id}' in parents and trashed = false",
            fields="nextPageToken,files(id, name, mimeType, createdTime, size)",
            pageToken=page_token,
            pageSize=500,
        ).execute()
        items = resp.get("files", [])
        for item in items:
            all_items.append(item)
            if item["mimeType"] == "application/vnd.google-apps.folder":
                all_items.extend(_list_all(service, item["id"]))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return all_items


def _trash_item(service, item, dry_run: bool):
    """Trash a single Drive item (file or folder)."""
    if dry_run:
        log.info("[dry-run] Would trash: %s (id=%s, type=%s)", item["name"], item["id"], item["mimeType"].split(".")[-1])
        return
    try:
        service.files().update(fileId=item["id"], body={"trashed": True}).execute()
        log.info("Trashed: %s (id=%s)", item["name"], item["id"])
    except Exception as e:
        log.warning("Failed to trash %s: %s", item["name"], e)


def cleanup_old_shorts(service, days: int, dry_run: bool):
    """Remove shorts folders older than specified days."""
    from datetime import datetime, timezone, timedelta
    from utils.drive_auth import find_or_create_folder

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    root_id = find_or_create_folder(service, "yt-clips")
    shorts_id = find_or_create_folder(service, "shorts", root_id)

    # List subfolders under shorts/
    resp = service.files().list(
        q=f"'{shorts_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
        fields="files(id, name, createdTime)",
    ).execute()
    folders = resp.get("files", [])
    removed = 0
    for f in folders:
        created = f.get("createdTime", "")
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if created_dt < cutoff:
                _trash_item(service, f, dry_run)
                removed += 1
        except (ValueError, AttributeError):
            log.warning("Could not parse date '%s' for %s", created, f["name"])

    if dry_run:
        log.info("[dry-run] Would remove %d old shorts folders (older than %d days)", removed, days)
    else:
        log.info("Removed %d old shorts folders (older than %d days)", removed, days)


def cleanup_all(service, dry_run: bool):
    """Remove everything under yt-clips/ on Drive."""
    from utils.drive_auth import find_or_create_folder

    root_id = find_or_create_folder(service, "yt-clips")
    all_items = _list_all(service, root_id)
    # Also trash the root folder itself (but not if it's the only thing)
    for item in all_items:
        _trash_item(service, item, dry_run)
    _trash_item(service, {"name": "yt-clips", "id": root_id, "mimeType": "folder"}, dry_run)
    log.info("%s: %d items under yt-clips/", "Would remove" if dry_run else "Removed", len(all_items) + 1)


def cleanup_orphans(service, dry_run: bool):
    """Remove orphaned files in yt-clips root (not in standard structure)."""
    from utils.drive_auth import find_or_create_folder

    root_id = find_or_create_folder(service, "yt-clips")
    resp = service.files().list(
        q=f"'{root_id}' in parents and trashed = false",
        fields="files(id, name, mimeType)",
    ).execute()
    items = resp.get("files", [])
    known_folders = {"shorts", "input", "transcripts", "highlights", "output", "photos", "utils", "automation", "face_os", "tools", "logs", "data", "temp"}
    removed = 0
    for item in items:
        is_folder = item["mimeType"] == "application/vnd.google-apps.folder"
        if is_folder and item["name"] in known_folders:
            continue
        _trash_item(service, item, dry_run)
        removed += 1
    log.info("%s: %d orphan items", "Would remove" if dry_run else "Removed", removed)


def cleanup_duplicates(service, dry_run: bool):
    """Find and trash duplicate files/folders with the same name."""
    from utils.drive_auth import find_or_create_folder

    root_id = find_or_create_folder(service, "yt-clips")
    all_items = _list_all(service, root_id)
    name_map = {}
    for item in all_items:
        name_map.setdefault(item["name"].lower(), []).append(item)
    removed = 0
    for name, items in name_map.items():
        if len(items) > 1:
            items.sort(key=lambda x: x.get("createdTime", ""), reverse=True)
            for dup in items[1:]:
                _trash_item(service, dup, dry_run)
                removed += 1
    log.info("%s: %d duplicate items", "Would remove" if dry_run else "Removed", removed)


def main():
    parser = argparse.ArgumentParser(description="Clean up Google Drive data for yt-clips")
    parser.add_argument("--old-shorts", type=int, nargs="?", const=30, default=None,
                        help="Remove shorts folders older than N days (default: 30)")
    parser.add_argument("--orphan-files", action="store_true", help="Remove orphaned root files")
    parser.add_argument("--duplicates", action="store_true", help="Remove duplicate files/folders")
    parser.add_argument("--all", action="store_true", help="Remove everything under yt-clips/")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no deletions")
    args = parser.parse_args()

    if not any([args.old_shorts is not None, args.orphan_files, args.duplicates, args.all]):
        parser.print_help()
        print("\nError: Specify at least one cleanup mode.")
        sys.exit(1)

    service = _get_service()
    if not service:
        sys.exit(1)

    if args.all:
        cleanup_all(service, args.dry_run)
    else:
        if args.old_shorts is not None:
            cleanup_old_shorts(service, args.old_shorts, args.dry_run)
        if args.orphan_files:
            cleanup_orphans(service, args.dry_run)
        if args.duplicates:
            cleanup_duplicates(service, args.dry_run)

    log.info("Cleanup complete.")


if __name__ == "__main__":
    main()
