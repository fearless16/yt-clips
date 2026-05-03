"""
clean_drive.py — Utility to remove duplicate or stale job files from Google Drive.
Run this if the Colab worker is getting confused by old jobs.
"""
from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("clean_drive", cfg["logging"]["log_file"], cfg["logging"]["level"])


def clean():
    """Remove all remote_job.json files from Drive and local disk."""
    log.info("🧹 Cleaning up job files on Google Drive...")

    try:
        from utils.drive_auth import get_drive_service, FILESYSTEM_MODE

        service = get_drive_service()
        if not service:
            log.error("Authentication failed — cannot clean Drive.")
            return

        if service == FILESYSTEM_MODE:
            # Colab filesystem mode — just delete local files
            import os
            colab_base = Path(os.environ.get("COLAB_SYNC_PATH", "/content/drive/MyDrive/yt-clips"))
            remote_job = colab_base / "remote_job.json"
            if remote_job.exists():
                remote_job.unlink()
                log.info("✅ Cleaned remote_job.json from Drive mount.")
            else:
                log.info("ℹ️ No job files found on Drive mount.")
        else:
            # Drive API mode
            from utils.drive_auth import find_or_create_folder

            # 1. Find the yt-clips folder(s)
            results = service.files().list(
                q="name = 'yt-clips' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
                fields="files(id, name)",
            ).execute()
            folders = results.get("files", [])

            if not folders:
                log.info("ℹ️ No 'yt-clips' folder found on Drive.")
            else:
                for folder in folders:
                    folder_id = folder["id"]
                    log.info(f"📂 Checking folder: {folder['name']} ({folder_id})")

                    # 2. Find all remote_job.json files in this folder
                    results = service.files().list(
                        q=f"name = 'remote_job.json' and '{folder_id}' in parents and trashed = false",
                        fields="files(id, name)",
                    ).execute()
                    files = results.get("files", [])

                    if not files:
                        log.info("   ✅ No job files found.")
                        continue

                    log.info(f"   🗑️ Found {len(files)} job files. Deleting...")
                    for f in files:
                        service.files().delete(fileId=f["id"]).execute()
                        log.info(f"      Deleted: {f['id']}")

            # Also clean result files
            for folder in folders:
                folder_id = folder["id"]
                results = service.files().list(
                    q=f"name = 'remote_job_result.json' and '{folder_id}' in parents and trashed = false",
                    fields="files(id, name)",
                ).execute()
                for f in results.get("files", []):
                    service.files().delete(fileId=f["id"]).execute()
                    log.info(f"      Deleted result: {f['id']}")

        # 3. Clean local copies
        for local_file in ["remote_job.json", "remote_job_result.json"]:
            p = Path(local_file)
            if p.exists():
                p.unlink()
                log.info(f"✅ Local '{local_file}' deleted.")

        log.info("✨ Drive is now clean. Ready for a fresh start!")

    except Exception as e:
        log.error(f"❌ Error during cleanup: {e}")
        log.info("👉 Tip: Run 'gcloud auth application-default login' if you have permission issues.")


if __name__ == "__main__":
    clean()
