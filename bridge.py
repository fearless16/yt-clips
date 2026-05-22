"""
bridge.py — The local-to-cloud bridge.
Writes a job file that the Colab watcher will pick up.

Delivery cascade:
  1. Direct Drive API (via shared utils/drive_auth.py)
  2. Tunnel (if colab_url.txt exists)
  3. Local file fallback
  4. Manual injection block
"""
import argparse
import json
import time
import sys
from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("bridge", cfg["logging"]["log_file"], cfg["logging"]["level"])


def push_job(url: str, flags: list):
    """Push a pipeline job to the Colab worker via the best available channel."""
    job = {
        "url": url,
        "flags": flags,
        "timestamp": time.time(),
        "status": "pending",
    }

    # ─── Attach secrets (client_secrets.json, yt_token.json) ────────────────
    for secret_file in ["client_secrets.json", "yt_token.json"]:
        secret_path = Path(secret_file)
        if secret_path.exists() and secret_path.stat().st_size > 0:
            job[secret_file] = secret_path.read_text(encoding="utf-8")
            log.info(f"🔑 Attached {secret_file} ({secret_path.stat().st_size} bytes)")

    # ─── Mode 1: Tunnel Bridge (PREFERRED for Kaggle) ──────────────────────────
    if _push_via_tunnel(job):
        return

    # ─── Mode 2: Direct Google Drive API ──────────────────────────────────────
    if _push_via_drive_api(job):
        return

    # ─── Mode 3: Local File Fallback ─────────────────────────────────────────
    job_path = Path("remote_job.json").absolute()
    with open(job_path, "w") as f:
        json.dump(job, f, indent=2)
    log.info(f"📂 Job saved to local folder for sync: {job_path}")

    # ─── Mode 4: Manual Injection Block ──────────────────────────────────────
    log.info("═" * 50)
    log.info("💎 MANUAL INJECTION (If all else fails)")
    log.info("═" * 50)
    inject_code = (
        f"import json; job={json.dumps(job)}; "
        f"f=open('remote_job.json','w'); json.dump(job,f); f.close(); print('✨ Injected!')"
    )
    print(inject_code)
    log.info("═" * 50)


def _push_via_drive_api(job: dict) -> bool:
    """Attempt to push job via Google Drive API. Returns True on success."""
    try:
        from utils.drive_auth import get_drive_service, find_or_create_folder, FILESYSTEM_MODE
        from googleapiclient.http import MediaIoBaseUpload
        import io

        log.info("📡 Attempting Direct Drive API upload...")
        service = get_drive_service()

        if not service or service == FILESYSTEM_MODE:
            raise ConnectionError("Drive API not available (Colab FS mode or no creds)")

        # Find or create yt-clips folder
        folder_id = find_or_create_folder(service, "yt-clips")

        # Check for existing remote_job.json to UPDATE (prevents duplicates)
        results = service.files().list(
            q=f"name = 'remote_job.json' and '{folder_id}' in parents and trashed = false",
            fields="files(id)",
        ).execute()
        existing = results.get("files", [])

        media = MediaIoBaseUpload(
            io.BytesIO(json.dumps(job, indent=2).encode("utf-8")),
            mimetype="application/json",
        )

        if existing:
            file_id = existing[0]["id"]
            service.files().update(fileId=file_id, media_body=media).execute()
            log.info("✅ Existing job file updated on Drive!")
        else:
            file_metadata = {"name": "remote_job.json", "parents": [folder_id]}
            service.files().create(body=file_metadata, media_body=media, fields="id").execute()
            log.info("✨ New job file created on Drive!")

        log.info("🚀 Job beamed to Drive. (Note: Colab mount may take 30-60s to sync)")
        return True

    except Exception as e:
        log.info(f"ℹ️ Direct Drive API not ready ({e}).")
        log.info("👉 Tip: Run 'gcloud auth application-default login' to fix this permanently.")
        return False


def _push_via_tunnel(job: dict) -> bool:
    """Attempt to push job via Colab/Kaggle tunnel URL. Returns True on success."""
    colab_url = None
    for fname in ["colab_url.txt", "kaggle_url.txt"]:
        fpath = Path(fname)
        if fpath.exists():
            colab_url = fpath.read_text().strip()
            break

    if not colab_url or not colab_url.startswith("http"):
        return False

    try:
        import requests

        log.info(f"📡 Sending via Tunnel: {colab_url}")
        response = requests.post(
            f"{colab_url}/job",
            json=job,
            headers={"bypass-tunnel-reminder": "true"},
            timeout=10,
        )
        if response.status_code in (200, 202):
            log.info("✅ Job received by remote tunnel instantly!")
            return True
        else:
            log.warning(f"⚠️ Tunnel returned error {response.status_code}")
            return False

    except Exception as e:
        log.info(f"ℹ️ Tunnel delivery skipped ({e})")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a pipeline job to Colab.")
    parser.add_argument("url", help="YouTube URL to process remotely")
    args, unknown = parser.parse_known_args()
    push_job(args.url, unknown)
