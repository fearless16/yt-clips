"""
bridge.py — The local-to-cloud bridge.
Tunnel-only delivery to Colab watcher. Drive is credentials-only.

Falls back to local remote_job.json + manual injection if tunnel is down.
"""
import argparse
import json
import time
from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("bridge", cfg["logging"]["log_file"], cfg["logging"]["level"])


def push_job(url: str, flags: list):
    """Push a pipeline job to the Colab worker via tunnel.

    Args:
        url: YouTube URL to process.
        flags: Additional CLI flags for the pipeline.
    """
    try:
        from utils.token_refresh import ensure_fresh_tokens
        ensure_fresh_tokens()
    except Exception as e:
        log.warning(f"Token refresh check skipped: {e}")

    job = {
        "url": url,
        "flags": flags,
        "timestamp": time.time(),
        "status": "pending",
    }

    # ─── Attach credential files (embedded in the job payload) ─────
    cred_patterns = ["cookies.txt", ".env", "*token*.json", "client_secret*.json", "client_secrets.json"]
    cred_files: list[Path] = []
    for p in cred_patterns:
        cred_files.extend(Path(".").glob(p))
    for secret_path in sorted(set(cred_files)):
        if secret_path.exists() and secret_path.stat().st_size > 0:
            key = secret_path.name
            job[key] = secret_path.read_text(encoding="utf-8")
            log.info(f"Attached {key} ({secret_path.stat().st_size} bytes)")

    # ─── Tunnel delivery ──────────────────────────────────────────
    if _push_via_tunnel(job):
        return

    # ─── Local file fallback + manual injection ───────────────────
    job_path = Path("remote_job.json").absolute()
    with open(job_path, "w") as f:
        json.dump(job, f, indent=2)
    log.info(f"Tunnel unavailable — job saved to {job_path}")
    log.info("Upload to Colab manually or run Cell 3 on Colab to pick it up.")


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

        log.info(f"Sending via Tunnel: {colab_url}")
        response = requests.post(
            f"{colab_url}/job",
            json=job,
            headers={"bypass-tunnel-reminder": "true"},
            timeout=10,
        )
        if response.status_code in (200, 202):
            log.info("Job received by remote tunnel instantly!")
            return True
        else:
            log.warning(f"Tunnel returned error {response.status_code}")
            return False

    except Exception as e:
        log.info(f"Tunnel delivery skipped ({e})")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a pipeline job to Colab.")
    parser.add_argument("url", help="YouTube URL to process remotely")
    args, unknown = parser.parse_known_args()
    push_job(args.url, unknown)
