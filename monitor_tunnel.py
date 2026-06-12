"""
monitor_tunnel.py — Poll tunnel health/files, auto-resume on failure.
Prevents duplicate uploads at all costs.
Usage:
    .venv/bin/python monitor_tunnel.py <tunnel-url> <video-url>
"""
import json
import ssl
import sys
import time
import urllib.request

TUNNEL_URL = sys.argv[1] if len(sys.argv) > 1 else "https://wiry-rubble-boring.ngrok-free.dev"
VIDEO_URL = sys.argv[2] if len(sys.argv) > 2 else "https://www.youtube.com/watch?v=AMJZhbYwEgk"
POLL_SEC = 20

ctx = ssl._create_unverified_context()

_AUTH_FILES = [
    "cookies.txt", ".env", "client_secrets.json",
    "yt_channel_token.json", "yt_analytics_token.json",
    "drive_token.json",
]


def _get(path: str) -> dict | None:
    try:
        r = urllib.request.urlopen(f"{TUNNEL_URL}{path}", timeout=15, context=ctx)
        return json.loads(r.read().decode())
    except Exception as e:
        print(f"  [ERR] {path}: {e}")
        return None


def _post(path: str, data: dict) -> dict | None:
    try:
        body = json.dumps(data).encode()
        r = urllib.request.urlopen(
            f"{TUNNEL_URL}{path}", data=body, timeout=15, context=ctx,
        )
        return json.loads(r.read().decode())
    except Exception as e:
        print(f"  [ERR] POST {path}: {e}")
        return None


def determine_skip_flags(files: dict) -> list[str]:
    flags = []
    if files.get("transcripts"):
        if any(f["name"].endswith(".json") for f in files["transcripts"]):
            flags.append("--skip-transcribe")
            print("  ✅ Checkpoint: transcript exists → --skip-transcribe")
    if files.get("input"):
        if any(f["name"].endswith(".mp4") for f in files["input"]):
            flags.append("--skip-download")
            print("  ✅ Checkpoint: video exists → --skip-download")
    if files.get("highlights"):
        if any(f["name"].endswith(".yaml") for f in files["highlights"]):
            flags.extend(["--skip-download", "--skip-transcribe", "--skip-highlight"])
            print("  ✅ Checkpoint: highlights exist → skip-download/transcribe/highlight")
    if files.get("shorts"):
        if any(f["name"].endswith(".mp4") for f in files["shorts"]):
            flags.extend(["--skip-download", "--skip-transcribe", "--skip-highlight", "--skip-export"])
            print("  ✅ Checkpoint: clips exist → skip-download/transcribe/highlight/export")
    return flags


def remove_auth_files():
    """Delete auth creds on remote so re-submit CANNOT upload again."""
    print("  🚫 Removing auth files to prevent duplicate upload...")
    for f in _AUTH_FILES:
        _post("/exec", {"cmd": f"rm -f {f}"})


def kill_remote_pipeline():
    print("  🔪 Killing remote pipeline processes...")
    _post("/exec", {"cmd": "pkill -9 -f 'automation.cli' 2>/dev/null; pkill -9 -f 'python.*automation.cli' 2>/dev/null; echo done"})


def kill_remote_watcher():
    """Kill the entire watcher server on Colab."""
    print("  🛑 Killing remote watcher...")
    _post("/exec", {"cmd": "pkill -9 -f watcher.py 2>/dev/null; pkill -9 -f 'python.*watcher' 2>/dev/null; echo done"})


def resubmit(flags: list[str]):
    print(f"  🔄 Re-submitting with flags: {flags}")
    job = {"url": VIDEO_URL, "flags": flags}
    try:
        r = urllib.request.urlopen(
            f"{TUNNEL_URL}/job", data=json.dumps(job).encode(),
            timeout=30, context=ctx,
        )
        if r.status == 202:
            print("  ✅ Job re-submitted successfully")
        else:
            print(f"  ⚠️  Resubmit returned {r.status}")
    except Exception as e:
        print(f"  [ERR] Resubmit failed: {e}")


def check_uploaded_flag(files: dict) -> bool:
    """Check shorts dir for uploaded marker metadata."""
    shorts = files.get("shorts", [])
    if any(f["name"].endswith("_metadata.json") for f in shorts):
        return True
    return False


def main():
    print(f"Monitoring tunnel: {TUNNEL_URL}")
    print(f"Video: {VIDEO_URL}")
    print(f"Poll interval: {POLL_SEC}s")
    print(f"⚠️  DUPLICATE UPLOAD PREVENTION: Auth files wiped on any re-submit")
    print()

    while True:
        health = _get("/health")
        if health is None:
            time.sleep(POLL_SEC)
            continue

        processing = health.get("processing", False)
        files = _get("/files") or {}
        status_data = _get("/download/status.json") or {}
        result_data = _get("/download/remote_job_result.json")

        status = status_data.get("status", "unknown") if status_data else "unknown"
        print(f"[{time.strftime('%H:%M:%S')}] processing={processing} status={status}")

        for dirname in ("input", "transcripts", "highlights", "shorts"):
            entries = files.get(dirname, [])
            if entries:
                names = ", ".join(f["name"] for f in entries)
                print(f"  📁 {dirname}/: {names}")

        # ── Pipeline finished ──
        if result_data:
            rstatus = result_data.get("status")
            rc = result_data.get("returncode")
            print(f"  🏁 Result: status={rstatus} returncode={rc}")
            if rstatus == "done" and rc == 0:
                print("  ✅ Pipeline completed successfully!")
                kill_remote_watcher()
                print("  🛑 Watcher killed. Pipeline fully done.")
                return 0
            else:
                print("  ❌ Pipeline failed. Resuming from last checkpoint (no re-upload)...")
                flags = determine_skip_flags(files)
                remove_auth_files()
                kill_remote_pipeline()
                time.sleep(3)
                resubmit(flags)
                print("  👍 Resubmitted. Continuing to monitor...")
                time.sleep(POLL_SEC)
                continue

        # ── Status is failed but no result file yet ──
        if status in ("failed", "interrupted"):
            print("  ❌ Pipeline reported failure. Resuming from checkpoint...")
            flags = determine_skip_flags(files)
            remove_auth_files()
            kill_remote_pipeline()
            time.sleep(3)
            resubmit(flags)
            time.sleep(POLL_SEC)
            continue

        # ── Processing stopped unexpectedly ──
        if not processing and status == "idle":
            print("  ⚠️  Pipeline went idle unexpectedly. Checking for result...")
            time.sleep(5)
            result_retry = _get("/download/remote_job_result.json")
            if result_retry:
                continue
            flags = determine_skip_flags(files)
            remove_auth_files()
            kill_remote_pipeline()
            resubmit(flags)
            time.sleep(POLL_SEC)
            continue

        print()
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    sys.exit(main())
