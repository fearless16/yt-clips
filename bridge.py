"""
bridge.py — Tunnel-only job bridge to Colab watcher.
Credentials are embedded in the job payload and sent via POST /job.
No Drive file fallback (tunnel-only delivery).
"""
import argparse
import json
import ssl
import urllib.request
from pathlib import Path

CRED_FILES = {
    "cookies.txt", ".env", "client_secrets.json",
    "yt_channel_token.json", "yt_analytics_token.json",
    "drive_token.json",
}


def push_job(url: str, flags: list, tunnel_url: str):
    """Push a pipeline job to the Colab worker via tunnel.

    Args:
        url: YouTube URL to process.
        flags: Additional CLI flags for the pipeline.
        tunnel_url: Public tunnel URL of the Colab watcher.
    """
    job = {"url": url, "flags": flags}

    for cred_file in CRED_FILES:
        p = Path(cred_file)
        if p.exists():
            job[cred_file] = p.read_text(encoding="utf-8")

    try:
        ctx = ssl._create_unverified_context()
        body = json.dumps(job).encode()
        r = urllib.request.urlopen(
            f"{tunnel_url}/job", data=body, timeout=30, context=ctx,
        )
        if r.status == 202:
            print(f"Job submitted to {tunnel_url}")
            return
        resp = json.loads(r.read().decode())
        print(f"Tunnel returned {r.status}: {resp.get('error', 'unknown')}")
    except Exception as e:
        print(f"Failed to submit via tunnel: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a pipeline job to Colab (tunnel-only).")
    parser.add_argument("url", help="YouTube URL to process remotely")
    parser.add_argument("--tunnel-url", required=True, help="Public tunnel URL (e.g. https://xxx.serveo.net)")
    args, unknown = parser.parse_known_args()
    push_job(args.url, unknown, args.tunnel_url)
