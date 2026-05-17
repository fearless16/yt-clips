"""
kaggle_monitor.py — Mac-side monitor for Kaggle pipeline.

Polls the Kaggle watcher via tunnel and shows clean progress output.
No unicode, no token waste — just the info you need.

Usage:
    python kaggle_monitor.py                          # auto-detect tunnel URL
    python kaggle_monitor.py --url https://xxx.loca.lt  # explicit URL
    python kaggle_monitor.py --files                  # list files on Kaggle
    python kaggle_monitor.py --log                    # tail the watcher log
    python kaggle_monitor.py --exec "ls -la"          # run a command on Kaggle
"""

import argparse
import json
import re
import ssl
import sys
import time
import urllib.request
import urllib.error

# localtunnel uses Let's Encrypt but Mac Python often lacks the cert chain
_SSL_CTX = ssl._create_unverified_context()


def api_call(tunnel_url: str, path: str, method: str = "GET", data: dict = None, timeout: int = 60) -> dict:
    """Make a request to the Kaggle watcher."""
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(f"{tunnel_url}{path}", data=body, headers={"Content-Type": "application/json"})
    req.get_method = lambda: method
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}"}
    except Exception as e:
        return {"error": str(e)}


def clean_log(text: str) -> str:
    """Strip unicode box-drawing chars, clean up output."""
    # Remove box drawing chars
    text = re.sub(r'[\u2500-\u257f\u2580-\u259f\u25a0-\u25ff\u2600-\u26ff\u2700-\u27bf\u2b50-\u2b55\U0001f000-\U0001f9ff]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    # Remove empty lines
    lines = [l.rstrip() for l in text.split('\n') if l.strip()]
    return '\n'.join(lines)


def get_tunnel_url() -> str:
    """Read tunnel URL from kaggle_url.txt."""
    try:
        return open("kaggle_url.txt").read().strip()
    except FileNotFoundError:
        print("ERROR: kaggle_url.txt not found. Run Kaggle Cell 7 first.")
        sys.exit(1)


def cmd_health(tunnel_url: str):
    """Show pipeline status."""
    result = api_call(tunnel_url, "/health")
    status = result.get("status", "unknown")
    processing = result.get("processing", False)
    print(f"Status: {status} | Processing: {processing}")
    return result


def cmd_files(tunnel_url: str):
    """List files on Kaggle."""
    result = api_call(tunnel_url, "/files")
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    total_mb = 0
    for folder, files in result.items():
        if files:
            print(f"\n{folder}/")
            for f in files:
                sz_mb = f["size"] / 1e6
                total_mb += sz_mb
                print(f"  {f['name']}  ({sz_mb:.1f} MB)")
    print(f"\nTotal: {total_mb:.1f} MB")


def cmd_log(tunnel_url: str, lines: int = 30):
    """Tail the watcher log."""
    result = api_call(tunnel_url, "/exec", method="POST", data={"cmd": f"tail -{lines} watcher.log 2>&1"})
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    stdout = result.get("stdout", "")
    if stdout:
        print(clean_log(stdout))


def cmd_exec(tunnel_url: str, command: str):
    """Run a command on Kaggle."""
    result = api_call(tunnel_url, "/exec", method="POST", data={"cmd": command})
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    if result.get("stdout"):
        print(result["stdout"], end="")
    if result.get("stderr"):
        print(result["stderr"], end="", file=sys.stderr)
    sys.exit(result.get("returncode", 0))


def cmd_monitor(tunnel_url: str, interval: int = 15):
    """Continuously monitor the pipeline."""
    print(f"Monitoring Kaggle pipeline (poll every {interval}s)")
    print(f"Tunnel: {tunnel_url}")
    print("Press Ctrl+C to stop\n")

    last_log = ""
    try:
        while True:
            # Health check
            health = api_call(tunnel_url, "/health")
            processing = health.get("processing", False)

            # Get latest log lines
            result = api_call(tunnel_url, "/exec", method="POST", data={"cmd": "tail -5 watcher.log 2>&1"})
            log_text = clean_log(result.get("stdout", ""))

            # Only print if something changed
            if log_text != last_log:
                for line in log_text.split('\n'):
                    line = line.strip()
                    if line:
                        # Highlight phases and status
                        if "PHASE" in line:
                            print(f"\n>>> {line}")
                        elif "complete" in line.lower() or "success" in line.lower() or line.startswith("OK"):
                            print(f"  {line}")
                        elif "error" in line.lower() or "fail" in line.lower():
                            print(f"  ERR: {line}")
                        elif "progress" in line.lower() or "%" in line:
                            print(f"  {line}")
                        elif "download" in line.lower() or "transcri" in line.lower():
                            print(f"  {line}")
                        else:
                            print(f"  {line}")
                last_log = log_text

            if not processing and health.get("status") == "ok":
                print("\nPipeline finished or idle.")
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped monitoring.")


def main():
    parser = argparse.ArgumentParser(description="Monitor Kaggle pipeline from Mac")
    parser.add_argument("--url", help="Tunnel URL (default: read from kaggle_url.txt)")
    parser.add_argument("--files", action="store_true", help="List files on Kaggle")
    parser.add_argument("--log", action="store_true", help="Tail watcher log")
    parser.add_argument("--health", action="store_true", help="Check health status")
    parser.add_argument("--exec", metavar="CMD", help="Run a shell command on Kaggle")
    parser.add_argument("--monitor", action="store_true", help="Continuously monitor pipeline")
    parser.add_argument("--interval", type=int, default=15, help="Poll interval in seconds (default: 15)")
    args = parser.parse_args()

    tunnel_url = args.url or get_tunnel_url()

    if args.files:
        cmd_files(tunnel_url)
    elif args.log:
        cmd_log(tunnel_url)
    elif args.health:
        cmd_health(tunnel_url)
    elif args.exec:
        cmd_exec(tunnel_url, args.exec)
    elif args.monitor:
        cmd_monitor(tunnel_url, args.interval)
    else:
        # Default: show health + latest log
        cmd_health(tunnel_url)
        print()
        cmd_log(tunnel_url, 15)


if __name__ == "__main__":
    main()
