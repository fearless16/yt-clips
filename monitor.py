#!/usr/bin/env python3
"""
monitor.py — Poll Colab pipeline status via tunnel.

Usage:
    python monitor.py              # one-shot status
    python monitor.py --watch      # poll every 30s (Ctrl+C to stop)
    python monitor.py --watch 10   # poll every 10s
"""
import sys, time, json, ssl, urllib.request, urllib.error

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

TUNNEL_URL = None


def get_tunnel_url():
    global TUNNEL_URL
    if TUNNEL_URL:
        return TUNNEL_URL
    try:
        with open("colab_url.txt") as f:
            TUNNEL_URL = f.read().strip().rstrip("/")
            return TUNNEL_URL
    except FileNotFoundError:
        return None


def api(path, timeout=10):
    url = get_tunnel_url()
    if not url:
        return None, "No tunnel URL (colab_url.txt missing)"
    try:
        req = urllib.request.Request(f"{url}{path}")
        resp = urllib.request.urlopen(req, timeout=timeout, context=CTX)
        return json.loads(resp.read()), None
    except urllib.error.URLError as e:
        return None, f"Tunnel unreachable: {e}"
    except Exception as e:
        return None, str(e)


def api_post(cmd, timeout=30):
    url = get_tunnel_url()
    if not url:
        return None, "No tunnel URL"
    try:
        data = json.dumps({"cmd": cmd, "timeout": timeout - 5}).encode()
        req = urllib.request.Request(
            f"{url}/exec", data=data,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=timeout, context=CTX)
        return json.loads(resp.read()), None
    except Exception as e:
        return None, str(e)


def check_health():
    data, err = api("/health")
    if err:
        return f"TUNNEL DOWN — {err}"
    status = data.get("status", "?")
    busy = data.get("processing", False)
    return f"Tunnel: {status} | Processing: {'YES' if busy else 'no'}"


def get_log_tail(n=15):
    cmd = f"tail -{n} /content/drive/.shortcut-targets-by-id/1SoiNzjDmEhjOmABnFxo6omfb-YB3Q7Cu/yt-clips/logs/pipeline.log 2>/dev/null || echo 'No log file'"
    data, err = api_post(cmd)
    if err:
        return [f"Error: {err}"]
    raw = data.get("stdout", "")
    lines = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            ts = obj.get("ts", "")[:19]  # trim to YYYY-MM-DD HH:MM:SS
            level = obj.get("level", "")
            msg = obj.get("msg", "")
            lines.append(f"  {ts} [{level:5s}] {msg}")
        except json.JSONDecodeError:
            lines.append(f"  {line}")
    return lines


def get_process_status():
    data, err = api_post("ps aux | grep pipeline | grep -v grep | head -1")
    if err:
        return "unknown"
    proc = data.get("stdout", "").strip()
    if proc:
        return "RUNNING"
    return "IDLE"


def get_phase_summary():
    cmd = (
        'grep -oP "Phase [0-9.]+ .*" /content/drive/.shortcut-targets-by-id/'
        '1SoiNzjDmEhjOmABnFxo6omfb-YB3Q7Cu/yt-clips/logs/pipeline.log 2>/dev/null | tail -1'
    )
    data, err = api_post(cmd)
    if err:
        return "unknown"
    return data.get("stdout", "").strip() or "no phases found"


def get_output_clips():
    cmd = (
        'ls -lhS /content/drive/.shortcut-targets-by-id/'
        '1SoiNzjDmEhjOmABnFxo6omfb-YB3Q7Cu/yt-clips/shorts/'
        '2>/dev/null | head -20'
    )
    data, err = api_post(cmd)
    if err:
        return []
    return [l for l in data.get("stdout", "").strip().split("\n") if l]


def print_status():
    print(f"\n{'='*60}")
    print(f"  COLAB PIPELINE MONITOR")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # 1. Health
    print(f"\n  {check_health()}")

    # 2. Process
    proc = get_process_status()
    print(f"  Pipeline process: {proc}")

    # 3. Latest phase
    phase = get_phase_summary()
    print(f"  Latest phase: {phase}")

    # 4. Log tail
    lines = get_log_tail(12)
    print(f"\n--- Log (last {len(lines)} entries) ---")
    for l in lines:
        print(l)

    # 5. Output clips
    clips = get_output_clips()
    if clips:
        print(f"\n--- Output clips ---")
        for c in clips[:10]:
            print(f"  {c}")

    print()


def main():
    watch = False
    interval = 30

    if "--watch" in sys.argv:
        watch = True
        idx = sys.argv.index("--watch")
        if idx + 1 < len(sys.argv):
            try:
                interval = int(sys.argv[idx + 1])
            except ValueError:
                pass

    if watch:
        print(f"Watching every {interval}s (Ctrl+C to stop)")
        try:
            while True:
                print_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_status()


if __name__ == "__main__":
    main()
