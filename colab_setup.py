"""colab_setup.py — One-shot Colab setup: deps + worker + tunnel.

Usage:
    Runtime -> T4 GPU
    !python colab_setup.py

Exits cleanly after setup.  Watcher + tunnel run as nohup'd daemons.
Tunnel URL saved to colab_url.txt for bridge.py to pick up.
"""
import os, subprocess, sys, time
from pathlib import Path

WATCHER_LOG = "watcher.log"
TUNNEL_LOG = "tunnel.log"
URL_FILE = "colab_url.txt"


def run(cmd, timeout=120):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        err = r.stderr.strip()[-200:] if r.stderr else r.stdout.strip()[-200:]
        print(f"  \u26a0 {cmd[:50]}... ({err})")
    return r


def kill_old():
    run("pkill -f 'python watcher.py' 2>/dev/null || true")
    run("pkill -f serveo 2>/dev/null || true")
    run("fuser -k 5000/tcp 2>/dev/null || true")
    time.sleep(2)


def wait_for_watcher(timeout=15):
    """Poll /health endpoint until watcher responds."""
    for _ in range(timeout):
        r = subprocess.run(
            "curl -sf http://localhost:5000/health 2>/dev/null",
            shell=True, capture_output=True, text=True,
        )
        if r.returncode == 0:
            return True
        time.sleep(1)
    return False


def extract_tunnel_url(timeout=60):
    """Tail tunnel.log looking for the public URL."""
    url = None
    for _ in range(timeout):
        if not Path(TUNNEL_LOG).exists():
            time.sleep(1)
            continue
        lines = open(TUNNEL_LOG).read().splitlines()
        for line in lines:
            # serveo: "Forwarding HTTP traffic from https://xxx.serveo.net"
            # localhost.run: "https://xxx.lhrtunnel.com"
            # bore: no stdout check needed
            for word in line.split():
                w = word.strip().rstrip(",.;")
                if w.startswith("https://") and ("serveo" in w or "lhrtunnel" in w or "trycloudflare" in w):
                    url = w
                    break
        if url:
            break
        time.sleep(1)
    return url


print("=" * 55)
print("  yt-clips — Colab Setup")
print("=" * 55)

# ─── Mount Drive ──────────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

REPO_DIR = "/content/drive/MyDrive/yt-clips-repo"
ENV_DIR = "/content/drive/MyDrive/yt-clips"
REPO = "https://github.com/fearless16/yt-clips.git"

if Path(f"{REPO_DIR}/.git").exists():
    os.chdir(REPO_DIR)
    print(f"  Working dir: {REPO_DIR}")
else:
    print("  First-time setup — cloning repo to Drive...")
    run(f"git clone {REPO} {REPO_DIR}", timeout=60)
    os.chdir(REPO_DIR)
    print(f"  Working dir: {REPO_DIR}")

# ─── Secrets ──────────────────────────────────────────────────────────────
for d in [Path(ENV_DIR, ".env"), Path(REPO_DIR, ".env")]:
    if d.exists():
        for line in open(d):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
        print(f"  .env loaded from {d.parent.name}/")
        break

try:
    from google.colab import userdata
    for key in ["OPENROUTER_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY"]:
        val = userdata.get(key)
        if val:
            os.environ[key] = val
            print(f"  {key} loaded from secrets")
except:
    pass

# ─── Pull latest code ─────────────────────────────────────────────────────
print("  Pulling latest code...")
run("git pull origin main 2>&1", timeout=30)

# ─── Install deps ─────────────────────────────────────────────────────────
print("  System deps (aria2, ffmpeg, curl)...")
run("apt-get install -y -qq aria2 ffmpeg > /dev/null 2>&1")

print("  Python deps...")
run("pip install -q yt-dlp faster-whisper rich PyYAML opencv-python-headless numpy "
    "filterpy scipy google-genai google-generativeai openai python-dotenv "
    "pyngrok ultralytics torch --extra-index-url https://download.pytorch.org/whl/cu121")

# torchvision compat
try:
    import utils.torchvision_compat  # noqa: F401
    print("  torchvision compat applied")
except:
    pass

gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null",
                     shell=True, capture_output=True, text=True).stdout.strip()
print(f"  GPU: {gpu or 'NONE! Use T4 GPU runtime'}")

# Create folders
for folder in ["input", "temp", "transcripts", "highlights", "shorts", "logs", "photos"]:
    Path(folder).mkdir(exist_ok=True)

# ─── Start Watcher (nohup daemon) ─────────────────────────────────────────
kill_old()

watcher_cmd = f"nohup {sys.executable} watcher.py > {WATCHER_LOG} 2>&1 &"
subprocess.run(watcher_cmd, shell=True)
time.sleep(3)

if wait_for_watcher(30):
    pid = subprocess.run("pgrep -f 'python watcher.py'", shell=True,
                         capture_output=True, text=True).stdout.strip()
    print(f"  Watcher OK (PID: {pid.split()[0]})")
else:
    print("  Watcher FAILED — /health unreachable on port 5000")
    if Path(WATCHER_LOG).exists():
        print(open(WATCHER_LOG).read().strip()[-500:])
    sys.exit(1)

# ─── Start Tunnel (ngrok static domain) ───────────────────────────────────
NGROK_AUTH = os.environ.get("NGROK_TOKEN")
STATIC_DOMAIN = "wiry-rubble-boring.ngrok-free.dev"

if NGROK_AUTH:
    from pyngrok import ngrok as _ngrok
    _ngrok.set_auth_token(NGROK_AUTH)
    try:
        tunnel = _ngrok.connect(5000, domain=STATIC_DOMAIN)
        url = tunnel.public_url
    except Exception as e:
        if "ERR_NGROK_334" in str(e):
            url = f"https://{STATIC_DOMAIN}"
            print(f"  Tunnel already active at {url}")
        else:
            tunnel = _ngrok.connect(5000)
            url = tunnel.public_url
            print(f"  Static domain failed, using random URL")
    Path(URL_FILE).write_text(url)
    print(f"  Tunnel URL: {url}")
    print(f"  Saved to: {URL_FILE}")
else:
    print("  NGROK_TOKEN not found — tunnel skipped")

# ─── Done ─────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  WATCHER + TUNNEL RUNNING IN BACKGROUND")
print("=" * 55)
print()
print("On your Mac, run:")
print('  python bridge.py "https://youtu.be/VIDEO_ID"')
print()
print("Check logs anytime:")
print(f"  !tail -f {WATCHER_LOG}")
print(f"  !tail -f {TUNNEL_LOG}")
print(f"  !curl -s http://localhost:5000/health")
