"""colab_setup.py — One-shot Colab setup.
Runtime → T4 GPU → !python colab_setup.py → done.
"""
import os, subprocess, sys, time
from pathlib import Path

def run(cmd, timeout=120):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print(f"  \u26a0 {cmd[:60]}... ({r.stderr.strip()[-200:] or r.stdout.strip()[-200:]})")
    return r

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
else:
    print("  Cloning repo...")
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
        print(f"  .env loaded from Drive")
        break

try:
    from google.colab import userdata
    for key in ["NGROK_TOKEN", "OPENROUTER_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY"]:
        val = userdata.get(key)
        if val:
            os.environ[key] = val
            print(f"  {key} from secrets")
except:
    pass

# ─── Pull latest code ─────────────────────────────────────────────────────
run("git pull origin main 2>&1", timeout=30)

# ─── Install deps ─────────────────────────────────────────────────────────
print("  Installing deps...")
run("apt-get install -y -qq aria2 ffmpeg > /dev/null 2>&1")
run("pip install -q yt-dlp faster-whisper rich PyYAML opencv-python-headless numpy "
    "filterpy scipy google-genai google-generativeai openai python-dotenv "
    "pyngrok ultralytics torch youtube-transcript-api --extra-index-url https://download.pytorch.org/whl/cu121")

try:
    import utils.torchvision_compat
except:
    pass

gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True,
                     capture_output=True, text=True).stdout.strip()
print(f"  GPU: {gpu or 'NONE — use T4 runtime'}")

for folder in ["input", "temp", "transcripts", "highlights", "shorts", "logs", "photos"]:
    Path(folder).mkdir(exist_ok=True)

# ─── Start Watcher ────────────────────────────────────────────────────────
run("pkill -f 'python watcher.py' 2>/dev/null || true", timeout=5)
run("fuser -k 5000/tcp 2>/dev/null || true", timeout=5)
time.sleep(2)

subprocess.Popen(f"nohup {sys.executable} watcher.py > watcher.log 2>&1 &", shell=True)
time.sleep(3)

watcher_ok = False
for _ in range(20):
    r = subprocess.run("curl -sf http://localhost:5000/health", shell=True,
                       capture_output=True, text=True)
    if r.returncode == 0:
        watcher_ok = True
        break
    time.sleep(1)

if watcher_ok:
    pid = subprocess.run("pgrep -f 'python watcher.py'", shell=True,
                         capture_output=True, text=True).stdout.strip()
    print(f"  Watcher OK (PID: {pid.split()[0]})")
else:
    print("  Watcher FAILED")
    if Path("watcher.log").exists():
        print(open("watcher.log").read().strip()[-500:])
    sys.exit(1)

# ─── Start Tunnel ─────────────────────────────────────────────────────────
NGROK_AUTH = os.environ.get("NGROK_TOKEN")
STATIC_DOMAIN = "wiry-rubble-boring.ngrok-free.dev"

if NGROK_AUTH:
    from pyngrok import ngrok
    ngrok.set_auth_token(NGROK_AUTH)
    try:
        tunnel = ngrok.connect(5000, domain=STATIC_DOMAIN)
        url = tunnel.public_url
    except Exception as e:
        if "ERR_NGROK_334" in str(e):
            url = f"https://{STATIC_DOMAIN}"
        else:
            tunnel = ngrok.connect(5000)
            url = tunnel.public_url
    Path("colab_url.txt").write_text(url)
    print(f"  Tunnel: {url}")
else:
    print("  NGROK_TOKEN not found — no tunnel")

# ─── Done ─────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  READY — tunnel + watcher running")
print("=" * 55)
print()
print("On your Mac:")
print('  python bridge.py "https://youtu.be/VIDEO_ID"')
print()
print("Colab logs:")
print("  !tail -f watcher.log")
print("  !curl -s http://localhost:5000/health")
