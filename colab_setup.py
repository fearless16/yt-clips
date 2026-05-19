"""colab_setup.py — One-shot Colab setup: deps + worker + tunnel.

Usage:
    Runtime -> T4 GPU
    !python colab_setup.py

Uses serveo.net for tunnel (reliable, no npm needed).
"""
import os, subprocess, sys, time
from pathlib import Path

def run(cmd, timeout=120):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        err = r.stderr.strip()[-200:] if r.stderr else r.stdout.strip()[-200:]
        print(f"  \u26a0 {cmd[:50]}... ({err})")
    return r

def kill_old():
    run("pkill -f 'python watcher.py' 2>/dev/null || true")
    run("pkill -f 'lt --port' 2>/dev/null || true")
    run("pkill -f serveo 2>/dev/null || true")
    run("fuser -k 5000/tcp 2>/dev/null || true")
    time.sleep(2)

print("=" * 55)
print("  yt-clips — Colab Setup")
print("=" * 55)

# Mount Drive + find project
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
for p in ["/content/drive/MyDrive/yt-clips", "/content/drive/My Drive/yt-clips"]:
    if Path(p).exists():
        os.chdir(p)
        print(f"  Working dir: {p}")
        break
else:
    os.chdir("/content")

# Load .env from Drive
if Path(".env").exists():
    for line in open(".env"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()
    print("  .env loaded from Drive")

# Load Colab secrets (override .env)
try:
    from google.colab import userdata
    for key in ["OPENROUTER_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY"]:
        val = userdata.get(key)
        if val:
            os.environ[key] = val
            print(f"  {key} loaded from secrets")
except:
    pass

# Pull latest code
print("  Pulling latest code...")
run("git pull origin main 2>&1", timeout=30)

# Install system deps
print("  System deps (aria2, ffmpeg, ssh)...")
run("apt-get install -y -qq aria2 ffmpeg openssh-client > /dev/null 2>&1")

# Install Python deps
print("  Python deps...")
run("pip install -q yt-dlp faster-whisper rich PyYAML opencv-python-headless numpy "
    "filterpy scipy google-genai google-generativeai openai python-dotenv "
    "ultralytics torch --extra-index-url https://download.pytorch.org/whl/cu121")

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

# Start watcher + tunnel
kill_old()

subprocess.Popen([sys.executable, "watcher.py"], stdout=open("watcher.log","w"), stderr=subprocess.STDOUT)
time.sleep(3)

pid = subprocess.run("pgrep -f 'python watcher.py'", shell=True, capture_output=True, text=True).stdout.strip()
if pid:
    print(f"  Watcher OK (PID: {pid.split()[0]})")
else:
    print("  Watcher FAILED!")
    print(open("watcher.log").read().strip()[-300:])

# Use serveo (reliable, no npm needed)
print("  Starting tunnel (serveo.net)...")
subprocess.Popen(
    ["ssh", "-o", "StrictHostKeyChecking=no", "-R", "80:localhost:5000", "serveo.net"],
    stdout=open("tunnel.log","w"), stderr=subprocess.STDOUT,
)
time.sleep(8)

# Extract URL
url = None
for i in range(20):
    time.sleep(2)
    for line in open("tunnel.log").read().splitlines():
        if "serveousercontent.com" in line or "https://" in line:
            for word in line.split():
                if "https://" in word:
                    url = word.strip().rstrip(",.")
                    break
    if url:
        break

if url:
    Path("colab_url.txt").write_text(url)
    print(f"  Tunnel URL: {url}")
else:
    print("  No tunnel URL found. tunnel.log:")
    for l in open("tunnel.log").read().strip().splitlines()[-5:]:
        print(f"    {l}")

print()
print("=" * 55)
print("  WORKER IS ONLINE!")
print("=" * 55)
print("\nOn your Mac, run:")
print('  python bridge.py "https://youtu.be/VIDEO_ID"')

# Monitor mode
print("\nMonitoring watcher.log...")
try:
    last_pos = 0
    last_inode = None
    while True:
        time.sleep(10)
        try:
            st = Path("watcher.log").stat()
        except FileNotFoundError:
            continue
        if last_inode is not None and st.st_ino != last_inode:
            last_pos = 0
        last_inode = st.st_ino
        with open("watcher.log", "r") as f:
            f.seek(last_pos)
            for l in f.readlines():
                if l.strip():
                    print(f"  {l.strip()}")
            last_pos = f.tell()
except KeyboardInterrupt:
    print("\nStopped.")
