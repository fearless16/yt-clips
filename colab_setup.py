"""colab_setup.py — One-shot Colab setup: deps + worker + tunnel.

Usage:
    Runtime → Change runtime type → T4 GPU
    !python colab_setup.py
"""
import os, subprocess, sys, time
from pathlib import Path

def install(cmd, desc):
    print(f"  -> {desc}...")
    subprocess.run(cmd, shell=True, capture_output=True)

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

# Install deps
install("apt-get install -y -qq aria2 ffmpeg > /dev/null 2>&1", "aria2 + ffmpeg")
install("npm install -g localtunnel > /dev/null 2>&1", "localtunnel")
install("curl -fsSL https://deno.land/x/install/install.sh | sh > /dev/null 2>&1", "Deno")
os.environ["PATH"] += ":/root/.deno/bin"
install("pip install -q yt-dlp faster-whisper rich PyYAML opencv-python-headless numpy "
        "filterpy scipy google-genai google-generativeai openai python-dotenv "
        "ultralytics torch --extra-index-url https://download.pytorch.org/whl/cu121",
        "Python + PyTorch + YOLO")

gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null",
                     shell=True, capture_output=True, text=True).stdout.strip()
print(f"  GPU: {gpu or 'NONE! Use T4 GPU runtime'}")

# Load API key
try:
    from google.colab import userdata
    key = userdata.get("GOOGLE_API_KEY") or userdata.get("AI_API_KEY")
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        os.environ["AI_API_KEY"] = key
        print("  ✅ API key loaded from Colab secrets")
except:
    print("  ⚠️  No API key in secrets. Gemini will have low rate limits.")
    print("     Add GOOGLE_API_KEY via 🔑 tab (left sidebar)")

# Create folders
for folder in ["input", "temp", "transcripts", "highlights", "shorts", "logs"]:
    Path(folder).mkdir(exist_ok=True)

# Start watcher + tunnel
subprocess.run("pkill -f 'python watcher.py' 2>/dev/null || true", shell=True)
subprocess.run("pkill -f 'lt --port' 2>/dev/null || true", shell=True)
time.sleep(1)
subprocess.Popen([sys.executable, "watcher.py"], stdout=open("watcher.log","w"), stderr=subprocess.STDOUT)
time.sleep(2)
subprocess.Popen(["lt", "--port", "5000"], stdout=open("tunnel.log","w"), stderr=subprocess.STDOUT)
time.sleep(5)

with open("tunnel.log") as f:
    for line in f:
        if "://" in line.strip():
            with open("colab_url.txt", "w") as out:
                out.write(line.strip())
            print(f"\n  🔗 Tunnel: {line.strip()}")
            break

print()
print("=" * 55)
print("  WORKER IS ONLINE!")
print("=" * 55)
print("\nOn your Mac, run:")
print('  ./automate.sh "https://youtu.be/VIDEO_ID"')
print("  -> Select option 2 (Remote Run)")
print("\nMonitoring watcher.log...")

try:
    last_pos = 0
    while True:
        time.sleep(10)
        if Path("watcher.log").exists():
            with open("watcher.log", "r") as f:
                f.seek(0, 2)
                if f.tell() < last_pos:
                    last_pos = 0
                f.seek(last_pos)
                for l in f.readlines():
                    if l.strip():
                        print(f"  {l.strip()}")
                last_pos = f.tell()
except KeyboardInterrupt:
    print("\nStopped.")
