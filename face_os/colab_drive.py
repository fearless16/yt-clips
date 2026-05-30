# Face OS — Colab Server (git + Drive .env flow)
# Run each cell in order on Google Colab (Runtime → T4 GPU).

# ════════════════════════════════════════════════════════════════════════════
# CELL 1: Mount Drive + git clone/pull + load .env
# ════════════════════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

import os
from pathlib import Path

REPO_DIR = "/content/drive/MyDrive/yt-clips-repo"
ENV_DIR = "/content/drive/MyDrive/yt-clips"
REPO = "https://github.com/fearless16/yt-clips.git"

if Path(f"{REPO_DIR}/.git").exists():
    os.chdir(REPO_DIR)
    !git pull origin main 2>&1
else:
    os.chdir("/content/drive/MyDrive")
    !git clone {REPO} {REPO_DIR} 2>&1

os.chdir(REPO_DIR)
print(f"Working dir: {REPO_DIR}")
print(f"Files: {len(list(os.walk('.')))} dirs")
!ls face_os/ 2>/dev/null || echo "face_os/ not found"

loaded = False
for d in [Path(ENV_DIR, ".env"), Path(REPO_DIR, ".env")]:
    if d.exists():
        for line in open(d):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
        print(f".env loaded from {d.parent.name}/")
        loaded = True
        break
if not loaded:
    print("No .env found — check Drive yt-clips/ or yt-clips-repo/")

# ════════════════════════════════════════════════════════════════════════════
# CELL 2: Install deps
# ════════════════════════════════════════════════════════════════════════════
%cd /content/drive/MyDrive/yt-clips-repo
!apt-get install -y -qq aria2 ffmpeg > /dev/null 2>&1
!pip install -q yt-dlp faster-whisper rich PyYAML opencv-python-headless numpy \
    filterpy scipy google-genai google-generativeai openai python-dotenv \
    pyngrok ultralytics flask mediapipe torch --extra-index-url https://download.pytorch.org/whl/cu121

# ════════════════════════════════════════════════════════════════════════════
# CELL 3: Start tunnel (ngrok) + server
# ════════════════════════════════════════════════════════════════════════════
import os, time

NGROK_AUTH = os.environ.get("NGROK_TOKEN")
STATIC_DOMAIN = "wiry-rubble-boring.ngrok-free.dev"
TUNNEL_URL = None

if NGROK_AUTH:
    from pyngrok import ngrok
    ngrok.set_auth_token(NGROK_AUTH)
    try:
        tunnel = ngrok.connect(5000, domain=STATIC_DOMAIN)
        TUNNEL_URL = tunnel.public_url
    except Exception as e:
        if "ERR_NGROK_334" in str(e):
            TUNNEL_URL = f"https://{STATIC_DOMAIN}"
            print("Tunnel already active — reusing existing URL")
        else:
            tunnel = ngrok.connect(5000)
            TUNNEL_URL = tunnel.public_url
            print("Static domain failed, using random URL")
    print(f"\n{'='*60}")
    print(f"TUNNEL_URL = {TUNNEL_URL}")
    print(f"{'='*60}")
else:
    print("NGROK_TOKEN not found in .env — tunnel skipped")

# Start server in background
!nohup python face_os/colab_server.py > /tmp/faceos_server.log 2>&1 &
time.sleep(3)
!cat /tmp/faceos_server.log
print("\nREADY — run on local:")
print(f"  python run_on_colab.py {TUNNEL_URL or '<tunnel_url>'} --gpu")
