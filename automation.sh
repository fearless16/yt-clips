#!/usr/bin/env bash
# automation.sh — Colab T4 GPU setup: deps + watcher + tunnel
# Usage:  In Colab cell:  !bash automation.sh
set -euo pipefail

SCRIPT_VERSION="2.0.0"
START_TS=$(date +%s)

echo "═══ yt-clips automation v$SCRIPT_VERSION ═══"

# ── Verify GPU ────────────────────────────────────────────────────────
GPU_NAME=$(python3 -c "
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
" 2>/dev/null)
echo "  GPU: $GPU_NAME"

# ── Mount Drive ───────────────────────────────────────────────────────
echo ""
echo "--- Step 1/5: Mount Drive + Git Clone ---"
DRIVE_DIR=""
for p in "/content/drive/MyDrive/yt-clips" "/content/drive/My Drive/yt-clips"; do
    [ -d "$p" ] && { DRIVE_DIR="$p"; break; }
done
if [ -z "$DRIVE_DIR" ]; then
    python3 -c "from google.colab import drive; drive.mount('/content/drive', force_remount=True)" 2>/dev/null
    for p in "/content/drive/MyDrive/yt-clips" "/content/drive/My Drive/yt-clips"; do
        [ -d "$p" ] && { DRIVE_DIR="$p"; break; }
    done
fi
if [ -z "$DRIVE_DIR" ]; then echo "ERROR: yt-clips/ not found on Drive"; exit 1; fi
echo "  Secrets dir: $DRIVE_DIR"

# Git clone / pull code (code comes from git, not Drive)
REPO="/content/yt-clips"
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull origin main 2>/dev/null && echo "  git pull ✓" || echo "  git pull failed"
else
    git clone https://github.com/fearless16/yt-clips "$REPO"
    echo "  git clone ✓"
fi
cd "$REPO"

# ── API Keys (copy from Drive to repo) ──────────────────────────────
echo ""
echo "--- Step 2/5: API Keys ---"
SECRET_FILES=".env drive_token.json yt_channel_token.json yt_analytics_token.json client_secrets.json cookies.txt channel_logo.png face_landmarker.task"
for fname in $SECRET_FILES; do
    src="$DRIVE_DIR/$fname"
    if [ -f "$src" ]; then
        cp "$src" "$REPO/$fname"
        echo "  Copied: $fname"
    fi
done
[ -f .env ] && { set -a; source .env; set +a; echo "  .env loaded ✓"; } || echo "  WARNING: no .env found"

# ── Deps ──────────────────────────────────────────────────────────────
echo ""
echo "--- Step 3/5: System Deps ---"
apt-get update -qq && apt-get install -y -qq aria2 ffmpeg >/dev/null 2>&1
echo "  aria2 ffmpeg ✓"

echo "  deno..."
curl -fsSL https://deno.land/x/install/install.sh | sh -s -- -y 2>&1 | tail -1
export PATH="$HOME/.deno/bin:$PATH"

echo ""
echo "--- Step 4/5: Python Deps ---"
pip install -q torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu121
pip install -q \
    yt-dlp faster-whisper youtube-transcript-api mediapipe \
    rich PyYAML opencv-python-headless numpy \
    filterpy scipy openai python-dotenv Pillow requests \
    ultralytics gfpgan basicsr realesrgan \
    google-api-python-client google-auth-httplib2 google-auth-oauthlib face_recognition 2>&1 | tail -1

# ── Start Watcher + Tunnel ───────────────────────────────────────────
echo ""
echo "--- Step 5/5: Start Watcher + Tunnel ---"
mkdir -p input temp transcripts highlights shorts logs photos

pkill -9 -f 'python.*watcher.py' 2>/dev/null || true
fuser -k 5000/tcp 2>/dev/null || true
pkill -f 'serveo\|localhost.run\|localtunnel\|ngrok' 2>/dev/null || true
sleep 2

# Start watcher directly (not via automation.watcher — that suppresses output)
python3 watcher.py > watcher.log 2>&1 &
WATCHER_PID=$!
sleep 3

# Verify watcher is up
if curl -s http://localhost:5000/health >/dev/null 2>&1; then
    echo "  Watcher: OK (PID $WATCHER_PID)"
else
    echo "  Watcher: FAILED — check watcher.log"
    tail -5 watcher.log 2>/dev/null
fi

# Start tunnel
python3 -c "
import sys, os, time, json, urllib.request
sys.path.insert(0, '.')
os.chdir('$REPO')

# Load .env
if os.path.exists('.env'):
    for line in open('.env'):
        line = line.strip()
        if line and '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip()

from automation.tunnel import start_tunnel
url = start_tunnel(port=5000)
if url:
    print(url)
    open('colab_url.txt','w').write(url.strip())
    # Verify
    ok = False
    for _ in range(5):
        try:
            r = urllib.request.urlopen(f'{url}/health', timeout=5)
            if r.status == 200:
                ok = True
                break
        except: time.sleep(2)
    print(f'  Health: {\"OK\" if ok else \"TIMEOUT\"}')
else:
    print('FAILED')
" > tunnel_output.txt 2>&1 &

# Wait for tunnel URL
URL=""
for i in $(seq 1 45); do
    URL=$(head -1 tunnel_output.txt 2>/dev/null)
    if [ -n "$URL" ] && [ "$URL" != "FAILED" ]; then
        break
    fi
    sleep 1
done

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "═══ COLAB WORKER ONLINE ═══"
echo "  Watcher: jobs delivered via tunnel (POST /job)"
if [ -n "$URL" ]; then
    echo "  Tunnel: $URL"
else
    echo "  Tunnel: N/A (still starting in background — check Cell 2)"
fi
echo ""
echo "  Next → Open Colab.ipynb Cell 2 for graphical dashboard"
echo "  Or → run python -m automation.cli --setup-colab again"
echo "═══ $(( $(date +%s) - START_TS ))s ═══"
