"""colab_tunnel.py — Tunnel-only Colab cells for the automation pipeline.

Copy-paste each cell into a Colab notebook (Runtime → T4 GPU).
"""

# ═══════════════════════════════════════════════════════════════════════════
# CELL 1 — Mount Drive, git clone/pull, load .env
# ═══════════════════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount("/content/drive")

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

try:
    from google.colab import userdata
    for key in ["NGROK_TOKEN"]:
        val = userdata.get(key)
        if val:
            os.environ[key] = val
            print(f"{key} from Colab secrets")
except:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# CELL 2 — Start ngrok tunnel to watcher (port 5000)
# ═══════════════════════════════════════════════════════════════════════════
!pip install -q pyngrok

import os
from pyngrok import ngrok

NGROK_AUTH = os.environ.get("NGROK_TOKEN")
STATIC_DOMAIN = "wiry-rubble-boring.ngrok-free.dev"

if NGROK_AUTH:
    ngrok.set_auth_token(NGROK_AUTH)
    try:
        tunnel = ngrok.connect(5000, domain=STATIC_DOMAIN)
        url = tunnel.public_url
    except Exception as e:
        if "ERR_NGROK_334" in str(e):
            url = f"https://{STATIC_DOMAIN}"
            print("Tunnel already active — reusing existing URL")
        else:
            tunnel = ngrok.connect(5000)
            url = tunnel.public_url
            print("Static domain failed, using random URL")
    print(f"\n{'='*60}")
    print(f"TUNNEL_URL = {url}")
    print(f"{'='*60}")
    print(f"\nTunnel active → localhost:5000 (watcher)")
    print(f"Keep this Colab running.")
    print(f"\nIn your local terminal:")
    print(f"  python bridge.py <video_url>")
else:
    print("NGROK_TOKEN not found in .env or secrets")
