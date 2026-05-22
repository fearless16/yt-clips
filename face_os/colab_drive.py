# Face OS — Colab Server (Google Drive Mount Flow)
# Run each cell in order on Google Colab.

# ════════════════════════════════════════════════════════════════════════════
# CELL 1: Mount Drive + Verify
# ════════════════════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

import os
CODE_DIR = "/content/drive/MyDrive/yt-clips"

if not os.path.exists(CODE_DIR):
    print(f"ERROR: {CODE_DIR} not found!")
    print("Run this on your LOCAL machine first:")
    print("  python push_code.py")
else:
    print(f"Found: {CODE_DIR}")
    os.chdir(CODE_DIR)
    print(f"Files: {len(list(os.walk('.')))} dirs")
    !ls face_os/ 2>/dev/null || echo "face_os/ not found — run 'python push_code.py' on local first"

# ════════════════════════════════════════════════════════════════════════════
# CELL 2: Install deps
# ════════════════════════════════════════════════════════════════════════════
%cd /content/drive/MyDrive/yt-clips
!pip install -q flask mediapipe face-recognition opencv-python-headless numpy scipy

# ════════════════════════════════════════════════════════════════════════════
# CELL 3: Start tunnel + server
# ════════════════════════════════════════════════════════════════════════════
import subprocess, re, time

# Start tunnel
tunnel_proc = subprocess.Popen(
    ["ssh", "-o", "StrictHostKeyChecking=no", "-R", "80:localhost:5000", "serveo.net"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
TUNNEL_URL = None
for _ in range(30):
    line = tunnel_proc.stdout.readline()
    m = re.search(r"(https?://\S+\.serveo\.net)", line)
    if m:
        TUNNEL_URL = m.group(1)
        break
    time.sleep(1)

if TUNNEL_URL:
    print(f"\n{'='*60}")
    print(f"TUNNEL_URL = {TUNNEL_URL}")
    print(f"{'='*60}")
else:
    print("TUNNEL FAILED. Try: !ssh -R 80:localhost:5000 serveo.net")

# Start server in background
!nohup python face_os/colab_server.py > /tmp/faceos_server.log 2>&1 &
time.sleep(3)
!cat /tmp/faceos_server.log
print("\nREADY — copy TUNNEL_URL above, then run on local:")
print("  python run_on_colab.py <TUNNEL_URL> --gpu")
