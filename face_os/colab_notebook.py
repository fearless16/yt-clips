# Face OS — Colab GPU Compute Server
# Run this cell to set up Face OS on Colab and expose it via tunnel.

# ── Step 1: Clone repo + install deps ──────────────────────────────────────
!git clone https://github.com/YOUR_USER/yt-clips.git 2>/dev/null || true
%cd yt-clips
!pip install -q flask pyngrok mediapipe opencv-python-headless numpy

# ── Step 2: Start tunnel (pick one method) ─────────────────────────────────

# Option A: ngrok (recommended, stable)
# !pip install -q pyngrok
# from pyngrok import ngrok
# public_url = ngrok.connect(5000).public_url
# print(f"Tunnel URL: {public_url}")

# Option B: serveo (no install needed)
import subprocess, re, time
tunnel_proc = subprocess.Popen(
    ["ssh", "-o", "StrictHostKeyChecking=no", "-R", "80:localhost:5000", "serveo.net"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
for _ in range(20):
    line = tunnel_proc.stdout.readline()
    m = re.search(r"(https?://[a-zA-Z0-9.-]+\.serveo\.net)", line)
    if m:
        public_url = m.group(1)
        print(f"Tunnel URL: {public_url}")
        break
    time.sleep(1)

# ── Step 3: Start Face OS server ──────────────────────────────────────────
!python face_os/colab_server.py &
time.sleep(3)
print("Server ready. Use the tunnel URL above from your local machine.")

# ── Step 4: Test from local machine ────────────────────────────────────────
# In your local terminal:
#   from face_os.colab_client import ColabClient
#   client = ColabClient("https://xxxx.serveo.net")
#   client.enroll("expectation.png", "photos/")
#   result = client.process("clips_test/test_clip.mp4", max_frames=30)
#   print(result["telemetry"])
