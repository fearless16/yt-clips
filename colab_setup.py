"""
colab_setup.py — yt-clips Colab GPU Worker Setup

Usage (on Google Colab):
    1. Runtime → Change runtime type → T4 GPU
    2. Upload this file + all .py files + utils/ to /content/
       OR sync from Drive (see Step 3)
    3. Run: !python colab_setup.py

What it does:
    - Mounts Google Drive
    - Installs all deps (ffmpeg, aria2, Deno, Python pkgs, PyTorch + CUDA, YOLO, GFPGAN)
    - Writes GPU-optimized config.yaml
    - Starts watcher.py + localtunnel
    - Shows tunnel URL — use with ./automate.sh → Remote Run
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, desc=None):
    if desc:
        print(f"  -> {desc}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: exit code {result.returncode}")
    return result


def main():
    print("=" * 55)
    print("  yt-clips -- Colab GPU Worker Setup")
    print("=" * 55)

    # --- 1. Mount Drive ---------------------------------------------------
    print("\n--- Step 1: Google Drive Mount ---")
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)

    # Try to find project
    project_name = "yt-clips"
    project_path = None
    for p in [
        f"/content/drive/MyDrive/{project_name}",
        f"/content/drive/My Drive/{project_name}",
    ]:
        if Path(p).exists():
            project_path = p
            break

    if project_path:
        os.chdir(project_path)
        print(f"  Working dir: {project_path}")
    else:
        print(f"  No '{project_name}' folder in Drive.")
        print("  Using /content/ -- upload files manually via sidebar.")
        os.chdir("/content")
        project_path = "/content"

    # --- 2. System Dependencies -------------------------------------------
    print("\n--- Step 2: System Dependencies ---")
    run(
        "apt-get update -qq && apt-get install -y -qq aria2 ffmpeg "
        "nasm yasm build-essential > /dev/null 2>&1",
        "Installing aria2, ffmpeg, build tools",
    )

    run(
        "curl -fsSL https://deno.land/x/install/install.sh | sh > /dev/null 2>&1",
        "Installing Deno (bot bypass)",
    )
    os.environ["PATH"] += ":/root/.deno/bin"

    # --- 3. Python Dependencies -------------------------------------------
    print("\n--- Step 3: Python Dependencies ---")

    base = "yt-dlp faster-whisper rich PyYAML opencv-python-headless numpy filterpy scipy"
    api = "google-api-python-client google-auth-httplib2 google-auth-oauthlib requests Pillow"
    ai = "google-genai google-generativeai openai python-dotenv"
    test = "pytest pytest-timeout"

    run(f"pip install -q {base} {api} {ai} {test} 2>&1 | tail -1",
        "Installing Python packages (base + API + AI)")

    # GPU packages -- use separate commands for index-url
    run(
        "pip install -q ultralytics torch --extra-index-url "
        "https://download.pytorch.org/whl/cu121 > /dev/null 2>&1",
        "Installing PyTorch + YOLOv8 (CUDA)",
    )
    run("pip install -q gfpgan basicsr > /dev/null 2>&1",
        "Installing GFPGAN face enhancement")

    # Verify GPU
    gpu = run("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null",
              desc=None).stdout.strip()
    if gpu:
        print(f"  GPU: {gpu}")
    else:
        print("  WARNING: No GPU! Set Runtime -> T4 GPU and restart.")

    # --- 4. Localtunnel ---------------------------------------------------
    print("\n--- Step 4: Localtunnel ---")
    run("npm install -g localtunnel > /dev/null 2>&1", "Installing localtunnel")

    # --- 5. Configure (patch existing config from Drive) ------------------
    print("\n--- Step 5: Patching Config for Colab GPU ---")
    for folder in ["input", "temp", "transcripts", "highlights", "shorts", "logs"]:
        Path(folder).mkdir(exist_ok=True)

    # Don't overwrite the Drive-synced config — just patch GPU settings
    import yaml
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    patched = 0
    # GPU overrides
    gpu_overrides = {
        "transcription": {"device": "cuda", "compute_type": "float16"},
        "premium": {"enabled": True, "face_enhancement": True, "frame_interpolation": True},
        "export": {"encoder": "h264_nvenc"},
        "testing": {"enabled": False},
    }
    for section, values in gpu_overrides.items():
        if section not in cfg:
            cfg[section] = {}
        for k, v in values.items():
            if cfg[section].get(k) != v:
                cfg[section][k] = v
                patched += 1
    # Load API key from Colab secrets
    try:
        from google.colab import userdata
        key = userdata.get("GOOGLE_API_KEY") or userdata.get("AI_API_KEY")
        if key:
            os.environ["GOOGLE_API_KEY"] = key
            os.environ["AI_API_KEY"] = key
            print("  ✅ GOOGLE_API_KEY loaded from Colab secrets")
        else:
            print("  ⚠️  No GOOGLE_API_KEY in Colab secrets. Gemini will have low rate limits.")
    except Exception:
        print("  ⚠️  No GOOGLE_API_KEY in Colab secrets. Gemini will have low rate limits.")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"  Config patched ({patched} GPU overrides applied)")

    # --- 6. Start Watcher + Tunnel ----------------------------------------
    print("\n--- Step 6: Starting Watcher + Tunnel ---")

    os.system("pkill -f 'python watcher.py' 2>/dev/null || true")
    os.system("pkill -f 'lt --port' 2>/dev/null || true")
    time.sleep(1)

    subprocess.Popen(
        [sys.executable, "watcher.py"],
        stdout=open("watcher.log", "w"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(2)

    subprocess.Popen(
        ["lt", "--port", "5000"],
        stdout=open("tunnel.log", "w"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(5)

    # Extract tunnel URL
    if Path("tunnel.log").exists():
        with open("tunnel.log") as f:
            for line in f:
                line = line.strip()
                if "://" in line:
                    with open("colab_url.txt", "w") as out:
                        out.write(line)
                    print(f"\n  TUNNEL URL: {line}")
                    print(f"  (saved to colab_url.txt)")
                    break

    print()
    print("=" * 55)
    print("  COLAB WORKER IS ONLINE!")
    print("=" * 55)
    print()
    print("  On your Mac, run:")
    print(f'    ./automate.sh "https://youtu.be/VIDEO_ID"')
    print("    -> Select option 2 (Remote Run)")
    print()
    print("  Waiting for jobs...")

    # Monitor watcher log
    try:
        while True:
            time.sleep(30)
            if Path("watcher.log").exists():
                with open("watcher.log") as f:
                    lines = f.readlines()
                    for line in lines[-3:]:
                        s = line.strip()
                        if s:
                            print(f"  {s}")
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
