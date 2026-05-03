"""
test_upload.py — Verification script for YouTube API.
"""
import os
import subprocess
import json
from upload import upload_video
from utils.logger import get_logger
from utils.config import load_config

cfg = load_config()
log = get_logger("test_upload", cfg["logging"]["log_file"], cfg["logging"]["level"])

def create_test_video():
    log.info("🎞️ Creating 1-second test video...")
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1", 
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "test_video.mp4"
    ]
    subprocess.run(cmd, capture_output=True)
    
    meta = {
        "title": "API Test Upload",
        "description": "Testing the automated pipeline credentials.",
        "tags": ["test", "api"]
    }
    with open("test_meta.json", "w") as f:
        json.dump(meta, f)
    
    return "test_video.mp4", "test_meta.json"

if __name__ == "__main__":
    v, m = create_test_video()
    try:
        log.info("📡 Starting verification upload...")
        upload_video(v, m, privacy="private")
        log.info("✨ VERIFICATION SUCCESSFUL! Your YouTube API is alive.")
    except Exception as e:
        log.error(f"❌ VERIFICATION FAILED: {e}")
    finally:
        # Cleanup
        if os.path.exists(v): os.remove(v)
        if os.path.exists(m): os.remove(m)
