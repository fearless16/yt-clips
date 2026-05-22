"""colab_client.py — Local client for Face OS Colab compute server.

Sends pipeline commands to Colab via tunnel, receives results.
Uses curl for HTTP (Python SSL incompatible with serveo tunnels).

Usage:
    from face_os.colab_client import ColabClient

    client = ColabClient("https://xxxx.serveousercontent.com")

    # Enroll
    client.enroll("expectation.png", "photos/")

    # Process
    result = client.process("clips_test/test_clip.mp4", max_frames=30)
    print(result["telemetry"])
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class ColabClient:
    """Client for Face OS Colab compute server (uses curl)."""

    def __init__(self, base_url: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        return self._get("/health")

    def gpu_info(self) -> dict:
        return self._get("/gpu")

    def enroll(self, reference_path: str = None, photos_dir: str = None, drive_ref: str = None, drive_photos: str = None) -> dict:
        """Enroll identity.

        Args:
            reference_path: Local reference image path (uploaded via curl)
            photos_dir: Local photos directory (uploaded via curl)
            drive_ref: Drive path on Colab (e.g. /content/drive/MyDrive/yt-clips/expectation.png)
            drive_photos: Drive path on Colab (e.g. /content/drive/MyDrive/yt-clips/photos)
        """
        if drive_ref:
            return self._run([
                "curl", "-sk", "--max-time", str(self.timeout), "-X", "POST",
                "-F", f"drive_path={drive_ref}",
                f"{self.base_url}/enroll"
            ])
        args = ["curl", "-sk", "--max-time", str(self.timeout), "-X", "POST"]
        if reference_path:
            args += ["-F", f"reference=@{reference_path}"]
        if photos_dir and Path(photos_dir).is_dir():
            for photo in sorted(Path(photos_dir).glob("*.png")):
                args += ["-F", f"photos=@{photo}"]
        args.append(f"{self.base_url}/enroll")
        return self._run(args)

    def process(self, video_path: str = None, max_frames: int = 30, drive_video: str = None) -> dict:
        """Process video.

        Args:
            video_path: Local video path (uploaded via curl)
            max_frames: Max frames to process
            drive_video: Drive path on Colab (e.g. /content/drive/MyDrive/yt-clips/clips_test/test_clip.mp4)
        """
        if drive_video:
            return self._run([
                "curl", "-sk", "--max-time", str(self.timeout), "-X", "POST",
                "-F", f"drive_path={drive_video}",
                "-F", f"max_frames={max_frames}",
                f"{self.base_url}/process"
            ])
        args = ["curl", "-sk", "--max-time", str(self.timeout), "-X", "POST"]
        if video_path:
            args += ["-F", f"video=@{video_path}"]
        args += ["-F", f"max_frames={max_frames}"]
        args.append(f"{self.base_url}/process")
        return self._run(args)

    def get_telemetry(self) -> dict:
        return self._get("/telemetry")

    def get_frame_telemetry(self, start: int = 0, limit: int = 100) -> dict:
        return self._get(f"/telemetry/frames?start={start}&limit={limit}")

    def reset(self) -> dict:
        return self._post("/reset")

    def _get(self, path: str) -> dict:
        return self._run(["curl", "-sk", "--max-time", str(self.timeout), f"{self.base_url}{path}"])

    def _post(self, path: str) -> dict:
        return self._run(["curl", "-sk", "--max-time", str(self.timeout), "-X", "POST", f"{self.base_url}{path}"])

    def _run(self, args: list) -> dict:
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=self.timeout + 10)
            if r.returncode != 0:
                return {"error": f"curl failed: {r.stderr.strip()}"}
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON: {r.stdout[:200]}"}
        except subprocess.TimeoutExpired:
            return {"error": f"Timeout after {self.timeout}s"}
        except Exception as e:
            return {"error": str(e)}
