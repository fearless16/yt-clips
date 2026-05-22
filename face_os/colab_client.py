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

    def enroll(self, reference_path: str, photos_dir: Optional[str] = None) -> dict:
        args = ["curl", "-sk", "--max-time", str(self.timeout), "-X", "POST"]
        args += ["-F", f"reference=@{reference_path}"]
        if photos_dir and Path(photos_dir).is_dir():
            for photo in sorted(Path(photos_dir).glob("*.png")):
                args += ["-F", f"photos=@{photo}"]
        args.append(f"{self.base_url}/enroll")
        return self._run(args)

    def process(self, video_path: str, max_frames: int = 30) -> dict:
        args = ["curl", "-sk", "--max-time", str(self.timeout), "-X", "POST"]
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
