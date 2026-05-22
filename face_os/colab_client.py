"""colab_client.py — Local client for Face OS Colab compute server.

Sends pipeline commands to Colab via tunnel, receives results.

Usage:
    from face_os.colab_client import ColabClient

    client = ColabClient("https://xxxx.ngrok.io")  # or serveo/localtunnel

    # Enroll
    client.enroll("expectation.png", "photos/")

    # Process
    result = client.process("clips_test/test_clip.mp4", max_frames=30)
    print(result["telemetry"])

    # Get per-frame telemetry
    frames = client.get_frame_telemetry(start=0, limit=10)
"""

import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


class ColabClient:
    """Client for Face OS Colab compute server."""

    def __init__(self, base_url: str, timeout: int = 600):
        """
        Args:
            base_url: Colab server URL (from tunnel)
            timeout: HTTP timeout in seconds (default 600 for long video processing)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        """Check server health."""
        return self._get("/health")

    def gpu_info(self) -> dict:
        """Get GPU info from Colab."""
        return self._get("/gpu")

    def enroll(self, reference_path: str, photos_dir: Optional[str] = None) -> dict:
        """Enroll identity on Colab.

        Args:
            reference_path: Path to reference image (expectation.png)
            photos_dir: Directory of photo files (optional)

        Returns:
            Dict with enrollment status
        """
        import mimetypes

        # Build multipart form
        boundary = "----FaceOSBoundary"
        body = b""

        # Reference image
        ref_data = Path(reference_path).read_bytes()
        mime = mimetypes.guess_type(reference_path)[0] or "image/png"
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="reference"; filename="{Path(reference_path).name}"\r\n'.encode()
        body += f"Content-Type: {mime}\r\n\r\n".encode()
        body += ref_data + b"\r\n"

        # Photos
        if photos_dir and Path(photos_dir).is_dir():
            for photo in sorted(Path(photos_dir).glob("*.png")):
                photo_data = photo.read_bytes()
                body += f"--{boundary}\r\n".encode()
                body += f'Content-Disposition: form-data; name="photos"; filename="{photo.name}"\r\n'.encode()
                body += f"Content-Type: image/png\r\n\r\n".encode()
                body += photo_data + b"\r\n"

        body += f"--{boundary}--\r\n".encode()

        return self._post_multipart("/enroll", body, boundary)

    def process(self, video_path: str, max_frames: int = 30) -> dict:
        """Process video on Colab.

        Args:
            video_path: Path to input video
            max_frames: Max frames to process

        Returns:
            Dict with processing results, telemetry, per-frame data
        """
        import mimetypes

        boundary = "----FaceOSBoundary"
        body = b""

        # Video file
        video_data = Path(video_path).read_bytes()
        mime = mimetypes.guess_type(video_path)[0] or "video/mp4"
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="video"; filename="{Path(video_path).name}"\r\n'.encode()
        body += f"Content-Type: {mime}\r\n\r\n".encode()
        body += video_data + b"\r\n"

        # max_frames
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="max_frames"\r\n\r\n'.encode()
        body += f"{max_frames}\r\n".encode()

        body += f"--{boundary}--\r\n".encode()

        return self._post_multipart("/process", body, boundary)

    def get_telemetry(self) -> dict:
        """Get aggregate telemetry from last run."""
        return self._get("/telemetry")

    def get_frame_telemetry(self, start: int = 0, limit: int = 100) -> dict:
        """Get per-frame telemetry from last run."""
        return self._get(f"/telemetry/frames?start={start}&limit={limit}")

    def reset(self) -> dict:
        """Reset pipeline state."""
        return self._post("/reset")

    # ── HTTP helpers ──────────────────────────────────────────────────────

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                return json.loads(body)
            except Exception:
                return {"error": f"HTTP {e.code}: {body}"}
        except Exception as e:
            return {"error": str(e)}

    def _post(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        try:
            req = urllib.request.Request(url, method="POST", data=b"")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                return json.loads(body)
            except Exception:
                return {"error": f"HTTP {e.code}: {body}"}
        except Exception as e:
            return {"error": str(e)}

    def _post_multipart(self, path: str, body: bytes, boundary: str) -> dict:
        url = f"{self.base_url}{path}"
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body_bytes = e.read().decode()
            try:
                return json.loads(body_bytes)
            except Exception:
                return {"error": f"HTTP {e.code}: {body_bytes}"}
        except Exception as e:
            return {"error": str(e)}
