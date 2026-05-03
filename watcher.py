"""
watcher.py — The Colab-side worker.
Watches for remote_job.json and triggers the pipeline.

Features:
  - File-based job polling (Drive mount)
  - Direct HTTP API for instant tunnel delivery
  - Job result tracking (writes remote_job_result.json)
  - Proper logging and error handling
"""
import json
import time
import os
import subprocess
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("watcher", cfg["logging"]["log_file"], cfg["logging"]["level"])

# Thread-safe job queue
pending_jobs = []
_job_lock = threading.Lock()


class JobHandler(BaseHTTPRequestHandler):
    """HTTP handler for receiving jobs via tunnel."""

    def do_POST(self):
        if self.path == "/job":
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                job = json.loads(post_data.decode("utf-8"))

                with _job_lock:
                    pending_jobs.append(job)

                # Also write to disk for persistence
                with open("remote_job.json", "w") as f:
                    json.dump(job, f, indent=2)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "received"}).encode("utf-8"))
                log.info("📥 Job received via HTTP tunnel")

            except json.JSONDecodeError as e:
                log.error(f"Invalid JSON in POST: {e}")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
            except Exception as e:
                log.error(f"Error handling POST: {e}")
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        """Health check + job status endpoint."""
        if self.path == "/status":
            result_file = Path("remote_job_result.json")
            status = {}
            if result_file.exists():
                try:
                    with open(result_file, "r") as f:
                        status = json.load(f)
                except (json.JSONDecodeError, OSError):
                    status = {"status": "unknown"}
            else:
                status = {"status": "idle"}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode("utf-8"))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "alive"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP logs — we use our own logger."""
        pass


def run_server(port: int):
    """Start the HTTP API server for tunnel delivery."""
    server = HTTPServer(("0.0.0.0", port), JobHandler)
    log.info(f"📡 API Server listening on port {port}...")
    server.serve_forever()


def _read_job_file(job_file: Path) -> dict:
    """Safely read and parse a job JSON file."""
    try:
        with open(job_file, "r") as f:
            content = f.read().strip()
            if not content or content == "{}":
                return {}
            return json.loads(content)
    except json.JSONDecodeError as e:
        log.warning(f"Corrupt job file (will retry): {e}")
        return {}
    except OSError as e:
        log.debug(f"Could not read job file: {e}")
        return {}


def _write_result(job: dict, success: bool, error_msg: str = ""):
    """Write job result for status tracking."""
    result = {
        "url": job.get("url", ""),
        "status": "success" if success else "failed",
        "error": error_msg,
        "completed_at": time.time(),
        "flags": job.get("flags", []),
    }
    try:
        with open("remote_job_result.json", "w") as f:
            json.dump(result, f, indent=2)
    except OSError as e:
        log.warning(f"Could not write result file: {e}")


def _clear_job_file(job_file: Path):
    """Remove or clear the job file after processing."""
    try:
        if job_file.exists():
            job_file.unlink()
    except OSError:
        # If unlink fails on Drive mount, at least clear the content
        try:
            with open(job_file, "w") as f:
                f.write("{}")
        except OSError as e:
            log.warning(f"Could not clear job file: {e}")


def watch():
    """Main watch loop — poll for jobs from queue and file system."""
    log.info("👀 Colab Worker started. Watching for jobs via Drive + Direct API...")
    job_file = Path("remote_job.json")

    while True:
        job = None

        # Priority 1: Jobs from HTTP tunnel (in-memory queue)
        with _job_lock:
            if pending_jobs:
                job = pending_jobs.pop(0)

        # Priority 2: Jobs from Drive mount (file-based)
        if not job and job_file.exists():
            job = _read_job_file(job_file)

        if job and job.get("status") == "pending":
            url = job["url"]
            flags = job.get("flags", [])
            log.info(f"✨ NEW JOB DETECTED: {url}")
            log.info(f"🚩 Flags: {' '.join(flags) if flags else '(none)'}")

            # Mark as processing immediately to prevent re-pick
            job["status"] = "processing"
            try:
                with open(job_file, "w") as f:
                    json.dump(job, f, indent=2)
            except OSError:
                pass

            # Run the pipeline
            log.info("🚀 Starting Pipeline...")
            cmd = ["python", "pipeline.py", url] + flags
            result = subprocess.run(cmd)

            if result.returncode == 0:
                log.info("✅ Pipeline Finished Successfully!")
                _write_result(job, success=True)
            else:
                log.error(f"❌ Pipeline Failed! (exit code {result.returncode})")
                _write_result(job, success=False, error_msg=f"Exit code {result.returncode}")

            _clear_job_file(job_file)

        time.sleep(2)


if __name__ == "__main__":
    if os.environ.get("USE_API") == "1":
        port = int(os.environ.get("PORT", 5000))
        threading.Thread(target=run_server, args=(port,), daemon=True).start()

    watch()
