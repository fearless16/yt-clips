"""
watcher.py — Colab Worker: Listens for pipeline jobs via HTTP tunnel + file poll.
"""
import json
import os
import subprocess
import sys
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

PORT = int(os.environ.get("PORT", "5000"))
JOB_FILE = "remote_job.json"
RESULT_FILE = "remote_job_result.json"

job_queue = []
processing_lock = threading.Lock()
currently_processing = False


class JobHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "processing": currently_processing}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/job":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                job = json.loads(body)
                url = job.get("url", "")
                if not url:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Missing url"}).encode())
                    return

                job_queue.append(job)
                print(f"[WATCHER] Job received via tunnel: {url}")

                if not currently_processing:
                    threading.Thread(target=process_queue, daemon=True).start()

                self.send_response(202)
                self.end_headers()
                self.wfile.write(json.dumps({"status": "accepted"}).encode())
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def process_queue():
    global currently_processing
    while job_queue:
        with processing_lock:
            currently_processing = True
            if not job_queue:
                break
            job = job_queue.pop(0)
        url = job.get("url", "")
        flags = job.get("flags", [])

        print(f"\n{'='*55}")
        print(f"  PROCESSING: {url}")
        print(f"{'='*55}\n")

        cmd = [sys.executable, "pipeline.py", url] + flags
        result = None
        try:
            result = subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n  Job interrupted by user")
            with open(RESULT_FILE, "w") as f:
                json.dump({
                    "status": "failed",
                    "returncode": -1,
                    "url": url,
                    "error": "KeyboardInterrupt",
                }, f, indent=2)
            continue

        with open(RESULT_FILE, "w") as f:
            json.dump({
                "status": "done" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "url": url,
            }, f, indent=2)

        status = "OK" if result.returncode == 0 else "FAILED"
        print(f"  Job {status} (exit={result.returncode})\n")

    with processing_lock:
        currently_processing = False


def poll_job_file():
    global currently_processing
    job_path = Path(JOB_FILE)
    while True:
        if job_path.exists() and not currently_processing:
            try:
                with open(job_path) as f:
                    job = json.load(f)
                url = job.get("url", "")
                if url:
                    print(f"[WATCHER] Job detected via file: {url}")
                    job_queue.append(job)
                    threading.Thread(target=process_queue, daemon=True).start()
                job_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[WATCHER] File poll error: {e}")
                job_path.unlink(missing_ok=True)
        time.sleep(10)


if __name__ == "__main__":
    print(f"[WATCHER] Starting on port {PORT}...")

    poller = threading.Thread(target=poll_job_file, daemon=True)
    poller.start()

    server = HTTPServer(("0.0.0.0", PORT), JobHandler)
    print(f"[WATCHER] HTTP server: http://0.0.0.0:{PORT}")
    print(f"[WATCHER] POST /job to submit a pipeline job")
    print(f"[WATCHER] GET /health to check status")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[WATCHER] Shutting down...")
        server.shutdown()
