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
STATUS_FILE = "status.json"
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "7200"))  # 2 hours default

job_queue = []
processing_lock = threading.Lock()
currently_processing = False


def _ts() -> str:
    """Return current timestamp string."""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_info(msg: str):
    """Print timestamped log message."""
    print(f"[{_ts()}] {msg}")


def write_status(status: str, url: str = "", message: str = ""):
    """Write a status file so the notebook/Mac client can track progress."""
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump({
                "status": status,
                "url": url,
                "message": message,
                "timestamp": time.time(),
            }, f, indent=2)
    except Exception:
        pass


class JobHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "processing": currently_processing}).encode())
        elif parsed.path == "/files":
            import os
            data = {}
            for name in ["input", "transcripts", "highlights", "shorts", "photos"]:
                p = Path(name)
                if p.exists():
                    files = []
                    for f in sorted(p.iterdir()):
                        if f.is_file() and not f.name.startswith("."):
                            files.append({"name": f.name, "size": f.stat().st_size})
                    data[name] = files
                else:
                    data[name] = []
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(data, indent=2).encode())
        elif parsed.path.startswith("/download/"):
            file_path = parsed.path[len("/download/"):]
            target = Path(file_path)
            if target.exists() and target.is_file() and target.stat().st_size > 0:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
                self.send_header("Content-Length", str(target.stat().st_size))
                self.end_headers()
                with open(target, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"File not found: {file_path}"}).encode())
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

                # ─── Save secrets delivered via tunnel ───────────────────────
                for secret_file in ["client_secrets.json", "yt_token.json"]:
                    if secret_file in job and job[secret_file]:
                        Path(secret_file).write_text(job[secret_file], encoding="utf-8")
                        log_info(f"🔑 Saved {secret_file} ({len(job[secret_file])} bytes)")
                        del job[secret_file]  # don't pass to pipeline

                job_queue.append(job)
                log_info(f"Job received via tunnel: {url}")

                with processing_lock:
                    if not currently_processing:
                        threading.Thread(target=process_queue, daemon=True).start()

                self.send_response(202)
                self.end_headers()
                self.wfile.write(json.dumps({"status": "accepted"}).encode())
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif parsed.path == "/exec":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                req = json.loads(body)
                cmd = req.get("cmd", "")
                if not cmd:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Missing cmd"}).encode())
                    return
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif parsed.path == "/write":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                req = json.loads(body)
                file_path = req.get("path", "")
                content = req.get("content", "")
                if not file_path:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Missing path"}).encode())
                    return
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                Path(file_path).write_text(content, encoding="utf-8")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"status": "written", "path": file_path, "size": len(content)}).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def process_queue():
    global currently_processing
    # Ensure deno is in PATH for yt-dlp JS challenge solver
    deno_bin = str(Path.home() / ".deno" / "bin")
    env = os.environ.copy()
    if deno_bin not in env.get("PATH", ""):
        env["PATH"] = deno_bin + ":" + env.get("PATH", "")

    # Ensure project root is on PYTHONPATH so automation.cli can resolve
    # root-level modules (download, transcribe, export, highlight, utils.*)
    project_root = str(Path(__file__).resolve().parent)
    existing_pp = env.get("PYTHONPATH", "")
    if project_root not in existing_pp:
        env["PYTHONPATH"] = project_root + (os.pathsep + existing_pp if existing_pp else "")

    while True:
        with processing_lock:
            if not job_queue:
                currently_processing = False
                return
            job = job_queue.pop(0)
            currently_processing = True
        url = job.get("url", "")
        flags = job.get("flags", [])

        cmd = [sys.executable, "-m", "automation.cli", url] + flags

        log_info(f"{'='*55}")
        log_info(f"  PROCESSING: {url}")
        log_info(f"{'='*55}")
        log_info(f"  Command: {' '.join(cmd[-3:])} ...")
        log_info(f"  Timeout: {JOB_TIMEOUT}s")
        write_status("running", url, "Pipeline executing...")
        t_start = time.time()
        result = None
        try:
            result = subprocess.run(cmd, env=env, cwd=project_root, timeout=JOB_TIMEOUT)
            elapsed = time.time() - t_start
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t_start
            log_info(f"Job timed out after {JOB_TIMEOUT}s ({elapsed:.0f}s elapsed): {url}")
            write_status("failed", url, f"Timed out after {JOB_TIMEOUT}s")
            with open(RESULT_FILE, "w") as f:
                json.dump({
                    "status": "failed",
                    "returncode": -1,
                    "url": url,
                    "error": f"TimeoutExpired ({JOB_TIMEOUT}s)",
                }, f, indent=2)
            continue
        except KeyboardInterrupt:
            log_info("Job interrupted by user")
            write_status("interrupted", url, "KeyboardInterrupt")
            with open(RESULT_FILE, "w") as f:
                json.dump({
                    "status": "failed",
                    "returncode": -1,
                    "url": url,
                    "error": "KeyboardInterrupt",
                }, f, indent=2)
            continue

        status = "done" if result.returncode == 0 else "failed"
        with open(RESULT_FILE, "w") as f:
            json.dump({
                "status": status,
                "returncode": result.returncode,
                "url": url,
            }, f, indent=2)

        status_label = "OK" if result.returncode == 0 else "FAILED"
        write_status(status, url, f"Exit code {result.returncode} ({elapsed:.0f}s)")
        log_info(f"[EXIT] Job {status_label} url={url} exit={result.returncode} elapsed={elapsed:.0f}s\n")


def poll_job_file():
    job_path = Path(JOB_FILE)
    while True:
        if job_path.exists():
            with processing_lock:
                if currently_processing:
                    time.sleep(5)
                    continue
            try:
                with open(job_path) as f:
                    job = json.load(f)
                url = job.get("url", "")
                if url:
                    # ─── Extract secrets delivered via Drive job ─────────────
                    for secret_file in ["client_secrets.json", "yt_token.json"]:
                        if secret_file in job and job[secret_file]:
                            Path(secret_file).write_text(job[secret_file], encoding="utf-8")
                            log_info(f"Saved {secret_file} ({len(job[secret_file])} bytes) from Drive job")
                            del job[secret_file]
                    log_info(f"Queued: {url}")
                    write_status("queued", url, "Job detected on Drive, queued for processing")
                    with processing_lock:
                        job_queue.append(job)
                    threading.Thread(target=process_queue, daemon=True).start()
                else:
                    log_info("Skipping job file: missing 'url' field")
                job_path.unlink(missing_ok=True)
            except json.JSONDecodeError as e:
                log_info(f"Invalid JSON in job file: {e}")
                job_path.unlink(missing_ok=True)
            except Exception as e:
                log_info(f"File poll error: {e}")
                job_path.unlink(missing_ok=True)
        time.sleep(10)


if __name__ == "__main__":
    log_info(f"Starting watcher on port {PORT}...")
    log_info(f"Job timeout: {JOB_TIMEOUT}s (set JOB_TIMEOUT env to change)")
    log_info(f"Polling: {JOB_FILE} every 10s")
    log_info(f"Status file: {STATUS_FILE}")
    write_status("idle", "", "Watcher started")

    poller = threading.Thread(target=poll_job_file, daemon=True)
    poller.start()

    server = HTTPServer(("0.0.0.0", PORT), JobHandler)
    log_info(f"HTTP server: http://0.0.0.0:{PORT}")
    log_info(f"POST /job to submit a pipeline job")
    log_info(f"GET /health to check status")
    log_info(f"GET /files to list working files")
    log_info(f"Watcher ready")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_info("Shutting down...")
        write_status("stopped", "", "Watcher shutting down")
        server.shutdown()
