"""
Tests for watcher.py — worker process_queue + job file poll.
"""
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, PropertyMock

import pytest


class TestProcessQueue:
    """process_queue must handle interruptions gracefully."""

    def test_keyboard_interrupt_does_not_crash_watcher(self):
        """
        When subprocess.run raises KeyboardInterrupt, process_queue
        should NOT let it propagate uncaught. It must release the
        processing lock and let the watcher continue serving.
        """
        from watcher import process_queue, job_queue, processing_lock, currently_processing

        job = {"url": "https://youtu.be/test", "flags": []}
        job_queue.append(job)

        interrupt_caught = threading.Event()
        lock_released = threading.Event()

        def run_watcher_thread():
            try:
                process_queue()
            except KeyboardInterrupt:
                interrupt_caught.set()
            finally:
                if not processing_lock.locked():
                    lock_released.set()

        with patch("watcher.subprocess.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()
            t = threading.Thread(target=run_watcher_thread, daemon=True)
            t.start()
            t.join(timeout=5)

        assert not interrupt_caught.is_set(), (
            "KeyboardInterrupt propagated out of process_queue"
        )
        assert lock_released.is_set(), (
            "processing_lock was not released after KeyboardInterrupt"
        )
        assert not currently_processing, (
            "currently_processing should be False after process_queue exits"
        )

        job_queue.clear()

    def test_process_queue_writes_result_even_on_interrupt(self):
        """
        Even when the subprocess is interrupted, process_queue should
        still write the result file so the caller knows the job failed.
        """
        from watcher import process_queue, job_queue, RESULT_FILE

        job_queue.append({"url": "https://youtu.be/test", "flags": []})

        with patch("watcher.subprocess.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()
            process_queue()

        result_path = Path(RESULT_FILE)
        assert result_path.exists(), (
            "Result file should exist even after KeyboardInterrupt"
        )
        with open(result_path) as f:
            result = json.load(f)
        assert result["status"] == "failed"

    def test_pipeline_failure_sets_result_status(self):
        """A non-zero exit code should be recorded as 'failed'."""
        from watcher import process_queue, job_queue, RESULT_FILE

        job_queue.append({"url": "https://youtu.be/test", "flags": []})

        with patch("watcher.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            process_queue()

        with open(RESULT_FILE) as f:
            result = json.load(f)
        assert result["status"] == "failed"
        assert result["returncode"] == 1

    def test_process_queue_pop_inside_lock(self):
        """process_queue should pop from job_queue inside the processing_lock."""
        with open("watcher.py", "r") as f:
            content = f.read()
        assert "with processing_lock" in content
        assert "job_queue.pop(0)" in content
        lines = content.split("\n")
        lock_depth = 0
        pop_inside_lock = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("with processing_lock"):
                lock_depth += 1
            elif stripped.startswith("with ") and not stripped.startswith("with processing_lock"):
                pass
            if "job_queue.pop(0)" in stripped and lock_depth > 0:
                pop_inside_lock = True
        assert pop_inside_lock, "job_queue.pop(0) must be inside processing_lock block"

    def test_concurrent_queue_access_safe(self):
        """Multiple threads should not cause race conditions on job_queue."""
        import random
        from watcher import job_queue, processing_lock

        ops_done = []

        def pusher():
            for i in range(20):
                with processing_lock:
                    job_queue.append({"url": f"https://youtu.be/{i}"})
                    ops_done.append(f"push_{i}")
                time.sleep(random.random() * 0.01)

        def popper():
            for _ in range(20):
                time.sleep(random.random() * 0.01)
                with processing_lock:
                    if job_queue:
                        job_queue.pop(0)
                        ops_done.append("pop")

        threads = [threading.Thread(target=pusher, daemon=True) for _ in range(3)]
        threads += [threading.Thread(target=popper, daemon=True) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        job_queue.clear()


class TestJobFilePoll:
    """poll_job_file must detect and load remote_job.json correctly."""

    def test_poll_detects_and_loads_job_file(self, tmp_path):
        """poll_job_file should detect the job file and add to queue."""
        from watcher import poll_job_file, job_queue, JOB_FILE

        job_queue.clear()

        job_data = {"url": "https://youtu.be/test", "flags": ["--sync"]}
        job_path = Path(tmp_path / JOB_FILE)
        job_path.write_text(json.dumps(job_data))

        with (
            patch("watcher.JOB_FILE", str(job_path)),
            patch("watcher.currently_processing", False),
            patch("watcher.time.sleep", side_effect=InterruptedError),
            patch("watcher.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value.start = MagicMock()
            try:
                poll_job_file()
            except InterruptedError:
                pass

        assert len(job_queue) > 0, (
            "Job file should have been added to queue"
        )
        assert job_queue[0]["url"] == "https://youtu.be/test"
        job_queue.clear()

    def test_poll_does_not_exist_when_processing(self, tmp_path):
        """poll_job_file should NOT pick up a new job while already processing."""
        from watcher import poll_job_file, job_queue, JOB_FILE

        job_queue.clear()

        job_data = {"url": "https://youtu.be/test", "flags": ["--sync"]}
        job_path = Path(tmp_path / JOB_FILE)
        job_path.write_text(json.dumps(job_data))

        with (
            patch("watcher.JOB_FILE", str(job_path)),
            patch("watcher.currently_processing", True),
            patch("watcher.time.sleep", side_effect=InterruptedError),
        ):
            try:
                poll_job_file()
            except InterruptedError:
                pass

        assert len(job_queue) == 0, (
            "Should not pick up job while currently_processing is True"
        )

    def test_poll_deletes_job_file_after_loading(self, tmp_path):
        """poll_job_file should remove the job file after successfully loading it."""
        from watcher import poll_job_file, job_queue, JOB_FILE

        job_queue.clear()
        job_data = {"url": "https://youtu.be/test", "flags": []}
        job_path = Path(tmp_path / JOB_FILE)
        job_path.write_text(json.dumps(job_data))

        with (
            patch("watcher.JOB_FILE", str(job_path)),
            patch("watcher.currently_processing", False),
            patch("watcher.time.sleep", side_effect=InterruptedError),
            patch("watcher.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value.start = MagicMock()
            try:
                poll_job_file()
            except InterruptedError:
                pass

        assert not job_path.exists(), (
            "Job file should have been deleted after loading"
        )
        job_queue.clear()


class TestJobHandler:
    """HTTP handler for incoming jobs."""

    def _make_handler(self, path="/job", body=b""):
        """Create a mock handler with proper rfile support."""
        from watcher import JobHandler

        handler = MagicMock(spec=JobHandler)
        handler.path = path
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = body
        handler.wfile = MagicMock()
        return handler

    def test_job_handler_rejects_empty_url(self):
        """POST /job with no url should return 400."""
        from watcher import JobHandler

        handler = self._make_handler(body=b'{"flags": []}')
        JobHandler.do_POST(handler)

        handler.send_response.assert_called_once_with(400)

    def test_job_handler_accepts_valid_job(self):
        """POST /job with a valid url should return 202."""
        from watcher import JobHandler

        handler = self._make_handler(
            body=b'{"url": "https://youtu.be/test", "flags": ["--sync"]}'
        )
        JobHandler.do_POST(handler)

        handler.send_response.assert_called_once_with(202)

    def test_job_handler_starts_processing_if_not_busy(self):
        """A new job should trigger process_queue via thread."""
        from watcher import JobHandler

        handler = self._make_handler(
            body=b'{"url": "https://youtu.be/test", "flags": ["--sync"]}'
        )

        with (
            patch("watcher.currently_processing", False),
            patch("watcher.threading.Thread") as mock_thread,
        ):
            JobHandler.do_POST(handler)

            mock_thread.assert_called_once()

    def test_job_handler_404_on_unknown_path(self):
        """GET/POST to unknown paths should return 404."""
        from watcher import JobHandler

        handler = self._make_handler(path="/unknown")
        JobHandler.do_GET(handler)

        handler.send_response.assert_called_once_with(404)
