"""worker.py — Safe parallel agent pool with memory backpressure.
Thread.Semaphore throttling, batch processing, graceful shutdown."""

import time
import queue
import threading
from concurrent.futures import Future

from .memory import ensure_free, memory_report

_log_prefix = "[worker]"


def _log(msg: str):
    import sys
    print(f"{_log_prefix} {msg}", file=sys.stderr, flush=True)


class _ControlledFuture:
    def __init__(self, future: Future):
        self._future = future

    def result(self, timeout=None):
        return self._future.result(timeout=timeout)

    def done(self):
        return self._future.done()

    def cancel(self):
        return self._future.cancel()


class ParallelPool:
    def __init__(self, max_workers: int = 2, max_memory_gb: float = 2.0):
        self._max_workers = max_workers
        self._max_memory_gb = max_memory_gb
        self._semaphore = threading.Semaphore(max_workers)
        self._active = 0
        self._lock = threading.Lock()
        self._shutdown = False
        self._stop_event = threading.Event()

    def submit(self, fn, *args, **kwargs):
        future: Future = Future()
        self._semaphore.acquire()
        with self._lock:
            if self._shutdown:
                self._semaphore.release()
                raise RuntimeError("pool is shut down")
            self._active += 1

        def _run():
            try:
                if self._stop_event.is_set():
                    return
                report = memory_report()
                if report["environment"] == "linux":
                    ensure_free(self._max_memory_gb, poll_interval=2.0, timeout=30.0)
                result = fn(*args, **kwargs)
                future.set_result(result)
            except BaseException as e:
                future.set_exception(e)
            finally:
                with self._lock:
                    self._active -= 1
                self._semaphore.release()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return _ControlledFuture(future)

    def map(self, fn, items):
        results = []
        futures = [self.submit(fn, item) for item in items]
        for f in futures:
            results.append(f.result())
        return results

    def batch_run(self, fn, items, batch_size: int = 4):
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            ensure_free(self._max_memory_gb, poll_interval=2.0, timeout=30.0)
            for item in batch:
                results.append(self.submit(fn, item))
        return [r.result() for r in results]

    def shutdown(self, wait: bool = True):
        self._stop_event.set()
        with self._lock:
            self._shutdown = True
        if wait:
            while self.active_count() > 0:
                time.sleep(0.1)

    def pool_size(self) -> int:
        return self._max_workers

    def active_count(self) -> int:
        with self._lock:
            return self._active
