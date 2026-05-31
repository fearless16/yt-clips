"""worker.py — Safe parallel agent pool with memory backpressure.

Uses threading + Semaphore for throttling (no process pool, avoids
serialisation overhead). Every batch calls ensure_free between batches.

Usage::

    from .worker import ParallelPool

    pool = ParallelPool(max_workers=2, max_memory_gb=2.0)
    pool.submit(fn, arg)           # returns _ControlledFuture
    pool.map(fn, [1, 2, 3])       # blocking, all results
    pool.batch_run(fn, items, 2)   # batched with memory backpressure
    pool.shutdown()
"""

import time
import logging
import threading
from concurrent.futures import Future

from .memory import ensure_free, memory_report

log = logging.getLogger("worker")


class _ControlledFuture:
    """Thin wrapper around concurrent.futures.Future.

    Exposes result(), done(), cancel().
    """

    def __init__(self, future: Future):
        self._future = future

    def result(self, timeout=None):
        return self._future.result(timeout=timeout)

    def done(self):
        return self._future.done()

    def cancel(self):
        return self._future.cancel()


class ParallelPool:
    """Thread-based parallel worker pool with memory throttling.

    Args:
        max_workers: Maximum number of concurrent threads.
        max_memory_gb: Minimum free GB required per batch.
    """

    def __init__(self, max_workers: int = 2, max_memory_gb: float = 2.0):
        self._max_workers = max_workers
        self._max_memory_gb = max_memory_gb
        self._semaphore = threading.Semaphore(max_workers)
        self._active = 0
        self._lock = threading.Lock()
        self._shutdown = False
        self._stop_event = threading.Event()

    def submit(self, fn, *args, **kwargs) -> _ControlledFuture:
        """Submit a function for execution. Returns _ControlledFuture.

        Raises RuntimeError if pool is shut down.
        """
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
        """Apply *fn* to every *items* element. Blocks until all done."""
        results = []
        futures = [self.submit(fn, item) for item in items]
        for f in futures:
            results.append(f.result())
        return results

    def batch_run(self, fn, items, batch_size: int = 4):
        """Process *items* in batches with memory backpressure between batches.

        Calls ensure_free after each batch before starting the next.
        """
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i: i + batch_size]
            ensure_free(self._max_memory_gb, poll_interval=2.0, timeout=30.0)
            for item in batch:
                results.append(self.submit(fn, item))
        return [r.result() for r in results]

    def shutdown(self, wait: bool = True):
        """Signal shutdown. If *wait*, block until all workers finish."""
        self._stop_event.set()
        with self._lock:
            self._shutdown = True
        if wait:
            while self.active_count() > 0:
                time.sleep(0.1)

    def pool_size(self) -> int:
        """Return the configured max worker count."""
        return self._max_workers

    def active_count(self) -> int:
        """Return number of currently executing workers."""
        with self._lock:
            return self._active
