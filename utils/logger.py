import logging
import json
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TimeElapsedColumn, TimeRemainingColumn, TaskID,
)
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.traceback import install as rich_traceback_install
from rich import box

rich_traceback_install(show_locals=False)


class _ReplaceStream:
    """Text stream wrapper that never raises on non-encodable output.

    On Windows the console defaults to cp1252, so an emoji in a log message
    raises UnicodeEncodeError inside the Rich handler, which logging swallows
    via handleError — silently dropping the record. Wrapping the write path
    with errors="replace" (the fix endorsed across Stack Overflow / Rich
    issues #3907/#3764) guarantees the record is always emitted.
    """

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        try:
            self._stream.write(text)
        except UnicodeEncodeError:
            # Sanitize to the sink's encoding (cp1252 on Windows consoles)
            # using replacement chars, so the record is always emitted.
            safe = text.encode("cp1252", errors="replace").decode("cp1252")
            try:
                self._stream.write(safe)
            except UnicodeEncodeError:
                self._stream.write(text.encode("ascii", errors="replace").decode("ascii"))
        return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _safe_stream(stream):
    """Return a stream that never raises on non-encodable chars.

    Prefer reconfiguring the real stdout/stderr to UTF-8 (per PEP 528 /
    community guidance). If that is unavailable, wrap writes with a
    replacement-error handler so log records are never silently dropped.
    """
    try:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
        return stream
    except (ValueError, OSError, AttributeError):
        return _ReplaceStream(stream)


_CONSOLE = Console(
    stderr=False,
    file=_safe_stream(sys.stdout),
    emoji=False,
    soft_wrap=True,
)

# Harden the process-wide streams once (defense-in-depth). Even if some other
# library prints directly to stdout/stderr, output stays UTF-8-safe on Windows.
for _s in (sys.stdout, sys.stderr):
    try:
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError, AttributeError):
        pass


class PhaseTracker:
    def __init__(self):
        self.phases: list[dict] = []
        self._current: Optional[str] = None
        self._start: Optional[datetime] = None

    def begin(self, name: str) -> None:
        self._current = name
        self._start = datetime.now()
        _CONSOLE.print()
        _CONSOLE.print(Panel(
            Text(name, style="bold cyan", justify="center"),
            border_style="cyan",
            width=70,
        ))

    def end(self, status: str = "done") -> float:
        if not self._current or not self._start:
            return 0.0
        elapsed = (datetime.now() - self._start).total_seconds()
        self.phases.append({"name": self._current, "elapsed_sec": round(elapsed, 1), "status": status})
        tag = "green" if status == "done" else "red"
        _CONSOLE.print(f"  [{tag}] {self._current} — {elapsed:.1f}s")
        self._current = None
        self._start = None
        return elapsed

    def summary_table(self) -> Table:
        table = Table(title="Pipeline Summary", box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Phase", style="cyan")
        table.add_column("Time", justify="right")
        table.add_column("Status", justify="center")
        total = 0.0
        for p in self.phases:
            tag = "green" if p["status"] == "done" else "red"
            table.add_row(p["name"], f"{p['elapsed_sec']:.1f}s", f"[{tag}]{p['status']}[/]")
            total += p["elapsed_sec"]
        table.add_row("[bold]TOTAL[/]", f"[bold]{total:.1f}s[/]", "")
        return table


class ProgressManager:
    def __init__(self):
        self._progress: Optional[Progress] = None
        self._tasks: dict[str, TaskID] = {}

    def __enter__(self):
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=_CONSOLE,
            transient=False,
        )
        self._progress.__enter__()
        return self

    def __exit__(self, *args):
        if self._progress:
            self._progress.__exit__(*args)

    def add(self, name: str, total: int = 100) -> TaskID:
        if not self._progress:
            raise RuntimeError("ProgressManager not active — use 'with'")
        tid = self._progress.add_task(name, total=total)
        self._tasks[name] = tid
        return tid

    def update(self, name: str, advance: int = 1, description: Optional[str] = None):
        tid = self._tasks.get(name)
        if tid is not None and self._progress:
            kwargs = {"advance": advance}
            if description:
                kwargs["description"] = description
            self._progress.update(tid, **kwargs)

    def stop(self, name: str):
        tid = self._tasks.pop(name, None)
        if tid is not None and self._progress:
            self._progress.update(tid, visible=False)


# Structured fields lifted from LogRecord.__dict__ into the JSON entry when present.
# Keeping this list explicit makes the on-disk log schema stable and queryable.
_STRUCTURED_FIELDS = (
    "stage",         # logical pipeline stage, e.g. "export", "seo"
    "status",        # "start" | "ok" | "failed" | "skipped" | "partial"
    "phase",         # human phase label, e.g. "phase 4 Export"
    "phase_index",   # 1-based position in the run
    "phase_total",   # total phases in the run
    "run_id",        # correlation id shared by every record in one run()
    "duration_ms",   # elapsed wall-clock for the stage
    "error_type",    # exception class name on failure
    "metadata",      # free-form dict of counts/model info/etc.
)


def _build_log_entry(handler: logging.Handler, record: logging.LogRecord) -> dict:
    """Build the structured JSON entry for a log record (shared by handlers)."""
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "level": record.levelname,
        "logger": record.name,
        "msg": record.getMessage(),
    }
    for field in _STRUCTURED_FIELDS:
        if hasattr(record, field):
            entry[field] = getattr(record, field)
    if record.exc_info and record.exc_info[0]:
        entry["exc"] = handler.format(record)
    return entry


class JsonFileHandler(logging.Handler):
    """Writes structured JSON lines to a log file."""

    def __init__(self, log_path: str):
        super().__init__(logging.DEBUG)
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = _build_log_entry(self, record)
            self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        self._file.close()
        super().close()


class JsonStreamHandler(logging.StreamHandler):
    """Writes structured JSON lines to a stream (e.g. stdout)."""
    def __init__(self, stream=None):
        super().__init__(stream)
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = _build_log_entry(self, record)
            self.stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str, log_file: str = "logs/pipeline.log", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Dynamic check for console log format
    console_format = "text"
    try:
        from utils.config import load_config
        cfg = load_config()
        if isinstance(cfg, dict) and "logging" in cfg:
            console_format = cfg["logging"].get("console_format", "text")
    except Exception:
        pass

    if console_format == "json":
        c_handler = JsonStreamHandler(sys.stdout)
    else:
        c_handler = RichHandler(
            console=_CONSOLE,
            show_time=False,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=False,
            show_level=False,
        )
    c_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(c_handler)

    f_handler = JsonFileHandler(log_file)
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)

    return logger


phase_tracker = PhaseTracker()


# ─── Structured phase instrumentation ─────────────────────────────────────────

def new_run_id() -> str:
    """Return a short correlation id for one pipeline run."""
    return uuid.uuid4().hex[:8]


class _PhaseHandle:
    """Yielded by :func:`run_phase`; lets the phase body attach result metadata.

    Example::

        with run_phase(log, "phase 4 Export", "export", run_id=rid) as ph:
            clips = export_all(...)
            ph.set(exported=len(clips))
    """

    __slots__ = ("metadata",)

    def __init__(self) -> None:
        self.metadata: dict = {}

    def set(self, **kwargs) -> None:
        """Attach key/value metadata recorded on the phase's completion log."""
        self.metadata.update(kwargs)


@contextmanager
def run_phase(
    logger: logging.Logger,
    name: str,
    stage: str,
    run_id: Optional[str] = None,
    phase_index: Optional[int] = None,
    phase_total: Optional[int] = None,
):
    """Instrument a pipeline phase with start/end/failure structured logging.

    Guarantees that a record carrying ``stage`` + ``duration_ms`` is emitted
    whether the phase succeeds OR fails — so timing and stage visibility are
    never lost on the error path. On failure the exception is logged AT THE
    SITE with ``exc_info=True``, ``error_type`` and the elapsed time, then
    re-raised so callers can record/aggregate it.

    Args:
        logger: The caller's logger (so log records are attributed to the
            caller's module — important for tests that patch ``module.log``).
        name: Human-readable phase label, e.g. ``"phase 4 Export"``.
        stage: Stable machine-queryable stage key, e.g. ``"export"``.
        run_id: Correlation id shared across the whole run.
        phase_index / phase_total: Position for progress visibility.

    Yields:
        A :class:`_PhaseHandle` whose ``.set(**meta)`` attaches metadata to the
        completion log (counts, model info, etc.).
    """
    def _extra(status: str, duration_ms: int, **extra) -> dict:
        d = {
            "stage": stage,
            "status": status,
            "phase": name,
            "duration_ms": duration_ms,
            "run_id": run_id,
        }
        if phase_index is not None:
            d["phase_index"] = phase_index
        if phase_total is not None:
            d["phase_total"] = phase_total
        d.update(extra)
        return d

    handle = _PhaseHandle()
    t0 = time.monotonic()
    prog = ""
    if phase_index is not None and phase_total is not None:
        prog = " [%d/%d]" % (phase_index, phase_total)
    # Start event: duration_ms=0 keeps the "stage implies duration_ms" invariant.
    logger.info("[%s]%s start", name, prog, extra=_extra("start", 0))

    try:
        yield handle
    except Exception as e:
        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.error(
            "[%s]%s FAILED after %dms: %s",
            name, prog, duration_ms, e,
            exc_info=True,
            extra=_extra(
                "failed", duration_ms,
                error_type=type(e).__name__,
                metadata=handle.metadata or None,
            ),
        )
        raise
    else:
        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "[%s]%s done %.1fs", name, prog, duration_ms / 1000.0,
            extra=_extra(
                "ok", duration_ms,
                metadata=handle.metadata or None,
            ),
        )
