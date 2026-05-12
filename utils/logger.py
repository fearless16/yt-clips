import logging
import json
import sys
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

_CONSOLE = Console(stderr=False)


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
            entry = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info and record.exc_info[0]:
                entry["exc"] = self.format(record)
            self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        self._file.close()
        super().close()


def get_logger(name: str, log_file: str = "logs/pipeline.log", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    c_handler = RichHandler(
        console=_CONSOLE,
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    c_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(c_handler)

    f_handler = JsonFileHandler(log_file)
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)

    return logger


phase_tracker = PhaseTracker()
