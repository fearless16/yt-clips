"""dashboard.py — Graphical memory dashboard for Colab/Kaggle.

Renders live RAM + GPU memory charts and a system status panel.
Works in three modes:
  - colab: matplotlib inline charts (via IPython display)
  - terminal: ASCII/Unicode fallback
  - headless: returns dict (for CLI --memory-report)

Usage::

    from .dashboard import Dashboard, render_report

    dash = Dashboard()
    dash.sample()                       # record a sample
    chart = dash.memory_sparkline()     # Unicode sparkline
    report = dash.report()              # rich text dashboard

    # One-shot report:
    text = render_report()
    print(text)
"""

import time
import shutil
import datetime
from pathlib import Path
from collections import deque
from threading import Lock

from .memory import memory_report, emit_graph, _read_meminfo, _ring
from .env import gpu_info

try:
    from . import VERSION
except ImportError:
    VERSION = "2.0.0"

_ENV_COLAB = False
_ENV_KAGGLE = False
try:
    import google.colab  # noqa: F401
    _ENV_COLAB = True
except ImportError:
    pass
try:
    import IPython.display
    _ENV_COLAB = True  # IPython display works in both
except ImportError:
    pass


def _colab_display(html: str):
    if _ENV_COLAB:
        from IPython.display import HTML, display
        display(HTML(html))


def _terminal_width() -> int:
    try:
        return shutil.get_terminal_size((80, 20)).columns
    except Exception:
        return 80


# ─── Dashboard ────────────────────────────────────────────────────────────

class Dashboard:
    """Live memory/GPU tracker with chart rendering.

    Samples are stored in a per-instance ring buffer (separate from
    memory.py's global _ring).  Call ``.sample()`` periodically,
    then ``.render()`` or ``.report()`` to display.

    Args:
        max_samples: Max history entries (default 120 = 10 min at 5s intervals).
    """

    def __init__(self, max_samples: int = 120):
        self._ram_pct = deque(maxlen=max_samples)
        self._gpu_pct = deque(maxlen=max_samples)
        self._timestamps = deque(maxlen=max_samples)
        self._lock = Lock()
        self._start = time.monotonic()

    def sample(self):
        """Record a RAM + GPU memory snapshot."""
        total, free, env = _read_meminfo()
        ram_pct = ((total - free) / total * 100) if total > 0 else 0.0
        gpu = gpu_info()
        gpu_total = gpu.get("memory_total_gb", 0)
        gpu_free = gpu.get("memory_free_gb", 0)
        gpu_pct = ((gpu_total - gpu_free) / gpu_total * 100) if gpu_total > 0 else 0.0
        now = time.monotonic() - self._start
        with self._lock:
            self._ram_pct.append(ram_pct)
            self._gpu_pct.append(gpu_pct)
            self._timestamps.append(now)

    def snapshot(self) -> dict:
        """Return current raw data as dict."""
        with self._lock:
            return {
                "ram_pct": list(self._ram_pct),
                "gpu_pct": list(self._gpu_pct),
                "timestamps": list(self._timestamps),
                "uptime": round(time.monotonic() - self._start, 1),
            }

    def clear(self):
        """Clear all history."""
        with self._lock:
            self._ram_pct.clear()
            self._gpu_pct.clear()
            self._timestamps.clear()

    def memory_sparkline(self, width: int = 30) -> str:
        """Unicode sparkline of RAM usage (last *width* samples)."""
        from .memory import _bars
        with self._lock:
            snap = list(self._ram_pct)
        if not snap:
            return ""
        vals = snap[-width:]
        mn, mx = min(vals), max(vals)
        span = mx - mn or 1.0
        return "".join(_bars[min(7, int((v - mn) / span * 7))] for v in vals)

    def gpu_sparkline(self, width: int = 30) -> str:
        """Unicode sparkline of GPU usage (last *width* samples)."""
        from .memory import _bars
        with self._lock:
            snap = list(self._gpu_pct)
        if not snap:
            return ""
        vals = snap[-width:]
        mn, mx = min(vals), max(vals)
        span = mx - mn or 1.0
        return "".join(_bars[min(7, int((v - mn) / span * 7))] for v in vals)

    def render(self, width: int | None = None) -> str:
        """Render a full text dashboard panel.

        Suitable for printing to terminal or colab cell output.
        """
        width = width or _terminal_width()
        sep = "─" * width
        mem = memory_report()
        gpu = gpu_info()
        ram_spark = self.memory_sparkline(30)
        gpu_spark = self.gpu_sparkline(30)
        uptime = datetime.timedelta(seconds=int(time.monotonic() - self._start))

        lines = [
            sep,
            f"  yt-clips v{VERSION}  │  uptime {uptime}".center(width),
            sep,
            f"  RAM  {mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB  ({100*mem['used_gb']/max(mem['total_gb'],0.01):.0f}%)  {ram_spark}",
            f"  GPU  {gpu['name']}  {gpu.get('memory_free_gb', 0):.1f} free  {gpu_spark}",
            f"  Env  {mem['environment']}  batch={mem['safe_batch_size']} workers={mem['safe_parallel_workers']}",
            sep,
        ]
        if gpu.get("memory_total_gb", 0) > 0:
            gpu_used = gpu["memory_total_gb"] - gpu.get("memory_free_gb", 0)
            gpu_bar = _pct_bar(gpu_used / max(gpu["memory_total_gb"], 0.01))
            lines.insert(4, f"  GPU ┃{gpu_bar}┃ {gpu_used:.1f}/{gpu['memory_total_gb']:.1f} GB")
        ram_used = mem["used_gb"]
        ram_bar = _pct_bar(ram_used / max(mem["total_gb"], 0.01))
        lines.insert(3, f"  RAM ┃{ram_bar}┃ {ram_used:.1f}/{mem['total_gb']:.1f} GB")
        return "\n".join(lines)

    def report(self) -> str:
        """Return a one-shot memory/GPU text report."""
        mem = memory_report()
        gpu = gpu_info()
        lines = [
            f"yt-clips v{VERSION}  env={mem['environment']}",
            f"  RAM:  {mem['used_gb']:.1f} / {mem['total_gb']:.1f} GB  ({100*mem['used_gb']/max(mem['total_gb'],0.01):.0f}%)",
            f"  GPU:  {gpu['name']}  {gpu.get('memory_free_gb', 0):.1f} GB free",
            f"  Safe: batch={mem['safe_batch_size']} workers={mem['safe_parallel_workers']}",
        ]
        spark = self.memory_sparkline(20)
        if spark:
            lines.insert(1, f"  RAM sparkline: {spark}")
        gs = self.gpu_sparkline(20)
        if gs:
            lines.insert(2, f"  GPU sparkline: {gs}")
        return "\n".join(lines)

    def html(self) -> str:
        """Return an HTML dashboard suitable for IPython.display.

        Only works in colab/Jupyter (charts via inline SVG/CSS bars).
        """
        mem = memory_report()
        gpu = gpu_info()
        uptime = datetime.timedelta(seconds=int(time.monotonic() - self._start))
        ram_used = mem["used_gb"]
        ram_total = mem["total_gb"]
        ram_pct = ram_used / max(ram_total, 0.01) * 100
        gpu_used = gpu["memory_total_gb"] - gpu.get("memory_free_gb", 0)
        gpu_total = gpu["memory_total_gb"]
        gpu_pct = gpu_used / max(gpu_total, 0.01) * 100 if gpu_total > 0 else 0

        bar_css = "height:20px;border-radius:4px;transition:width .5s;min-width:2px"
        ram_color = f"hsl({max(0, 120 - ram_pct * 1.2)}, 70%, 45%)"
        gpu_color = f"hsl({max(0, 120 - gpu_pct * 1.2)}, 70%, 45%)"

        return f"""<div style="font-family:monospace;font-size:13px;line-height:1.6;background:#1a1a2e;color:#e0e0e0;padding:16px;border-radius:8px;max-width:640px">
  <div style="font-size:15px;font-weight:bold;margin-bottom:8px;color:#00d4ff">
    yt-clips v{VERSION}  │  uptime {uptime}
  </div>
  <div style="margin:4px 0">
    <span style="color:#888">RAM</span>
    <div style="background:#333;border-radius:4px;margin:2px 0">
      <div style="width:{ram_pct}%;background:{ram_color};{bar_css}"></div>
    </div>
    <span>{ram_used:.1f} / {ram_total:.1f} GB ({ram_pct:.0f}%)</span>
  </div>
  <div style="margin:4px 0">
    <span style="color:#888">GPU  {gpu['name']}</span>
    <div style="background:#333;border-radius:4px;margin:2px 0">
      <div style="width:{gpu_pct}%;background:{gpu_color};{bar_css}"></div>
    </div>
    <span>{gpu_used:.1f} / {gpu_total:.1f} GB ({gpu_pct:.0f}%)</span>
  </div>
  <div style="margin-top:8px;color:#aaa;font-size:12px">
    batch={mem['safe_batch_size']}  workers={mem['safe_parallel_workers']}  env={mem['environment']}
  </div>
</div>"""

    def display(self):
        """Render HTML dashboard in colab/IPython, or text fallback."""
        try:
            from IPython.display import HTML, display as ipy_display
            ipy_display(HTML(self.html()))
        except ImportError:
            print(self.render())


# ─── Helper ───────────────────────────────────────────────────────────────

def _pct_bar(pct: float, width: int = 20) -> str:
    """Render a Unicode bar like ``███████░░░``."""
    filled = int(pct * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


# ─── Top-level helpers ────────────────────────────────────────────────────

_dash_global: Dashboard | None = None
_dash_lock = Lock()


def get_dashboard() -> Dashboard:
    """Return the module-level Dashboard singleton."""
    global _dash_global
    if _dash_global is None:
        with _dash_lock:
            if _dash_global is None:
                _dash_global = Dashboard()
    return _dash_global


def sample():
    """Record a RAM + GPU memory snapshot on the global dashboard."""
    get_dashboard().sample()


def render_dashboard() -> str:
    """Render the full dashboard text panel."""
    return get_dashboard().render()


def render_report() -> str:
    """One-shot memory/GPU text report."""
    return get_dashboard().report()


def show_dashboard():
    """Display the HTML dashboard in colab, or text fallback.

    Call this periodically (e.g. every 15s) in a colab loop.
    """
    dash = get_dashboard()
    dash.sample()
    dash.display()


def show_report():
    """Print a one-shot memory/GPU report to stdout."""
    print(render_report())
