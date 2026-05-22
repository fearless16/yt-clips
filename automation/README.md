# automation — yt-clips Pipeline

A standalone pipeline module for yt-clips with zero FaceOS dependencies.
Built for speed, low token usage, and parallel agent safety on Colab/Kaggle.

## Architecture

```
automation/
├── __init__.py      Package exports + VERSION
├── _cache.py        TTL + LRU cache (thread-safe), 4 global singletons
├── config.py        Cached YAML config with dot-notation get()
├── env.py           Colab/Kaggle detection, nvidia-smi GPU queries
├── memory.py        /proc/meminfo reader, ring buffer, sparkline, backpressure
├── transcript.py    YouTube transcript fetcher (API → yt-dlp, 1h cache)
├── watcher.py       Watcher subprocess lifecycle, /health polling
├── tunnel.py        TunnelKeeper daemon, auto-reconnect, 3 fallback methods
├── worker.py        ParallelPool (threading + Semaphore, batch backpressure)
├── orchestrator.py  8-phase pipeline runner
├── cli.py           CLI entry point with subcommands
└── README.md        This file
```

## Key Design Decisions

### 1. Caching (all external queries)
- `TTLCache` in `_cache.py` — LRU eviction + TTL expiry, thread-safe
- 4 pre-configured singletons: CONFIG, TRANSCRIPT, GPU, MEMORY
- Repeat calls within TTL return cached results, zero network/disk cost

### 2. Memory Tracking
- Only reads `/proc/meminfo` (Colab/Linux). macOS detected as `env="local"`
- `_RingBuffer` (60-sample deque) for usage history
- `emit_graph()` — 8-level Unicode sparkline
- `ensure_free()` — blocks until ≥N GB free, or timeout
- `safe_batch_size()` / `safe_workers()` — halve on low memory

### 3. Always-Up Tunnel
- `TunnelKeeper` — background daemon thread, heartbeat every 10s
- 3 consecutive health check failures → auto-reconnect
- Fallback chain: serveo.net → localhost.run → localtunnel
- Singleton helpers: `start_tunnel()`, `tunnel_status()`, `kill_tunnel()`

### 4. Parallel Workers
- `ParallelPool` — threading + `Semaphore`, no process pool
- `batch_run()` — calls `ensure_free()` between every batch
- `_ControlledFuture` — thin wrapper over `concurrent.futures.Future`

### 5. Lazy Imports
- Every module uses deferred imports for heavy packages (yaml, cv2, yt-dlp)
- `orchestrator.py` lazy-imports each pipeline phase
- CLI flags only import their required submodules

## CLI Usage

```bash
# Full pipeline
python -m automation.cli https://youtu.be/dQw4w9WgXcQ

# With sync + upload
python -m automation.cli https://youtu.be/dQw4w9WgXcQ --sync --upload

# Skip phases
python -m automation.cli https://youtu.be/dQw4w9WgXcQ --skip-download --skip-highlight

# Diagnostics
python -m automation.cli --memory-report
python -m automation.cli --gpu-info
python -m automation.cli --tunnel-status
python -m automation.cli --fetch-transcript https://youtu.be/dQw4w9WgXcQ

# Colab setup
python -m automation.cli --setup-colab

# Advanced
python -m automation.cli --sync-only
python -m automation.cli --auto-pilot https://youtube.com/@channel
python -m automation.cli --remote https://abc.lhr.life
```

## Programmatic API

```python
# Transcript
from automation.transcript import fetch
data = fetch("https://youtu.be/dQw4w9WgXcQ")
print(data["source"], len(data["segments"]))

# Memory
from automation.memory import memory_report, emit_graph
print(memory_report()["free_gb"], "GB free")
print(emit_graph())

# Tunnel
from automation.tunnel import start_tunnel, tunnel_status
start_tunnel()
print(tunnel_status())

# Workers
from automation.worker import ParallelPool
pool = ParallelPool(max_workers=2)
results = pool.map(lambda x: x * 2, [1, 2, 3])
pool.shutdown()

# Config
from automation.config import get
print(get("paths.input", "default"))

# Full pipeline
from automation.orchestrator import run
result = run("https://youtu.be/dQw4w9WgXcQ", auto_sync=True)
print(result.exported, result.failures)
```

## Running Tests

```bash
.venv/bin/python -m pytest tests/test_automation.py -v
```

45 tests covering all 9 submodules with edge cases:
- Cache TTL expiry, LRU eviction, thread safety
- Config missing-file fallback, dot-notation default
- Memory report keys, ensure_free type, sparkline empty
- Video ID extraction (5 formats), VTT parsing (2 timestamp formats)
- Env detection (local), GPU info shape, tunnel status shape
- Worker submit/map/batch/shutdown/shutdown-after-submit
- PipelineResult defaults
- CLI --help and --memory-report
