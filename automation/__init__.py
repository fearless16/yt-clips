"""automation — yt-clips automation pipeline.

Zero FaceOS deps. Built for speed, low token usage, parallel safety.

Submodules:
    _cache       TTL + LRU query cache (thread-safe)
    config       Cached YAML config loader
    env          Colab/Kaggle detection, GPU queries
    memory       RAM tracker with ring buffer, sparkline, backpressure
    transcript   YouTube transcript fetcher + LLM formatter
    scoring      LLM output quality scoring + evaluation
    watcher      Watcher subprocess lifecycle
    tunnel       Always-up TunnelKeeper daemon
    worker       ParallelPool with Semaphore throttle
    orchestator  8-phase pipeline runner
    cli          CLI entry point with subcommands

Usage:
    python -m automation.cli <url>              # full pipeline
    python -m automation.cli --memory-report    # RAM/GPU snapshot
    python -m automation.cli --tunnel-status    # tunnel health
    python -m automation.cli --fetch-transcript <url>  # transcript only
"""

VERSION = "2.0.0"
