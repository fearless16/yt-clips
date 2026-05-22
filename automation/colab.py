"""colab.py — Backward-compatible re-exports.

Split into env.py (env detection, GPU queries), watcher.py (watcher lifecycle),
and tunnel.py (TunnelKeeper daemon). All public names re-exported here.
"""
from .env import is_colab, is_kaggle, gpu_info, gpu_count, setup  # noqa: F401
from .watcher import WATCHER_PORT, start_watcher, kill_watcher  # noqa: F401
from .tunnel import TunnelKeeper, start_tunnel, tunnel_status, kill_tunnel, TUNNEL_URL_FILE  # noqa: F401
