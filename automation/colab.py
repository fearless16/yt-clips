"""colab.py — Backward-compatible re-exports.

env.py (env detection, GPU queries) public names re-exported here.
The remote Colab/Kaggle watcher + tunnel stack was removed: the worker daemon
exposed an unauthenticated RCE/file-read surface and is no longer used by the
local pipeline.
"""
from .env import is_colab, is_kaggle, gpu_info, gpu_count, setup  # noqa: F401
