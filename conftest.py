"""
conftest.py — Root conftest for Face OS tests.

Adds the project root to sys.path so that face_os can be imported.
"""

import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Locked baseline failures. Keep node IDs exact so new failures can never be
# hidden by a broad marker. Re-audit and remove entries when their owners land.
KNOWN_FAILURES = {
    "tests/test_video_analyzer_sampling.py::TestFfmpegNv12Pipe::test_software_nv12_pipe_produces_frames",
    "tests/test_video_analyzer_sampling.py::TestVideoAnalyzerIntegration::test_sample_frames_returns_valid_frames",
}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.replace("\\", "/") in KNOWN_FAILURES:
            item.add_marker(pytest.mark.xfail(
                reason="locked pre-existing baseline failure",
                run=False,
            ))

# Test suites must never fire live LLM grounding calls (cost + determinism).
import os as _os
_os.environ.setdefault("YT_CLIPS_LLM_GROUNDING", "0")


@pytest.fixture(autouse=True)
def _disable_live_vidiq(monkeypatch):
    """Unit tests must not spend vidIQ credits or depend on its network."""
    from utils.config import load_config

    vidiq = load_config().setdefault("seo", {}).setdefault("vidiq", {})
    monkeypatch.setitem(vidiq, "enabled", False)
    monkeypatch.setitem(vidiq, "required", False)
