"""faster-whisper/CTranslate2 has no AMD GPU path: whisper.cpp is the only ASR engine."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(r"faster[_-]whisper|ctranslate2|WhisperModel|BatchedInferencePipeline", re.I)
# face_os is a separate system (its Colab notebook text is owned there); .claude holds
# the register that documents this removal; tests may name the engine they assert is gone.
EXCLUDED = ("face_os", ".claude", "docs", "tests/")


def _tracked(*patterns):
    out = subprocess.run(["git", "ls-files", *patterns], cwd=ROOT, capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines() if not f.startswith(EXCLUDED) and f not in EXCLUDED]


def test_no_python_source_references_faster_whisper():
    offenders = [f for f in _tracked("*.py")
                 if FORBIDDEN.search((ROOT / f).read_text(encoding="utf-8", errors="ignore"))]
    assert offenders == []


def test_requirements_do_not_install_faster_whisper_or_ctranslate2():
    for name in _tracked("requirements*.txt"):
        assert not FORBIDDEN.search((ROOT / name).read_text(encoding="utf-8")), name


def test_config_has_no_dead_faster_whisper_keys():
    from utils.config import load_config
    t = load_config()["transcription"]
    for dead in ("device", "compute_type", "batch_size", "vad_filter"):
        assert dead not in t, f"transcription.{dead} is a faster-whisper option nothing reads"
    assert t["model"] == "large-v3-turbo"
    assert t["alignment_model"] == "large-v3-turbo"
    assert t["alignment_fallback_model"] != t["alignment_model"]


def test_configured_models_exist_on_disk():
    from utils.config import load_config
    from utils import whisper_cpp
    t = load_config()["transcription"]
    for key in ("model", "alignment_model", "alignment_fallback_model"):
        try:
            whisper_cpp.resolve_model(t[key])
        except whisper_cpp.WhisperCppUnavailable as exc:
            pytest.fail(f"transcription.{key}={t[key]!r}: {exc}")
