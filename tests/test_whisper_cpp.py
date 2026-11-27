"""whisper.cpp is the only ASR engine. Contract tests for utils.whisper_cpp."""

import json
import subprocess
from pathlib import Path

import pytest

from utils import whisper_cpp


def _tok(text, p=0.9):
    return {"text": text, "p": p, "offsets": {"from": 0, "to": 0}}


def _entry(text, start_ms, end_ms, tokens=None):
    return {"offsets": {"from": start_ms, "to": end_ms}, "text": text,
            "tokens": tokens if tokens is not None else [_tok("[_BEG_]", 0.5), _tok(text, 0.9)]}


# ── parse_words ────────────────────────────────────────────────────────────

def test_each_word_segment_becomes_one_word_with_seconds():
    data = {"transcription": [_entry(" भाई", 0, 310), _entry(" हेलो", 310, 700)]}
    words = whisper_cpp.parse_words(data)
    assert [w["word"] for w in words] == ["भाई", "हेलो"]
    assert (words[0]["start"], words[0]["end"]) == (0.0, 0.31)
    assert all(type(w[k]) is float for w in words for k in ("start", "end", "probability"))


def test_word_probability_is_mean_of_real_token_probabilities_only():
    entry = _entry(" रेंज", 0, 400, [_tok("[_BEG_]", 0.01), _tok(" र", 0.8), _tok("ें", 0.6), _tok("ज", 1.0)])
    (word,) = whisper_cpp.parse_words({"transcription": [entry]})
    assert word["probability"] == pytest.approx(0.8)


def test_word_without_tokens_has_no_probability_key():
    """Absent confidence must stay absent so alignment_flags says 'unavailable'."""
    (word,) = whisper_cpp.parse_words({"transcription": [_entry(" आउट", 0, 100, [])]})
    assert "probability" not in word


def test_empty_and_whitespace_segments_are_skipped_not_invented():
    data = {"transcription": [_entry("", 0, 0), _entry("   ", 0, 10), _entry(" ठीक", 10, 50)]}
    assert [w["word"] for w in whisper_cpp.parse_words(data)] == ["ठीक"]


def test_raw_timings_are_never_repaired():
    """A zero-length or overlapping word is passed through; alignment holds it."""
    data = {"transcription": [_entry(" a", 100, 100), _entry(" b", 90, 200)]}
    words = whisper_cpp.parse_words(data)
    assert (words[0]["start"], words[0]["end"]) == (0.1, 0.1)
    assert (words[1]["start"], words[1]["end"]) == (0.09, 0.2)


def test_parser_tolerates_missing_transcription_key():
    assert whisper_cpp.parse_words({}) == []


# ── group_words ────────────────────────────────────────────────────────────

def _w(text, start, end):
    return {"start": start, "end": end, "word": text, "probability": 0.9}


def test_group_words_splits_segments_at_pauses_and_keeps_words():
    words = [_w("a", 0, 0.3), _w("b", 0.3, 0.6), _w("c", 2.0, 2.3)]
    segments = whisper_cpp.group_words(words, max_gap=0.5)
    assert [s["text"] for s in segments] == ["a b", "c"]
    assert segments[0]["start"] == 0 and segments[0]["end"] == 0.6
    assert segments[0]["words"] == words[:2]


def test_group_words_caps_segment_length():
    words = [_w(str(i), i * 0.1, i * 0.1 + 0.1) for i in range(30)]
    segments = whisper_cpp.group_words(words, max_gap=0.5, max_words=10)
    assert [len(s["words"]) for s in segments] == [10, 10, 10]


def test_group_words_empty():
    assert whisper_cpp.group_words([]) == []


# ── decode_output (moved from transcribe.py) ───────────────────────────────

def test_decode_output_survives_truncated_multibyte_without_single_byte_codecs():
    raw = b'{"transcription": [{"text": "\xe0\xa4\x95\xe0\xa4"}]}'
    data, warnings = whisper_cpp.decode_output(raw)
    assert data["transcription"][0]["text"] == "क"
    assert any("utf-8" in w.lower() for w in warnings)


def test_transcribe_module_still_exports_decode_whisper_output():
    from transcribe import decode_whisper_output
    assert decode_whisper_output is whisper_cpp.decode_output


# ── find_cli / resolve_model ───────────────────────────────────────────────

def test_find_cli_prefers_rocm_then_vulkan(tmp_path, monkeypatch):
    rocm, vulkan = tmp_path / "rocm.exe", tmp_path / "vulkan.exe"
    vulkan.write_text("x")
    monkeypatch.setattr(whisper_cpp, "CLI_CANDIDATES", [("rocm", rocm), ("vulkan", vulkan)])
    assert whisper_cpp.find_cli() == ("vulkan", vulkan)
    rocm.write_text("x")
    assert whisper_cpp.find_cli() == ("rocm", rocm)


def test_find_cli_missing_raises_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper_cpp, "CLI_CANDIDATES", [("rocm", tmp_path / "nope.exe")])
    with pytest.raises(whisper_cpp.WhisperCppUnavailable, match="whisper-cli"):
        whisper_cpp.find_cli()


def test_unavailable_is_a_whisper_cpp_error():
    assert issubclass(whisper_cpp.WhisperCppUnavailable, whisper_cpp.WhisperCppError)


def test_resolve_model_maps_names_to_ggml_files(tmp_path):
    (tmp_path / "ggml-large-v3-turbo.bin").write_bytes(b"m")
    assert whisper_cpp.resolve_model("large-v3-turbo", tmp_path) == tmp_path / "ggml-large-v3-turbo.bin"


def test_resolve_model_missing_names_file_and_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not shell out"))
    with pytest.raises(whisper_cpp.WhisperCppError, match="ggml-medium.bin"):
        whisper_cpp.resolve_model("medium", tmp_path)


# ── build_command ──────────────────────────────────────────────────────────

def test_build_command_requests_word_segments_and_no_previous_context():
    cmd = whisper_cpp.build_command(Path("cli.exe"), Path("m.bin"), Path("a.wav"), "out",
                                    language="hi", beam_size=3, max_context=0, threads=8, processors=1)
    assert cmd[0] == "cli.exe"
    pairs = {cmd[i]: cmd[i + 1] for i in range(1, len(cmd) - 1)}
    assert pairs["-m"] == "m.bin" and pairs["-f"] == "a.wav" and pairs["-of"] == "out"
    assert pairs["-l"] == "hi" and pairs["-bs"] == "3" and pairs["-mc"] == "0"
    assert pairs["-ml"] == "1" and pairs["-t"] == "8" and pairs["-p"] == "1"
    assert "-sow" in cmd and "-ojf" in cmd and "--no-prints" in cmd


# ── run_whisper_cli ────────────────────────────────────────────────────────

@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    cli = tmp_path / "whisper-cli.exe"
    cli.write_text("x")
    (tmp_path / "ggml-small.bin").write_bytes(b"m")
    monkeypatch.setattr(whisper_cpp, "CLI_CANDIDATES", [("rocm", cli)])
    monkeypatch.setattr(whisper_cpp, "MODELS_DIR", tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"w")
    return scratch, wav


def _fake_run(payload=None, returncode=0, stderr=""):
    seen = {}

    def run(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        if payload is not None:
            base = cmd[cmd.index("-of") + 1]
            Path(base + ".json").write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        return subprocess.CompletedProcess(cmd, returncode, "", stderr)
    run.seen = seen
    return run


def test_run_returns_grouped_segments_backend_and_language(fake_env, monkeypatch):
    scratch, wav = fake_env
    payload = {"result": {"language": "hi"},
               "transcription": [_entry(" भाई", 0, 300), _entry(" हेलो", 300, 700)]}
    run = _fake_run(payload)
    monkeypatch.setattr(whisper_cpp.subprocess, "run", run)
    result = whisper_cpp.run_whisper_cli(wav, model_name="small", language="hi", scratch=scratch)
    assert result["backend"] == "rocm" and result["language"] == "hi"
    assert result["segments"][0]["text"] == "भाई हेलो"
    assert result["segments"][0]["words"][1]["start"] == 0.3
    assert run.seen["cmd"][run.seen["cmd"].index("-m") + 1].endswith("ggml-small.bin")


def test_run_nonzero_exit_raises_with_stderr_tail(fake_env, monkeypatch):
    scratch, wav = fake_env
    monkeypatch.setattr(whisper_cpp.subprocess, "run", _fake_run(None, 3, "HIP error: out of memory"))
    with pytest.raises(whisper_cpp.WhisperCppError, match="exited 3.*out of memory"):
        whisper_cpp.run_whisper_cli(wav, model_name="small", language="hi", scratch=scratch)


def test_run_timeout_raises_whisper_cpp_error(fake_env, monkeypatch):
    scratch, wav = fake_env

    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 5)
    monkeypatch.setattr(whisper_cpp.subprocess, "run", boom)
    with pytest.raises(whisper_cpp.WhisperCppError, match="timed out"):
        whisper_cpp.run_whisper_cli(wav, model_name="small", language="hi", scratch=scratch, timeout=5)


def test_run_never_reads_stale_json(fake_env, monkeypatch):
    scratch, wav = fake_env
    (scratch / "whisper_rocm_out.json").write_text('{"transcription": [{"text": "STALE"}]}')
    monkeypatch.setattr(whisper_cpp.subprocess, "run", _fake_run(None, 0))
    with pytest.raises(whisper_cpp.WhisperCppError, match="no JSON output"):
        whisper_cpp.run_whisper_cli(wav, model_name="small", language="hi", scratch=scratch)


def test_run_unparseable_json_raises_whisper_cpp_error(fake_env, monkeypatch):
    scratch, wav = fake_env

    def run(cmd, **kwargs):
        Path(cmd[cmd.index("-of") + 1] + ".json").write_bytes(b"{not json")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(whisper_cpp.subprocess, "run", run)
    with pytest.raises(whisper_cpp.WhisperCppError, match="unreadable"):
        whisper_cpp.run_whisper_cli(wav, model_name="small", language="hi", scratch=scratch)


# ── padding artifacts (measured on real GPU output) ────────────────────────
# whisper.cpp pads audio to a 30 s window and emits tail words at/after the end of
# the supplied audio: zero-length at the clip end, last word stretched to the window end.

def test_parse_words_drops_words_starting_at_or_after_audio_end_and_clamps_the_last_end():
    data = {"transcription": [_entry(" a", 0, 500), _entry(" b", 13000, 30000),
                              _entry(" c", 14000, 14000), _entry(" d", 14000, 30000)]}
    words = whisper_cpp.parse_words(data, audio_seconds=14.0)
    assert [w["word"] for w in words] == ["a", "b"]
    assert words[1]["end"] == 14.0 and words[0]["end"] == 0.5


def test_parse_words_without_audio_length_changes_nothing():
    data = {"transcription": [_entry(" d", 14000, 30000)]}
    assert whisper_cpp.parse_words(data)[0]["end"] == 30.0


def test_run_measures_real_wav_length_before_trimming_padding(fake_env, monkeypatch):
    import wave
    scratch, wav = fake_env
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 32000)  # 2.0 s
    payload = {"transcription": [_entry(" real", 100, 900), _entry(" pad", 2000, 30000)]}
    monkeypatch.setattr(whisper_cpp.subprocess, "run", _fake_run(payload))
    result = whisper_cpp.run_whisper_cli(wav, model_name="small", language="hi", scratch=scratch)
    assert [w["word"] for s in result["segments"] for w in s["words"]] == ["real"]


def test_wav_seconds_is_none_for_non_wav(tmp_path):
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"not a wav")
    assert whisper_cpp.wav_seconds(bad) is None
