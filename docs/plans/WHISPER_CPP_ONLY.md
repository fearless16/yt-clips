# Plan: whisper.cpp is the only ASR engine (remove faster-whisper)

Branch: `feat/whisper-cpp-only` · Baseline: 1298 passed, 4 xfailed (2026-11-27)
Rule: TDD only. Every production change is preceded by a test that fails for the right reason.

## Why
faster-whisper/CTranslate2 has no AMD GPU path on Windows (no gfx1201 build). The R9700 runs
whisper.cpp ROCm (turbo, 24 s per 5 min of audio). Today faster-whisper is still used in two places:

| Place | Use | Today |
|---|---|---|
| `transcribe.py:311-490` | fallback after whisper.cpp fails | CPU/CUDA, dead on this machine |
| `automation/speech_alignment.py:323` `transcribe_range` | word timing for every shortlisted clip | CPU int8, about 11 min per run |

North-star link (guardrails §1): alignment is the step that produces word timings every
published Short depends on. Moving it to GPU removes about 10 min per run and the CPU
`medium` failures (2 of 3 ranges). Measure: wall time of `transcribe_range` on a real
10 s range, before and after (register entry).

## Findings that shape the design
1. **Existing whisper.cpp parser is wrong for captions.** It treats every *token* as a
   *word*. Real JSON (`temp/whisper_rocm_out.json`) has sub-word and byte-fallback tokens
   (`" क"`, `"र"`, `"त"`, `"े"`, empty-text byte pieces). Alignment needs real words with
   `probability`. New parser merges tokens into words: a token whose text starts with a space
   begins a new word; empty-text and continuation tokens extend the current word.
2. whisper.cpp reports no per-word probability; it reports per-token `p`. Word probability =
   mean of its token `p` values (tokens with real text). Needed by `alignment_flags`.
3. Raw intervals are never repaired (existing policy: alignment holds or retries the fallback
   model). The parser passes timings through unchanged.
4. Models on disk: `ggml-large-v3-turbo.bin`, `ggml-small.bin`. No `ggml-medium.bin`. The
   alignment fallback model therefore becomes `small` (no 1.5 GB download).
5. faster-whisper `chunk_length` / `condition_on_previous_text` have the whisper.cpp
   equivalent `-mc 0` (no previous-text context). `chunk_length=8` has no equivalent: ranges
   are at most about 20 s, so a single 30 s window is used.
6. `face_os/colab_*.py` install `faster-whisper` in Colab notebook text. **face_os is out of
   scope (guardrails §0). Not touched. Reported to the owner.**

## Design
New single owner `utils/whisper_cpp.py`:
- `find_cli()` → `(backend, path)`, ROCm before Vulkan, else raises `WhisperCppUnavailable`
  with an actionable message (no silent CPU fallback to a missing engine).
- `resolve_model(name)` → ggml path (map + on-disk preference; download kept as today).
- `decode_output(raw)` (moved from `transcribe.decode_whisper_output`, re-exported there).
- `parse_result(data)` → `[{start,end,text,words:[{start,end,word,probability}]}]`.
- `run_whisper_cli(wav, model, language, scratch, *, beam_size, max_context, threads, timeout)`
  → parsed result; raises `WhisperCppError` carrying returncode and stderr tail (A13).

`transcribe.py`: `transcribe()` uses it only; failure raises with the real reason (callers
already catch and log in `automation/transcript.py`). Remove `_log_vram` (nvidia-smi),
the faster-whisper block, and dead config keys.

`speech_alignment.transcribe_range`: ffmpeg → wav → `run_whisper_cli` → shift by
`context_start`. `_MODELS` removed. Cache identity `version` 2→3 and `decoding` updated so old
faster-whisper caches are never reused.

Config (A11, dead keys): delete `device`, `compute_type`, `batch_size`, `vad_filter`; wire
`beam_size` into `-bs`; `alignment_fallback_model: small`.

Cleanup: `requirements.txt`, `dry_run.py` stubs (patch `utils.whisper_cpp.run_whisper_cli`),
`ARCHITECTURE.md`, cancer-register, `.venv` uninstall of `faster-whisper` + `ctranslate2`
(proves nothing imports them).

## TDD slices (each: RED → GREEN → refactor, tests run after each)
| # | Test first | Then implement |
|---|---|---|
| 1 | `tests/test_whisper_cpp.py`: parser merges tokens into words, probability = mean token `p`, skips `[_...]` specials, empty-text tokens extend word, passes timings unchanged | `parse_result` |
| 2 | `find_cli` order ROCm>Vulkan; missing → `WhisperCppUnavailable`; `resolve_model` | `find_cli`, `resolve_model` |
| 3 | `run_whisper_cli` builds command (`-bs`, `-mc 0`, `-l`, `-ojf`), unique scratch, raises with stderr on non-zero/timeout/missing JSON, never reads stale JSON | `run_whisper_cli` |
| 4 | `transcribe()` happy path writes segments + `source: whisper-<backend>`; failure raises real reason; no faster_whisper import | rewrite `transcribe.py` |
| 5 | `transcribe_range` uses whisper.cpp, offsets by context start, floats, `max_context=0`, raises on no words | rewrite `transcribe_range`, drop `_MODELS` |
| 6 | Guard test: no `faster_whisper`/`ctranslate2` in repo `.py`, `requirements.txt`, config keys (face_os excluded) | cleanup, dry_run stubs, config |
| 7 | Update/replace affected existing tests (`test_transcribe`, `test_transcript_sync`, `test_dry_run_safety`, `test_automation` config test) | — |

## Verification
1. `pytest tests/ -v --ignore=tests/test_face_detect.py` (baseline 1298 passed).
2. `dry_run.py https://youtu.be/test --skip-download --skip-transcribe`.
3. `pip uninstall faster-whisper ctranslate2`, rerun 1 and 2.
4. Real-hardware smoke: `transcribe_range` on a 10 s range from a cached real source with the
   ROCm CLI; record wall time and word count in the register.

## Out of scope / risks
- `face_os/` Colab notebooks still mention faster-whisper (fence).
- whisper.cpp heuristic word timestamps can have zero-length words; alignment already holds and
  retries the fallback model. Whether `--dtw` improves this is measured in step 4 of
  Verification, not assumed (A7).
