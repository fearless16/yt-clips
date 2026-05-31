"""
test_pipeline_overhaul.py — TDD verification suite for the 7-subsystem overhaul.

Tests the CROSS-CUTTING contracts between subsystems:
  1. Observability: run_phase emits structured records on success AND failure
  2. LLM orchestration: 429 cooldown, escalation, no degradation
  3. SEO: no generic fallback, Shorts preserved, retry queue works E2E
  4. Transcription: correction centralized across all sources
  5. Self-learning: retention/CTR activate scoring, pattern-key stable
  6. Upload: byte-safe, categoryId validated, bounded retries
  7. Face detection: batched, device-resolved, identity-locked

Run with: pytest tests/test_pipeline_overhaul.py -v
"""
import io
import json
import logging
import time
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OBSERVABILITY — run_phase contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunPhaseContract:
    """run_phase MUST emit start+done records with stage+duration_ms+run_id
    on success, and start+failed records with exc_info on failure."""

    def _capture_logger(self):
        from utils.logger import JsonStreamHandler
        buf = io.StringIO()
        handler = JsonStreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger(f"test_rp_{id(self)}")
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        return logger, buf

    def _entries(self, buf):
        return [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]

    def test_success_emits_start_and_ok_with_duration(self):
        from utils.logger import run_phase, new_run_id
        logger, buf = self._capture_logger()
        rid = new_run_id()
        with run_phase(logger, "test phase", "test_stage", run_id=rid) as ph:
            time.sleep(0.01)
            ph.set(items=42)
        entries = self._entries(buf)
        starts = [e for e in entries if e.get("status") == "start"]
        oks = [e for e in entries if e.get("status") == "ok"]
        assert len(starts) == 1 and starts[0]["stage"] == "test_stage"
        assert len(oks) == 1
        assert oks[0]["duration_ms"] >= 10
        assert oks[0]["run_id"] == rid
        assert oks[0]["metadata"]["items"] == 42

    def test_failure_emits_error_with_exc_info_and_reraises(self):
        from utils.logger import run_phase, new_run_id
        logger, buf = self._capture_logger()
        rid = new_run_id()
        with pytest.raises(ValueError):
            with run_phase(logger, "boom", "fail_stage", run_id=rid):
                raise ValueError("kaboom")
        entries = self._entries(buf)
        failed = [e for e in entries if e.get("status") == "failed"]
        assert len(failed) == 1
        assert failed[0]["error_type"] == "ValueError"
        assert "exc" in failed[0]
        assert failed[0]["run_id"] == rid

    def test_run_id_is_unique_per_call(self):
        from utils.logger import new_run_id
        ids = {new_run_id() for _ in range(100)}
        assert len(ids) == 100


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LLM ORCHESTRATION — health-aware racing + escalation
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMOrchestration:
    """The LLM client must honor 429 Retry-After, skip cooled providers in the
    racer, and return '' (escalation signal) on total failure."""

    def _reset(self):
        from utils.ai_client import AIClient
        with AIClient._shared_lock:
            AIClient._provider_failures.clear()
            AIClient._provider_cooldown_until.clear()
            AIClient._provider_token_buckets.clear()

    def test_429_sets_cooldown_from_retry_after(self):
        from utils.ai_client import AIClient
        self._reset()

        class FakeExc(Exception):
            status_code = 429
            response = MagicMock(headers={"retry-after": "60"})

        AIClient._note_provider_error("groq", FakeExc())
        assert AIClient._in_cooldown("groq")

    def test_racer_returns_empty_on_total_failure_not_generic(self):
        from utils.ai_client import AIClient
        self._reset()
        ai = AIClient()
        ai.opencode_api_key = None
        ai.groq_api_key = "k"
        ai.openrouter_api_key = ai.nvidia_api_key = None

        class FakeExc(Exception):
            status_code = 429
            response = MagicMock(headers={"retry-after": "120"})

        with patch.object(AIClient, "generate_groq", side_effect=FakeExc()):
            out = ai.generate_fastest_first("p", "s")
        assert out == ""  # escalation signal, NOT generic text

    def test_racer_records_success_and_clears_cooldown(self):
        from utils.ai_client import AIClient
        self._reset()
        ai = AIClient()
        ai.opencode_api_key = None
        ai.groq_api_key = "k"
        ai.openrouter_api_key = ai.nvidia_api_key = None
        with patch.object(AIClient, "generate_groq", return_value="OK"):
            out = ai.generate_fastest_first("p", "s")
        assert out == "OK"
        assert not AIClient._in_cooldown("groq")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SEO — escalate-not-degrade + Shorts fix + retry queue E2E
# ═══════════════════════════════════════════════════════════════════════════════

class TestSEOContract:
    """SEO must never produce generic fallback; Shorts keep short desc;
    retry_failed_seo closes the loop."""

    def test_no_generic_safe_defaults_in_tags(self):
        from automation.seo.seo import _enforce_limits
        out = _enforce_limits({
            "title": "T", "description": "D",
            "hashtags": ["#Shorts"],
            "search_terms": ["kohli cover drive"],
        }, fallback_terms=None)
        generic = {"cricket highlights", "cricket live match", "ipl match video",
                   "t20 cricket live", "best cricket moments"}
        assert not (set(out["search_terms"]) & generic)

    def test_shorts_preserves_llm_short_description(self):
        from automation.seo.seo import generate_clip_seo
        ai_json = json.dumps({
            "title": "KOHLI SIX! #Shorts",
            "description": "Kohli ne maara six! Subscribe! #Shorts",
            "search_terms": ["kohli six", "ipl live"],
            "hashtags": ["#Shorts", "#Kohli"],
        })
        with patch("utils.ai_client.AIClient.generate_fastest_first",
                   return_value=ai_json):
            res = generate_clip_seo("c1", "kohli six", "RCB vs CSK",
                                    is_shorts=True)
        assert "CHAPTERS" not in res["description"]
        assert "Disclaimer:" not in res["description"]
        assert "Kohli ne maara" in res["description"]
        assert res["is_shorts"] is True

    def test_total_failure_raises_seo_generation_error(self):
        from automation.seo.seo import generate_clip_seo, SEOGenerationError
        with patch("utils.ai_client.AIClient.generate_fastest_first",
                   return_value=""), \
             patch("utils.ai_client.AIClient.generate_text",
                   side_effect=RuntimeError("down")):
            with pytest.raises(SEOGenerationError):
                generate_clip_seo("c2", "kohli six", "RCB vs CSK")

    def test_retry_queue_e2e(self, tmp_path):
        from automation.seo.seo import (generate_seo_for_exported_clip,
                                         retry_failed_seo)
        # Force failure -> marker written
        with patch("utils.ai_client.AIClient.generate_fastest_first",
                   return_value=""), \
             patch("utils.ai_client.AIClient.generate_text",
                   side_effect=RuntimeError("down")):
            generate_seo_for_exported_clip("clipQ", "kohli six",
                                           str(tmp_path), video_title="RCB vs CSK")
        assert (tmp_path / "clipQ_seo_failed.json").exists()
        assert not (tmp_path / "clipQ_metadata.json").exists()
        # Marker is self-contained
        ctx = json.loads((tmp_path / "clipQ_seo_failed.json").read_text())
        assert ctx["transcript"] == "kohli six"

        # Now recover
        good = json.dumps({"title": "T", "description": "D",
                           "search_terms": ["a b"], "hashtags": ["#S"]})
        with patch("utils.ai_client.AIClient.generate_fastest_first",
                   return_value=good):
            r = retry_failed_seo(str(tmp_path))
        assert r["recovered"] == 1
        assert (tmp_path / "clipQ_metadata.json").exists()
        assert not (tmp_path / "clipQ_seo_failed.json").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRANSCRIPTION — centralized correction across all sources
# ═══════════════════════════════════════════════════════════════════════════════

class TestTranscriptionCorrection:
    """Spelling correction must run on ALL transcript sources (api/vtt/whisper)
    and must NOT corrupt common English words."""

    def test_correct_segments_fixes_mishearings(self):
        from utils.transcript_postproc import correct_segments
        segs = [{"text": "coaly and bumra played well"}]
        segs, n = correct_segments(segs)
        assert "Kohli" in segs[0]["text"]
        assert "Bumrah" in segs[0]["text"]
        assert n >= 2

    def test_sky_and_head_not_corrupted(self):
        from utils.transcript_postproc import correct_text
        text, n = correct_text("ball went over the head into the sky")
        assert "Travis" not in text
        assert "SKY" not in text
        assert n == 0

    def test_llm_validation_rejects_bad_output(self):
        from utils.transcript_postproc import validate_and_apply_llm_corrections
        segs = [{"text": "short"}, {"text": "hello"}]
        bad_map = {
            0: "",              # empty -> reject
            1: "x" * 9999,     # way too long -> reject
            99: "out of range", # invalid index -> reject
        }
        segs, applied, rejected = validate_and_apply_llm_corrections(segs, bad_map)
        assert applied == 0
        assert rejected == 3
        assert segs[0]["text"] == "short"  # unchanged

    def test_api_transcript_gets_corrected(self):
        """automation/transcript.py fetch() must correct api/vtt segments."""
        from automation.transcript import fetch
        from automation._cache import TRANSCRIPT_CACHE
        TRANSCRIPT_CACHE.clear()
        fake_segs = [{"start": 0, "end": 5, "text": "coaly hit six"}]
        with patch("automation.transcript._fetch_via_api",
                   return_value={"segments": fake_segs, "language": "en",
                                 "source": "api"}):
            result = fetch("https://www.youtube.com/watch?v=12345678901")
        assert "Kohli" in result["segments"][0]["text"]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SELF-LEARNING — retention/CTR activates scoring
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelfLearning:
    """Real Analytics signals (retention, CTR) must activate the previously-dead
    scoring branches and produce different scores than view-only."""

    def test_retention_ctr_changes_score(self):
        from automation.seo.seo_learner import SEOLearner
        learner = SEOLearner.__new__(SEOLearner)
        # View-only
        score_basic = learner._calculate_performance_score(
            {"viewCount": 1000, "likeCount": 50, "commentCount": 10})
        # With retention + CTR
        score_rich = learner._calculate_performance_score(
            {"viewCount": 1000, "likeCount": 50, "commentCount": 10,
             "retention": 0.7, "ctr": 0.08})
        # They must differ (the branches activate)
        assert score_rich != score_basic

    def test_pattern_key_excludes_numerics(self):
        from automation.seo.seo_learner import _stable_pattern_key
        f1 = {"has_pipe_format": True, "title_length": 45, "has_emoji": False}
        f2 = {"has_pipe_format": True, "title_length": 99, "has_emoji": False}
        # Keys should be IDENTICAL despite different title_length (numeric excluded)
        assert _stable_pattern_key(f1) == _stable_pattern_key(f2)

    def test_best_model_prefers_real_over_benchmark(self):
        from automation.seo.seo_learner import SEOLearner
        learner = SEOLearner.__new__(SEOLearner)
        learner.learned_insights = {
            "model_performance": {
                "groq/llama": {"count": 5, "avg_score": 0.8,
                               "provider": "groq", "model": "llama"},
            },
            "benchmark_history": [{
                "top_result": {"provider": "openrouter", "model": "gemini",
                               "score": 90},
            }],
            "current_best_provider": None, "current_best_model": None,
        }
        from automation.seo.seo_learner import MIN_CLIPS_FOR_PATTERN
        # Ensure count meets threshold
        learner.learned_insights["model_performance"]["groq/llama"]["count"] = \
            max(5, MIN_CLIPS_FOR_PATTERN)
        source = learner._recompute_best_model()
        assert source == "real"
        assert learner.learned_insights["current_best_provider"] == "groq"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. UPLOAD — byte-safe, categoryId validated, bounded retries
# ═══════════════════════════════════════════════════════════════════════════════

class TestUploadContract:
    """Upload must truncate by bytes (not chars), validate categoryId, and
    respect bounded retry + deadline config."""

    def test_truncate_bytes_multibyte_safe(self):
        from upload import _truncate_bytes
        # Hindi text: each char is 3 bytes in UTF-8
        text = "\u0915" * 2000  # 2000 chars = 6000 bytes
        result = _truncate_bytes(text, max_bytes=5000)
        assert len(result.encode("utf-8")) <= 5000
        # Must be valid UTF-8 (no mid-character split)
        result.encode("utf-8")

    def test_tag_limiter_accounts_for_quotes(self):
        from upload import _limit_youtube_tags
        # Multi-word tags get quoted (adds 2 bytes overhead each)
        tags = [f"cricket live match {i}" for i in range(50)]
        limited = _limit_youtube_tags(tags, max_chars=500)
        # Total chars including 2-byte quote overhead per multi-word tag
        total = sum(len(t) + 2 for t in limited if " " in t) + \
                sum(len(t) for t in limited if " " not in t) + \
                len(limited) - 1  # comma separators
        assert total <= 500

    def test_config_has_upload_reliability_tunables(self):
        from utils.config import load_config
        cfg = load_config()
        yt = cfg.get("youtube", {})
        assert yt.get("upload_chunk_size_mb") > 0
        assert yt.get("upload_max_retries") > 0
        assert yt.get("upload_deadline_seconds") > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FACE DETECTION — batched, device-resolved, identity-locked
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaceDetectionContract:
    """FaceDetector must resolve device, batch detections, and the premium
    analyzer must use detect_batch (not per-frame detect)."""

    def test_device_resolution_cpu_fallback(self):
        from premium_analyzer import FaceDetector
        assert FaceDetector._resolve_device("auto") == "cpu"  # no GPU in CI
        assert FaceDetector._resolve_device("cpu") == "cpu"

    def test_detect_batch_falls_back_per_frame_without_yolo(self):
        from premium_analyzer import FaceDetector
        fd = FaceDetector()  # backend=dnn (no ultralytics here)
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        results = fd.detect_batch(frames)
        assert len(results) == 3
        for xyxy, conf in results:
            assert isinstance(xyxy, np.ndarray)

    def test_identity_refs_load_from_photos_dir(self):
        from premium_analyzer import PremiumAnalyzer
        imgs = PremiumAnalyzer._load_identity_images(["photos", "expectation.png"])
        assert len(imgs) >= 1  # repo ships real reference photos

    def test_no_haar_cascade_in_codebase(self):
        """Regression guard: no Haar Cascade implementation anywhere."""
        repo = Path(__file__).resolve().parent.parent
        for py in repo.rglob("*.py"):
            if "face_os" in str(py) or "test_" in py.name or ".venv" in str(py):
                continue
            text = py.read_text(encoding="utf-8", errors="ignore")
            assert "haarcascade" not in text.lower(), f"Haar found in {py}"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CONFIG — all overhaul tunables present
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigTunables:
    """Every config key introduced by the overhaul must be present and have a
    sane default. This catches accidental key deletion or typos."""

    KEYS = [
        ("ai", "retry_base_delay_seconds"),
        ("ai", "retry_max_delay_seconds"),
        ("ai", "race_tier_timeout_seconds"),
        ("seo", "inject_viral_elements"),
        ("premium", "yolo_device"),
        ("premium", "yolo_batch_size"),
        ("premium", "identity_refs"),
        ("youtube", "upload_chunk_size_mb"),
        ("youtube", "upload_max_retries"),
        ("youtube", "upload_deadline_seconds"),
    ]

    def test_all_keys_present(self):
        from utils.config import load_config
        cfg = load_config()
        missing = []
        for path in self.KEYS:
            node = cfg
            for part in path:
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    missing.append(".".join(path))
                    break
        assert not missing, f"Missing config keys: {missing}"

    def test_seo_inject_viral_elements_default_off(self):
        from utils.config import load_config
        cfg = load_config()
        assert cfg["seo"]["inject_viral_elements"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# 9. DRY-RUN — validates the entire pipeline in <1s
# ═══════════════════════════════════════════════════════════════════════════════

class TestDryRun:
    """dry_run.py must complete successfully and pass its own validation."""

    def test_dry_run_full_passes(self):
        from dry_run import dry_run
        result = dry_run(
            url="https://youtu.be/4ylLhtICj1I",
            auto_upload=True, auto_schedule=True, auto_sync=True,
        )
        assert result["run_id"]
        assert len(result["exported"]) >= 1
        assert result["uploaded_count"] >= 1
        assert not result["failures"]
        assert result["validation"]["transcript_corrected"]
        assert result["validation"]["seo_marker_written"]
        assert result["validation"]["seo_recovered"] >= 1
        assert result["prompts_ok"], "Prompts module validation failed"

    def test_dry_run_validate_config_all_present(self):
        from dry_run import validate_config
        from automation.config import load as load_config
        checks = validate_config(load_config())
        missing = [k for k, ok, _ in checks if not ok]
        assert not missing, f"Missing: {missing}"



# ═══════════════════════════════════════════════════════════════════════════════
# 10. MODEL DIVERSITY — randomization + prefer_provider
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelDiversity:
    """The racer must shuffle models within tiers for diversity, and
    prefer_provider must boost that provider to front-of-tier."""

    def test_available_plan_shuffles_within_tiers(self):
        from utils.ai_client import AIClient
        ai = AIClient()
        ai.groq_api_key = "k"
        ai.openrouter_api_key = "k"
        ai.nvidia_api_key = "k"
        # Run multiple times and check we get different orderings
        orderings = set()
        for _ in range(20):
            plan = ai._available_plan()
            # Tier 1 has multiple groq models
            if plan:
                orderings.add(tuple(plan[0]))
        # With 3 groq models in tier 1, we should see >1 ordering across 20 calls
        assert len(orderings) > 1, "Models should be shuffled for diversity"

    def test_prefer_provider_boosts_to_front(self):
        from utils.ai_client import AIClient
        ai = AIClient()
        ai.groq_api_key = "k"
        ai.openrouter_api_key = "k"
        ai.nvidia_api_key = "k"
        # Tier 2 has nvidia + openrouter mixed
        # With prefer_provider="nvidia", nvidia should be first in tier 2
        found_nvidia_first = False
        for _ in range(10):
            plan = ai._available_plan(prefer_provider="nvidia")
            # Find the tier containing nvidia
            for tier in plan:
                nvidia_in_tier = [x for x in tier if x[0] == "nvidia"]
                if nvidia_in_tier:
                    assert tier[0][0] == "nvidia", \
                        f"nvidia should be first in its tier but got {tier[0]}"
                    found_nvidia_first = True
                    break
        assert found_nvidia_first

    def test_seo_does_not_mutate_shared_ai_singleton(self):
        """generate_clip_seo must NOT mutate ai._provider/_model (bug #1 fix)."""
        from automation.seo.seo import generate_clip_seo, ai as seo_ai
        original_provider = seo_ai._provider
        original_model = seo_ai._model
        good = json.dumps({"title": "T #Shorts", "description": "D #Shorts",
                           "search_terms": ["a b"], "hashtags": ["#Shorts"]})
        with patch("utils.ai_client.AIClient.generate_fastest_first",
                   return_value=good):
            generate_clip_seo("c1", "kohli six", "RCB vs CSK",
                              provider_override="openrouter",
                              model_override="some-model",
                              is_shorts=True)
        # Shared singleton must be unchanged
        assert seo_ai._provider == original_provider
        assert seo_ai._model == original_model
