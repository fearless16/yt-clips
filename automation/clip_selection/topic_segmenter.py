"""TopicSegmenter — zero-LLM topic boundary detection using embeddings.

Uses a local transformer model (sentence-transformers/all-MiniLM-L6-v2) to
compute per-segment embeddings, then detects topic boundaries by cosine
similarity drops between consecutive segments.

Design:
  - Lazy-loaded model (loaded on first call, cached globally)
  - Configurable threshold for boundary detection
  - Returns segments grouped by topic with start/end/text

Usage:
    from automation.clip_selection.topic_segmenter import TopicSegmenter
    segmenter = TopicSegmenter()
    topics = segmenter.segment(transcript_segments)
    # topics -> [{"topic_id": 0, "start": 0.0, "end": 45.0,
    #             "text": "...", "segments": [...], "embeddings": [...]}, ...]
"""

import copy
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import cosine

from utils.logger import get_logger

log = get_logger("topic_segmenter")

# ── Global model cache ──────────────────────────────────────────────

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from transformers import AutoModel, AutoTokenizer
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            log.info("Loading embedding model: %s ...", model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            _EMBEDDER = (tokenizer, model)
            log.info("Embedding model loaded")
        except Exception as e:
            log.warning("Failed to load embedding model: %s — topic segmentation disabled", e)
            return None
    return _EMBEDDER


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)


def _embed(texts: list[str], batch_size: int = 32) -> np.ndarray | None:
    embedder = _get_embedder()
    if embedder is None:
        return None
    tokenizer, model = embedder
    all_embeddings = []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            with torch.no_grad():
                output = model(**encoded)
            batch_embeddings = _mean_pooling(output, encoded["attention_mask"])
            all_embeddings.append(batch_embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0) if len(all_embeddings) > 1 else all_embeddings[0]
    except Exception as e:
        log.warning("Embedding failed: %s", e)
        return None


# ── TopicSegmenter ──────────────────────────────────────────────────

class TopicSegmenter:
    """Segment transcript into topics by embedding similarity."""

    def __init__(
        self,
        boundary_threshold: float = 0.25,
        min_segments_per_topic: int = 2,
    ):
        self.boundary_threshold = boundary_threshold
        self.min_segments_per_topic = min_segments_per_topic

    def segment(self, segments: list[dict]) -> list[dict[str, Any]]:
        """Group transcript segments into topics.

        Args:
            segments: List of transcript segments with start, end, text.

        Returns:
            List of topic dicts with topic_id, start, end, text, segments, embeddings.
        """
        if not segments:
            return []

        texts = [s.get("text", "") for s in segments]
        embeddings = _embed(texts)

        if embeddings is None:
            log.warning("Embeddings unavailable — returning single topic")
            full_text = " ".join(texts)
            return [{
                "topic_id": 0,
                "start": segments[0]["start"],
                "end": segments[-1]["end"],
                "text": full_text,
                "segments": copy.deepcopy(segments),
            }]

        boundaries = self._detect_boundaries(embeddings)
        topics = self._build_topics(segments, embeddings, boundaries)
        return topics

    def _detect_boundaries(self, embeddings: np.ndarray) -> list[int]:
        """Find topic boundaries using cosine similarity drops."""
        boundaries = [0]
        n = len(embeddings)
        for i in range(1, n):
            sim = 1.0 - cosine(embeddings[i - 1], embeddings[i])
            if sim < self.boundary_threshold:
                boundaries.append(i)
        if boundaries[-1] != n:
            boundaries.append(n)
        return boundaries

    def _build_topics(
        self, segments: list[dict], embeddings: np.ndarray, boundaries: list[int]
    ) -> list[dict[str, Any]]:
        """Build topic groups from boundary indices."""
        topics = []
        for t_idx in range(len(boundaries) - 1):
            start_i = boundaries[t_idx]
            end_i = boundaries[t_idx + 1]
            group_segs = segments[start_i:end_i]
            if len(group_segs) < self.min_segments_per_topic:
                continue
            topic = {
                "topic_id": t_idx,
                "start": group_segs[0]["start"],
                "end": group_segs[-1]["end"],
                "text": " ".join(s.get("text", "") for s in group_segs),
                "segments": copy.deepcopy(group_segs),
                "embeddings": embeddings[start_i:end_i],
            }
            topics.append(topic)

        if not topics:
            full_text = " ".join(s.get("text", "") for s in segments)
            topics.append({
                "topic_id": 0,
                "start": segments[0]["start"],
                "end": segments[-1]["end"],
                "text": full_text,
                "segments": copy.deepcopy(segments),
                "embeddings": embeddings,
            })

        return topics


# ── Convenience ─────────────────────────────────────────────────────

def segment_topics(segments: list[dict], **kwargs) -> list[dict[str, Any]]:
    """One-shot topic segmentation."""
    return TopicSegmenter(**kwargs).segment(segments)
