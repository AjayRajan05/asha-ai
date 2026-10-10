# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optimizer semantic-similarity gate (MiniLM cosine vs calibrated floor).

Similarity **below** ``optimizer_similarity.safe_max`` triggers revert in
:class:`asha.core.safety_constraints.SafetyConstraints`.  Heavy deps load lazily
on first :func:`compute_similarity` call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

from asha.core.ml.calibration import ThresholdBands, get_bands, load_thresholds

SimilarityMode = Literal["auto", "embedding", "jaccard"]


@dataclass(frozen=True)
class SimilarityResult:
    """Outcome of comparing original vs optimized prompt text."""

    score: float
    floor: float
    method: str
    is_safe: bool


def optimizer_similarity_bands(
    *,
    thresholds: Optional[dict] = None,
    method: str = "embedding",
) -> ThresholdBands:
    """Load similarity bands for the active scoring method.

    Jaccard (Base) → ``optimizer_similarity_lite``;
    MiniLM embedding → ``optimizer_similarity``.
    """
    key = "optimizer_similarity_lite" if method == "jaccard" else "optimizer_similarity"
    try:
        bands = get_bands(key, thresholds=thresholds or load_thresholds())
        if bands.higher_is_safer:
            return bands
        return ThresholdBands(
            safe_max=bands.safe_max,
            block_min=bands.block_min,
            warn_max=bands.warn_max,
            target_fpr=bands.target_fpr,
            source=bands.source,
            higher_is_safer=True,
        )
    except Exception:
        return ThresholdBands(
            safe_max=0.70,
            block_min=0.70,
            target_fpr=0.02,
            source="builtin",
            higher_is_safer=True,
        )


def jaccard_similarity(original: str, optimized: str) -> float:
    """Token Jaccard overlap: lite fallback when embeddings are unavailable."""
    original_tokens = set(original.lower().split())
    optimized_tokens = set(optimized.lower().split())
    if not original_tokens and not optimized_tokens:
        return 1.0
    intersection = len(original_tokens & optimized_tokens)
    union = len(original_tokens | optimized_tokens)
    if union == 0:
        return 1.0
    return intersection / union


def embedding_similarity(
    original: str,
    optimized: str,
    *,
    allow_hash_fallback: bool = False,
) -> float:
    """MiniLM cosine similarity (hash fallback only for offline tests)."""
    from asha.core.ml.embeddings import get_encoder

    enc = get_encoder(allow_hash_fallback=allow_hash_fallback)
    return enc.cosine_similarity(original, optimized)


def is_similarity_safe(score: float, bands: Optional[ThresholdBands] = None) -> bool:
    """True when similarity meets the calibrated floor (fail-closed on NaN)."""
    bands = bands or optimizer_similarity_bands(method="embedding")
    try:
        s = float(score)
    except (TypeError, ValueError):
        return False
    if s != s:
        return False
    if bands.higher_is_safer:
        return s >= bands.safe_max
    return s <= bands.safe_max


def compute_similarity(
    original: str,
    optimized: str,
    *,
    mode: SimilarityMode = "auto",
    allow_hash_fallback: bool = False,
    bands: Optional[ThresholdBands] = None,
) -> SimilarityResult:
    """Score original vs optimized and apply the configured similarity floor."""
    method = "jaccard"
    score = jaccard_similarity(original, optimized)

    # Lite default: Jaccard is primary. Embedding is opt-in upgrade (hardened extra).
    if mode == "embedding":
        try:
            score = embedding_similarity(
                original,
                optimized,
                allow_hash_fallback=allow_hash_fallback,
            )
            method = "embedding"
        except ImportError:
            raise
    elif mode == "auto" and os.environ.get("ASHA_OPTIMIZER_EMBEDDING", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        try:
            score = embedding_similarity(
                original,
                optimized,
                allow_hash_fallback=allow_hash_fallback,
            )
            method = "embedding"
        except ImportError:
            pass

    active_bands = bands or optimizer_similarity_bands(method=method)
    return SimilarityResult(
        score=score,
        floor=active_bands.safe_max,
        method=method,
        is_safe=is_similarity_safe(score, active_bands),
    )
