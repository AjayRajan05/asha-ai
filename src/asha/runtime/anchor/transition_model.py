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

"""Markov transition model over tool *categories* for ChainGuard."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from asha.core.ml.calibration import (
    ThresholdBands,
    Verdict,
    get_bands,
    load_thresholds,
)

# Default smoothing for unseen transitions (aligned with generator script).
_DEFAULT_SMOOTHING = 1e-4


def _normalize_category_matrix(
    matrix: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Collapse tool-name keys into abstract ChainGuard categories when needed."""
    from asha.runtime.anchor.tool_capabilities import KNOWN_CATEGORIES, categorize_tool

    known = set(KNOWN_CATEGORIES)
    # Already abstract → keep as-is (re-normalize rows).
    sample_keys = {str(k) for k in matrix.keys() if str(k) != "<END>"}
    if sample_keys and sample_keys.issubset(known | {"<END>"}):
        return {
            str(a): {str(b): float(p) for b, p in (nexts or {}).items()}
            for a, nexts in matrix.items()
        }

    counts: Dict[str, Dict[str, float]] = {}
    for raw_a, nexts in matrix.items():
        a = str(raw_a)
        if a == "<END>":
            continue
        cat_a = a if a in known else categorize_tool(a)
        bucket = counts.setdefault(cat_a, {})
        for raw_b, p in (nexts or {}).items():
            b = str(raw_b)
            if b == "<END>":
                cat_b = "<END>"
            else:
                cat_b = b if b in known else categorize_tool(b)
            bucket[cat_b] = bucket.get(cat_b, 0.0) + float(p)

    normalized: Dict[str, Dict[str, float]] = {}
    for a, nexts in counts.items():
        total = sum(nexts.values()) or 1.0
        normalized[a] = {b: v / total for b, v in nexts.items()}
    return normalized


class TransitionModel:
    """Category→category transition probabilities from benign agent runs."""

    def __init__(
        self,
        matrix: Optional[Mapping[str, Mapping[str, float]]] = None,
        *,
        smoothing: float = _DEFAULT_SMOOTHING,
    ) -> None:
        self.matrix: Dict[str, Dict[str, float]] = {
            a: dict(nexts) for a, nexts in (matrix or {}).items()
        }
        self.smoothing = smoothing

    @classmethod
    def from_json(cls, path: Union[str, Path], **kwargs: Any) -> "TransitionModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("transitions"), dict):
            data = data["transitions"]
        if not isinstance(data, dict):
            raise ValueError(f"transition matrix must be an object: {path}")
        return cls(_normalize_category_matrix(data), **kwargs)

    def transition_probability(self, a: str, b: str) -> float:
        return float(self.matrix.get(a, {}).get(b, self.smoothing))

    def sequence_probability(self, categories: Sequence[str]) -> float:
        if len(categories) < 2:
            return 1.0
        log_p = 0.0
        for a, b in zip(categories, categories[1:]):
            p = max(self.transition_probability(a, b), self.smoothing)
            log_p += math.log(p)
        return math.exp(log_p)

    def min_transition_probability(self, categories: Sequence[str]) -> float:
        """Rarest single-step transition in the sequence (anomaly signal)."""
        if len(categories) < 2:
            return 1.0
        return min(
            self.transition_probability(a, b)
            for a, b in zip(categories, categories[1:])
        )

    def rarest_transition(
        self, categories: Sequence[str]
    ) -> Optional[Tuple[str, str, float]]:
        if len(categories) < 2:
            return None
        best: Optional[Tuple[str, str, float]] = None
        for a, b in zip(categories, categories[1:]):
            p = self.transition_probability(a, b)
            if best is None or p < best[2]:
                best = (a, b, p)
        return best


def _candidate_matrix_paths() -> List[Path]:
    from asha.core.ml.model_store import candidate_model_paths, ensure_models

    ensure_models()
    return candidate_model_paths("chain_guard", "transition_matrix.json")


_model_singleton: Optional[TransitionModel] = None
_model_lock = threading.Lock()


def get_transition_model(*, reset: bool = False) -> TransitionModel:
    """Load process-wide transition model (generator matrix by default)."""
    global _model_singleton
    if reset:
        with _model_lock:
            _model_singleton = None
    if _model_singleton is not None:
        return _model_singleton
    with _model_lock:
        for path in _candidate_matrix_paths():
            if path.is_file():
                _model_singleton = TransitionModel.from_json(path)
                return _model_singleton
        # Empty matrix → every transition uses smoothing (fail-closed rarity).
        _model_singleton = TransitionModel({})
        return _model_singleton


def score_category_sequence(
    categories: Sequence[str],
    *,
    model: Optional[TransitionModel] = None,
) -> Dict[str, Any]:
    """Score a category sequence; return rarity signals + verdict.

    Threshold semantics (``chain_transition`` in thresholds.yaml):
      transition_prob >= safe_max  → SAFE (common benign transition)
      block_min <= p < safe_max    → REVIEW
      p < block_min                → BLOCK (rarer than calibrated floor)
    """
    mdl = model or get_transition_model()
    min_p = mdl.min_transition_probability(categories)
    rarest = mdl.rarest_transition(categories)
    bands = _chain_bands()

    # Map rarity → "anomaly probability" for bucket_probability (higher = worse).
    # anomaly = 1 - min_p clipped, so rare transitions score high.
    anomaly = max(0.0, min(1.0, 1.0 - min_p))
    # Custom banding for rarity (invert relative to injection-style bands).
    if min_p >= bands.safe_max:
        verdict = Verdict.SAFE
    elif min_p >= bands.block_min:
        verdict = Verdict.REVIEW
    else:
        verdict = Verdict.BLOCK

    return {
        "min_transition_probability": min_p,
        "anomaly_score": anomaly,
        "verdict": verdict,
        "rarest": rarest,
        "sequence_probability": mdl.sequence_probability(categories),
        "categories": list(categories),
    }


def _chain_bands() -> ThresholdBands:
    try:
        bands = get_bands("chain_transition", thresholds=load_thresholds(reload=True))
        if not bands.higher_is_safer:
            # Force rarity semantics even if YAML omitted the flag.
            return ThresholdBands(
                safe_max=bands.safe_max,
                block_min=bands.block_min,
                target_fpr=bands.target_fpr,
                source=bands.source,
                higher_is_safer=True,
            )
        return bands
    except Exception:
        return ThresholdBands(
            safe_max=0.05,
            block_min=0.01,
            source="builtin",
            higher_is_safer=True,
        )
