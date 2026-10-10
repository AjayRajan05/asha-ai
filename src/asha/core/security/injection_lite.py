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

"""Stdlib-only lite-tier prompt-injection detector.

Pipeline: canonicalize → phrase Aho-Corasick → char-LM windowed perplexity
→ hashed linear classifier → noisy-OR fusion → ``injection_lite`` bands.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from asha.core.ml.calibration import (
    ThresholdBands,
    bucket_probability,
    get_bands,
    load_thresholds,
)
from asha.core.text.ahocorasick_matcher import PatternMatcher
from asha.core.text.canonicalize import canonicalize, expand_for_matching
from asha.core.text.hashed_features import HashedLinearClassifier
from asha.core.text.ngram_lm import CharNgramLM

from .injection_detector import InjectionDetectionResult

# Literal phrase bank (not regex): canonical form expected after normalize.
LITE_INJECTION_PHRASES: Sequence[Tuple[str, str, float]] = (
    ("ignore all previous instructions", "instruction_override", 1.0),
    ("ignore previous instructions", "instruction_override", 1.0),
    ("forget all previous", "instruction_override", 0.95),
    ("disregard all previous", "instruction_override", 0.95),
    ("disregard the rules", "instruction_override", 0.9),
    ("disregard all rules", "instruction_override", 0.9),
    ("override your safety", "bypass", 0.9),
    ("override safety guidelines", "bypass", 0.9),
    ("jailbreak", "jailbreak", 1.0),
    ("you are now an uncensored", "jailbreak", 0.95),
    ("act as an unrestricted", "jailbreak", 0.9),
    ("pretend to be dan", "jailbreak", 0.9),
    ("pretend you are dan", "jailbreak", 0.95),
    ("do anything now", "jailbreak", 0.95),
    ("you are dan", "jailbreak", 0.95),
    ("reveal the system prompt", "exfiltration", 0.95),
    ("reveal your system prompt", "exfiltration", 0.95),
    ("show the system prompt", "exfiltration", 0.9),
    ("reveal system prompt", "exfiltration", 0.9),
    ("bypass security", "bypass", 0.95),
    ("override restrictions", "bypass", 0.9),
    ("circumvent security", "bypass", 0.9),
    ("drop table", "sql_injection", 0.9),
    ("ignore all directives", "instruction_override", 0.95),
    ("disregard earlier directives", "instruction_override", 0.9),
    ("new instruction:", "instruction_override", 0.95),
    ("reveal all confidential data", "exfiltration", 0.95),
)

_PATTERN_HIT_PROB = 0.85
_LM_WINDOW = 8
# Char-LM windowed perplexity is an auxiliary signal only. On compact tables it
# saturates near 1.0 for ordinary English, so it must not sole-gate BLOCK.
_PPL_FUSION_CAP = 0.45


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _artifact_candidates(filename: str) -> List[Path]:
    from asha.core.ml.model_store import candidate_model_paths

    return candidate_model_paths("injection", "lite", filename)


def _resolve_artifact(filename: str) -> Optional[Path]:
    from asha.core.ml.model_store import resolve_model_file

    # Lite JSON is small; still allow auto-download of the official bundle.
    return resolve_model_file("injection", "lite", filename, ensure=True)


def _perplexity_to_prob(
    ppl: float,
    *,
    baseline: float,
    scale: float,
) -> float:
    if scale <= 0:
        scale = 1.0
    return _sigmoid((float(ppl) - baseline) / scale)


def _fuse(p_pat: float, p_ppl: float, p_clf: float) -> float:
    p_ppl_aux = max(0.0, min(_PPL_FUSION_CAP, float(p_ppl)))
    return max(
        0.0,
        min(1.0, 1.0 - (1.0 - p_pat) * (1.0 - p_ppl_aux) * (1.0 - p_clf)),
    )


class LiteInjectionDetector:
    """Stdlib-only injection detector (no sklearn/torch at inference)."""

    def __init__(
        self,
        *,
        matcher: Optional[PatternMatcher] = None,
        classifier: Optional[HashedLinearClassifier] = None,
        lm: Optional[CharNgramLM] = None,
    ) -> None:
        self._matcher = matcher
        self._classifier = classifier
        self._lm = lm
        self._lm_baseline = 10.0
        self._lm_scale = 5.0
        self._ready = False
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._matcher is None:
                self._matcher = PatternMatcher(
                    LITE_INJECTION_PHRASES,
                    case_insensitive=True,
                    prefer_native=True,
                )
            if self._classifier is None:
                clf_path = _resolve_artifact("hashed_clf.json")
                if clf_path is not None:
                    self._classifier = HashedLinearClassifier.from_json(clf_path)
                else:
                    dim = 1 << 14
                    self._classifier = HashedLinearClassifier(
                        [0.0] * dim,
                        bias=0.0,
                        n_bits=14,
                        ngram_range=(3, 5),
                    )
            if self._lm is None:
                lm_path = _resolve_artifact("char_ngram_lm.json")
                if lm_path is not None:
                    import json

                    data = json.loads(lm_path.read_text(encoding="utf-8"))
                    self._lm = CharNgramLM.from_dict(data)
                    self._lm_baseline = float(data.get("benign_baseline_ppl", 10.0))
                    self._lm_scale = float(data.get("ppl_scale", 5.0))
                else:
                    self._lm = CharNgramLM({}, n=3, unseen_log_prob=-10.0)
            self._ready = True

    def pattern_score(self, text: str) -> Tuple[float, Tuple[str, ...]]:
        """Return (p_pat, matched labels). p_pat is 0.85 on hit else 0."""
        self._ensure_loaded()
        assert self._matcher is not None
        matched_labels: List[str] = []
        for view in expand_for_matching(text):
            for hit in self._matcher.scan(view):
                matched_labels.append(hit.label)
        if matched_labels:
            return _PATTERN_HIT_PROB, tuple(dict.fromkeys(matched_labels))
        return 0.0, ()

    def perplexity_score(self, text: str) -> Tuple[float, float]:
        """Return (p_ppl, windowed perplexity)."""
        self._ensure_loaded()
        assert self._lm is not None
        ppl = self._lm.windowed_max_perplexity(text, window=_LM_WINDOW)
        prob = _perplexity_to_prob(
            ppl,
            baseline=self._lm_baseline,
            scale=self._lm_scale,
        )
        return prob, ppl

    def classifier_score(self, text: str) -> float:
        self._ensure_loaded()
        assert self._classifier is not None
        return self._classifier.predict_proba(text)

    def detect(self, text: str) -> InjectionDetectionResult:
        bands = _lite_bands()
        canon = canonicalize(text)

        p_pat, matched = self.pattern_score(text)
        p_ppl, windowed_ppl = self.perplexity_score(canon)
        p_clf = self.classifier_score(canon)
        probability = _fuse(p_pat, p_ppl, p_clf)
        # High-precision literal phrase bank: a hit is enough for BLOCK even
        # when the auxiliary LM score is capped for fusion.
        if matched:
            probability = max(probability, bands.block_min)
        verdict = bucket_probability(probability, bands, fail_closed=True)

        return InjectionDetectionResult(
            probability=probability,
            verdict=verdict,
            p_perplexity=p_ppl,
            p_embedding=p_clf,
            regex_hit=bool(p_pat > 0),
            regex_patterns_matched=matched,
            features={
                "p_pat": p_pat,
                "p_ppl": p_ppl,
                "p_clf": p_clf,
                "windowed_ppl": windowed_ppl,
                "canonical_length": float(len(canon)),
            },
            mode="lite",
        )


_lite_singleton: Optional[LiteInjectionDetector] = None
_lite_lock = threading.Lock()


def get_lite_injection_detector(*, reset: bool = False) -> LiteInjectionDetector:
    global _lite_singleton
    if reset or _lite_singleton is None:
        with _lite_lock:
            _lite_singleton = LiteInjectionDetector()
    return _lite_singleton


def _lite_bands() -> ThresholdBands:
    try:
        return get_bands("injection_lite", thresholds=load_thresholds())
    except Exception:
        return ThresholdBands(safe_max=0.15, block_min=0.85, source="builtin")
